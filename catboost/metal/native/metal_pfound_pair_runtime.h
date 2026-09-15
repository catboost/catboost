#pragma once
#include "metal_pairwise_matrix_runtime.h"
#include "metal_query_sampler_runtime.h"
#include "metal_pfound_pair_kernels.h"
#include "metal_pfound_sparse_kernels.h"
#include <mutex>
#include <vector>

static const char* CBMMetalPFoundBridgeSource = R"METAL(
kernel void GeneratePFoundSamplingKeys(device uint* keys [[buffer(0)]],
    constant BootstrapParams& p [[buffer(1)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    ulong seed = BootstrapSeedForItem(row, p);
    keys[row] = BootstrapNextUint(seed);
}
kernel void GatherPFoundObservations(const device uint* ids [[buffer(0)]],
    const device float* target [[buffer(1)]], const device float* weights [[buffer(2)]],
    const device float* point [[buffer(3)]], device float* sampled_target [[buffer(4)]],
    device float* sampled_weight [[buffer(5)]], device float* sampled_point [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]], constant PFoundPairParams& p [[buffer(8)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const uint id = ids[row];
    sampled_target[row] = target[id]; sampled_weight[row] = weights[id]; sampled_point[row] = point[id];
    if (!isfinite(target[id]) || !isfinite(weights[id]) || weights[id] < 0.0f || !isfinite(point[id]))
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}
)METAL";

// PFoundF's changing weak target shares the resident matrix search/leaf solver.
// Only compact shape/offset metadata crosses to the host for CUDA task packing.
// Random keys use Metal's documented per-item/iteration streams; PFound's inner
// permutation seed and arithmetic follow the CUDA kernel independently of them.
class CBMPFoundPairRuntime : public CBMPairwiseMatrixRuntime {
public:
    CBMPFoundPairRuntime(id<MTLDevice> device,uint32_t rows,uint32_t groups,const uint32_t* offsets,
        const float* targets,const float* weights,uint32_t permutations,float decay,
        uint32_t maxLeaves,uint32_t candidates,uint32_t tile=32,uint64_t budget=(1ull<<30),bool compact=true,bool sparse=true)
        : CBMPFoundPairRuntime(device,rows,groups,offsets,targets,weights,permutations,decay,
            maxLeaves,candidates,tile,budget,MakeGeometry(rows,groups,offsets,permutations,budget,compact && sparse),compact) {}

    uint64_t AllocatedBytes() const override { return CBMPairwiseMatrixRuntime::AllocatedBytes()+OwnBytes+Sampler->AllocatedBytes(); }
    bool HasObjectiveValue() const override { return false; }
    void SetTargetPoint(id<MTLBuffer> cursor,const CBMBootstrapOptions& bootstrap,
                        uint32_t iteration,uint32_t dataset) override {
        ValidateBuffer(cursor,4ull*Rows);Require(dataset<64,"PFound dataset index must be below 64");
        LastCursor=cursor;LastBootstrap=bootstrap;LastIteration=iteration;DatasetIndex=dataset;FixedReady=false;
    }
    double TakeAuxiliaryGPUSeconds() override { const double result=GPUSeconds;GPUSeconds=0;return result; }
    void ClearStatus() override { CBMPairwiseMatrixRuntime::ClearStatus();*static_cast<uint32_t*>(Status.contents)=0; }
    void CheckStatus() const override {
        CBMPairwiseMatrixRuntime::CheckStatus();
        Require(!*static_cast<const uint32_t*>(Status.contents),"Nonfinite generated PFound target");
    }
    void SetGroupSampling(bool group) { GroupSampling=group; }
    const CBMQuerySamplerRuntime::Shape& SampledShape() const { return Shape; }
    id<MTLBuffer> SampledDocuments() const { return Sampler->SampledDocuments(); }
    uint32_t GeneratedStorageCount() const { return StorageCount; }
    bool UsesSparseCapacity() const { return SparseCapacity; }
    id<MTLBuffer> GeneratedPairs() const { return Pairs; }
    id<MTLBuffer> GeneratedEdges() const { return Edges; }
    id<MTLBuffer> GeneratedMatrix() const { return Matrix; }

    void EncodeEdges(id<MTLCommandBuffer> command,id<MTLBuffer> cursor,id<MTLBuffer> values,
        id<MTLBuffer> ids,uint32_t leaves,bool shifted,const CBMBootstrapOptions* bootstrap=nullptr,
        uint32_t absoluteIteration=0,uint64_t* dispatches=nullptr,bool trial=false) override {
        if (bootstrap) {
            Require(!shifted && !trial,"A weak PFound target requires the original cursor");
            ValidateCommand(command);ValidateBuffer(cursor,4ull*Rows);
            Generate(cursor,*bootstrap,absoluteIteration,false,dispatches);
            LastCursor=cursor;LastBootstrap=*bootstrap;LastIteration=absoluteIteration;FixedReady=false;
        } else {
            Require(FixedReady,"Prepare the fixed PFound leaf target before differentiation");
            CBMPairwiseMatrixRuntime::EncodeEdges(command,cursor,values,ids,leaves,shifted,nullptr,0,dispatches,trial);
        }
        LastConsumer=command;
    }
    void EncodeCandidates(id<MTLCommandBuffer> command,id<MTLBuffer> bins,id<MTLBuffer> ids,
        id<MTLBuffer> features,id<MTLBuffer> borders,id<MTLBuffer> types,uint32_t featureCount,
        uint32_t parents,bool gradientScore,float l2,float nonDiag,uint64_t* dispatches=nullptr,
        uint32_t firstCandidate=0,uint32_t candidateCount=0) override {
        CBMPairwiseMatrixRuntime::EncodeCandidates(command,bins,ids,features,borders,types,featureCount,
            parents,gradientScore,l2,nonDiag,dispatches,firstCandidate,candidateCount);LastConsumer=command;
    }
    void EncodeLeafLayout(id<MTLCommandBuffer> command,id<MTLBuffer> ids,uint32_t leaves,uint64_t* dispatches=nullptr) override {
        ValidateCommand(command);Require(LastCursor!=nil,"Prepare the weak PFound target before its leaves");
        // CUDA FillPairsAndWeightsAtPoint constructs a default GPU bootstrap:
        // Bayesian, temperature=1, independent of the weak target's settings.
        CBMBootstrapOptions leaf=LastBootstrap;leaf.bootstrap_type=1;leaf.bagging_temperature=1;leaf.subsample=1;
        Generate(LastCursor,leaf,LastIteration,true,dispatches);FixedReady=true;
        CBMPairwiseMatrixRuntime::EncodeLeafLayout(command,ids,leaves,dispatches);
        LastConsumer=command;
    }
private:
    struct Geometry { uint32_t Pairs,DensePairs,SampleRows;bool Sparse;uint64_t ExtraBytes,OwnBytes; };
    struct Params { uint32_t Rows,Groups,Tasks,Pairs,Permutations,SeedLow,SeedHigh,Reserved;float Decay;uint32_t R0,R1,R2; };
    struct BootstrapParams { uint32_t Rows,Type,SeedLow,SeedHigh,Iteration,Stream,R0,R1;float Temperature,Subsample,Lambda,Noise; };
    struct ShaderSet { std::unordered_map<std::string,id<MTLComputePipelineState>> Pipelines; };
    id<MTLDevice> Device;id<MTLCommandQueue> Queue;id<MTLCommandBuffer> LastConsumer;
    id<MTLBuffer> Target,Weight,Keys,QueryMask,SampleTarget,SampleWeight,SamplePoint,Exponents;
    id<MTLBuffer> Tasks,Matrix,Contributions,Multipliers,Pairs,Edges,Status,LastCursor;
    std::unique_ptr<CBMQuerySamplerRuntime> Sampler;std::shared_ptr<ShaderSet> Shaders;
    uint32_t Rows,Groups,PairCapacity,DensePairCapacity,Permutations,LastIteration=0,DatasetIndex=0,StorageCount=0;float Decay;uint64_t OwnBytes=0;double GPUSeconds=0;
    bool GroupSampling=false,FixedReady=false,Compact=true,SparseCapacity=false;CBMBootstrapOptions LastBootstrap={};CBMQuerySamplerRuntime::Shape Shape={};
    static void Require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
    // Literal validation messages allocate only when a check fails.
    static void Require(bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    }
    static Geometry MakeGeometry(uint32_t rows,uint32_t groups,const uint32_t* offsets,uint32_t permutations,uint64_t budget,bool allowSparse) {
        Require(rows && rows<=(1u<<24) && groups && groups<=rows && offsets && offsets[0]==0 && offsets[groups]==rows,
            "Invalid PFound original query metadata");
        Require(permutations && permutations<=10000,"Invalid PFound permutation capacity");
        const uint32_t cap=std::min<uint32_t>(2ull*rows/groups+8,1023);uint64_t pairs=0,sampledRows=0;
        for(uint32_t q=0;q<groups;++q) {
            Require(offsets[q]<offsets[q+1] && offsets[q+1]<=rows,"PFound offsets must increase strictly");
            const uint64_t n=std::min(offsets[q+1]-offsets[q],cap);pairs+=n*(n-1)/2;sampledRows+=n;
            Require(pairs<0xffffffffull,"PFound original pair IDs exceed uint32 capacity; reduce query count or size");
        }
        const uint32_t densePairs=std::max<uint64_t>(pairs,1);
        const uint64_t slots=sampledRows*permutations;
        const bool sparse=allowSparse && pairs>slots;
        pairs=std::max<uint64_t>(sparse ? slots : pairs,1);
        Require(pairs<=(1u<<24),"PFound stored pair capacity exceeds 2^24; reduce rows or permutations");
        const uint64_t own=12ull*rows+16*sampledRows+12ull*groups+(sparse ? 36 : 32)*pairs+4,extra=own+CBMQuerySamplerRuntime::RequiredBytes(rows,groups);
        Require(budget<=(1ull<<30) && extra<budget,"PFound generation exceeds its 1 GiB workspace budget");
        return {uint32_t(pairs),densePairs,uint32_t(sampledRows),sparse,extra,own};
    }
    CBMPFoundPairRuntime(id<MTLDevice> device,uint32_t rows,uint32_t groups,const uint32_t* offsets,
        const float* targets,const float* weights,uint32_t permutations,float decay,
        uint32_t maxLeaves,uint32_t candidates,uint32_t tile,uint64_t budget,Geometry geometry,bool compact)
        : CBMPairwiseMatrixRuntime(device,rows,geometry.Pairs,nullptr,nullptr,nullptr,maxLeaves,candidates,tile,budget-geometry.ExtraBytes,true),
          Device(device),Rows(rows),Groups(groups),PairCapacity(geometry.Pairs),DensePairCapacity(geometry.DensePairs),Permutations(permutations),Decay(decay) {
        Compact=compact;SparseCapacity=geometry.Sparse;
        Require(targets && weights && permutations && permutations<=10000 && std::isfinite(decay) && decay>=0 && decay<=1,
            "Invalid PFound target, permutation count or decay");
        for(uint32_t row=0;row<rows;++row)Require(std::isfinite(targets[row]) && std::isfinite(weights[row]) && weights[row]>=0,
            "Invalid PFound observation");
        Sampler=std::make_unique<CBMQuerySamplerRuntime>(device,rows,groups,offsets,budget);
        Target=Buffer(targets,4ull*rows);Weight=Buffer(weights,4ull*rows);Keys=Buffer(nullptr,4ull*rows);QueryMask=Buffer(nullptr,4ull*groups);
        SampleTarget=Buffer(nullptr,4ull*geometry.SampleRows);SampleWeight=Buffer(nullptr,4ull*geometry.SampleRows);
        SamplePoint=Buffer(nullptr,4ull*geometry.SampleRows);Exponents=Buffer(nullptr,4ull*geometry.SampleRows);
        Tasks=Buffer(nullptr,8ull*groups);Matrix=Buffer(nullptr,4ull*PairCapacity);
        if(SparseCapacity)Contributions=Buffer(nullptr,4ull*PairCapacity);
        Multipliers=Buffer(nullptr,4ull*PairCapacity);
        Pairs=Buffer(nullptr,8ull*PairCapacity);Edges=Buffer(nullptr,16ull*PairCapacity);Status=Buffer(nullptr,4);
        Require(OwnBytes==geometry.OwnBytes && AllocatedBytes()<=budget,"PFound allocation accounting mismatch");
        Queue=[device newCommandQueue];Require(Queue!=nil,"PFound queue allocation failed");Shaders=GetShaders(device);
    }
    void Generate(id<MTLBuffer> cursor,const CBMBootstrapOptions& config,uint32_t iteration,bool fixed,uint64_t* dispatches) {
        Require(!LastConsumer || LastConsumer.status==MTLCommandBufferStatusCompleted,
            "Complete previous PFound target consumers before replacing generated pairs");
        Require(config.bootstrap_type<=2 && std::isfinite(config.bagging_temperature) && config.bagging_temperature>=0 &&
            std::isfinite(config.subsample) && config.subsample>0 && config.subsample<=1,
            "PFound supports No, Bayesian and Bernoulli bootstrap");
        // Four stream slots per dataset keep sampling, query masks and pair
        // weights separate. Dataset zero preserves the original P1 protocol.
        const uint32_t domain=(fixed ? 0x50464c00u : 0x50465700u)+4*DatasetIndex;
        auto sampling=[Queue commandBuffer];Require(sampling!=nil,"PFound sampling command allocation failed");
        BootstrapParams key={Rows,0,config.random_seed_low,config.random_seed_high,iteration,domain,0,0,0,1,0,0};
        Dispatch(sampling,"GeneratePFoundSamplingKeys",{Keys},key,Rows,false,dispatches);
        const bool sampleQueries=!fixed && GroupSampling && config.bootstrap_type==2;
        BootstrapParams mask={Groups,sampleQueries ? 2u : 0u,config.random_seed_low,config.random_seed_high,iteration,domain+1,0,0,0,config.subsample,0,0};
        Dispatch(sampling,"GenerateBootstrapWeights",{QueryMask,Weight},mask,Groups,false,dispatches);
        const float fraction=!fixed && !GroupSampling && config.bootstrap_type==2 ? config.subsample : 1.f;
        Sampler->Encode(sampling,Keys,QueryMask,fraction,0,DensePairCapacity,dispatches);Wait(sampling);Shape=Sampler->ReadShape();
        const uint32_t* offsets=static_cast<const uint32_t*>(Sampler->SampledOffsets().contents);
        uint32_t* tasks=static_cast<uint32_t*>(Tasks.contents);uint32_t taskCount=0;
        for(uint32_t q=0;q<Shape.Groups;) {
            const uint32_t limit=std::min(Shape.Rows,offsets[q]+1024);
            const uint32_t next=limit==Shape.Rows ? Shape.Groups : uint32_t(std::upper_bound(offsets,offsets+Shape.Groups+1,limit)-offsets-1);
            Require(next>q && offsets[next]-offsets[q]<=1024,"Invalid PFound sampled task geometry");
            tasks[2*taskCount]=q;tasks[2*taskCount+1]=next;++taskCount;q=next;
        }
        const bool sparse=SparseCapacity && uint64_t(Shape.Rows)*Permutations<Shape.Pairs;
        StorageCount=sparse ? Shape.Rows*Permutations : Shape.Pairs;
        Require(StorageCount<=PairCapacity,"PFound selected storage exceeds its reserved capacity");
        auto command=[Queue commandBuffer];Require(command!=nil,"PFound target command allocation failed");
        auto clear=[command blitCommandEncoder];Require(clear!=nil,"PFound clear allocation failed");
        [clear fillBuffer:Status range:NSMakeRange(0,4) value:0];
        [clear fillBuffer:Matrix range:NSMakeRange(0,4ull*std::max(StorageCount,1u)) value:0];[clear endEncoding];
        // Independent per-tree seed for the CUDA inner permutation generator.
        // The domain is recorded in the same explicit Metal RNG protocol as
        // document sampling and Bayesian weights, rather than hidden host RNG.
        const uint64_t seed=SeedForOracle(config,iteration,fixed,DatasetIndex);
        Params p={Shape.Rows,Shape.Groups,taskCount,Shape.Pairs,Permutations,uint32_t(seed),uint32_t(seed>>32),0,Decay,0,0,0};
        Dispatch(command,"GatherPFoundObservations",{Sampler->SampledDocuments(),Target,Weight,cursor,SampleTarget,SampleWeight,SamplePoint,Status},p,Shape.Rows,false,dispatches);
        Dispatch(command,"PreparePFoundPairApprox",{SamplePoint,Sampler->SampledOffsets(),Exponents},p,Shape.Groups,true,dispatches);
        BootstrapParams boot={StorageCount,config.bootstrap_type==1 ? 1u : 0u,config.random_seed_low,config.random_seed_high,iteration,domain+2,0,0,config.bagging_temperature,1,0,0};
        if(sparse) {
            if(boot.Type==1 && boot.Temperature>20.f) {
                BootstrapParams validation=boot;validation.Rows=Shape.Pairs;
                Dispatch(command,"ValidatePFoundDenseBootstrap",{Status},validation,Shape.Pairs,false,dispatches);
            }
            Dispatch(command,"GeneratePFoundSparseContributions",{Exponents,SampleTarget,Sampler->SampledQueryIds(),Sampler->SampledOffsets(),Tasks,Sampler->SampledPairOffsets(),GeneratedSortKeys(),Contributions,GeneratedSortIndices()},p,taskCount,true,dispatches);
            EncodeGeneratedSort(command,StorageCount,dispatches);
            Dispatch(command,"ReducePFoundSparseContributions",{GeneratedSortedKeys(),GeneratedSortedIndices(),Contributions,Matrix,Status},p,StorageCount,false,dispatches);
            Dispatch(command,"BootstrapPFoundSparsePairs",{GeneratedSortedKeys(),Multipliers},boot,StorageCount,false,dispatches);
            Dispatch(command,"FinalizePFoundSparsePairs",{GeneratedSortedKeys(),Matrix,Multipliers,SampleTarget,SampleWeight,Exponents,Sampler->SampledOffsets(),Sampler->SampledPairOffsets(),Sampler->SampledDocuments(),Pairs,Edges,Status},p,StorageCount,false,dispatches);
        } else {
            Dispatch(command,"GeneratePFoundPairWeights",{Exponents,SampleTarget,Sampler->SampledQueryIds(),Sampler->SampledOffsets(),Tasks,Sampler->SampledPairOffsets(),Matrix},p,taskCount,true,dispatches);
            Dispatch(command,"GenerateBootstrapWeights",{Multipliers,Weight},boot,StorageCount,false,dispatches);
            Dispatch(command,"FinalizePFoundPairs",{Matrix,Multipliers,SampleTarget,SampleWeight,Exponents,Sampler->SampledOffsets(),Sampler->SampledPairOffsets(),Sampler->SampledDocuments(),Pairs,Edges,Status},p,StorageCount,false,dispatches);
        }
        if (Compact) {
            EncodeGeneratedSelection(command,StorageCount,Matrix,Multipliers,dispatches);Wait(command);CheckStatus();
            const uint32_t count=ReadGeneratedSelectionCount();
            command=[Queue commandBuffer];Require(command!=nil,"PFound compact import command allocation failed");
            EncodeGeneratedTarget(command,count,Pairs,Edges,Target,fixed,dispatches,StorageCount);
        } else EncodeGeneratedTarget(command,StorageCount,Pairs,Edges,Target,fixed,dispatches);
        Wait(command);CheckStatus();
    }
public:
    static uint64_t SeedForOracle(const CBMBootstrapOptions& config,uint32_t iteration,bool fixed,uint32_t dataset=0) {
        auto mix=[](uint64_t x){x+=0x9e3779b97f4a7c15ull;x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;x=(x^(x>>27))*0x94d049bb133111ebull;return x^(x>>31);};
        const uint64_t datasetSeed=dataset ? mix((uint64_t(dataset)<<32)|0x50464453ull) : 0;
        return mix((uint64_t(config.random_seed_high)<<32|config.random_seed_low)^mix(uint64_t(iteration))^datasetSeed^
            (fixed ? 0x50464c454146ull : 0x50465745414bull));
    }
private:
    static std::shared_ptr<ShaderSet> GetShaders(id<MTLDevice> device) {
        static std::mutex mutex;static std::unordered_map<uint64_t,std::shared_ptr<ShaderSet>> cache;
        std::lock_guard<std::mutex> guard(mutex);auto& result=cache[device.registryID];if(result)return result;
        auto built=std::make_shared<ShaderSet>();NSError* error=nil;MTLCompileOptions* options=[MTLCompileOptions new];
        options.languageVersion=MTLLanguageVersion3_0;options.fastMathEnabled=NO;
        auto library=[device newLibraryWithSource:[NSString stringWithFormat:@"%s\n%s\n%s\n%s",CBMMetalBootstrapSource,CBMMetalPFoundPairSource,CBMMetalPFoundBridgeSource,CBMMetalPFoundSparseSource] options:options error:&error];
        Require(library!=nil,error ? error.localizedDescription.UTF8String : "PFound compilation failed");
        for(const char* name:{"GeneratePFoundSamplingKeys","GenerateBootstrapWeights","GatherPFoundObservations","PreparePFoundPairApprox","GeneratePFoundPairWeights","FinalizePFoundPairs","GeneratePFoundSparseContributions","ReducePFoundSparseContributions","BootstrapPFoundSparsePairs","FinalizePFoundSparsePairs","ValidatePFoundDenseBootstrap"}) {
            auto fn=[library newFunctionWithName:[NSString stringWithUTF8String:name]];Require(fn!=nil,"Missing PFound function");
            auto pipeline=[device newComputePipelineStateWithFunction:fn error:&error];Require(pipeline && pipeline.maxTotalThreadsPerThreadgroup>=256,"PFound pipeline allocation failed");
            built->Pipelines.emplace(name,pipeline);
        }
        result=built;return result;
    }
    id<MTLBuffer> Buffer(const void* source,uint64_t size) {
        Require(size && size<=Device.maxBufferLength,"PFound buffer exceeds device limits");
        auto result=source ? [Device newBufferWithBytes:source length:size options:MTLResourceStorageModeShared]
            :[Device newBufferWithLength:size options:MTLResourceStorageModeShared];
        Require(result!=nil,"PFound buffer allocation failed");if(!source)std::memset(result.contents,0,size);OwnBytes+=size;return result;
    }
    void ValidateCommand(id<MTLCommandBuffer> command) const {Require(command && command.device==Device && command.status==MTLCommandBufferStatusNotEnqueued && command.retainedReferences,"PFound needs an uncommitted retained command");}
    void ValidateBuffer(id<MTLBuffer> buffer,uint64_t size) const {Require(buffer && buffer.device==Device && buffer.length>=size,"Invalid PFound buffer");}
    void Wait(id<MTLCommandBuffer> command) {
        [command commit];[command waitUntilCompleted];Require(command.status==MTLCommandBufferStatusCompleted,command.error ? command.error.localizedDescription.UTF8String : "PFound GPU command failed");
        GPUSeconds+=std::max(0.,command.GPUEndTime-command.GPUStartTime);
    }
    template<class T>void Dispatch(id<MTLCommandBuffer> command,const char* name,std::initializer_list<id<MTLBuffer>> buffers,
        const T& params,uint32_t count,bool grouped,uint64_t* dispatches) {
        if(!count)return;auto encoder=[command computeCommandEncoder];Require(encoder!=nil,"PFound encoder allocation failed");
        [encoder setComputePipelineState:Shaders->Pipelines.at(name)];NSUInteger index=0;
        for(auto buffer:buffers)[encoder setBuffer:buffer offset:0 atIndex:index++];[encoder setBytes:&params length:sizeof(params) atIndex:index];
        [encoder dispatchThreadgroups:MTLSizeMake(grouped ? count : (uint64_t(count)+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];if(dispatches)++*dispatches;
    }
};
