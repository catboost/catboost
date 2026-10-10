#include "../native/metal_pfound_pair_runtime.h"
struct Params {
    uint32_t Rows,Groups,Features,Candidates,Parents,Leaves,Permutations,Iterations;
    uint32_t LeafMethod,ScoreMethod,Bootstrap,SeedLow,SeedHigh,Absolute,GroupSampling,Passes;
    float L2,NonDiag,Temperature,Subsample;uint32_t Compact;
};
extern "C" int cbm_pfound_runtime_probe(const Params* input,uint64_t budget,
    const uint8_t* bins,const uint32_t* parentIds,const uint32_t* leafIds,const float* point,
    const float* target,const float* objectWeights,const uint32_t* groupOffsets,
    const uint32_t* sortedRows,const uint32_t* leafOffsets,const uint32_t* features,const uint32_t* borders,const uint8_t* types,
    uint32_t* shapes,uint32_t* documents,uint32_t* pairs,float* edges,float* matrices,
    float* scores,float* leaves,float* weights,uint64_t* allocated,double* seconds,char* error,uint32_t capacity) {
    @autoreleasepool {try {
        if(!input)throw std::runtime_error("Null PFound parameters");const auto& p=*input;
        if(!p.Passes || p.Passes>10 || p.Iterations>100 || p.LeafMethod>1 || p.ScoreMethod>1 || p.GroupSampling>1 || p.Compact>1)
            throw std::runtime_error("Invalid PFound probe iterations/method");
        auto device=MTLCreateSystemDefaultDevice();
        CBMPFoundPairRuntime runtime(device,p.Rows,p.Groups,groupOffsets,target,objectWeights,p.Permutations,.85f,p.Leaves,p.Candidates,8,budget,p.Compact,false);
        runtime.SetGroupSampling(p.GroupSampling);
        uint32_t pairCapacity=0,cap=std::min<uint32_t>(2ull*p.Rows/p.Groups+8,1023);
        for(uint32_t q=0;q<p.Groups;++q){uint32_t n=std::min(groupOffsets[q+1]-groupOffsets[q],cap);pairCapacity+=n*(n-1)/2;}
        pairCapacity=std::max(pairCapacity,1u);
        auto buffer=[&](const void* source,uint64_t size)->id<MTLBuffer>{
            auto value=source ? [device newBufferWithBytes:source length:size options:MTLResourceStorageModeShared]
                :[device newBufferWithLength:size options:MTLResourceStorageModeShared];
            if(!value)throw std::runtime_error("PFound probe buffer allocation failed");if(!source)std::memset(value.contents,0,size);return value;
        };
        auto data=buffer(bins,uint64_t(p.Rows)*p.Features),parent=buffer(parentIds,4ull*p.Rows),ids=buffer(leafIds,4ull*p.Rows);
        auto cursor=buffer(point,4ull*p.Rows),object=buffer(objectWeights,4ull*p.Rows),rows=buffer(sortedRows,4ull*p.Rows),off=buffer(leafOffsets,4ull*(p.Leaves+1));
        auto cf=buffer(features,4ull*p.Candidates),cb=buffer(borders,4ull*p.Candidates),ct=buffer(types,p.Candidates);
        auto values=buffer(nullptr,4ull*p.Leaves),leafWeight=buffer(nullptr,4ull*p.Leaves);
        auto queue=[device newCommandQueue];uint64_t dispatches=0;
        auto wait=[](id<MTLCommandBuffer> command){[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);};
        auto copy=[&](uint32_t index){const auto s=runtime.SampledShape();shapes[4*index]=s.Rows;shapes[4*index+1]=s.Groups;shapes[4*index+2]=s.Pairs;shapes[4*index+3]=runtime.PairCount();
            std::memcpy(documents+uint64_t(index)*p.Rows,runtime.SampledDocuments().contents,4ull*s.Rows);
            std::memcpy(pairs+uint64_t(index)*2*pairCapacity,runtime.GeneratedPairs().contents,8ull*s.Pairs);
            std::memcpy(edges+uint64_t(index)*4*pairCapacity,runtime.GeneratedEdges().contents,16ull*s.Pairs);
            std::memcpy(matrices+uint64_t(index)*pairCapacity,runtime.GeneratedMatrix().contents,4ull*s.Pairs);};
        for(uint32_t pass=0;pass<p.Passes;++pass){
            CBMBootstrapOptions boot={};boot.bootstrap_type=p.Bootstrap;boot.random_seed_low=p.SeedLow;boot.random_seed_high=p.SeedHigh;
            boot.bagging_temperature=p.Temperature;boot.subsample=p.Subsample;
            runtime.ClearStatus();std::memset(values.contents,0,4ull*p.Leaves);
            auto command=[queue commandBuffer];
            runtime.EncodeEdges(command,cursor,values,parent,p.Parents,false,&boot,p.Absolute+pass,&dispatches);
            copy(2*pass);
            runtime.EncodeCandidates(command,data,parent,cf,cb,ct,p.Features,p.Parents,p.ScoreMethod,p.L2,p.NonDiag,&dispatches);
            wait(command);runtime.CheckStatus();std::memcpy(scores+uint64_t(pass)*2*p.Candidates,runtime.CandidateScores().contents,8ull*p.Candidates);
            command=[queue commandBuffer];runtime.EncodeLeafLayout(command,ids,p.Leaves,&dispatches);copy(2*pass+1);
            runtime.EncodeLeafWeights(command,object,rows,off,leafWeight,p.Leaves,&dispatches);
            for(uint32_t step=0;step<p.Iterations;++step){
                runtime.EncodeEdges(command,cursor,values,ids,p.Leaves,true,nullptr,0,&dispatches);
                runtime.EncodeLeafProjection(command,p.Leaves,p.LeafMethod,&dispatches);
                runtime.EncodeLeafDirection(command,p.Leaves,p.L2,p.NonDiag,&dispatches);
                runtime.EncodeLeafUpdate(command,values,leafWeight,values,p.Leaves,1,&dispatches);
            }
            runtime.EncodeCenterSolvedPoint(command,values,p.Leaves,&dispatches);wait(command);runtime.CheckStatus();
            std::memcpy(leaves+uint64_t(pass)*p.Leaves,values.contents,4ull*p.Leaves);std::memcpy(weights+uint64_t(pass)*p.Leaves,leafWeight.contents,4ull*p.Leaves);
        }
        *allocated=runtime.AllocatedBytes();*seconds=runtime.TakeAuxiliaryGPUSeconds();
        if(runtime.HasObjectiveValue() || runtime.TakeAuxiliaryGPUSeconds()!=0)throw std::runtime_error("Invalid PFound metric or timing contract");
        return 0;
    }catch(const std::exception& e){if(error && capacity){std::strncpy(error,e.what(),capacity-1);error[capacity-1]=0;}return 1;}}
}
