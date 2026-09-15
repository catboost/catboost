#pragma once
#include "metal_pairwise_matrix_runtime.h"
#include "metal_query_cross_entropy_kernels.h"
#include "metal_qce_candidate_kernels.h"
#include <vector>
#include <mutex>
#include <array>

static const char* CBMMetalQCEBridge = R"METAL(
kernel void PrepareQCEQueries(device float* point [[buffer(0)]],
    const device uint* offsets [[buffer(1)]], const device float* scales [[buffer(2)]],
    const device float* multipliers [[buffer(3)]], device uchar* active [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]], constant QCEParams& p [[buffer(6)]],
    uint query [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    if (!lane) active[query]=uchar(multipliers[query]>0.0f);
    const uint row=offsets[query]+lane;
    if (row<offsets[query+1] && !isfinite(point[row]*scales[query])) {
        atomic_fetch_or_explicit(status,1u,memory_order_relaxed);point[row]=0.0f;
    }
}
kernel void ValidateQCEStatistics(const device float4* rows [[buffer(0)]],
    const device float4* groups [[buffer(1)]], device atomic_uint* status [[buffer(2)]],
    constant QCEParams& p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index<p.rows && (!all(isfinite(rows[index])) || any(rows[index].yzw<0.0f)))
        atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
    if (index<p.groups && (!all(isfinite(groups[index])) || any(groups[index].yzw<0.0f)))
        atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
}
kernel void ReduceQCELoss(const device float4* stats [[buffer(0)]],
    const device float* weights [[buffer(1)]], device float4* result [[buffer(2)]],
    constant QCEParams& p [[buffer(3)]], uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]]) {
    threadgroup float4 high_parts[256],low_parts[256];
    float4 high=float4(0.0f),low=float4(0.0f);
    for (uint row=group*256+tid;row<p.rows;row+=groups*256)
        QCECandidateAdd4(high,low,float4(stats[row].w,weights[row],0.0f,0.0f)/float(p.rows));
    high_parts[tid]=high;low_parts[tid]=low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint width=128;width;width>>=1) {
        if(tid<width) {
            high=high_parts[tid];low=low_parts[tid];
            QCECandidateAdd4(high,low,high_parts[tid+width]);QCECandidateAdd4(high,low,low_parts[tid+width]);
            high_parts[tid]=high;low_parts[tid]=low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!tid)result[group]=float4(high_parts[0].xy,low_parts[0].xy);
}
)METAL";

// Pipeline compilation is shared by training targets and evaluation pools.
struct CBMQueryCrossEntropyShaders {
    id<MTLLibrary> Library;
    std::unordered_map<std::string,id<MTLComputePipelineState>> Pipelines;
    static void Require(bool value,const std::string& message) { if(!value)throw std::runtime_error(message); }
    // Literal validation messages allocate only when a check fails.
    static void Require(bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    }
    static std::string Error(NSError* error) { return error ? error.localizedDescription.UTF8String : "QueryCrossEntropy compilation failed"; }
    explicit CBMQueryCrossEntropyShaders(id<MTLDevice> device,uint32_t queryThreads) {
        Require(queryThreads==32 || queryThreads==64 || queryThreads==128 || queryThreads==256,"Invalid QCE query threadgroup size");
        MTLCompileOptions* options=[MTLCompileOptions new];
        if(@available(macOS 13.0,*))options.languageVersion=MTLLanguageVersion3_0;
        else throw std::runtime_error("QueryCrossEntropy requires macOS 13 or newer");
        options.fastMathEnabled=NO;NSError* error=nil;
        NSString* source=[NSString stringWithFormat:@"#define CBM_QCE_THREADS %u\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s",queryThreads,CBMMetalQueryCrossEntropySource,
            CBMMetalLeafMatrixSource,CBMMetalQCECandidateSource,CBMMetalPairwiseScoreSource,CBMMetalBootstrapSource,
            CBMMetalPairwiseMatrixSource,CBMMetalPairMatrixBridge,CBMMetalQCEBridge];
        Library=[device newLibraryWithSource:source options:options error:&error];Require(Library!=nil,Error(error));
        for(const char* name:{"PreparePairMatrixPoint","GenerateBootstrapWeights","PrepareQCEQueries","QueryCrossEntropyStatistics",
            "ValidateQCEStatistics","CacheQCECandidateLeafSums","AccumulateQCECandidateMatrices","FinalizeQCECandidateMatrices",
            "RegularizePairwiseSplitMatrix","SolveLeafMatrix","ScorePairwiseSplitSolution","ExportSimplePairwiseLeaves","StorePairMatrixScores","SelectPairwiseSplitWinner",
            "ReducePairMatrixLeafWeights","RegularizeLeafMatrix","ApplyLeafMatrixRidge","UpdateLeafMatrixPoint","ReduceLeafMatrixDirectionalDot","ReduceQCELoss"}) {
            auto fn=[Library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(fn!=nil,std::string("Missing QueryCrossEntropy function ")+name);
            auto pipeline=[device newComputePipelineStateWithFunction:fn error:&error];
            Require(pipeline && pipeline.maxTotalThreadsPerThreadgroup>=256,Error(error));Pipelines.emplace(name,pipeline);
        }
    }
};
inline std::shared_ptr<CBMQueryCrossEntropyShaders> CBMGetQueryCrossEntropyShaders(id<MTLDevice> device,uint32_t queryThreads) {
    static std::mutex mutex;
    static std::unordered_map<uint64_t,std::array<std::shared_ptr<CBMQueryCrossEntropyShaders>,4>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const uint32_t index=queryThreads==32 ? 0 : queryThreads==64 ? 1 : queryThreads==128 ? 2 : 3;
    auto& value=cache[device.registryID][index];
    if(!value)value=std::make_shared<CBMQueryCrossEntropyShaders>(device,queryThreads);
    return value;
}

// Full point diagonal plus query Laplacian. Query/candidate tiling bounds the
// cache without enumerating O(query_size^2) document pairs. Leaf points keep
// their absolute mean and original document weights throughout Newton walks.
class CBMQueryCrossEntropyRuntime : public CBMFullMatrixRuntime {
public:
    CBMQueryCrossEntropyRuntime(id<MTLDevice> device,uint32_t rows,uint32_t groups,
        const float* targets,const float* weights,const uint32_t* offsets,const float* scales,
        float alpha,uint32_t maxLeaves,uint32_t candidates,uint32_t requestedTile=8,
        uint32_t requestedQueryTile=64,uint64_t budget=(1ull<<30),uint32_t minimumQueryThreads=32,bool metricOnly=false)
        : Device(device),Rows(rows),Groups(groups),MaxLeaves(maxLeaves),Candidates(candidates),Alpha(alpha),MetricOnly(metricOnly) {
        Require(device && rows && rows<=(1u<<24) && groups && groups<=rows && targets && weights && offsets && scales,
            "Invalid QueryCrossEntropy target metadata");
        Require(maxLeaves && maxLeaves<=256 && candidates && candidates<=(1u<<20) && requestedTile && requestedTile<=64 && requestedQueryTile,
            "Invalid QueryCrossEntropy leaf or tile capacity");
        Require(std::isfinite(alpha) && alpha>=0 && alpha<=1,"QueryCrossEntropy alpha must be in [0, 1]");
        Require(offsets[0]==0 && offsets[groups]==rows,"QueryCrossEntropy offsets must span all rows");
        Require(minimumQueryThreads==32 || minimumQueryThreads==64 || minimumQueryThreads==128 || minimumQueryThreads==256,
            "QueryCrossEntropy threadgroups must be 32, 64, 128 or 256 threads");
        QueryThreads=minimumQueryThreads;
        double mass=0;
        for(uint32_t q=0;q<groups;++q) {
            Require(offsets[q]<offsets[q+1] && offsets[q+1]<=rows && offsets[q+1]-offsets[q]<=256 && std::isfinite(scales[q]),
                "QueryCrossEntropy needs queries of 1..256 rows and finite scales");
            while(QueryThreads<offsets[q+1]-offsets[q])QueryThreads*=2;
        }
        for(uint32_t row=0;row<rows;++row) {
            Require(std::isfinite(targets[row]) && targets[row]>=0 && targets[row]<=1 && std::isfinite(weights[row]) && weights[row]>=0,
                "QueryCrossEntropy needs targets in [0, 1] and finite nonnegative weights");mass+=weights[row];
        }
        Require(mass>0 && mass<=std::numeric_limits<float>::max(),"QueryCrossEntropy needs positive finite weight mass");
        Tile=std::min(candidates,requestedTile);QueryTile=std::min(groups,requestedQueryTile);
        LossGroups=std::min<uint32_t>((rows+255)/256,4096);
        while(RequiredBytes(Tile,QueryTile)>budget && (Tile>1 || QueryTile>1)) {
            if(QueryTile>1)QueryTile=(QueryTile+1)/2;else --Tile;
        }
        Require(budget<=(1ull<<30) && RequiredBytes(Tile,QueryTile)<=budget,"QueryCrossEntropy exceeds its 1 GiB workspace budget");
        Targets=Buffer(targets,4ull*rows);OriginalWeights=Buffer(weights,4ull*rows);Point=Buffer(nullptr,4ull*rows);RowStats=Buffer(nullptr,16ull*rows);
        Offsets=Buffer(offsets,4ull*(groups+1));Scales=Buffer(scales,4ull*groups);GroupStats=Buffer(nullptr,16ull*groups);
        Singles=Buffer(nullptr,4ull*groups);Multipliers=Buffer(nullptr,4ull*groups);Active=Buffer(nullptr,groups);Dummy=Buffer(nullptr,8);
        const uint64_t values=uint64_t(Tile)*maxLeaves,cells=values*maxLeaves;
        if(!MetricOnly) {
            Cache=Buffer(nullptr,32ull*Tile*QueryTile*maxLeaves);
            GradientSums=Buffer(nullptr,8*values);HessianSums=Buffer(nullptr,8*cells);
            Gradient=Buffer(nullptr,4*values);Hessian=Buffer(nullptr,4*cells);Direction=Buffer(nullptr,4*values);Workspace=Buffer(nullptr,8*cells);
            TileScores=Buffer(nullptr,8ull*Tile);TileStatus=Buffer(nullptr,4ull*Tile);Scores=Buffer(nullptr,8ull*candidates);
            TrialStatus=Buffer(nullptr,4);SelectedIndex=Buffer(nullptr,4);SelectedScore=Buffer(nullptr,8);
        }
        Status=Buffer(nullptr,4);
        LossPartials=Buffer(nullptr,16ull*LossGroups);
        Require(Bytes==RequiredBytes(Tile,QueryTile),"QueryCrossEntropy allocation accounting mismatch");
        Shaders=CBMGetQueryCrossEntropyShaders(device,QueryThreads);
    }
    uint32_t CandidateTile() const { return Tile; }
    uint32_t QueryTileSize() const { return QueryTile; }
    id<MTLBuffer> Statistics() const { return RowStats; }
    id<MTLBuffer> QueryStatistics() const { return GroupStats; }
    id<MTLBuffer> QueryMask() const { return Active; }

    // Evaluation reuses exactly the CUDA target's scale and single-class
    // semantics. Only metric alpha can differ from the training loss.
    void EncodeMetric(id<MTLCommandBuffer> command,const float* point,uint64_t* dispatches=nullptr) {
        EncodeMetricWithAlpha(command,point,Alpha,dispatches);
    }
    void EncodeMetricWithAlpha(id<MTLCommandBuffer> command,const float* point,float alpha,uint64_t* dispatches=nullptr) {
        CheckCommand(command);Require(point,"QueryCrossEntropy metric predictions are required");
        Require(std::isfinite(alpha) && alpha>=0 && alpha<=1,"QueryCrossEntropy metric alpha must be in [0, 1]");
        for(uint32_t row=0;row<Rows;++row)Require(std::isfinite(point[row]),"QueryCrossEntropy metric predictions must be finite");
        std::memcpy(Point.contents,point,4ull*Rows);
        Dispatch(command,"GenerateBootstrapWeights",{Multipliers,Dummy},BootstrapParams{Groups,0,0,0,0,0,0,0,1,1,0,0},Groups,false,dispatches);
        const QueryParams p={Rows,Groups,1,alpha};
        Dispatch(command,"PrepareQCEQueries",{Point,Offsets,Scales,Multipliers,Active,Status},p,Groups,true,dispatches);
        Dispatch(command,"QueryCrossEntropyStatistics",{Targets,OriginalWeights,Point,Offsets,Scales,RowStats,GroupStats,Singles},p,Groups,true,dispatches);
        Dispatch(command,"ValidateQCEStatistics",{RowStats,GroupStats,Status},p,Rows,false,dispatches);
        Dispatch(command,"ReduceQCELoss",{RowStats,OriginalWeights,LossPartials},p,LossGroups,true,dispatches);
        HasStatistics=true;
    }

    // The shared controller calls this at each weak target and leaf trial.
    // Only the weak target has a bootstrap pointer; original queries always
    // supply leaf estimation and the objective used for accepting a step.
    void EncodeEdges(id<MTLCommandBuffer> command,id<MTLBuffer> cursor,id<MTLBuffer> values,
        id<MTLBuffer> ids,uint32_t leaves,bool shifted,const CBMBootstrapOptions* bootstrap=nullptr,
        uint32_t absoluteIteration=0,uint64_t* dispatches=nullptr,bool trial=false) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(cursor,4ull*Rows);CheckBuffer(ids,4ull*Rows);CheckBuffer(values,4ull*leaves);
        Require(!bootstrap || bootstrap->bootstrap_type==0 || bootstrap->bootstrap_type==2,"QueryCrossEntropy supports No or Bernoulli query bootstrap only");
        const float subsample=bootstrap ? bootstrap->subsample : 1.f;
        Require(std::isfinite(subsample) && subsample>0 && subsample<=1,"QueryCrossEntropy subsample must be in (0, 1]");
        auto state=trial ? TrialStatus : Status;
        Dispatch(command,"PreparePairMatrixPoint",{cursor,values,ids,Point,state},PointParams{Rows,Groups,leaves,uint32_t(shifted)},Rows,false,dispatches);
        const BootstrapParams b={Groups,bootstrap ? bootstrap->bootstrap_type : 0,
            bootstrap ? bootstrap->random_seed_low : 0,bootstrap ? bootstrap->random_seed_high : 0,
            absoluteIteration,0,0,0,1,subsample,0,0};
        Dispatch(command,"GenerateBootstrapWeights",{Multipliers,Dummy},b,Groups,false,dispatches);
        const QueryParams p={Rows,Groups,leaves,Alpha};
        Dispatch(command,"PrepareQCEQueries",{Point,Offsets,Scales,Multipliers,Active,state},p,Groups,true,dispatches);
        Dispatch(command,"QueryCrossEntropyStatistics",{Targets,OriginalWeights,Point,Offsets,Scales,RowStats,GroupStats,Singles},p,Groups,true,dispatches);
        Dispatch(command,"ValidateQCEStatistics",{RowStats,GroupStats,state},p,Rows,false,dispatches);
        HasStatistics=true;
    }
    void EncodeCandidates(id<MTLCommandBuffer> command,id<MTLBuffer> bins,id<MTLBuffer> ids,
        id<MTLBuffer> features,id<MTLBuffer> borders,id<MTLBuffer> types,uint32_t featureCount,
        uint32_t parents,bool gradientScore,float l2,float nonDiag,uint64_t* dispatches=nullptr,
        uint32_t firstCandidate=0,uint32_t candidateCount=0) {
        CheckCommand(command);CheckLeaves(2ull*parents);CheckRegularization(l2,nonDiag);
        Require(HasStatistics && !gradientScore && featureCount,"QueryCrossEntropy needs a Newton weak target; L2 score is unsupported like CUDA");
        CheckBuffer(bins,uint64_t(featureCount)*Rows);CheckBuffer(ids,4ull*Rows);
        CheckBuffer(features,4ull*Candidates);CheckBuffer(borders,4ull*Candidates);CheckBuffer(types,Candidates);
        Require(firstCandidate<Candidates,"First full-matrix candidate is out of range");
        if (!candidateCount) candidateCount=Candidates-firstCandidate;
        Require(candidateCount<=Candidates-firstCandidate,"Full-matrix candidate range exceeds capacity");
        const uint32_t candidateEnd=firstCandidate+candidateCount;
        LayoutLeaves=0;LayoutIds=nil;
        const LeafParams leaf={2*parents,1,0,0,l2,nonDiag,1e-20f,1};
        for(uint32_t first=NextActiveCandidate(firstCandidate,candidateEnd);first<candidateEnd;) {
            const uint32_t count=ActiveCandidateCount(first,std::min(Tile,candidateEnd-first));
            ClearProjection(command,TileStatus);
            Project(command,bins,ids,features,borders,types,CandidateParams{Rows,Groups,parents,count,featureCount,first,0,0},TileStatus,dispatches);
            Dispatch(command,"RegularizePairwiseSplitMatrix",{Hessian,Workspace},leaf,count,true,dispatches);
            Dispatch(command,"SolveLeafMatrix",{Workspace,Gradient,Direction,TileStatus},leaf,count,true,dispatches);
            Dispatch(command,"ScorePairwiseSplitSolution",{Hessian,Gradient,Direction,TileScores},leaf,count,true,dispatches);
            Dispatch(command,"StorePairMatrixScores",{TileScores,TileStatus,Scores,Status},TileParams{first,count,leaf.Leaves,0},count,false,dispatches);
            first=NextActiveCandidate(first+count,candidateEnd);
        }
    }
    void EncodeSimpleLeafValues(id<MTLCommandBuffer> command,id<MTLBuffer> values,
        id<MTLBuffer> weights,uint32_t leaves,bool oneHot,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);Require(leaves>=2,"Simple leaves require a split");
        CheckBuffer(values,4ull*leaves);CheckBuffer(weights,4ull*leaves);
        Require(values!=weights,"Simple value and weight outputs cannot alias");
        Dispatch(command,"ExportSimplePairwiseLeaves",{Direction,Hessian,values,weights},
            LeafParams{leaves,0,uint32_t(oneHot),0,0,0,0,1},leaves,false,dispatches);
    }
    void EncodeLeafLayout(id<MTLCommandBuffer> command,id<MTLBuffer> ids,uint32_t leaves,uint64_t* =nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(ids,4ull*Rows);LayoutIds=ids;LayoutLeaves=leaves;
    }
    void EncodeLeafProjection(id<MTLCommandBuffer> command,uint32_t leaves,bool gradientMethod,uint64_t* dispatches=nullptr) {
        CheckCommand(command);Require(HasStatistics && LayoutLeaves==leaves && LayoutIds && !gradientMethod,
            "Prepare the fixed QueryCrossEntropy Newton leaf layout first");
        ClearProjection(command,nil);
        Project(command,Dummy,LayoutIds,Dummy,Dummy,Dummy,CandidateParams{Rows,Groups,leaves,1,0,0,0,0},Status,dispatches);
    }
    void EncodeCenterSolvedPoint(id<MTLCommandBuffer> command,id<MTLBuffer> point,uint32_t leaves,uint64_t* =nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(point,4ull*leaves); // QCE keeps its free mean.
    }
    void EncodeLoss(id<MTLCommandBuffer> command,id<MTLBuffer> ids,uint32_t leaves,bool,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(ids,4ull*Rows);Require(HasStatistics,"Prepare QueryCrossEntropy statistics first");
        Dispatch(command,"ReduceQCELoss",{RowStats,OriginalWeights,LossPartials},QueryParams{Rows,Groups,leaves,Alpha},LossGroups,true,dispatches);
    }
    std::pair<double,double> ReadLoss() const {
        CheckStatus();const float* values=static_cast<const float*>(LossPartials.contents);double loss=0,mass=0;
        for(uint32_t i=0;i<LossGroups;++i) {
            for(uint32_t j=0;j<4;++j)Require(std::isfinite(values[4*i+j]),"Nonfinite QueryCrossEntropy loss reduction");
            loss+=double(values[4*i])+values[4*i+2];mass+=double(values[4*i+1])+values[4*i+3];
        }
        Require(loss>=0 && mass>0 && std::isfinite(loss) && std::isfinite(mass),"Invalid QueryCrossEntropy objective");
        return {loss*Rows,mass*Rows};
    }
    uint64_t AllocatedBytes() const { return Bytes; }
    id<MTLBuffer> CandidateScores() const { return Scores; }
    id<MTLBuffer> LeafGradient() const { return Gradient; }
    id<MTLBuffer> LeafHessian() const { return Hessian; }
    id<MTLBuffer> LeafDirection() const { return Direction; }
    void ClearStatus() { *static_cast<uint32_t*>(Status.contents)=0; }
    void CheckStatus() const {
        const uint32_t code=*static_cast<const uint32_t*>(Status.contents);
        Require(!code,"Invalid GPU QueryCrossEntropy matrix state (status "+std::to_string(code)+")");
    }
    void EncodeSelectWinner(id<MTLCommandBuffer> command,id<MTLBuffer> features,id<MTLBuffer> featureWeights,
        uint32_t featureCount,float previousScore,bool packedWeights=false,uint64_t* dispatches=nullptr) {
        CheckCommand(command);Require(featureCount && std::isfinite(previousScore),"Invalid QueryCrossEntropy selection metadata");
        CheckBuffer(features,4ull*Candidates);CheckBuffer(featureWeights,(packedWeights?8ull:4ull)*featureCount);
        Dispatch(command,"SelectPairwiseSplitWinner",{Scores,features,featureWeights,SelectedIndex,SelectedScore,CandidateMask?CandidateMask:Scores},
            SelectionParams{Candidates,featureCount,uint32_t(packedWeights),uint32_t(CandidateMask!=nil),previousScore,0,0,0},1,true,dispatches);
    }
    Selected ReadWinner() const {
        CheckStatus();const uint32_t index=*static_cast<const uint32_t*>(SelectedIndex.contents);
        const float* score=static_cast<const float*>(SelectedScore.contents);
        Require(index<Candidates && std::isfinite(score[0]) && std::isfinite(score[1]),"No finite QueryCrossEntropy split winner");
        return {index,score[0],score[1]};
    }
    void EncodeLeafWeights(id<MTLCommandBuffer> command,id<MTLBuffer> originalWeights,id<MTLBuffer> rows,
        id<MTLBuffer> offsets,id<MTLBuffer> result,uint32_t leaves,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(originalWeights,4ull*Rows);CheckBuffer(rows,4ull*Rows);
        CheckBuffer(offsets,4ull*(leaves+1));CheckBuffer(result,4ull*leaves);
        Require(result!=originalWeights && result!=rows && result!=offsets,"QueryCrossEntropy leaf weight output cannot alias its inputs");
        Dispatch(command,"ReducePairMatrixLeafWeights",{originalWeights,rows,offsets,result,Status},PointParams{Rows,Groups,leaves,0},leaves,true,dispatches);
    }
    void EncodeLeafRidge(id<MTLCommandBuffer> command,id<MTLBuffer> point,uint32_t leaves,float l2,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckRegularization(l2,0);CheckBuffer(point,4ull*leaves);
        Require(point!=Gradient,"QueryCrossEntropy ridge point cannot alias the projected gradient");
        Dispatch(command,"ApplyLeafMatrixRidge",{Gradient,point,Status},
            LeafParams{leaves,1,0,0,l2,0,1e-20f,1},leaves,false,dispatches);
    }
    void EncodeLeafDirection(id<MTLCommandBuffer> command,uint32_t leaves,float l2,float nonDiag,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckRegularization(l2,nonDiag);
        const LeafParams p={leaves,1,0,0,l2,nonDiag,1e-20f,1};
        Dispatch(command,"RegularizeLeafMatrix",{Hessian,Workspace},p,leaves*leaves,false,dispatches);
        Dispatch(command,"SolveLeafMatrix",{Workspace,Gradient,Direction,Status},p,1,true,dispatches);
    }
    void EncodeLeafUpdate(id<MTLCommandBuffer> command,id<MTLBuffer> point,id<MTLBuffer> weights,
        id<MTLBuffer> updated,uint32_t leaves,float step,uint64_t* dispatches=nullptr,bool trial=false) {
        CheckCommand(command);CheckLeaves(leaves);Require(std::isfinite(step),"Invalid QueryCrossEntropy leaf step");
        for (auto buffer : {point,weights,updated}) CheckBuffer(buffer,4ull*leaves);
        Require(updated!=weights,"QueryCrossEntropy leaf updates cannot overwrite original weights");
        Dispatch(command,"UpdateLeafMatrixPoint",{point,Direction,weights,updated,trial ? TrialStatus : Status},
            LeafParams{leaves,1,0,0,0,0,1e-20f,step},leaves,false,dispatches);
    }
    void EncodeBeginTrial(id<MTLCommandBuffer> command) {
        CheckCommand(command);
        auto clear=[command blitCommandEncoder];Require(clear!=nil,"QueryCrossEntropy trial reset encoder allocation failed");
        [clear fillBuffer:TrialStatus range:NSMakeRange(0,4) value:0];[clear endEncoding];
    }
    void EncodeDirectionDot(id<MTLCommandBuffer> command,id<MTLBuffer> output,uint32_t leaves,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(output,8);
        Dispatch(command,"ReduceLeafMatrixDirectionalDot",{Gradient,Direction,output},
            LeafParams{leaves,1,0,0,0,0,0,1},1,true,dispatches);
    }
    double ReadTrialLoss() const {
        CheckStatus();
        if (*static_cast<const uint32_t*>(TrialStatus.contents)) return std::numeric_limits<double>::infinity();
        return ReadLoss().first;
    }

private:
    struct SelectionParams { uint32_t Candidates,Features,Packed,R0;float Previous,R1,R2,R3; };
    struct QueryParams { uint32_t Rows,Groups,Leaves;float Alpha; };
    struct CandidateParams { uint32_t Rows,Groups,Parents,Candidates,Features,First,GroupBegin,GroupCount; };
    struct LeafParams { uint32_t Leaves,Diagonal,R0,R1;float L2,NonDiag,MinWeight,Step; };
    struct PointParams { uint32_t Rows,Groups,Leaves,Shift; };
    struct TileParams { uint32_t First,Count,Leaves,SupportOnly; };
    struct BootstrapParams { uint32_t Rows,Type,SeedLow,SeedHigh,Iteration,Stream,R0,R1;float Temperature,Subsample,Lambda,Noise; };
    id<MTLDevice> Device;std::shared_ptr<CBMQueryCrossEntropyShaders> Shaders;
    id<MTLBuffer> Targets,OriginalWeights,Point,RowStats,Offsets,Scales,GroupStats,Singles,Multipliers,Active,Dummy;
    id<MTLBuffer> Cache,GradientSums,HessianSums,Gradient,Hessian,Direction,Workspace,TileScores,TileStatus,Scores,Status,TrialStatus,SelectedIndex,SelectedScore,LossPartials,LayoutIds;
    uint32_t Rows,Groups,MaxLeaves,Candidates,Tile,QueryTile,LossGroups,LayoutLeaves=0,QueryThreads=32;
    float Alpha;uint64_t Bytes=0;bool HasStatistics=false,MetricOnly=false;
    static void Require(bool value,const std::string& message) { if(!value)throw std::runtime_error(message); }
    // Literal validation messages allocate only when a check fails.
    static void Require(bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    }
    static std::string Error(NSError* error) { return error ? error.localizedDescription.UTF8String : "QueryCrossEntropy Metal compilation failed"; }
    static void CheckRegularization(float l2,float nonDiag) { Require(std::isfinite(l2) && l2>=0 && std::isfinite(nonDiag) && nonDiag>=0,"Invalid QueryCrossEntropy regularization"); }
    void CheckLeaves(uint64_t leaves) const { Require(!MetricOnly,"A metric workspace cannot train leaves");Require(leaves && leaves<=MaxLeaves,"QueryCrossEntropy leaf count exceeds capacity"); }
    void CheckCommand(id<MTLCommandBuffer> command) const { Require(command && command.device==Device && command.status==MTLCommandBufferStatusNotEnqueued && command.retainedReferences,"QueryCrossEntropy needs an uncommitted retained command on its device"); }
    void CheckBuffer(id<MTLBuffer> buffer,uint64_t bytes) const { Require(buffer && buffer.device==Device && buffer.length>=bytes,"QueryCrossEntropy buffer is missing, too short or on another device"); }
    uint64_t RequiredBytes(uint32_t tile,uint32_t queries) const {
        if(MetricOnly)return 28ull*Rows+33ull*Groups+16+16ull*LossGroups;
        const uint64_t values=uint64_t(tile)*MaxLeaves,cells=values*MaxLeaves;
        return 28ull*Rows+33ull*Groups+32+32ull*tile*queries*MaxLeaves+16*values+20*cells+12ull*tile+8ull*Candidates+16ull*LossGroups;
    }
    id<MTLBuffer> Buffer(const void* source,uint64_t bytes) {
        Require(bytes && bytes<=Device.maxBufferLength,"QueryCrossEntropy allocation exceeds device limits");
        auto value=source ? [Device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Require(value!=nil,"QueryCrossEntropy allocation failed");if(!source)std::memset(value.contents,0,bytes);Bytes+=bytes;return value;
    }
    void ClearProjection(id<MTLCommandBuffer> command,id<MTLBuffer> status) {
        auto clear=[command blitCommandEncoder];Require(clear!=nil,"QueryCrossEntropy reset encoder allocation failed");
        for(auto buffer:{GradientSums,HessianSums})[clear fillBuffer:buffer range:NSMakeRange(0,buffer.length) value:0];
        if(status)[clear fillBuffer:status range:NSMakeRange(0,status.length) value:0];[clear endEncoding];
    }
    void Project(id<MTLCommandBuffer> command,id<MTLBuffer> bins,id<MTLBuffer> ids,id<MTLBuffer> features,
        id<MTLBuffer> borders,id<MTLBuffer> types,CandidateParams p,id<MTLBuffer> status,uint64_t* dispatches) {
        const uint32_t leaves=p.Features ? 2*p.Parents : p.Parents,cells=p.Candidates*leaves*leaves;
        for(uint32_t first=0;first<Groups;first+=QueryTile) {
            p.GroupBegin=first;p.GroupCount=std::min(QueryTile,Groups-first);
            DispatchGrid(command,"CacheQCECandidateLeafSums",{RowStats,Offsets,bins,ids,features,borders,types,Active,Cache,status},
                p,MTLSizeMake(leaves,p.GroupCount,p.Candidates),true,dispatches);
            Dispatch(command,"AccumulateQCECandidateMatrices",{Cache,GroupStats,GradientSums,HessianSums,status},p,cells,false,dispatches);
        }
        Dispatch(command,"FinalizeQCECandidateMatrices",{GradientSums,HessianSums,Gradient,Hessian},p,cells,false,dispatches);
    }
    template<class T>
    void Dispatch(id<MTLCommandBuffer> command,const char* name,std::initializer_list<id<MTLBuffer>> buffers,
        const T& params,uint32_t count,bool grouped,uint64_t* dispatches) {
        DispatchGrid(command,name,buffers,params,MTLSizeMake(count,1,1),grouped,dispatches);
    }
    template<class T>
    void DispatchGrid(id<MTLCommandBuffer> command,const char* name,std::initializer_list<id<MTLBuffer>> buffers,
        const T& params,MTLSize grid,bool grouped,uint64_t* dispatches) {
        auto encoder=[command computeCommandEncoder];Require(encoder!=nil,"QueryCrossEntropy compute encoder allocation failed");
        [encoder setComputePipelineState:Shaders->Pipelines.at(name)];NSUInteger index=0;
        for(auto buffer:buffers)[encoder setBuffer:buffer offset:0 atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index];
        const bool queryKernel=std::strcmp(name,"QueryCrossEntropyStatistics")==0 || std::strcmp(name,"PrepareQCEQueries")==0 ||
            std::strcmp(name,"CacheQCECandidateLeafSums")==0;
        if(grouped)[encoder dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(queryKernel ? QueryThreads : 256,1,1)];
        else[encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];if(dispatches)++*dispatches;
    }
};
