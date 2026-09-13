#pragma once

// Resident full-matrix PairLogitPairwise target/search primitives. Caller owns
// command submission, tree topology, leaf walk and original document weights.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_pairwise_matrix_kernels.h"
#include "metal_pairwise_candidate_kernels.h"
#include "metal_leaf_matrix_kernels.h"
#include "metal_pairwise_score_kernels.h"
#include "metal_bootstrap_kernels.h"
#include "metal_sort.h"
#include "metal_trainer.h"
#include "metal_full_matrix_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <limits>
#include <utility>
#include <stdexcept>
#include <string>
#include <unordered_map>

static const char* CBMMetalPairMatrixBridge = R"METAL(
struct PairMatrixPointParams { uint rows, pairs, leaves, apply_shift; };
struct PairMatrixTileParams { uint first, count, leaves, support_only; };
kernel void BuildGeneratedPairSelection(const device float* raw [[buffer(0)]],
    const device float* multipliers [[buffer(1)]], device uint* keys [[buffer(2)]],
    device uint* indices [[buffer(3)]], constant PairMatrixPointParams& p [[buffer(4)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge>=p.pairs) return;
    keys[edge]=abs(raw[edge]*multipliers[edge])>1e-20f ? 0u : 1u;
    indices[edge]=edge;
}
kernel void CountGeneratedPairSelection(const device uint* sorted_keys [[buffer(0)]],
    device uint* count [[buffer(1)]], constant PairMatrixPointParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    if (row) return;
    uint begin=0,end=p.pairs;
    while(begin<end){const uint mid=begin+(end-begin)/2;if(sorted_keys[mid]==0)begin=mid+1;else end=mid;}
    count[0]=begin;
}
kernel void ImportGeneratedPairMatrix(const device uint2* pairs [[buffer(0)]],
    const device float4* source [[buffer(1)]], const device float* target [[buffer(2)]],
    const device uint* selected [[buffer(3)]], device uint* winners [[buffer(4)]], device uint* losers [[buffer(5)]],
    device float* weights [[buffer(6)]], device float4* edges [[buffer(7)]],
    device atomic_uint* status [[buffer(8)]], constant PairMatrixPointParams& p [[buffer(9)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge >= p.pairs) return;
    const uint input=p.leaves ? selected[edge] : edge;
    if (p.leaves && input>=p.leaves) {
        atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
        winners[edge]=0;losers[edge]=0;weights[edge]=0;edges[edge]=float4(0);return;
    }
    const uint2 pair = pairs[input];
    const float4 value = source[input];
    const bool valid = pair.x < p.rows && pair.y < p.rows && pair.x != pair.y &&
        all(isfinite(value)) && value.y >= 0.0f && value.z >= 0.0f;
    if (!valid) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        winners[edge] = 0; losers[edge] = 0; weights[edge] = 0; edges[edge] = float4(0);
        return;
    }
    const bool swap = p.apply_shift && target[pair.x] < target[pair.y];
    winners[edge] = swap ? pair.y : pair.x;
    losers[edge] = swap ? pair.x : pair.y;
    weights[edge] = value.z;
    edges[edge] = float4(swap ? -value.x : value.x, value.yzw);
}
kernel void PreparePairMatrixPoint(const device float* cursor [[buffer(0)]],
    const device float* values [[buffer(1)]], const device uint* ids [[buffer(2)]],
    device float* point [[buffer(3)]], device atomic_uint* status [[buffer(4)]],
    constant PairMatrixPointParams& p [[buffer(5)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    bool valid = !p.apply_shift || ids[row] < p.leaves;
    float value = cursor[row];
    if (valid && p.apply_shift) value += values[ids[row]];
    valid = valid && isfinite(value);
    if (!valid) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    point[row] = valid ? value : 0.0f;
}
kernel void SamplePairMatrixWeights(const device float* original [[buffer(0)]],
    const device float* multipliers [[buffer(1)]], device float* sampled [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant PairMatrixPointParams& p [[buffer(4)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge >= p.pairs) return;
    const float value = original[edge] * multipliers[edge];
    const bool valid = isfinite(value) && value >= 0.0f;
    if (!valid) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    sampled[edge] = valid ? value : 0.0f;
}
kernel void StorePairMatrixScores(const device float2* scores [[buffer(0)]],
    const device uint* tile_status [[buffer(1)]], device float2* all_scores [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant PairMatrixTileParams& p [[buffer(4)]],
    uint candidate [[thread_position_in_grid]]) {
    if (candidate >= p.count) return;
    uint code = tile_status[candidate];
    if (!all(isfinite(scores[candidate]))) code |= 4u;
    if (code) atomic_fetch_or_explicit(status, code, memory_order_relaxed);
    all_scores[p.first + candidate] = code ? float2(NAN) : scores[candidate];
}
kernel void ReducePairMatrixLeafWeights(const device float* weights [[buffer(0)]],
    const device uint* rows [[buffer(1)]], const device uint* offsets [[buffer(2)]],
    device float* leaf_weights [[buffer(3)]], device atomic_uint* status [[buffer(4)]],
    constant PairMatrixPointParams& p [[buffer(5)]], uint tid [[thread_position_in_threadgroup]],
    uint leaf [[threadgroup_position_in_grid]]) {
    threadgroup float2 sums[256];
    float2 sum = float2(0.0f);
    if (offsets[leaf] > offsets[leaf+1] || offsets[leaf+1] > p.rows) {
        if (!tid) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        return;
    }
    for (uint index = offsets[leaf]+tid; index < offsets[leaf+1]; index += 256) {
        const uint row = rows[index];
        if (row >= p.rows || !isfinite(weights[row]) || weights[row] < 0.0f) {
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed); continue;
        }
        sum = LeafMatrixAdd(sum, float2(weights[row], 0.0f));
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width=128; width; width>>=1) {
        if (tid<width) sums[tid]=LeafMatrixAdd(sums[tid],sums[tid+width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) {
        leaf_weights[leaf] = sums[0].x+sums[0].y;
        if (!isfinite(leaf_weights[leaf])) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
    }
}
kernel void ReducePairMatrixLoss(const device float4* edges [[buffer(0)]],
    const device uint* winners [[buffer(1)]], const device uint* losers [[buffer(2)]],
    const device uint* ids [[buffer(3)]], device float4* partials [[buffer(4)]],
    constant PairwiseMatrixParams& p [[buffer(5)]], uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]]) {
    threadgroup float4 high_parts[256], low_parts[256];
    float4 high=float4(0.0f),low=float4(0.0f);
    for (uint edge=group*256+tid; edge<p.pairs; edge+=groups*256) {
        if (winners[edge]>=p.rows || losers[edge]>=p.rows) continue;
        if (p.reserved0 && ids[winners[edge]]==ids[losers[edge]]) continue;
        PairMatrixAdd4(high,low,float4(edges[edge].w,edges[edge].z,0.0f,0.0f)/float(p.pairs));
    }
    high_parts[tid]=high;low_parts[tid]=low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width=128;width;width>>=1) {
        if (tid<width) {
            high=high_parts[tid];low=low_parts[tid];
            PairMatrixAdd4(high,low,high_parts[tid+width]);PairMatrixAdd4(high,low,low_parts[tid+width]);
            high_parts[tid]=high;low_parts[tid]=low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) partials[group]=float4(high_parts[0].xy,low_parts[0].xy);
}
)METAL";

class CBMPairwiseMatrixRuntime : public CBMFullMatrixRuntime {
public:
    CBMPairwiseMatrixRuntime(id<MTLDevice> device, uint32_t rows, uint32_t pairs,
        const uint32_t* winners, const uint32_t* losers, const float* weights,
        uint32_t maxLeaves, uint32_t candidates, uint32_t requestedTile=32,
        uint64_t budget=(1ull<<30), bool generated=false)
        : Device(device), Rows(rows), Pairs(pairs), PairCapacity(pairs), MaxLeaves(maxLeaves), Candidates(candidates), Generated(generated) {
        Require(device && rows && rows <= (1u<<24) && pairs && pairs <= (1u<<24) &&
            (generated ? !winners && !losers && !weights : winners && losers && weights),
            "Invalid pairwise matrix target dimensions");
        Require(maxLeaves && maxLeaves <= 256 && candidates && candidates <= (1u<<20) && requestedTile && requestedTile <= 64,
            "Invalid pairwise matrix leaf/candidate capacity");
        LossGroups = std::min<uint32_t>((pairs+255)/256,4096);
        Tile = std::min(candidates,requestedTile);
        while (Tile > 1 && (uint64_t(Tile)*Pairs > (1u<<24) || RequiredBytes(Tile) > budget)) --Tile;
        Require(uint64_t(Tile)*Pairs <= (1u<<24) && RequiredBytes(Tile) <= budget && budget <= (1ull<<30),
            "Pairwise matrix runtime exceeds the 1 GiB workspace budget");
        double mass=0;
        for (uint32_t edge=0;!generated && edge<pairs;++edge) {
            Require(winners[edge]<rows && losers[edge]<rows && winners[edge]!=losers[edge] &&
                std::isfinite(weights[edge]) && weights[edge]>=0, "Invalid supplied pair endpoints or weight");
            mass+=weights[edge];
        }
        Require(generated || (mass>0 && mass<=std::numeric_limits<float>::max()), "Pairwise matrix target needs positive finite edge mass");
        Winners=Buffer(winners,4ull*pairs); Losers=Buffer(losers,4ull*pairs); OriginalWeights=Buffer(weights,4ull*pairs);
        Point=Buffer(nullptr,4ull*rows); Edges=Buffer(nullptr,16ull*pairs);
        Multipliers=Buffer(nullptr,4ull*pairs); SampledWeights=Buffer(nullptr,4ull*pairs);
        const uint64_t entries=uint64_t(Tile)*pairs,cells=uint64_t(Tile)*maxLeaves*maxLeaves;
        KeyA=Buffer(nullptr,4*entries);KeyB=Buffer(nullptr,4*entries);
        IndexA=Buffer(nullptr,4*entries);IndexB=Buffer(nullptr,4*entries);
        Offsets=Buffer(nullptr,4*(cells+1));Cells=Buffer(nullptr,32*cells);
        Gradient=Buffer(nullptr,4ull*Tile*maxLeaves);Hessian=Buffer(nullptr,4*cells);
        Direction=Buffer(nullptr,4ull*Tile*maxLeaves);Workspace=Buffer(nullptr,8*cells);
        TileScores=Buffer(nullptr,8ull*Tile);TileStatus=Buffer(nullptr,4ull*Tile);
        TrialStatus=Buffer(nullptr,4);SelectedIndex=Buffer(nullptr,4);SelectedScore=Buffer(nullptr,8);
        Scores=Buffer(nullptr,8ull*candidates);Status=Buffer(nullptr,4);LossPartials=Buffer(nullptr,16ull*LossGroups);
        Sort=std::make_unique<CBMSortU32Workspace>(device,entries);Bytes+=Sort->AllocatedBytes();
        Require(Bytes==RequiredBytes(Tile), "Pairwise matrix allocation accounting mismatch");
        MTLCompileOptions* options=[MTLCompileOptions new];
        if (@available(macOS 13.0,*)) options.languageVersion=MTLLanguageVersion3_0;
        else throw std::runtime_error("Pairwise matrix training requires macOS 13 or newer");
        options.fastMathEnabled=NO;NSError* error=nil;
        NSString* source=[NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s", CBMMetalPairwiseMatrixSource,
            CBMMetalPairwiseCandidateSource,CBMMetalLeafMatrixSource,CBMMetalPairwiseScoreSource,CBMMetalBootstrapSource,CBMMetalPairMatrixBridge];
        Library=[device newLibraryWithSource:source options:options error:&error];
        Require(Library!=nil,Error(error));
        for (const char* name : {"BuildGeneratedPairSelection","CountGeneratedPairSelection","ImportGeneratedPairMatrix","PreparePairMatrixPoint","GenerateBootstrapWeights","SamplePairMatrixWeights",
            "ComputePairwiseMatrixEdges","BuildPairwiseCandidateKeys","BuildPairwiseCandidateOffsets",
            "ReducePairwiseCandidateCells","AssemblePairwiseCandidateMatrices","RegularizePairwiseSplitMatrix",
            "SolveLeafMatrix","CenterPairwiseSplitSolution","ScorePairwiseSplitSolution","ExportSimplePairwiseLeaves","StorePairMatrixScores",
            "BuildPairwiseLeafKeys","BuildPairwiseCellOffsets","ReducePairwiseLeafCells","AssemblePairwiseLeafMatrix",
            "RegularizeLeafMatrix","UpdateLeafMatrixPoint","ReduceLeafMatrixDirectionalDot",
            "ReducePairMatrixLeafWeights","ReducePairMatrixLoss","SelectPairwiseSplitWinner"}) {
            auto function=[Library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function!=nil,std::string("Missing pairwise matrix function ")+name);
            auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline && pipeline.maxTotalThreadsPerThreadgroup>=256,Error(error));
            Pipelines.emplace(name,pipeline);
        }
        if (generated) { Pairs=0; LossGroups=0; }
    }

    uint64_t AllocatedBytes() const { return Bytes; }
    uint32_t CandidateTile() const { return Tile; }
    id<MTLBuffer> CandidateScores() const { return Scores; }
    id<MTLBuffer> LeafGradient() const { return Gradient; }
    id<MTLBuffer> LeafHessian() const { return Hessian; }
    id<MTLBuffer> LeafDirection() const { return Direction; }
    uint32_t PairCount() const { return Pairs; }
    id<MTLBuffer> GeneratedSortKeys() const { return KeyA; }
    id<MTLBuffer> GeneratedSortIndices() const { return IndexA; }
    id<MTLBuffer> GeneratedSortedKeys() const { return KeyB; }
    id<MTLBuffer> GeneratedSortedIndices() const { return IndexB; }
    void EncodeGeneratedSort(id<MTLCommandBuffer> command,uint32_t count,uint64_t* dispatches=nullptr) {
        CheckCommand(command);Require(Generated && count<=PairCapacity,"Generated contribution sort exceeds capacity");
        Sort->Encode(command,KeyA,IndexA,count,KeyB,IndexB,dispatches);
    }
    // CUDA FilterZeroEntries stably sorts a nonzero flag before gathering.
    // Reuse matrix-search radix storage and only one four-bit pass for flags.
    // Filtering precedes multiplication by the original document/query weight.
    void EncodeGeneratedSelection(id<MTLCommandBuffer> command,uint32_t pairs,id<MTLBuffer> raw,
        id<MTLBuffer> multipliers,uint64_t* dispatches=nullptr) {
        CheckCommand(command);Require(Generated && pairs<=PairCapacity,"Generated selection exceeds pair capacity");
        CheckBuffer(raw,4ull*pairs);CheckBuffer(multipliers,4ull*pairs);
        const PointParams p={Rows,pairs,0,0};
        Dispatch(command,"BuildGeneratedPairSelection",{raw,multipliers,KeyA,IndexA},p,pairs,false,dispatches);
        Sort->Encode(command,KeyA,IndexA,pairs,KeyB,IndexB,dispatches,4);
        Dispatch(command,"CountGeneratedPairSelection",{KeyB,SelectedIndex},p,1,false,dispatches);
        SelectionCommand=command;SelectionSourcePairs=pairs;
    }
    uint32_t ReadGeneratedSelectionCount() const {
        Require(SelectionCommand && SelectionCommand.status==MTLCommandBufferStatusCompleted,
            "Complete generated selection before reading its count");
        CheckStatus();const uint32_t count=*static_cast<const uint32_t*>(SelectedIndex.contents);
        Require(count<=SelectionSourcePairs,"Invalid generated pair selection count");return count;
    }
    // Generated targets may change logical length while the bounded allocation
    // stays resident. A fixed leaf target orients endpoints by relevance; weak
    // PFound targets retain their signed gradient in sampled document order.
    void EncodeGeneratedTarget(id<MTLCommandBuffer> command,uint32_t pairs,id<MTLBuffer> endpoints,
        id<MTLBuffer> edges,id<MTLBuffer> target,bool fixedLeaves,uint64_t* dispatches=nullptr,uint32_t selectedSourcePairs=0) {
        CheckCommand(command);Require(Generated && pairs<=PairCapacity,"Generated pair count exceeds its reserved capacity");
        Require(!selectedSourcePairs || (selectedSourcePairs==SelectionSourcePairs && pairs==ReadGeneratedSelectionCount()),
            "Generated import must use its completed selection");
        CheckBuffer(endpoints,8ull*std::max(pairs,selectedSourcePairs));CheckBuffer(edges,16ull*std::max(pairs,selectedSourcePairs));CheckBuffer(target,4ull*Rows);
        Dispatch(command,"ImportGeneratedPairMatrix",{endpoints,edges,target,IndexB,Winners,Losers,OriginalWeights,Edges,Status},
            PointParams{Rows,pairs,selectedSourcePairs,uint32_t(fixedLeaves)},pairs,false,dispatches);
        Pairs=pairs;LossGroups=std::min<uint32_t>((pairs+255)/256,4096);LayoutLeaves=0;LayoutIds=nil;
    }
    void ClearStatus() { *static_cast<uint32_t*>(Status.contents)=0; }
    void CheckStatus() const {
        const uint32_t code=*static_cast<const uint32_t*>(Status.contents);
        Require(!code,"Invalid GPU pairwise matrix state (status "+std::to_string(code)+")");
    }

    // Original edges for leaves/metrics; sampled edges for a frozen weak target.
    // Sampling is by EDGE before derivatives, unlike ordinary PairLogit.
    void EncodeEdges(id<MTLCommandBuffer> command,id<MTLBuffer> cursor,id<MTLBuffer> values,
        id<MTLBuffer> ids,uint32_t leaves,bool shifted,const CBMBootstrapOptions* bootstrap=nullptr,
        uint32_t absoluteIteration=0,uint64_t* dispatches=nullptr,bool trial=false) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(cursor,4ull*Rows);CheckBuffer(ids,4ull*Rows);CheckBuffer(values,4ull*leaves);
        const PointParams point={Rows,Pairs,leaves,uint32_t(shifted)};
        id<MTLBuffer> state = trial ? TrialStatus : Status;
        Dispatch(command,"PreparePairMatrixPoint",{cursor,values,ids,Point,state},point,Rows,false,dispatches);
        id<MTLBuffer> weights=OriginalWeights;
        if (bootstrap && bootstrap->bootstrap_type) {
            Require(bootstrap->bootstrap_type<=3 && std::isfinite(bootstrap->bagging_temperature) && bootstrap->bagging_temperature>=0 &&
                std::isfinite(bootstrap->subsample) && bootstrap->subsample>0 && bootstrap->subsample<=1 &&
                (bootstrap->bootstrap_type!=3 || bootstrap->subsample<1), "Invalid pairwise edge bootstrap; MVS is unsupported");
            const BootstrapParams b={Pairs,bootstrap->bootstrap_type,bootstrap->random_seed_low,bootstrap->random_seed_high,
                absoluteIteration,0,0,0,bootstrap->bagging_temperature,bootstrap->subsample,0,0};
            Dispatch(command,"GenerateBootstrapWeights",{Multipliers,OriginalWeights},b,Pairs,false,dispatches);
            Dispatch(command,"SamplePairMatrixWeights",{OriginalWeights,Multipliers,SampledWeights,Status},point,Pairs,false,dispatches);
            weights=SampledWeights;
        }
        Dispatch(command,"ComputePairwiseMatrixEdges",{Point,Winners,Losers,weights,Edges,state},
            MatrixParams{Rows,Pairs,leaves,0,0,0,0,0},Pairs,false,dispatches);
    }

    void EncodeCandidates(id<MTLCommandBuffer> command,id<MTLBuffer> bins,id<MTLBuffer> ids,
        id<MTLBuffer> features,id<MTLBuffer> borders,id<MTLBuffer> types,uint32_t featureCount,
        uint32_t parents,bool gradientScore,float l2,float nonDiag,uint64_t* dispatches=nullptr,
        uint32_t firstCandidate=0,uint32_t candidateCount=0) {
        CheckCommand(command);CheckLeaves(2ull*parents);CheckRegularization(l2,nonDiag);
        Require(featureCount,"Pairwise candidates need features");
        CheckBuffer(bins,uint64_t(featureCount)*Rows);CheckBuffer(ids,4ull*Rows);
        CheckBuffer(features,4ull*Candidates);CheckBuffer(borders,4ull*Candidates);CheckBuffer(types,Candidates);
        Require(firstCandidate<Candidates,"First full-matrix candidate is out of range");
        if (!candidateCount) candidateCount=Candidates-firstCandidate;
        Require(candidateCount<=Candidates-firstCandidate,"Full-matrix candidate range exceeds capacity");
        const uint32_t candidateEnd=firstCandidate+candidateCount;
        LayoutLeaves=0;
        const LeafParams leaf={2*parents,0,0,0,l2,nonDiag,1e-20f,1};
        for (uint32_t first=firstCandidate;first<candidateEnd;first+=Tile) {
            const uint32_t count=std::min(Tile,candidateEnd-first),entries=count*Pairs,cells=count*4*parents*parents;
            const CandidateParams p={Rows,Pairs,parents,count,featureCount,first,uint32_t(gradientScore),0};
            auto clear=[command blitCommandEncoder];[clear fillBuffer:TileStatus range:NSMakeRange(0,4ull*Tile) value:0];[clear endEncoding];
            Dispatch(command,"BuildPairwiseCandidateKeys",{bins,ids,Winners,Losers,features,borders,types,KeyA,IndexA,TileStatus},p,entries,false,dispatches);
            Sort->Encode(command,KeyA,IndexA,entries,KeyB,IndexB,dispatches);
            Dispatch(command,"BuildPairwiseCandidateOffsets",{KeyB,Offsets},p,cells+1,false,dispatches);
            Dispatch(command,"ReducePairwiseCandidateCells",{Offsets,IndexB,Edges,Cells},p,cells,true,dispatches);
            Dispatch(command,"AssemblePairwiseCandidateMatrices",{Cells,Gradient,Hessian,TileStatus},p,cells,false,dispatches);
            Dispatch(command,"RegularizePairwiseSplitMatrix",{Hessian,Workspace},leaf,count,true,dispatches);
            Dispatch(command,"SolveLeafMatrix",{Workspace,Gradient,Direction,TileStatus},leaf,count,true,dispatches);
            Dispatch(command,"CenterPairwiseSplitSolution",{Direction},leaf,count,true,dispatches);
            Dispatch(command,"ScorePairwiseSplitSolution",{Hessian,Gradient,Direction,TileScores},leaf,count,true,dispatches);
            Dispatch(command,"StorePairMatrixScores",{TileScores,TileStatus,Scores,Status},TileParams{first,count,leaf.Leaves,0},count,false,dispatches);
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
    void EncodeSelectWinner(id<MTLCommandBuffer> command,id<MTLBuffer> features,id<MTLBuffer> featureWeights,
        uint32_t featureCount,float previousScore,bool packedWeights=false,uint64_t* dispatches=nullptr) {
        CheckCommand(command);Require(featureCount && std::isfinite(previousScore),"Invalid pairwise selection metadata");
        CheckBuffer(features,4ull*Candidates);CheckBuffer(featureWeights,(packedWeights?8ull:4ull)*featureCount);
        Dispatch(command,"SelectPairwiseSplitWinner",{Scores,features,featureWeights,SelectedIndex,SelectedScore},
            SelectionParams{Candidates,featureCount,uint32_t(packedWeights),0,previousScore,0,0,0},1,true,dispatches);
    }
    Selected ReadWinner() const {
        CheckStatus();const uint32_t index=*static_cast<const uint32_t*>(SelectedIndex.contents);
        const float* score=static_cast<const float*>(SelectedScore.contents);
        Require(index<Candidates && std::isfinite(score[0]) && std::isfinite(score[1]),"No finite pairwise split winner");
        return {index,score[0],score[1]};
    }

    void EncodeLeafLayout(id<MTLCommandBuffer> command,id<MTLBuffer> ids,uint32_t leaves,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(ids,4ull*Rows);
        const MatrixParams p={Rows,Pairs,leaves,0,0,0,0,0};
        Dispatch(command,"BuildPairwiseLeafKeys",{ids,Winners,Losers,KeyA,IndexA,Status},p,Pairs,false,dispatches);
        Sort->Encode(command,KeyA,IndexA,Pairs,KeyB,IndexB,dispatches);
        Dispatch(command,"BuildPairwiseCellOffsets",{KeyB,Offsets},p,leaves*leaves+1,false,dispatches);
        LayoutLeaves=leaves;LayoutIds=ids;
    }
    void EncodeLeafProjection(id<MTLCommandBuffer> command,uint32_t leaves,bool gradientMethod,uint64_t* dispatches=nullptr) {
        CheckCommand(command);Require(LayoutLeaves==leaves && LayoutIds,"Prepare the fixed pairwise leaf layout first");
        const MatrixParams p={Rows,Pairs,leaves,uint32_t(gradientMethod),0,0,0,0};
        Dispatch(command,"ReducePairwiseLeafCells",{Offsets,IndexB,Edges,Cells},p,leaves*leaves,true,dispatches);
        Dispatch(command,"AssemblePairwiseLeafMatrix",{Cells,Gradient,Hessian,Status},p,leaves*leaves,false,dispatches);
    }
    void EncodeLeafWeights(id<MTLCommandBuffer> command,id<MTLBuffer> originalWeights,id<MTLBuffer> rows,
        id<MTLBuffer> offsets,id<MTLBuffer> result,uint32_t leaves,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(originalWeights,4ull*Rows);CheckBuffer(rows,4ull*Rows);
        CheckBuffer(offsets,4ull*(leaves+1));CheckBuffer(result,4ull*leaves);
        Require(result!=originalWeights && result!=rows && result!=offsets,"Pairwise leaf weight output cannot alias its inputs");
        Dispatch(command,"ReducePairMatrixLeafWeights",{originalWeights,rows,offsets,result,Status},PointParams{Rows,Pairs,leaves,0},leaves,true,dispatches);
    }
    void EncodeLeafDirection(id<MTLCommandBuffer> command,uint32_t leaves,float l2,float nonDiag,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckRegularization(l2,nonDiag);
        const LeafParams p={leaves,0,0,0,l2,nonDiag,1e-20f,1};
        Dispatch(command,"RegularizeLeafMatrix",{Hessian,Workspace},p,leaves*leaves,false,dispatches);
        Dispatch(command,"SolveLeafMatrix",{Workspace,Gradient,Direction,Status},p,1,true,dispatches);
    }
    void EncodeLeafUpdate(id<MTLCommandBuffer> command,id<MTLBuffer> point,id<MTLBuffer> weights,
        id<MTLBuffer> updated,uint32_t leaves,float step,uint64_t* dispatches=nullptr,bool trial=false) {
        CheckCommand(command);CheckLeaves(leaves);Require(std::isfinite(step),"Invalid pairwise leaf step");
        for (auto buffer : {point,weights,updated}) CheckBuffer(buffer,4ull*leaves);
        Require(updated!=weights,"Pairwise leaf updates cannot overwrite original weights");
        Dispatch(command,"UpdateLeafMatrixPoint",{point,Direction,weights,updated,trial ? TrialStatus : Status},
            LeafParams{leaves,0,0,0,0,0,1e-20f,step},leaves,false,dispatches);
    }
    void EncodeBeginTrial(id<MTLCommandBuffer> command) {
        CheckCommand(command);
        auto clear=[command blitCommandEncoder];Require(clear!=nil,"Pairwise trial reset encoder allocation failed");
        [clear fillBuffer:TrialStatus range:NSMakeRange(0,4) value:0];[clear endEncoding];
    }
    void EncodeDirectionDot(id<MTLCommandBuffer> command,id<MTLBuffer> output,uint32_t leaves,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(output,8);
        Dispatch(command,"ReduceLeafMatrixDirectionalDot",{Gradient,Direction,output},
            LeafParams{leaves,0,0,0,0,0,0,1},1,true,dispatches);
    }
    double ReadTrialLoss() const {
        CheckStatus();
        if (*static_cast<const uint32_t*>(TrialStatus.contents)) return std::numeric_limits<double>::infinity();
        return ReadLoss().first;
    }
    // Only the final fixed-last-coordinate solution is centered; trial points
    // retain their last coordinate at zero throughout leaf estimation.
    void EncodeCenterSolvedPoint(id<MTLCommandBuffer> command,id<MTLBuffer> point,uint32_t leaves,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(point,4ull*leaves);
        Dispatch(command,"CenterPairwiseSplitSolution",{point},LeafParams{leaves,0,0,0,0,0,0,1},1,true,dispatches);
    }
    void EncodeLoss(id<MTLCommandBuffer> command,id<MTLBuffer> ids,uint32_t leaves,bool supportOnly,uint64_t* dispatches=nullptr) {
        CheckCommand(command);CheckLeaves(leaves);CheckBuffer(ids,4ull*Rows);
        Dispatch(command,"ReducePairMatrixLoss",{Edges,Winners,Losers,ids,LossPartials},
            MatrixParams{Rows,Pairs,leaves,0,uint32_t(supportOnly),0,0,0},LossGroups,true,dispatches);
    }
    std::pair<double,double> ReadLoss() const {
        CheckStatus();const float* values=static_cast<const float*>(LossPartials.contents);double loss=0,mass=0;
        for (uint32_t i=0;i<LossGroups;++i) {
            for (uint32_t j=0;j<4;++j) Require(std::isfinite(values[4*i+j]),"Nonfinite pairwise matrix loss reduction");
            loss+=double(values[4*i])+values[4*i+2];mass+=double(values[4*i+1])+values[4*i+3];
        }
        Require(std::isfinite(loss) && std::isfinite(mass) && loss>=0 && mass>=0,"Invalid pairwise matrix objective");
        return {loss*Pairs,mass*Pairs};
    }

private:
    struct SelectionParams { uint32_t Candidates,Features,Packed,R0;float Previous,R1,R2,R3; };
    struct MatrixParams { uint32_t Rows,Pairs,Leaves,LeafMethod,R0,R1,R2,R3; };
    struct CandidateParams { uint32_t Rows,Pairs,Parents,Candidates,Features,First,LeafMethod,Reserved; };
    struct LeafParams { uint32_t Leaves,Diagonal,R0,R1;float L2,NonDiag,MinWeight,Step; };
    struct PointParams { uint32_t Rows,Pairs,Leaves,Shift; };
    struct TileParams { uint32_t First,Count,Leaves,SupportOnly; };
    struct BootstrapParams { uint32_t Rows,Type,SeedLow,SeedHigh,Iteration,Stream,R0,R1;float Temperature,Subsample,Lambda,Noise; };
    id<MTLDevice> Device;id<MTLLibrary> Library;id<MTLCommandBuffer> SelectionCommand;
    id<MTLBuffer> Winners,Losers,OriginalWeights,Point,Edges,Multipliers,SampledWeights;
    id<MTLBuffer> KeyA,KeyB,IndexA,IndexB,Offsets,Cells,Gradient,Hessian,Direction,Workspace,TileScores,TileStatus,Scores,Status,LossPartials,LayoutIds,SelectedIndex,SelectedScore,TrialStatus;
    std::unordered_map<std::string,id<MTLComputePipelineState>> Pipelines;
    std::unique_ptr<CBMSortU32Workspace> Sort;
    uint32_t Rows,Pairs,PairCapacity,MaxLeaves,Candidates,Tile,LossGroups,LayoutLeaves=0;
    bool Generated;
    uint32_t SelectionSourcePairs=0;
    uint64_t Bytes=0;
    static void Require(bool value,const std::string& message) { if (!value) throw std::runtime_error(message); }
    // Literal validation messages allocate only when a check fails.
    static void Require(bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    }
    static std::string Error(NSError* error) { return error ? error.localizedDescription.UTF8String : "Pairwise Metal compilation failed"; }
    static void CheckRegularization(float l2,float nonDiag) { Require(std::isfinite(l2) && l2>=0 && std::isfinite(nonDiag) && nonDiag>=0,"Invalid pairwise matrix regularization"); }
    void CheckLeaves(uint64_t leaves) const { Require(leaves && leaves<=MaxLeaves,"Pairwise leaf count exceeds capacity"); }
    void CheckCommand(id<MTLCommandBuffer> command) const { Require(command && command.device==Device && command.status==MTLCommandBufferStatusNotEnqueued && command.retainedReferences,"Pairwise runtime needs an uncommitted retained command on its device"); }
    void CheckBuffer(id<MTLBuffer> buffer,uint64_t bytes) const { Require(buffer && buffer.device==Device && buffer.length>=bytes,"Pairwise buffer is missing, too short or on another device"); }
    uint64_t RequiredBytes(uint32_t tile) const {
        const uint64_t entries=uint64_t(tile)*PairCapacity,cells=uint64_t(tile)*MaxLeaves*MaxLeaves;
        uint64_t sort=12*entries,elements=((entries+255)/256)*16;
        while (true) { sort+=4*elements;if (elements==1)break;elements=(elements+255)/256; }
        return 36ull*PairCapacity+4ull*Rows+16*entries+48*cells+8ull*tile*MaxLeaves+12ull*tile+8ull*Candidates+24+16ull*std::min<uint32_t>((PairCapacity+255)/256,4096)+sort;
    }
    id<MTLBuffer> Buffer(const void* source,uint64_t bytes) {
        Require(bytes && bytes<=Device.maxBufferLength,"Pairwise allocation exceeds device limits");
        auto value=source ? [Device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Require(value!=nil,"Pairwise matrix allocation failed");if (!source)std::memset(value.contents,0,bytes);Bytes+=bytes;return value;
    }
    template<class T>
    void Dispatch(id<MTLCommandBuffer> command,const char* name,std::initializer_list<id<MTLBuffer>> buffers,
        const T& params,uint32_t count,bool grouped,uint64_t* dispatches) {
        if (!count) return;
        auto encoder=[command computeCommandEncoder];Require(encoder!=nil,"Pairwise compute encoder allocation failed");
        [encoder setComputePipelineState:Pipelines.at(name)];NSUInteger index=0;
        for (auto buffer:buffers)[encoder setBuffer:buffer offset:0 atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index];
        if (grouped)[encoder dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        else [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];if (dispatches)++*dispatches;
    }
};
