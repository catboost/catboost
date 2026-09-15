#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_ordered_trainer.h"
#include "metal_bootstrap_kernels.h"
#include "metal_score_noise_kernels.h"
#include "metal_kernels.h"
#include "metal_additional_objective_kernels.h"
#include "metal_objective_kernels.h"
#include "metal_backtracking_kernels.h"
#include "metal_exact_leaf_kernels.h"
#include "metal_sort.h"
#include "metal_ordered_kernels.h"
#include "metal_ordered_session_kernels.h"
#include "metal_ordered_backtracking.h"
#include "metal_deep_partition_kernels.h"
#include "metal_ordered_histogram_runtime.h"
#include "metal_querywise_kernels.h"
#include "metal_pairwise_kernels.h"
#include "metal_ordered_query_kernels.h"
#include "metal_ordered_yeti_runtime.h"
#include "metal_custom_objective.h"
#include "metal_ordered_combination_runtime.h"
#include "metal_langevin.h"
#include "metal_langevin_kernels.h"
#include "metal_ordered_langevin_kernels.h"
#include <cstring>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
// Activity is metadata only: retained split columns still route every cursor.
// Keep this additive kernel private so the legacy shared score ABI is unchanged.
static const char* OrderedActivitySource = R"METAL(
kernel void OrderedSessionMaskCandidates(device float2* scores [[buffer(0)]],
    const device uint2* candidates [[buffer(1)]], const device uchar* active [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint candidate [[thread_position_in_grid]]) {
    if (candidate < count && !active[candidates[candidate].x]) scores[candidate] = float2(FLT_MAX);
}
)METAL";
constexpr uint64_t MemoryLimit = uint64_t(1) << 30;
constexpr uint64_t HistogramLimit = uint64_t(32) << 20;
static_assert(CBMMetalKernelAbiVersion == 2, "Review Ordered runtime after shared Metal ABI changes");
using KernelParams = CBMMetalKernelParams;
struct OrderedParams {
    uint32_t Rows, Features, Folds, Leaves, Candidates, PackedRows, TestOnly, ScoreFunction;
    float L2;
    uint32_t Normalize;
    float ScoreBefore, LearningRate;
};
struct StepParams { uint32_t Tasks, Leaves, CursorCount, SelectedPermutation; };
struct BacktrackingParams { float Step; uint32_t Type, AddRidge, Normalize; };
struct NativeQueryParams { uint32_t Rows, Groups, Objective, ApplyValues; float Beta, Lambda; uint32_t Leaves, Reserved; };
struct NativePairParams { uint32_t Rows, Pairs, Objective, ApplyValues, Leaves, Reserved0, Reserved1, Reserved2; };
struct OrderedLangevinParams { uint32_t ActiveTasks, Query, Trial, Initial; };
struct OrderedTargetOptions {
    const CBMQueryOptions* Query = nullptr;
    const CBMPairOptions* Pair = nullptr;
    const CBMYetiRankOptions* Yeti = nullptr;
    const uint32_t* Winners = nullptr;
    const uint32_t* Losers = nullptr;
    const float* PairWeights = nullptr;
    const char* CustomSource = nullptr;
    const CBMCombinationOptions* Combination = nullptr;
    const CBMCombinationComponent* Components = nullptr;
};
struct BootstrapParams {
    uint32_t Rows, Type, SeedLow, SeedHigh, Iteration, Stream, Reserved0, Reserved1;
    float Temperature, Subsample, MVSLambda, NoiseScale;
};
struct LangevinWeakParams { BootstrapParams Random; uint32_t Offset, Stride, FilterBootstrap, Reserved; };
using SplitState = CBMMetalSplitState;
static_assert(sizeof(CBMOrderedParams) == 80 && sizeof(KernelParams) == 96 &&
              sizeof(OrderedParams) == 48 && sizeof(SplitState) == 32);
void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
// Literal validation messages allocate only when a check fails.
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void Text(char* destination, size_t capacity, const char* value) {
    if (!destination || !capacity) return;
    const size_t count = std::min(capacity - 1, std::strlen(value));
    std::memcpy(destination, value, count); destination[count] = 0;
}
std::string Error(NSError* error) { return error ? error.localizedDescription.UTF8String : "unknown Metal error"; }

// Context/command boilerplate is deliberately private and small so a common
// runtime can replace it without changing Ordered's mathematical state.
struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    explicit Runtime(const char* customSource = nullptr) {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device && Device.hasUnifiedMemory && [Device supportsFamily:MTLGPUFamilyApple7],
                "Ordered training requires an Apple Silicon Metal GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Ordered command queue allocation failed");
        MTLCompileOptions* options = [MTLCompileOptions new];
        if (@available(macOS 13.0, *)) options.languageVersion = MTLLanguageVersion3_0;
        else throw std::runtime_error("Ordered training requires macOS 13 or newer");
        options.fastMathEnabled = NO;
        NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s",
            CBMMetalBootstrapSource, CBMMetalScoreNoiseSource, CBMMetalSource,
            CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
            CBMMetalBacktrackingSource, CBMMetalExactLeafSource, CBMMetalOrderedSource,
            CBMMetalOrderedSessionSource, CBMMetalOrderedBacktrackingSource,
            CBMMetalDeepPartitionSource, CBMMetalOrderedHistogramSource, OrderedActivitySource,
            CBMMetalQuerywiseSource, CBMMetalPairwiseSource, CBMMetalOrderedQuerySource,
            CBMMetalYetiRankSource, CBMMetalOrderedYetiSource];
        source = [source stringByAppendingFormat:@"\n%s\n%s", CBMMetalLangevinSource, CBMMetalOrderedLangevinSource];
        if (customSource) source = [[NSString stringWithUTF8String:CBMCustomObjectivePrefix(customSource).c_str()] stringByAppendingString:source];
        NSError* error = nil;
        id<MTLLibrary> library = [Device newLibraryWithSource:source options:options error:&error];
        Require(library != nil, "Ordered shader compilation failed: " + Error(error));
        const char* names[] = {"OrderedSessionDerivatives", "OrderedCandidateStatistics", "ScoreOrderedCandidates",
            "OrderedSessionFindWinner", "OrderedSessionUpdateLeafIds", "OrderedSessionEstimateLeaves",
            "OrderedSessionApplyValues", "OrderedSessionPublish", "ReduceObjectiveLoss",
            "GenerateBootstrapWeights", "ApplyOrderedBootstrap", "OrderedQualityStatistics",
            "GenerateScoreFeatureNoise", "OrderedSessionFeatureNoise", "OrderedSessionMvsInput",
            "ReduceBootstrapStatistics", "ComputeMvsThresholds", "GenerateMvsBootstrapWeights",
            "OrderedSessionGatherExact", "PrepareExactResiduals", "MakeExactLeafKeys", "BuildExactLeafOffsets",
            "ReduceExactLeafPartials", "PrefixExactLeafTiles", "SelectExactLeafQuantile", "FinalizeExactLeafValues",
            "OrderedBacktrackingDirections", "OrderedBacktrackingCandidate", "OrderedBacktrackingObjective",
            "CountDeepPartitionBits", "ScanDeepPartitionTiles", "ScanDeepPartitionBlocks", "BuildDeepPartitionOffsets", "ScatterDeepPartitionRows",
            "InitializeOrderedHistogramOccurrences", "UpdateOrderedHistogramLeafIds", "ResetOrderedHistogramJobs", "BuildOrderedHistogramJobs",
            "OrderedHistogramArguments", "ClearOrderedHistogram", "ComputeOrderedHistogram", "ScanOrderedHistogram",
            "SubtractOrderedHistogramSibling", "ExtractOrderedHistogramCandidates", "ScatterOrderedHistogramScores",
            "ClearOrderedPartitionCandidateStatistics", "ComputeOrderedPartitionCandidates", "OrderedSessionMaskCandidates",
            "QueryRmseDerivatives", "QuerySoftMaxDerivatives", "PairLogitEdgeDerivatives", "ReducePairwiseRows",
            "OrderedQueryPreparePoint", "OrderedQueryPublishDerivatives", "OrderedQueryEstimateLeaves",
            "OrderedQueryBacktrackingDirections", "OrderedQueryBacktrackingObjective", "OrderedQueryCenterLeaves",
            "OrderedPairQueryStatistics", "PrepareYetiRankApprox", "YetiRankPointwise", "OrderedYetiPublishDerivatives",
            "AddLangevinWeakNoise", "OrderedLangevinStatistics", "OrderedLangevinDirections"};
        for (const char* name : names) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing Ordered kernel: ") + name);
            id<MTLComputePipelineState> pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline != nil, "Ordered pipeline failed: " + Error(error));
            Require(pipeline.maxTotalThreadsPerThreadgroup >= 256, "Ordered kernels require 256 threads");
            Pipelines.emplace(name, pipeline);
        }
    }
    id<MTLBuffer> Buffer(uint64_t bytes, const void* source = nullptr) {
        Require(bytes <= MemoryLimit && bytes <= Device.maxBufferLength, "Ordered buffer exceeds memory limit");
        const auto length = std::max<uint64_t>(bytes, 1);
        id<MTLBuffer> buffer = source && bytes
            ? [Device newBufferWithBytes:source length:length options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:length options:MTLResourceStorageModeShared];
        Require(buffer != nil, "Ordered Metal buffer allocation failed");
        return buffer;
    }
};
Runtime& Context() { static Runtime context; return context; }
struct Binding {
    id<MTLBuffer> Buffer;
    uint64_t Offset;
    Binding(id<MTLBuffer> buffer, uint64_t offset = 0) : Buffer(buffer), Offset(offset) {}
};
struct Command {
    using BindingType = Binding;
    CBMTrainStats& Stats;
    id<MTLCommandBuffer> Buffer;
    Runtime* PipelineContext;
    explicit Command(CBMTrainStats& stats, Runtime* pipelineContext = nullptr)
        : Stats(stats), Buffer([Context().Queue commandBuffer]), PipelineContext(pipelineContext ? pipelineContext : &Context()) {
        Require(Buffer != nil, "Ordered command buffer allocation failed");
    }
    template<class P> void Dispatch(const char* name, std::initializer_list<Binding> inputs, const P& params,
        uint64_t width, bool grouped = false, uint64_t height = 1, uint64_t depth = 1,
        const StepParams* extra = nullptr, const void* final = nullptr, size_t finalSize = 0) {
        Require(width && width <= UINT32_MAX && height && depth, "Invalid Ordered dispatch");
        id<MTLComputeCommandEncoder> encoder = [Buffer computeCommandEncoder];
        Require(encoder != nil, "Ordered command encoder allocation failed");
        [encoder setComputePipelineState:PipelineContext->Pipelines.at(name)];
        NSUInteger index = 0;
        for (const auto& item : inputs) [encoder setBuffer:item.Buffer offset:item.Offset atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index++];
        if (extra) [encoder setBytes:extra length:sizeof(*extra) atIndex:index++];
        if (final) [encoder setBytes:final length:finalSize atIndex:index];
        if (grouped) [encoder dispatchThreadgroups:MTLSizeMake(width, height, depth) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [encoder dispatchThreads:MTLSizeMake(width, height, depth) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding]; ++Stats.kernel_dispatches;
    }
    void Zero(id<MTLBuffer> destination) {
        id<MTLBlitCommandEncoder> encoder = [Buffer blitCommandEncoder];
        Require(encoder != nil, "Ordered blit encoder allocation failed");
        [encoder fillBuffer:destination range:NSMakeRange(0, destination.length) value:0];
        [encoder endEncoding];
    }
    template<class P> void DispatchIndirect(const char* name, std::initializer_list<Binding> inputs,
        const P& params, id<MTLBuffer> arguments, uint64_t offset) {
        id<MTLComputeCommandEncoder> encoder = [Buffer computeCommandEncoder];
        Require(encoder != nil, "Ordered indirect command encoder allocation failed");
        [encoder setComputePipelineState:PipelineContext->Pipelines.at(name)];
        NSUInteger index = 0;
        for (const auto& item : inputs) [encoder setBuffer:item.Buffer offset:item.Offset atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index];
        [encoder dispatchThreadgroupsWithIndirectBuffer:arguments indirectBufferOffset:offset
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding]; ++Stats.kernel_dispatches;
    }
    void Wait() {
        [Buffer commit]; [Buffer waitUntilCompleted];
        Require(Buffer.status == MTLCommandBufferStatusCompleted, "Ordered command failed: " + Error(Buffer.error));
        const double seconds = Buffer.GPUEndTime - Buffer.GPUStartTime;
        if (std::isfinite(seconds) && seconds >= 0 && Buffer.GPUStartTime > 0) Stats.gpu_seconds += seconds;
    }
};

class Session {
public:
    CBMOrderedParams P;
    CBMTrainStats Stats = {};
    std::mutex Mutex;
    uint32_t Completed = 0;
    float Loss = 0;
    uint32_t FoldCount = 0, LearnPermutations = 0, CursorCount = 0, Tasks = 0, MaxLeaves = 0;
    std::vector<CBMOrderedFold> Descriptors;
    std::vector<uint32_t> PermutationTaskOffsets, PermutationFoldCounts, PermutationPackedRows;
    Session(const CBMOrderedParams* params, const uint8_t* bins, const float* targets, const float* weights,
        const float* initial, const uint32_t* features, const uint32_t* borders, const uint32_t* permutations,
        const uint8_t* candidateTypes = nullptr, uint32_t groupCount = 0, const uint32_t* groupOffsets = nullptr, double groupGrowth = 0, uint32_t binBanks = 1, const OrderedTargetOptions* targetOptions = nullptr) {
        Require(params != nullptr, "Ordered parameters are required");
        P = *params;
        if ((P.objective <= 14 || P.objective == 19 || P.objective == 20) && P.leaf_method == 3) {
            Require(P.leaf_iterations == 1, "Ordered Simple estimation requires one leaf iteration");
            P.leaf_method = 1;
        }
        Require(P.rows >= 4 && P.rows <= (1u << 24), "Ordered rows must be in [4,16777216]");
        Require(P.features && uint64_t(P.rows) * P.features <= UINT32_MAX, "Invalid Ordered feature dimensions");
        Require(P.candidates <= uint64_t(P.features) * (candidateTypes ? 256 : 255) && P.iterations && P.iterations <= 100000,
                "Invalid Ordered candidate or iteration count");
        Require(P.depth <= 16 && (P.objective <= 11 || ((P.objective <= 14 || P.objective == 17 || P.objective == 19 || P.objective == 20) && targetOptions)) && P.score_function <= 1 && P.leaf_method <= 2,
                "Unsupported Ordered depth, objective, score or leaf method");
        Require(P.leaf_iterations && P.leaf_iterations <= 1000 && P.permutations && P.permutations <= 64 && P.normalize <= 1,
                "Invalid Ordered leaf iteration, permutation or normalization option");
        Require(!P.reserved0 && !P.reserved1 && !P.reserved2, "Ordered reserved parameters must be zero");
        // Retain the caller's Simple identity after its Gradient1 mapping.
        // CUDA accepts signed QuerySoftMax weak weights for this estimator.
        P.reserved2 = params->leaf_method == 3;
        Require(binBanks == 1 || binBanks == P.permutations, "Ordered feature banks must be shared or match every permutation");
        BinBanks = binBanks;
        // Internal-only leaf stride; caller-reserved fields remain required zero.
        P.reserved0 = BinBanks > 1 ? P.rows : 0;
        Require(std::isfinite(P.learning_rate) && P.learning_rate > 0 && P.learning_rate <= 1 &&
                std::isfinite(P.l2) && P.l2 >= 0 && std::isfinite(P.bias), "Invalid Ordered training scalars");
        Require(std::isfinite(P.objective_param) && (P.objective != 4 || P.objective_param >= 0) &&
                (P.objective != 5 || (P.objective_param >= 0 && P.objective_param <= 1)) &&
                (P.objective != 6 || P.objective_param >= 1) &&
                (P.objective != 7 || (P.objective_param > 1 && P.objective_param < 2)) &&
                ((P.objective != 8 && P.objective != 9) || (P.objective_param >= 0 && P.objective_param <= 1)),
                "Invalid Ordered objective parameter");
        Require(P.leaf_method != 0 || P.objective >= 12 || (P.objective < 8 && (P.objective != 6 || P.objective_param >= 2)),
                "Newton leaf estimation is unsupported for this Ordered objective");
        Require(P.leaf_method != 2 || (P.objective >= 9 && P.objective <= 11), "Ordered Exact supports Quantile, MAE and MAPE only");
        Require(bins && (targets || P.objective == 14) && permutations && (!P.candidates || (features && borders)), "Ordered input buffers are required");
        // Reject an impossible core geometry before allocating the host
        // feature maps or planning histogram tiles.
        Require(uint64_t(P.rows) * P.features * BinBanks + uint64_t(P.features) * 16 + uint64_t(P.candidates) * 20 <= MemoryLimit,
                "Ordered working set exceeds 1 GiB; reduce rows, features or candidates");
        std::vector<int8_t> featureKinds(P.features, -1);
        for (uint32_t candidate = 0; candidate < P.candidates; ++candidate) {
            const uint32_t type = candidateTypes ? candidateTypes[candidate] : 0;
            Require(features[candidate] < P.features && type <= 1 && borders[candidate] <= 255 - (type == 0),
                    "Invalid Ordered numeric/one-hot split candidate");
            auto& kind = featureKinds[features[candidate]];
            Require(kind < 0 || static_cast<uint32_t>(kind) == type, "An Ordered feature cannot mix numeric and one-hot candidates");
            kind = type;
        }
        // Upstream option validation normalizes a requested zero ridge to 1e-20.
        P.l2 = std::max(P.l2, 1e-20f);
        if (groupCount || groupOffsets) {
            Require(std::isfinite(groupGrowth) && groupGrowth > 1, "Ordered group fold growth must exceed one");
            Require(groupOffsets && groupCount >= 4 && groupCount <= P.rows,
                    "Ordered requires at least four groups with offsets");
            Require(groupOffsets[0] == 0 && groupOffsets[groupCount] == P.rows,
                    "Ordered group offsets must cover every row");
            for (uint32_t group = 0; group < groupCount; ++group)
                Require(groupOffsets[group] < groupOffsets[group + 1], "Ordered group offsets must be strictly increasing");
        }
        LearnPermutations = P.permutations > 1 ? P.permutations - 1 : 1;
        uint64_t cursorCount = 0;
        for (uint32_t permutation = 0; permutation < P.permutations; ++permutation) {
            std::vector<uint32_t> groupEnds;
            if (groupCount) {
                std::vector<bool> seen(groupCount, false);
                uint32_t position = 0;
                while (position < P.rows) {
                    const uint32_t first = permutations[uint64_t(permutation) * P.rows + position];
                    const auto bound = std::lower_bound(groupOffsets, groupOffsets + groupCount, first);
                    Require(bound != groupOffsets + groupCount && *bound == first,
                            "Ordered permutations must preserve whole groups and their row order");
                    const uint32_t group = bound - groupOffsets;
                    const uint32_t size = groupOffsets[group + 1] - first;
                    Require(!seen[group] && uint64_t(position) + size <= P.rows,
                            "Ordered permutation repeats a group or exceeds its row count");
                    seen[group] = true;
                    for (uint32_t row = 0; row < size; ++row)
                        Require(permutations[uint64_t(permutation) * P.rows + position + row] == first + row,
                                "Ordered permutations must preserve whole groups and their row order");
                    position += size; groupEnds.push_back(position);
                }
            }
            if (permutation >= LearnPermutations) continue;
            const auto folds = groupCount ? CBMCreateGroupedOrderedFolds(P.rows, groupEnds, groupGrowth, P.min_fold_size)
                                         : CBMCreateNumericOrderedFolds(P.rows, P.fold_growth, P.min_fold_size);
            Require(!folds.empty(), "Ordered requires at least one prefix fold");
            PermutationTaskOffsets.push_back(Descriptors.size());
            PermutationFoldCounts.push_back(folds.size());
            const uint32_t packedRows = folds.back().CursorOffset + folds.back().QualityEnd;
            PermutationPackedRows.push_back(packedRows);
            FoldCount = std::max(FoldCount, static_cast<uint32_t>(folds.size()));
            PermutationCursorCount = std::max(PermutationCursorCount, packedRows);
            for (const auto& fold : folds) {
                Require(cursorCount + fold.QualityEnd <= UINT32_MAX, "Ordered cursor index exceeds uint32");
                Descriptors.push_back({fold.EstimateEnd, fold.QualityEnd, uint32_t(cursorCount), permutation});
                cursorCount += fold.QualityEnd;
            }
        }
        Require(cursorCount + P.rows <= UINT32_MAX, "Ordered cursor index exceeds uint32");
        Descriptors.push_back({P.rows, P.rows, uint32_t(cursorCount), P.permutations - 1});
        cursorCount += P.rows; CursorCount = cursorCount; Tasks = Descriptors.size(); MaxLeaves = 1u << P.depth;
        std::unique_ptr<CBMOrderedHistogramPlan> histogramPlan;
        if (P.depth && P.candidates) {
            uint32_t foldSlots = 1, span = 2, cacheLeaves = MaxLeaves / 2;
            while (foldSlots < FoldCount) foldSlots <<= 1;
            for (uint32_t candidate = 0; candidate < P.candidates; ++candidate) span = std::max(span, std::min(256u, borders[candidate] + 2));
            while (cacheLeaves > 1 && uint64_t(cacheLeaves) * foldSlots * span * 32 > (uint64_t(128) << 20)) cacheLeaves >>= 1;
            histogramPlan = std::make_unique<CBMOrderedHistogramPlan>(P.rows, PermutationCursorCount, FoldCount,
                MaxLeaves / 2, P.features, P.candidates, features, borders, uint64_t(128) << 20, true, cacheLeaves, candidateTypes);
        }
        const uint64_t statPerCandidate = uint64_t(MaxLeaves) * FoldCount * 32;
        BatchSize = P.candidates ? std::max<uint64_t>(1, std::min<uint64_t>(P.candidates, HistogramLimit / statPerCandidate)) : 1;
        const uint64_t statBytes = statPerCandidate * BatchSize;
        const uint64_t dataCells = uint64_t(P.rows) * P.features;
        ExactTiles = std::min(256u, (P.rows + 255) / 256);
        // Two radix sorts coexist until each task command completes; include
        // both sort-owned key/payload/rank and prefix buffers in peak memory.
        const uint64_t exactBytes = P.leaf_method == 2 ? 66ull * P.rows +
            uint64_t(MaxLeaves) * (24ull * ExactTiles + 16) + 16384 : 0;
        const uint64_t bytes = dataCells * BinBanks + uint64_t(P.rows) * (16 + BinBanks * 4 + P.permutations * 4) +
            cursorCount * 16 + uint64_t(Tasks) * MaxLeaves * 8 + statBytes + uint64_t(P.candidates) * 20 +
            uint64_t(P.features) * 16 + uint64_t(Tasks) * 16 + uint64_t(groupCount) * 13 + 32768 + exactBytes
            + (histogramPlan ? histogramPlan->Bytes + dataCells * BinBanks + uint64_t(BatchSize) * 8 : 0);
        Require(bytes <= MemoryLimit, "Ordered working set exceeds 1 GiB; reduce depth, permutations, rows or features");
        WorkingBytes = bytes;
        std::vector<bool> seen(P.rows);
        for (uint32_t permutation = 0; permutation < P.permutations; ++permutation) {
            std::fill(seen.begin(), seen.end(), false);
            for (uint32_t position = 0; position < P.rows; ++position) {
                const uint32_t row = permutations[uint64_t(permutation) * P.rows + position];
                Require(row < P.rows && !seen[row], "Each Ordered permutation must contain every row once"); seen[row] = true;
            }
        }
        std::vector<float> unitWeights(P.rows, 1), pairTargets;
        if (targetOptions && (targetOptions->Query || targetOptions->Pair || targetOptions->Yeti || targetOptions->Combination)) {
            Require(P.objective == 19 || (groupCount >= 4 && groupOffsets), "Ordered query objectives require whole-query grouping");
            IsQuery = true;
            if (P.objective == 12 || P.objective == 13) {
                Require(targetOptions->Query && !targetOptions->Pair && !targetOptions->Yeti,
                        "Ordered query objective options do not match");
                QueryOptions = *targetOptions->Query;
                Require(QueryOptions.group_count == groupCount && !QueryOptions.reserved &&
                        std::isfinite(QueryOptions.beta) && std::isfinite(QueryOptions.lambda), "Invalid Ordered query options");
            } else if (P.objective == 14) {
                Require(targetOptions->Pair && !targetOptions->Query && !targetOptions->Yeti,
                        "Ordered PairLogit objective options do not match");
                const auto& pair = *targetOptions->Pair;
                Require(pair.group_count == groupCount && pair.pair_count && pair.pair_count <= UINT32_MAX / 2 &&
                        !pair.reserved0 && !pair.reserved1 && targetOptions->Winners && targetOptions->Losers && targetOptions->PairWeights,
                        "Invalid Ordered PairLogit supplied pair arrays");
                std::vector<double> mass(P.rows, 0);
                double totalPairMass = 0;
                for (uint32_t edge = 0; edge < pair.pair_count; ++edge) {
                    const uint32_t win = targetOptions->Winners[edge], lose = targetOptions->Losers[edge];
                    const float weight = targetOptions->PairWeights[edge];
                    Require(win < P.rows && lose < P.rows && win != lose && std::isfinite(weight) && weight >= 0,
                            "Invalid Ordered PairLogit endpoint or weight");
                    Require(std::upper_bound(groupOffsets, groupOffsets + groupCount + 1, win) ==
                            std::upper_bound(groupOffsets, groupOffsets + groupCount + 1, lose),
                            "Ordered PairLogit edges must remain inside one query");
                    mass[win] += weight; mass[lose] += weight; totalPairMass += weight;
                }
                Require(totalPairMass > 0 && std::isfinite(totalPairMass) && totalPairMass * 2 < 1e30,
                        "Ordered PairLogit requires positive finite total incident mass");
                for (uint32_t row = 0; row < P.rows; ++row) unitWeights[row] = mass[row];
                weights = unitWeights.data(); pairTargets.assign(P.rows, 0); targets = pairTargets.data();
            } else if (P.objective == 19) {
                Require(targetOptions->Combination && targetOptions->Components && !targetOptions->Query && !targetOptions->Pair && !targetOptions->Yeti,
                        "Ordered Combination objective options do not match");
                CombinationOptions = *targetOptions->Combination;
                Require(CombinationOptions.component_count && CombinationOptions.component_count <= 128 && !CombinationOptions.reserved,
                        "Invalid Ordered Combination component count/options");
                for (uint32_t component = 0; component < CombinationOptions.component_count; ++component) {
                    const auto& c = targetOptions->Components[component];
                    CombinationYetiCount += c.objective == 17;
                    Require(c.objective < 12 || (groupCount && groupOffsets), "Ordered Combination query components require group offsets");
                }
            } else {
                Require(P.objective == 17 && targetOptions->Yeti && !targetOptions->Query && !targetOptions->Pair,
                        "Ordered YetiRank objective options do not match");
                Require(P.leaf_method == 0, "Ordered YetiRank requires Newton leaves");
                YetiOptions = *targetOptions->Yeti;
                Require(YetiOptions.group_count == groupCount && YetiOptions.legacy_prefix_centering <= 1 &&
                        YetiOptions.permutations && YetiOptions.permutations <= 10000 &&
                        std::isfinite(YetiOptions.decay) && YetiOptions.decay >= 0 && YetiOptions.decay <= 1,
                        "Invalid Ordered YetiRank options");
            }
        }
        if (!weights) weights = unitWeights.data();
        double total = 0;
        for (uint32_t row = 0; row < P.rows; ++row) {
            Require(std::isfinite(targets[row]) && std::isfinite(weights[row]) && weights[row] >= 0 &&
                    (!initial || std::isfinite(initial[row])), "Ordered inputs must be finite with nonnegative weights");
            if (P.objective == 1) Require(targets[row] == 0 || targets[row] == 1, "Logloss targets must be zero or one");
            if (P.objective == 2) Require(targets[row] >= 0 && targets[row] <= 1, "CrossEntropy targets must be in [0,1]");
            if (P.objective == 3 || P.objective == 7 || P.objective == 13) Require(targets[row] >= 0, "Poisson/Tweedie targets must be nonnegative");
            total += weights[row];
        }
        Require(total > 0 && total < 1e30, "Ordered total weight must be positive and below 1e30");
        std::vector<uint32_t> candidatePairs(uint64_t(P.candidates) * 2), types(P.candidates, 0);
        for (uint32_t candidate = 0; candidate < P.candidates; ++candidate) {
            types[candidate] = candidateTypes ? candidateTypes[candidate] : 0;
            candidatePairs[candidate * 2] = features[candidate];
            candidatePairs[candidate * 2 + 1] = borders[candidate] | (types[candidate] << 31);
        }
        std::vector<uint8_t> rowBins(dataCells * BinBanks);
        for (uint32_t bank = 0; bank < BinBanks; ++bank)
            for (uint32_t feature = 0; feature < P.features; ++feature)
                for (uint32_t row = 0; row < P.rows; ++row)
                    rowBins[bank * dataCells + uint64_t(row) * P.features + feature] = bins[bank * dataCells + uint64_t(feature) * P.rows + row];
        std::vector<float> cursors(CursorCount), published(P.rows, P.bias), featureOptions(uint64_t(P.features) * 4, 0);
        if (initial) std::copy(initial, initial + P.rows, published.begin());
        for (const auto& task : Descriptors)
            for (uint32_t position = 0; position < task.QualityEnd; ++position)
                cursors[task.CursorOffset + position] = published[permutations[uint64_t(task.Reserved) * P.rows + position]];
        for (uint32_t feature = 0; feature < P.features; ++feature) featureOptions[feature * 4] = featureOptions[feature * 4 + 1] = 1;
        if (P.objective == 20) {
            Require(targetOptions && targetOptions->CustomSource && P.leaf_method <= 1, "Ordered Custom requires compiled source and Newton/Gradient leaves");
            CustomContext = std::make_unique<Runtime>(targetOptions->CustomSource);
        }
        auto& context = Context(); Text(Stats.device_name, sizeof(Stats.device_name), context.Device.name.UTF8String);
        Bins = context.Buffer(dataCells * BinBanks, rowBins.data()); Targets = context.Buffer(P.rows * 4ull, targets);
        Weights = context.Buffer(P.rows * 4ull, weights); Permutations = context.Buffer(uint64_t(P.rows) * P.permutations * 4, permutations);
        TaskBuffer = context.Buffer(Tasks * 16ull, Descriptors.data()); Cursor = context.Buffer(CursorCount * 4ull, cursors.data());
        NextCursor = context.Buffer(CursorCount * 4ull); Published = context.Buffer(P.rows * 4ull, published.data());
        NextPublished = context.Buffer(P.rows * 4ull); Derivatives = context.Buffer(CursorCount * 8ull);
        LeafIds = context.Buffer(uint64_t(P.rows) * BinBanks * 4); RawValues = context.Buffer(uint64_t(Tasks) * MaxLeaves * 4);
        LeafWeights = context.Buffer(uint64_t(Tasks) * MaxLeaves * 4); CandidatePairs = context.Buffer(P.candidates * 8ull, candidatePairs.data());
        CandidateTypes = context.Buffer(P.candidates * 4ull, types.data()); FeatureOptions = context.Buffer(P.features * 16ull, featureOptions.data());
        Statistics = context.Buffer(statBytes); Scores = context.Buffer(P.candidates * 8ull);
        if (histogramPlan) {
            Histogram = std::make_unique<CBMOrderedHistogramWorkspace>(context, std::move(*histogramPlan));
            for (uint32_t bank = 0; bank < BinBanks; ++bank)
                FeatureBanks.push_back(context.Buffer(dataCells, bins + bank * dataCells));
            TileScores = context.Buffer(uint64_t(BatchSize) * 8);
        }
        Winner = context.Buffer(sizeof(SplitState)); Status = context.Buffer(4);
        LossGroups = std::min(4096u, (P.rows + 255) / 256); LossPartials = context.Buffer(LossGroups * 4ull);
        if (P.leaf_method == 2) {
            ExactTargets = context.Buffer(P.rows * 4ull); ExactWeights = context.Buffer(P.rows * 4ull);
            ExactPredictions = context.Buffer(P.rows * 4ull); ExactLeafIds = context.Buffer(P.rows * 4ull);
            ExactResiduals = context.Buffer(P.rows * 4ull); ExactEffective = context.Buffer(P.rows * 4ull);
            ExactKeysA = context.Buffer(P.rows * 4ull); ExactRowsA = context.Buffer(P.rows * 4ull);
            ExactKeysB = context.Buffer(P.rows * 4ull); ExactRowsB = context.Buffer(P.rows * 4ull);
            ExactOffsets = context.Buffer((MaxLeaves + 1ull) * 4); ExactPartials = context.Buffer(uint64_t(MaxLeaves) * ExactTiles * 16);
            ExactTileOffsets = context.Buffer(uint64_t(MaxLeaves) * ExactTiles * 8); ExactTotals = context.Buffer(MaxLeaves * 8ull);
            ExactSelected = context.Buffer(MaxLeaves * 4ull);
        }
        K = {}; K.Rows = P.rows; K.Objective = P.objective; K.TotalWeight = total; K.ObjectiveParam = P.objective_param;
        if (IsQuery) {
            std::vector<uint32_t> singletonOffsets;
            if (!groupCount) {
                singletonOffsets.resize(P.rows + 1); for (uint32_t row = 0; row <= P.rows; ++row) singletonOffsets[row] = row;
                groupCount = P.rows; groupOffsets = singletonOffsets.data();
            }
            CreateQueryState(targetOptions, groupCount, groupOffsets, targets, weights, permutations);
        }
        Loss = ReadLoss(Published);
    }
    void CreateQueryState(const OrderedTargetOptions* options, uint32_t groupCount, const uint32_t* groupOffsets,
        const float* targets, const float* weights, const uint32_t* permutations) {
        Require(WorkingBytes + uint64_t(CursorCount) * 20 + uint64_t(Tasks) * 32 + 4 <= MemoryLimit,
                "Ordered query cursor arrays exceed the 1 GiB workspace guard");
        if (options->Pair) Require(WorkingBytes + uint64_t(CursorCount) * 24 + uint64_t(options->Pair->pair_count) * 44 <= MemoryLimit,
                "Ordered pair query metadata exceeds the 1 GiB workspace guard");
        std::vector<float> expandedTargets(CursorCount), expandedWeights(CursorCount);
        std::vector<uint32_t> offsets(1, 0), winners, losers, rowOffsets(CursorCount + 1, 0);
        std::vector<float> edgeWeights;
        std::vector<std::array<uint32_t, 2>> queryEdges;
        std::vector<CBMOrderedYetiBlock> yetiBlocks;
        std::vector<CBMOrderedCombinationBlock> combinationBlocks;
        std::vector<CBMOrderedFold> estimateTasks = Descriptors;
        std::vector<std::vector<uint32_t>> originalEdges(groupCount);
        if (options->Pair) for (uint32_t e = 0; e < options->Pair->pair_count; ++e) {
            const uint32_t group = std::upper_bound(groupOffsets, groupOffsets + groupCount + 1, options->Winners[e]) - groupOffsets - 1;
            originalEdges[group].push_back(e);
        }
        YetiWeakBlocks.resize(LearnPermutations);
        for (uint32_t taskId = 0; taskId < Tasks; ++taskId) {
            const auto& task = Descriptors[taskId];
            const uint32_t beginQuery = offsets.size() - 1;
            uint32_t estimateQueryEnd = beginQuery, prefixEdges = 0;
            std::vector<uint32_t> localOffsets(1, 0);
            for (uint32_t position = 0; position < task.QualityEnd;) {
                const uint32_t firstRow = permutations[uint64_t(task.Reserved) * P.rows + position];
                const uint32_t originalGroup = std::lower_bound(groupOffsets, groupOffsets + groupCount, firstRow) - groupOffsets;
                Require(originalGroup < groupCount && groupOffsets[originalGroup] == firstRow,
                        "Ordered query metadata requires whole-query histories");
                const uint32_t rows = groupOffsets[originalGroup + 1] - firstRow;
                Require(position + rows <= task.QualityEnd && !(position < task.EstimateEnd && position + rows > task.EstimateEnd),
                        "Ordered query fold cuts a query");
                const uint32_t edgeBegin = winners.size();
                for (const auto edge : originalEdges[originalGroup]) {
                    winners.push_back(task.CursorOffset + position + options->Winners[edge] - firstRow);
                    losers.push_back(task.CursorOffset + position + options->Losers[edge] - firstRow);
                    edgeWeights.push_back(options->PairWeights[edge]);
                    if (position < task.EstimateEnd) ++prefixEdges;
                }
                queryEdges.push_back({edgeBegin, static_cast<uint32_t>(winners.size())});
                for (uint32_t i = 0; i < rows; ++i) {
                    expandedTargets[task.CursorOffset + position + i] = targets[firstRow + i];
                    expandedWeights[task.CursorOffset + position + i] = weights[firstRow + i];
                }
                position += rows; offsets.push_back(task.CursorOffset + position); localOffsets.push_back(position);
                if (position <= task.EstimateEnd) estimateQueryEnd = offsets.size() - 1;
            }
            const bool active = P.objective == 19 || taskId + 1 == Tasks || (P.objective == 14 ? prefixEdges > 0
                : estimateQueryEnd - beginQuery < task.EstimateEnd);
            if (!active) estimateTasks[taskId].EstimateEnd = 0;
            HostQueryTaskRanges.push_back({beginQuery, active ? estimateQueryEnd : beginQuery,
                static_cast<uint32_t>(offsets.size() - 1), 0});
            if (P.objective == 17 || P.objective == 19) {
                auto block = [&](uint32_t begin, uint32_t end) {
                    if (begin == end) return UINT32_MAX;
                    CBMOrderedYetiBlock item; item.CursorOffset = task.CursorOffset + begin;
                    for (uint32_t boundary : localOffsets) if (boundary >= begin && boundary <= end)
                        item.GroupOffsets.push_back(boundary - begin);
                    if (P.objective == 19) {
                        CBMOrderedCombinationBlock combination;
                        combination.CursorOffset = item.CursorOffset; combination.GroupOffsets = item.GroupOffsets;
                        for (uint32_t position = begin; position < end; ++position)
                            combination.OriginalRows.push_back(permutations[uint64_t(task.Reserved) * P.rows + position]);
                        const uint32_t id = combinationBlocks.size(); combinationBlocks.push_back(std::move(combination)); return id;
                    }
                    const uint32_t id = yetiBlocks.size(); yetiBlocks.push_back(std::move(item)); return id;
                };
                if (taskId + 1 < Tasks) {
                    YetiWeakBlocks[task.Reserved].push_back(block(0, task.EstimateEnd));
                    YetiWeakBlocks[task.Reserved].push_back(block(task.EstimateEnd, task.QualityEnd));
                }
                if (active) YetiLeafBlocks.push_back(block(0, task.EstimateEnd));
            }
        }
        QueryCount = offsets.size() - 1; FlatPairCount = winners.size();
        Require(winners.size() <= UINT32_MAX / 2, "Ordered duplicated pairs exceed GPU indexing");
        const uint64_t bytes = uint64_t(CursorCount) * 20 + uint64_t(QueryCount) * 12 + 4 + uint64_t(Tasks) * 32
            + (P.objective == 14 ? uint64_t(FlatPairCount) * 44 + uint64_t(CursorCount + 1ull) * 4 + uint64_t(QueryCount) * 8 : 0)
            + (P.objective == 17 ? CBMOrderedYetiWorkspace::PlannedBytes(yetiBlocks) : 0)
            + (P.objective == 19 ? uint64_t(CursorCount) * 4 + CBMOrderedCombinationWorkspace::PlannedBytes(combinationBlocks,
                CombinationOptions, options->Components, P.rows, options->Winners, options->Losers, options->PairWeights) : 0);
        Require(WorkingBytes + bytes <= MemoryLimit, "Ordered query histories exceed the 1 GiB workspace guard");
        auto& context = Context();
        QueryTargets = context.Buffer(CursorCount * 4ull, expandedTargets.data());
        QueryWeights = context.Buffer(CursorCount * 4ull, expandedWeights.data());
        QueryPoint = context.Buffer(CursorCount * 4ull); QueryGradient = context.Buffer(CursorCount * 4ull);
        QueryHessian = context.Buffer(CursorCount * 4ull); QueryOffsets = context.Buffer(offsets.size() * 4ull, offsets.data());
        QueryStatistics = context.Buffer(QueryCount * 8ull); QueryTaskRanges = context.Buffer(Tasks * 16ull, HostQueryTaskRanges.data());
        EstimationTasks = context.Buffer(Tasks * 16ull, estimateTasks.data());
        if (P.objective == 14) {
            for (uint32_t e = 0; e < FlatPairCount; ++e) { ++rowOffsets[winners[e] + 1]; ++rowOffsets[losers[e] + 1]; }
            for (uint32_t row = 0; row < CursorCount; ++row) rowOffsets[row + 1] += rowOffsets[row];
            auto next = rowOffsets;
            std::vector<uint32_t> incidence(uint64_t(FlatPairCount) * 2);
            std::vector<int32_t> signs(uint64_t(FlatPairCount) * 2);
            for (uint32_t e = 0; e < FlatPairCount; ++e) {
                const uint32_t win = next[winners[e]]++, lose = next[losers[e]]++;
                incidence[win] = incidence[lose] = e; signs[win] = 1; signs[lose] = -1;
            }
            PairWinners = context.Buffer(FlatPairCount * 4ull, winners.data()); PairLosers = context.Buffer(FlatPairCount * 4ull, losers.data());
            PairWeights = context.Buffer(FlatPairCount * 4ull, edgeWeights.data()); PairRowOffsets = context.Buffer(rowOffsets.size() * 4ull, rowOffsets.data());
            PairIncidence = context.Buffer(incidence.size() * 4ull, incidence.data()); PairSigns = context.Buffer(signs.size() * 4ull, signs.data());
            PairEdges = context.Buffer(FlatPairCount * 16ull); PairQueryRanges = context.Buffer(QueryCount * 8ull, queryEdges.data());
        }
        if (P.objective == 17) Yeti = std::make_unique<CBMOrderedYetiWorkspace>(context, yetiBlocks,
            YetiOptions.permutations, YetiOptions.decay, YetiOptions.legacy_prefix_centering != 0);
        if (P.objective == 19) {
            Combination = std::make_unique<CBMOrderedCombinationWorkspace>(context, combinationBlocks, CombinationOptions,
                options->Components, targets, weights, P.rows, options->Winners, options->Losers, options->PairWeights);
            QueryGradientWeights = context.Buffer(CursorCount * 4ull);
        }
        WorkingBytes += bytes;
    }
    void EncodeQueryDerivatives(Command& command, id<MTLBuffer> values, uint32_t mode, uint32_t selected,
        bool structure, bool objectiveOnly = false, id<MTLBuffer> predictions = nil, bool trial = false) {
        LangevinStructureCall = structure;
        const auto step = StepConfiguration(Pending.Active ? 1u << Pending.Selected.size() : 1, selected);
        command.Dispatch("OrderedQueryPreparePoint", {Cursor, predictions ? predictions : Published, values, Permutations,
            LeafIds, TaskBuffer, QueryPoint, Status}, P, P.rows, false, Tasks, 1, &step, &mode, sizeof(mode));
        if (P.objective == 19) {
            const auto& blocks = structure ? YetiWeakBlocks[selected] : YetiLeafBlocks;
            auto& seeds = structure ? YetiSeeds : YetiLeafSeeds;
            auto& position = structure ? YetiSeedPosition : YetiLeafSeedPosition;
            const uint32_t count = objectiveOnly || Combination->HasYetiSeedCallback() ? 0 : CombinationYetiCount;
            Require(position + uint64_t(blocks.size()) * count <= seeds.size(), "Ordered Combination Yeti seed packet is missing or exhausted");
            if (structure) { command.Zero(QueryGradient); command.Zero(QueryHessian); command.Zero(QueryGradientWeights); }
            uint32_t blockIndex = 0;
            for (uint32_t block : blocks) {
                if (block != UINT32_MAX) Combination->EncodeBlock(command, block, QueryPoint, QueryGradient, QueryHessian,
                    QueryGradientWeights, count ? seeds.data() + position : nullptr, count, trial, objectiveOnly);
                else if (Combination->HasYetiSeedCallback() && !objectiveOnly) {
                    for (uint32_t c = 0; c < CombinationYetiCount; ++c) {
                        uint64_t unused; Require(!CombinationSeedCallback(CombinationSeedContext, &unused), "Ordered Combination seed callback failed");
                    }
                }
                position += count;
                if (LangevinEnabled && structure && (++blockIndex % 2 == 0)) LangevinSeed(CBM_LANGEVIN_WEAK_SEED_CACHE);
            }
        } else if (P.objective == 17) {
            if (structure) { command.Zero(QueryGradient); command.Zero(QueryHessian); }
            Require(!objectiveOnly, "YetiRank does not support objective backtracking");
            auto& seeds = structure ? YetiSeeds : YetiLeafSeeds;
            auto& position = structure ? YetiSeedPosition : YetiLeafSeedPosition;
            const auto& blocks = structure ? YetiWeakBlocks[selected] : YetiLeafBlocks;
            Require(LangevinEnabled || position + blocks.size() <= seeds.size(), "Ordered YetiRank oracle seed packet is missing or exhausted");
            uint32_t blockIndex = 0;
            for (const uint32_t block : blocks) {
                const uint64_t seed = LangevinEnabled ? LangevinSeed(structure ? CBM_LANGEVIN_YETI_WEAK : CBM_LANGEVIN_YETI_LEAF) : seeds[position++];
                if (block != UINT32_MAX) Yeti->EncodeBlock(command, block, QueryPoint, QueryTargets, QueryWeights,
                    QueryGradient, QueryHessian, Status, seed);
                if (LangevinEnabled && structure && (++blockIndex % 2 == 0)) LangevinSeed(CBM_LANGEVIN_WEAK_SEED_CACHE);
            }
        } else if (P.objective == 14) {
            const NativePairParams pairs = {CursorCount, FlatPairCount, 14, 0, 1, 0, 0, 0};
            command.Dispatch("PairLogitEdgeDerivatives", {QueryPoint, PairWinners, PairLosers, PairWeights, PairEdges}, pairs, FlatPairCount);
            if (!objectiveOnly) command.Dispatch("ReducePairwiseRows", {PairRowOffsets, PairIncidence, PairSigns, PairEdges,
                QueryGradient, QueryHessian, QueryWeights}, pairs, CursorCount, true);
            command.Dispatch("OrderedPairQueryStatistics", {PairEdges, PairQueryRanges, QueryStatistics}, QueryCount, QueryCount, true);
        } else {
            const NativeQueryParams query = {CursorCount, QueryCount, P.objective, 0, QueryOptions.beta, QueryOptions.lambda, 1, 0};
            command.Dispatch(P.objective == 12 ? "QueryRmseDerivatives" : "QuerySoftMaxDerivatives",
                {QueryTargets, QueryWeights, QueryPoint, QueryOffsets, QueryGradient, QueryHessian, QueryStatistics}, query, QueryCount, true);
        }
        if (structure) command.Dispatch("OrderedQueryPublishDerivatives", {QueryGradient, QueryHessian,
            P.objective == 17 ? QueryHessian : (P.objective == 19 ? QueryGradientWeights : QueryWeights), Derivatives, Status},
            P, CursorCount, false, 1, 1, &step);
    }
    float ReadQueryLoss(id<MTLBuffer> predictions) {
        Command command(Stats, CustomContext.get());
        EncodeQueryDerivatives(command, RawValues, 2, 0, false, true, predictions); command.Wait(); CheckStatus();
        if (Combination) return Combination->ReadMetric(YetiLeafBlocks.back());
        const auto& range = HostQueryTaskRanges.back();
        const float* statistics = static_cast<const float*>(QueryStatistics.contents);
        double value = 0, mass = 0;
        for (uint32_t q = range[0]; q < range[2]; ++q) { value += statistics[2 * q]; mass += statistics[2 * q + 1]; }
        Require(std::isfinite(value) && value >= 0 && std::isfinite(mass) && mass > 0, "Ordered query objective became invalid");
        value /= mass;
        if (P.objective == 12) value = std::sqrt(value);
        Require(std::isfinite(value), "Ordered query objective became nonfinite"); return value;
    }
    void YetiSeedShape(uint32_t selected, uint32_t* weakCount, uint32_t* leafCount) const {
        Require((Yeti || (Combination && CombinationYetiCount)) && selected < LearnPermutations && weakCount && leafCount, "Ordered YetiRank seed shape requires a valid learning permutation");
        const uint32_t components = Combination ? CombinationYetiCount : 1;
        *weakCount = YetiWeakBlocks[selected].size() * components;
        *leafCount = YetiLeafBlocks.size() * (P.leaf_iterations == 1 ? 1 : P.leaf_iterations + 1) * components;
    }
    void SetYetiSeeds(uint32_t count, const uint64_t* seeds, bool leafOnly) {
        Require(!Failed && (Yeti || (Combination && CombinationYetiCount)) && seeds, "Ordered YetiRank seed packet is required");
        const uint32_t components = Combination ? CombinationYetiCount : 1;
        if (leafOnly) {
            Require(Pending.Active && YetiLeafSeedPosition == 0, "Set Ordered YetiRank leaf seeds after begin and before finish");
            const uint32_t expected = YetiLeafBlocks.size() * (P.leaf_iterations == 1 ? 1 : P.leaf_iterations + 1) * components;
            Require(count == expected, "Ordered YetiRank leaf seed packet has the wrong size");
            YetiLeafSeeds.assign(seeds, seeds + count);
        } else {
            Require(!Pending.Active && Completed < P.iterations, "Set Ordered YetiRank weak seeds before begin");
            Require(count && count <= 2 * FoldCount * components, "Ordered YetiRank weak seed packet has the wrong size");
            YetiSeeds.assign(seeds, seeds + count); YetiSeedPosition = 0;
            YetiLeafSeeds.clear(); YetiLeafSeedPosition = 0;
        }
    }
    void SetCombinationSeedCallback(CBMCombinationYetiSeedCallback callback, void* context) {
        Require(!Failed && Combination && (!callback || Pending.Active), "Configure Ordered Combination leaf seed callback after begin");
        Combination->SetYetiSeedCallback(callback, context); CombinationSeedCallback = callback; CombinationSeedContext = context;
    }
    double ReadCombinationObjective(bool allowNonfinite = false) const {
        double result = 0;
        const float* masses = TaskMass ? static_cast<const float*>(TaskMass.contents) : nullptr;
        for (uint32_t task = 0; task < YetiLeafBlocks.size(); ++task) {
            const double value = Combination->ReadObjective(YetiLeafBlocks[task], allowNonfinite);
            result += P.normalize ? (masses[task] > 0 ? value / masses[task] : 0) : value;
        }
        return result;
    }
    void EstimateCombinationBacktracking(const StepParams& step) {
        auto project = [&](Command& command) {
            command.Dispatch("OrderedQueryBacktrackingDirections", {QueryGradient, QueryHessian, QueryWeights, Permutations,
                LeafIds, EstimationTasks, RawValues, Directions, LeafWeights, DirectionDot, TaskMass, Status},
                P, step.Leaves, true, Tasks, 1, &step);
        };
        Command initial(Stats, CustomContext.get());
        EncodeQueryDerivatives(initial, RawValues, 1, step.SelectedPermutation, false); project(initial);
        initial.Wait(); CheckStatus();
        double current = AddObjectiveRidge(ReadCombinationObjective(), RawValues, step.Leaves), dot = ReadDirectionDot(step.Leaves);
        BacktrackingParams backtracking = {1, BacktrackingType, P.reserved1, P.normalize};
        bool updated = false, newDirection = false;
        for (uint32_t attempt = 0; attempt < P.leaf_iterations || (!updated && attempt < 100); ++attempt) {
            Command trial(Stats, CustomContext.get());
            if (newDirection) project(trial);
            trial.Dispatch("OrderedBacktrackingCandidate", {RawValues, Directions, LeafWeights, TrialValues},
                P, uint64_t(Tasks) * step.Leaves, false, 1, 1, &step, &backtracking, sizeof(backtracking));
            EncodeQueryDerivatives(trial, TrialValues, 1, step.SelectedPermutation, false, false, nil, true);
            trial.Wait(); CheckStatus();
            if (newDirection) dot = ReadDirectionDot(step.Leaves);
            const double candidate = AddObjectiveRidge(ReadCombinationObjective(true), TrialValues, step.Leaves);
            const double threshold = current + (BacktrackingType == 2 ? 1e-5 * backtracking.Step * dot : 0);
            if (std::isfinite(candidate) && candidate >= threshold) {
                std::swap(RawValues, TrialValues); current = candidate; updated = true; newDirection = true; backtracking.Step = 1;
            } else { backtracking.Step *= .5f; newDirection = false; }
        }
    }
    StepParams StepConfiguration(uint32_t leaves = 1, uint32_t selected = 0) const { return {Tasks, leaves, CursorCount, selected}; }
    void Info(CBMStepInfo& info) const {
        info = {}; info.completed_iterations = Completed; info.finished = Completed == P.iterations;
        info.loss = Loss; info.stats = Stats;
    }
    float ReadLoss(id<MTLBuffer> predictions) {
        if (P.objective == 17) return 0;
        if (IsQuery) return ReadQueryLoss(predictions);
        Command command(Stats, CustomContext.get());
        command.Dispatch("ReduceObjectiveLoss", {Targets, Weights, predictions, LossPartials}, K, LossGroups, true);
        command.Wait();
        const float* partials = static_cast<const float*>(LossPartials.contents);
        double sum = 0; for (uint32_t group = 0; group < LossGroups; ++group) sum += partials[group];
        if (P.objective == 0) sum = std::sqrt(sum);
        Require(std::isfinite(sum), "Ordered objective became nonfinite"); return sum;
    }
    void CheckStatus() const {
        Require(!*static_cast<const uint32_t*>(Status.contents), "Ordered GPU arithmetic became nonfinite");
        if (Combination) Combination->CheckStatus();
    }
    void ConfigureRidge(uint32_t enabled) {
        Require(!Completed && !Pending.Active && !Failed && enabled <= 1,
                "Configure Ordered ridge with a boolean flag before the first tree");
        P.reserved1 = enabled;
    }
    double AddObjectiveRidge(double value, id<MTLBuffer> values, uint32_t leaves) const {
        if (P.reserved1) {
            const float* points = static_cast<const float*>(values.contents);
            for (uint32_t task = 0; task < Tasks; ++task) for (uint32_t leaf = 0; leaf < leaves; ++leaf) {
                const double point = points[uint64_t(task) * MaxLeaves + leaf];
                value -= 0.5 * double(P.l2) * point * point;
            }
        }
        return value;
    }
    void ConfigureBootstrap(const CBMBootstrapOptions* options, uint32_t testOnly) {
        Require(!Completed && !Pending.Active && !Failed && options, "Configure Ordered bootstrap before the first step");
        Require(options->bootstrap_type <= 4 && testOnly <= 1, "Unknown Ordered bootstrap type");
        Require(options->mvs_reg_is_set <= 1 && options->initial_mvs_lambda_is_set <= 1 && !options->reserved0 && !options->reserved1,
                "Invalid Ordered bootstrap flags");
        Require((!options->mvs_reg_is_set || (std::isfinite(options->mvs_reg) && options->mvs_reg >= 0)) &&
                (!options->initial_mvs_lambda_is_set || (std::isfinite(options->initial_mvs_lambda) && options->initial_mvs_lambda >= 0)),
                "Invalid Ordered MVS regularization");
        Require(std::isfinite(options->bagging_temperature) && options->bagging_temperature >= 0 &&
                std::isfinite(options->subsample) && options->subsample > 0 && options->subsample <= 1 &&
                (options->bootstrap_type != 3 || options->subsample < 1) && options->iteration_offset <= UINT32_MAX - P.iterations,
                "Invalid Ordered bootstrap parameters");
        const uint32_t mvsGroups = std::min(4096u, (PermutationCursorCount + 255) / 256);
        const uint64_t mvsBytes = options->bootstrap_type == 4 ? PermutationCursorCount * 4ull +
            ((PermutationCursorCount + 8191) / 8192) * 4ull + mvsGroups * 8ull : 0;
        const uint64_t extra = options->bootstrap_type ? uint64_t(PermutationCursorCount) * 4 + uint64_t(CursorCount) * 8 + mvsBytes : 0;
        Require(WorkingBytes + extra + NoiseBytes + BacktrackingBytes <= MemoryLimit, "Ordered bootstrap exceeds 1 GiB workspace");
        Bootstrap = *options; BootstrapTestOnly = testOnly; BootstrapBytes = extra;
        if (extra) { Multipliers = Context().Buffer(PermutationCursorCount * 4ull); Sampled = Context().Buffer(CursorCount * 8ull); }
        else { Multipliers = nil; Sampled = nil; }
        MvsGroups = mvsGroups; HasMvsLambda = options->initial_mvs_lambda_is_set; MvsLambda = options->initial_mvs_lambda;
        if (mvsBytes) {
            MvsInput = Context().Buffer(PermutationCursorCount * 4ull);
            MvsThresholds = Context().Buffer(((PermutationCursorCount + 8191) / 8192) * 4ull);
            MvsStatistics = Context().Buffer(MvsGroups * 8ull);
        } else { MvsInput = nil; MvsThresholds = nil; MvsStatistics = nil; }
    }
    void ConfigureNoise(const CBMScoreNoiseOptions* options) {
        Require(!Completed && !Pending.Active && !Failed && options, "Configure Ordered score noise before the first step");
        Require(std::isfinite(options->random_strength) && options->random_strength >= 0 &&
                !options->reserved0 && !options->reserved1 && !options->reserved2, "Invalid Ordered random_strength");
        const uint64_t extra = options->random_strength > 0 ? P.features * 4ull + FoldCount * 8ull : 0;
        Require(WorkingBytes + BootstrapBytes + extra + BacktrackingBytes <= MemoryLimit, "Ordered score noise exceeds 1 GiB workspace");
        RandomStrength = options->random_strength; NoiseBytes = extra;
        if (extra) { FeatureNoise = Context().Buffer(P.features * 4ull); QualityStatistics = Context().Buffer(FoldCount * 8ull); }
        else { FeatureNoise = nil; QualityStatistics = nil; }
    }
    void BootstrapState(uint32_t* iteration, float* lambda, uint32_t* valid) const {
        Require(!Pending.Active && !Failed, "Ordered bootstrap state is available only between completed trees");
        Require(iteration && lambda && valid, "Ordered bootstrap state outputs are required");
        *iteration = Bootstrap.iteration_offset + Completed; *valid = HasMvsLambda; *lambda = HasMvsLambda ? MvsLambda : 0;
    }
    void ConfigureBacktracking(uint32_t type) {
        Require(!Completed && !Pending.Active && !Failed && type <= 2, "Configure valid Ordered backtracking before the first step");
        Require(P.objective != 17 || type == 0, "Ordered YetiRank requires No leaf backtracking");
        const uint64_t extra = type && P.leaf_iterations > 1 && P.leaf_method != 2 ? uint64_t(Tasks) * (MaxLeaves * 16ull + 12) : 0;
        Require(WorkingBytes + BootstrapBytes + NoiseBytes + extra <= MemoryLimit, "Ordered backtracking exceeds 1 GiB workspace");
        BacktrackingType = type; BacktrackingBytes = extra;
        if (extra) {
            TrialValues = Context().Buffer(Tasks * MaxLeaves * 4ull);
            Directions = Context().Buffer(Tasks * MaxLeaves * 4ull);
            DirectionDot = Context().Buffer(Tasks * MaxLeaves * 8ull);
            TaskMass = Context().Buffer(Tasks * 4ull);
            BacktrackingLoss = Context().Buffer(Tasks * 8ull);
        } else { TrialValues = nil; Directions = nil; DirectionDot = nil; TaskMass = nil; BacktrackingLoss = nil; }
    }
    uint64_t LangevinSeed(uint32_t event) {
        uint64_t seed = 0;
        Require(LangevinSeedCallback && !LangevinSeedCallback(LangevinContext, event, &seed),
                "Ordered Langevin seed callback failed");
        return seed;
    }
    static int LangevinCombinationSeed(void* context, uint64_t* seed) {
        try {
            auto* session = static_cast<Session*>(context);
            *seed = session->LangevinSeed(session->LangevinStructureCall ? CBM_LANGEVIN_YETI_WEAK : CBM_LANGEVIN_YETI_LEAF);
            return 0;
        } catch (...) { return -1; }
    }
    void ConfigureLangevin(float temperature, CBMLangevinNoiseCallback noise,
        CBMLangevinSeedCallback seed, void* context) {
        Require(!Completed && !Pending.Active && !Failed && !LangevinEnabled && noise && seed &&
                std::isfinite(temperature) && temperature >= 0,
                "Configure finite nonnegative Ordered Langevin temperature and callbacks before training");
        std::vector<uint32_t> active;
        const auto* tasks = IsQuery ? static_cast<const CBMOrderedFold*>(EstimationTasks.contents) : Descriptors.data();
        for (uint32_t task = 0; task < Tasks; ++task) if (tasks[task].EstimateEnd) active.push_back(task);
        Require(!active.empty(), "Ordered Langevin requires an active estimation task");
        const uint64_t packed = uint64_t(active.size()) * MaxLeaves;
        const uint64_t extra = packed * 32 + uint64_t(Tasks) * (MaxLeaves * 16ull + 12) + active.size() * 4ull;
        Require(WorkingBytes + BootstrapBytes + NoiseBytes + BacktrackingBytes + extra <= MemoryLimit,
                "Ordered Langevin workspace exceeds 1 GiB");
        auto& contextRuntime = Context();
        LangevinTaskIds = contextRuntime.Buffer(active.size() * 4ull, active.data());
        LangevinStatistics = contextRuntime.Buffer(packed * 16);
        LangevinGradientNoise = contextRuntime.Buffer(packed * 8); LangevinHessianNoise = contextRuntime.Buffer(packed * 8);
        LangevinTrialValues = contextRuntime.Buffer(uint64_t(Tasks) * MaxLeaves * 4);
        LangevinDirections = contextRuntime.Buffer(uint64_t(Tasks) * MaxLeaves * 4);
        LangevinDirectionDot = contextRuntime.Buffer(uint64_t(Tasks) * MaxLeaves * 8);
        LangevinTaskMass = contextRuntime.Buffer(Tasks * 4ull); LangevinLoss = contextRuntime.Buffer(Tasks * 8ull);
        LangevinActiveTasks = active.size(); WorkingBytes += extra;
        LangevinTemperature = temperature; LangevinNoiseCallback = noise; LangevinSeedCallback = seed;
        LangevinContext = context; LangevinEnabled = true;
        if (CombinationYetiCount) {
            CombinationSeedCallback = LangevinCombinationSeed; CombinationSeedContext = this;
            Combination->SetYetiSeedCallback(CombinationSeedCallback, CombinationSeedContext);
        }
    }
    void EncodeLangevinWeak(Command& command, uint32_t selected) {
        LangevinSeed(CBM_LANGEVIN_WEAK_SEED_CACHE);
        const float coefficient = LangevinTemperature == 0 ? 0 : std::sqrt(2.0 / P.learning_rate / LangevinTemperature);
        Require(std::isfinite(coefficient), "Ordered Langevin weak coefficient exceeds float32");
        const uint32_t taskOffset = PermutationTaskOffsets[selected];
        for (uint32_t fold = 0; fold < PermutationFoldCounts[selected]; ++fold) {
            const auto& task = Descriptors[taskOffset + fold];
            for (uint32_t side = 0; side < 2; ++side) {
                const uint32_t begin = side ? task.EstimateEnd : 0, count = side ? task.QualityEnd - task.EstimateEnd : task.EstimateEnd;
                if (!count || coefficient == 0) continue;
                LangevinWeakParams params = {{count, 0, Bootstrap.random_seed_low, Bootstrap.random_seed_high,
                    Bootstrap.iteration_offset + Completed, 0x4c470000u ^ (2 * fold + side), 0, 0, 0, 1, 0, coefficient},
                    2 * (task.CursorOffset + begin), 2, 0, 0};
                command.Dispatch("AddLangevinWeakNoise", {Derivatives, Weights}, params, count);
            }
        }
    }
    void WriteLangevinNoise(uint32_t event, uint32_t count, bool hessian = false, bool accumulate = false) {
        std::vector<double> noise(count);
        Require(!LangevinNoiseCallback(LangevinContext, event, count, noise.data()), "Ordered Langevin noise callback failed");
        auto& cached = hessian ? LangevinHostHessianNoise : LangevinHostGradientNoise;
        if (!accumulate) cached.assign(count, 0);
        Require(cached.size() == count, "Ordered Langevin cached noise dimension differs");
        float* output = static_cast<float*>((hessian ? LangevinHessianNoise : LangevinGradientNoise).contents);
        for (uint32_t index = 0; index < count; ++index) {
            Require(std::isfinite(noise[index]), "Ordered Langevin noise is nonfinite");
            cached[index] += noise[index]; const float high = cached[index], low = cached[index] - high;
            Require(std::isfinite(high) && std::isfinite(low), "Ordered Langevin noise exceeds float32");
            output[2 * index] = high; output[2 * index + 1] = low;
        }
    }
    void EncodeLangevinPoint(Command& command, const StepParams& step, id<MTLBuffer> point, bool trial) {
        if (IsQuery) EncodeQueryDerivatives(command, point, 1, step.SelectedPermutation, false, false, nil, trial);
        OrderedLangevinParams params = {LangevinActiveTasks, uint32_t(IsQuery), uint32_t(trial), 0};
        command.Dispatch("OrderedLangevinStatistics", {Targets, Weights, Cursor, Permutations, LeafIds,
            IsQuery ? EstimationTasks : TaskBuffer, point, IsQuery ? QueryGradient : Targets,
            IsQuery ? QueryHessian : Targets, IsQuery ? QueryWeights : Weights, LangevinTaskIds,
            LangevinStatistics, LeafWeights, LangevinTaskMass, Status}, P, step.Leaves, true, LangevinActiveTasks, 1,
            &step, &params, sizeof(params));
        if (P.objective == 17 || Combination) return;
        if (IsQuery) command.Dispatch("OrderedQueryBacktrackingObjective", {QueryStatistics, QueryTaskRanges,
            LangevinTaskMass, LangevinLoss}, P, Tasks, true, 1, 1, &step);
        else command.Dispatch("OrderedBacktrackingObjective", {Targets, Weights, Cursor, Permutations, LeafIds, TaskBuffer,
            point, LangevinTaskMass, LangevinLoss}, P, Tasks, true, 1, 1, &step);
    }
    double ReadLangevinObjective(id<MTLBuffer> point, uint32_t leaves, bool trial) const {
        double value = 0;
        if (Combination) {
            const float* mass = static_cast<const float*>(LangevinTaskMass.contents);
            const uint32_t* ids = static_cast<const uint32_t*>(LangevinTaskIds.contents);
            for (uint32_t task = 0; task < YetiLeafBlocks.size(); ++task) {
                const double part = Combination->ReadObjective(YetiLeafBlocks[task], trial);
                value += P.normalize ? (mass[ids[task]] > 0 ? part / mass[ids[task]] : 0) : part;
            }
        } else if (P.objective != 17) {
            const float* data = static_cast<const float*>(LangevinLoss.contents);
            for (uint32_t task = 0; task < Tasks; ++task) value += double(data[2 * task]) + data[2 * task + 1];
        }
        return AddObjectiveRidge(value, point, leaves);
    }
    double PrepareLangevinDirection(const StepParams& step, bool initial) {
        const OrderedLangevinParams params = {LangevinActiveTasks, uint32_t(IsQuery), 0, uint32_t(initial)};
        Command command(Stats, CustomContext.get());
        command.Dispatch("OrderedLangevinDirections", {LangevinStatistics, LangevinGradientNoise, LangevinHessianNoise,
            LangevinTaskIds, LangevinDirections, LangevinDirectionDot, Status}, P,
            uint64_t(LangevinActiveTasks) * step.Leaves, false, 1, 1, &step, &params, sizeof(params));
        command.Wait(); CheckStatus();
        const float* dots = static_cast<const float*>(LangevinDirectionDot.contents);
        double result = 0;
        for (uint32_t task = 0; task < Tasks; ++task) for (uint32_t leaf = 0; leaf < step.Leaves; ++leaf) {
            const uint64_t at = 2 * (uint64_t(task) * MaxLeaves + leaf); result += double(dots[at]) + dots[at + 1];
        }
        Require(std::isfinite(result), "Ordered Langevin direction dot became nonfinite"); return result;
    }
    void EstimateLangevin(const StepParams& step) {
        const uint32_t count = LangevinActiveTasks * step.Leaves;
        Command initialize(Stats, CustomContext.get());
        initialize.Zero(LangevinTaskMass); initialize.Zero(LangevinDirectionDot); initialize.Zero(LangevinDirections);
        EncodeLangevinPoint(initialize, step, RawValues, false); initialize.Wait(); CheckStatus();
        double current = ReadLangevinObjective(RawValues, step.Leaves, false);
        Require(std::isfinite(current), "Ordered Langevin initial objective became nonfinite");
        WriteLangevinNoise(CBM_LANGEVIN_INITIAL_GRADIENT, count);
        WriteLangevinNoise(CBM_LANGEVIN_INITIAL_HESSIAN, count, true);
        double dot = PrepareLangevinDirection(step, true);
        BacktrackingParams backtracking = {1, BacktrackingType, P.reserved1, P.normalize};
        if (P.leaf_iterations == 1) {
            Command move(Stats, CustomContext.get());
            move.Dispatch("OrderedBacktrackingCandidate", {RawValues, LangevinDirections, LeafWeights, LangevinTrialValues},
                P, uint64_t(Tasks) * step.Leaves, false, 1, 1, &step, &backtracking, sizeof(backtracking));
            move.Wait(); std::swap(RawValues, LangevinTrialValues); return;
        }
        bool updated = false;
        for (uint32_t attempt = 0; attempt < P.leaf_iterations || (!updated && attempt < 100); ++attempt) {
            Command trial(Stats, CustomContext.get());
            trial.Dispatch("OrderedBacktrackingCandidate", {RawValues, LangevinDirections, LeafWeights, LangevinTrialValues},
                P, uint64_t(Tasks) * step.Leaves, false, 1, 1, &step, &backtracking, sizeof(backtracking));
            EncodeLangevinPoint(trial, step, LangevinTrialValues, true); trial.Wait(); CheckStatus();
            WriteLangevinNoise(CBM_LANGEVIN_TRIAL_GRADIENT, count);
            const double candidate = ReadLangevinObjective(LangevinTrialValues, step.Leaves, true);
            const double threshold = current + (BacktrackingType == 2 ? 1e-5 * backtracking.Step * dot : 0);
            if (!BacktrackingType || (std::isfinite(candidate) && candidate >= threshold)) {
                WriteLangevinNoise(CBM_LANGEVIN_ACCEPTED_GRADIENT, count, false, true);
                std::swap(RawValues, LangevinTrialValues); current = candidate; updated = true; backtracking.Step = 1;
                dot = PrepareLangevinDirection(step, false);
            } else backtracking.Step *= .5f;
        }
    }
    double ReadBacktrackingValue() const {
        const float* data = static_cast<const float*>(BacktrackingLoss.contents);
        double sum = 0; for (uint32_t task = 0; task < Tasks; ++task) sum += double(data[task * 2]) + data[task * 2 + 1];
        return sum;
    }
    double ReadDirectionDot(uint32_t leaves) const {
        const float* data = static_cast<const float*>(DirectionDot.contents);
        double sum = 0;
        for (uint32_t task = 0; task < Tasks; ++task) for (uint32_t leaf = 0; leaf < leaves; ++leaf) {
            const uint64_t offset = (uint64_t(task) * MaxLeaves + leaf) * 2;
            sum += double(data[offset]) + data[offset + 1];
        }
        Require(std::isfinite(sum), "Ordered backtracking direction dot became nonfinite"); return sum;
    }
    void EstimateBacktracking(const StepParams& step) {
        auto direction = [&](Command& command) {
            if (IsQuery) {
                EncodeQueryDerivatives(command, RawValues, 1, step.SelectedPermutation, false);
                command.Dispatch("OrderedQueryBacktrackingDirections", {QueryGradient, QueryHessian, QueryWeights, Permutations,
                    LeafIds, EstimationTasks, RawValues, Directions, LeafWeights, DirectionDot, TaskMass, Status},
                    P, step.Leaves, true, Tasks, 1, &step);
                return;
            }
            command.Dispatch("OrderedBacktrackingDirections", {Targets, Weights, Cursor, Permutations, LeafIds, TaskBuffer,
                RawValues, Directions, LeafWeights, DirectionDot, TaskMass, Status}, P, step.Leaves, true, Tasks, 1, &step);
        };
        auto objective = [&](Command& command, id<MTLBuffer> values) {
            if (IsQuery) {
                EncodeQueryDerivatives(command, values, 1, step.SelectedPermutation, false, true);
                command.Dispatch("OrderedQueryBacktrackingObjective", {QueryStatistics, QueryTaskRanges, TaskMass, BacktrackingLoss},
                    P, Tasks, true, 1, 1, &step);
                return;
            }
            command.Dispatch("OrderedBacktrackingObjective", {Targets, Weights, Cursor, Permutations, LeafIds, TaskBuffer,
                values, TaskMass, BacktrackingLoss}, P, Tasks, true, 1, 1, &step);
        };
        Command initialize(Stats, CustomContext.get()); direction(initialize); objective(initialize, RawValues); initialize.Wait(); CheckStatus();
        double current = AddObjectiveRidge(ReadBacktrackingValue(), RawValues, step.Leaves), dot = ReadDirectionDot(step.Leaves);
        Require(std::isfinite(current), "Ordered initial backtracking objective became nonfinite");
        BacktrackingParams backtracking = {1, BacktrackingType, P.reserved1, P.normalize};
        bool updated = false, newDirection = false;
        // CUDA counts rejected trials against the budget, but extends up to
        // 100 attempts until the first successful update.
        for (uint32_t attempt = 0; attempt < P.leaf_iterations || (!updated && attempt < 100); ++attempt) {
            Command trial(Stats, CustomContext.get());
            if (newDirection) direction(trial);
            trial.Dispatch("OrderedBacktrackingCandidate", {RawValues, Directions, LeafWeights, TrialValues},
                P, uint64_t(Tasks) * step.Leaves, false, 1, 1, &step, &backtracking, sizeof(backtracking));
            objective(trial, TrialValues); trial.Wait(); CheckStatus();
            if (newDirection) dot = ReadDirectionDot(step.Leaves);
            const double candidate = AddObjectiveRidge(ReadBacktrackingValue(), TrialValues, step.Leaves);
            const double threshold = current + (BacktrackingType == 2 ? 1e-5 * backtracking.Step * dot : 0);
            if (std::isfinite(candidate) && candidate >= threshold) {
                std::swap(RawValues, TrialValues); current = candidate; updated = true; newDirection = true; backtracking.Step = 1;
            } else { backtracking.Step *= .5f; newDirection = false; }
        }
    }
    void EstimateExact(uint32_t leaves) {
        for (uint32_t taskId = 0; taskId < Tasks; ++taskId) {
            const auto& task = Descriptors[taskId];
            const StepParams descriptor = {task.EstimateEnd, task.QualityEnd, task.CursorOffset, task.Reserved};
            const CBMExactLeafParams exact = {task.EstimateEnd, leaves, uint32_t(P.objective == 11),
                std::min(ExactTiles, (task.EstimateEnd + 255) / 256), P.objective == 9 ? P.objective_param : .5f, 0, 0, 0};
            const uint64_t outputOffset = uint64_t(taskId) * MaxLeaves * 4;
            Command command(Stats, CustomContext.get());
            command.Dispatch("OrderedSessionGatherExact", {Targets, Weights, Cursor, Permutations, LeafIds, ExactTargets,
                ExactWeights, ExactPredictions, ExactLeafIds}, P, task.EstimateEnd, false, 1, 1, &descriptor);
            command.Dispatch("PrepareExactResiduals", {ExactTargets, ExactWeights, ExactPredictions, ExactResiduals,
                ExactEffective, ExactKeysA, ExactRowsA}, exact, exact.Rows);
            CBMEncodeSortU32(command.Buffer, ExactKeysA, ExactRowsA, exact.Rows, ExactKeysB, ExactRowsB, &Stats.kernel_dispatches);
            command.Dispatch("MakeExactLeafKeys", {ExactRowsB, ExactLeafIds, ExactKeysA}, exact, exact.Rows);
            CBMEncodeSortU32(command.Buffer, ExactKeysA, ExactRowsB, exact.Rows, ExactKeysB, ExactRowsA, &Stats.kernel_dispatches);
            command.Dispatch("BuildExactLeafOffsets", {ExactKeysB, ExactOffsets}, exact, uint64_t(leaves) + 1);
            command.Dispatch("ReduceExactLeafPartials", {ExactRowsA, ExactEffective, ExactWeights, ExactOffsets, ExactPartials},
                exact, exact.TilesPerLeaf, true, leaves);
            command.Dispatch("PrefixExactLeafTiles", {ExactPartials, ExactTileOffsets, ExactTotals,
                {LeafWeights, outputOffset}, ExactSelected}, exact, leaves, true);
            command.Dispatch("SelectExactLeafQuantile", {ExactRowsA, ExactEffective, ExactOffsets, ExactTileOffsets, ExactTotals, ExactSelected},
                exact, exact.TilesPerLeaf, true, leaves);
            command.Dispatch("FinalizeExactLeafValues", {ExactRowsA, ExactResiduals, ExactOffsets, ExactTotals, ExactSelected,
                {RawValues, outputOffset}}, exact, leaves);
            command.Wait();
            const float* raw = static_cast<const float*>(RawValues.contents) + uint64_t(taskId) * MaxLeaves;
            for (uint32_t leaf = 0; leaf < leaves; ++leaf) Require(std::isfinite(raw[leaf]), "Ordered Exact leaf solve became nonfinite");
        }
    }
    void BeginTree(uint32_t selected) {
        Require(!Failed && !Pending.Active && Completed < P.iterations,
                "Ordered session is failed, already has an active tree, or has no remaining iterations");
        Require(selected < LearnPermutations, "Invalid Ordered search permutation");
        if (!LangevinEnabled && (Yeti || CombinationYetiCount)) {
            const uint32_t components = Combination ? CombinationYetiCount : 1;
            Require(YetiSeedPosition == 0 && YetiSeeds.size() == YetiWeakBlocks[selected].size() * components,
                    "Ordered YetiRank weak seed packet does not match the selected permutation");
        }
        try {
            const uint32_t foldCount = PermutationFoldCounts[selected];
            const uint32_t packedRows = PermutationPackedRows[selected];
            const uint32_t taskOffset = PermutationTaskOffsets[selected];
            StepParams step = StepConfiguration(1, selected);
            Command initialize(Stats, CustomContext.get()); initialize.Zero(LeafIds); initialize.Zero(RawValues); initialize.Zero(LeafWeights); initialize.Zero(Status);
            if (IsQuery) EncodeQueryDerivatives(initialize, RawValues, 0, selected, true);
            else initialize.Dispatch("OrderedSessionDerivatives", {Targets, Weights, Cursor, Permutations, TaskBuffer, Derivatives, Status},
                P, P.rows, false, Tasks, 1, &step);
            if (LangevinEnabled) EncodeLangevinWeak(initialize, selected);
            if (Histogram) Histogram->Initialize(initialize, Binding(Permutations, uint64_t(selected) * P.rows * 4),
                Binding(TaskBuffer, uint64_t(taskOffset) * 16), Descriptors[taskOffset].CursorOffset, foldCount, packedRows);
            initialize.Wait(); CheckStatus();
            OrderedParams sampling = {P.rows, P.features, foldCount, step.Leaves, 0, CursorCount, BootstrapTestOnly, 0,
                P.l2, P.normalize, 0, P.learning_rate};
            BootstrapParams bootstrap = {packedRows, Bootstrap.bootstrap_type, Bootstrap.random_seed_low,
                Bootstrap.random_seed_high, Bootstrap.iteration_offset + Completed, 0, 0, 0,
                Bootstrap.bagging_temperature, Bootstrap.subsample, 0, 0};
            float noiseScale = 0;
            if (QualityStatistics) {
                Command noise(Stats, CustomContext.get());
                noise.Dispatch("OrderedQualityStatistics", {Derivatives, {TaskBuffer, uint64_t(taskOffset) * 16},
                    QualityStatistics}, sampling, foldCount, true); noise.Wait();
                const float* statistics = static_cast<const float*>(QualityStatistics.contents);
                double squareSum = 0, count = 0;
                for (uint32_t fold = 0; fold < foldCount; ++fold) { squareSum += statistics[fold * 2]; count += statistics[fold * 2 + 1]; }
                const double exponent = std::log(double(P.rows)) - double(bootstrap.Iteration) * P.learning_rate;
                const double modelMultiplier = exponent >= 0 ? 1 / (1 + std::exp(-exponent)) : std::exp(exponent) / (1 + std::exp(exponent));
                noiseScale = RandomStrength * std::sqrt(squareSum / (count + 1e-100)) * modelMultiplier;
                Require(std::isfinite(noiseScale), "Ordered score noise scale became nonfinite");
            }
            if (Multipliers) {
                if (MvsInput) {
                    Command prepare(Stats, CustomContext.get());
                    prepare.Dispatch("OrderedSessionMvsInput", {Derivatives, {TaskBuffer, uint64_t(taskOffset) * 16}, MvsInput},
                        bootstrap, packedRows);
                    if (!Bootstrap.mvs_reg_is_set && !HasMvsLambda)
                        prepare.Dispatch("ReduceBootstrapStatistics", {MvsInput, MvsStatistics}, bootstrap, MvsGroups, true);
                    prepare.Wait();
                    if (!Bootstrap.mvs_reg_is_set && !HasMvsLambda) {
                        const float* values = static_cast<const float*>(MvsStatistics.contents);
                        double mean = 0; for (uint32_t group = 0; group < MvsGroups; ++group) mean += values[group * 2];
                        MvsLambda = mean * mean; HasMvsLambda = true;
                        Require(std::isfinite(MvsLambda), "Ordered MVS initial regularization became nonfinite");
                    }
                    bootstrap.MVSLambda = Bootstrap.mvs_reg_is_set ? Bootstrap.mvs_reg : MvsLambda;
                }
                Command sample(Stats, CustomContext.get());
                if (MvsInput) {
                    sample.Dispatch("ComputeMvsThresholds", {MvsInput, MvsThresholds}, bootstrap, (packedRows + 8191) / 8192, true);
                    sample.Dispatch("GenerateMvsBootstrapWeights", {Multipliers, MvsInput, MvsThresholds}, bootstrap, packedRows);
                } else sample.Dispatch("GenerateBootstrapWeights", {Multipliers, Derivatives}, bootstrap, packedRows);
                sample.Dispatch("ApplyOrderedBootstrap", {Derivatives, Multipliers,
                    {TaskBuffer, uint64_t(taskOffset) * 16}, Sampled}, sampling, P.rows, false, foldCount);
                sample.Wait();
            }
            Pending = {}; Pending.Active = true; Pending.SelectedPermutation = selected;
            Pending.Bootstrap = bootstrap; Pending.NoiseScale = noiseScale;
            Pending.Finished = !P.depth || !HasActiveCandidates();
            Pending.Exhausted = P.depth && !HasActiveCandidates();
        } catch (...) { Failed = true; throw; }
    }
    void StructureInfo(CBMStructureInfo& info) const {
        info = {}; info.depth = Pending.Selected.size(); info.finished = Pending.Finished;
        info.has_split = Pending.HasSplit;
        if (Pending.HasSplit) {
            info.feature = Pending.LastWinner.Feature; info.bin = Pending.LastWinner.Bin;
            info.type = Pending.LastWinner.Type; info.score = Pending.LastWinner.Score; info.gain = Pending.LastWinner.Gain;
        }
    }
    void GrowTree(CBMStructureInfo* info) {
        Require(!Failed && Pending.Active && info, "Ordered grow requires an active tree and output");
        Pending.HasSplit = false;
        if (Pending.Finished) { StructureInfo(*info); return; }
        try {
            const uint32_t selected = Pending.SelectedPermutation, level = Pending.Selected.size();
            StepParams step = StepConfiguration(1u << level, selected);
            OrderedParams hist = {P.rows, P.features, PermutationFoldCounts[selected], step.Leaves, 0, CursorCount, 1,
                P.objective == 19 || (P.objective == 13 && P.reserved2)
                    ? CBMOrderedScoreSignedRightMassClamp : 0u,
                P.l2, P.normalize, Pending.ScoreBefore, P.learning_rate};
            UpdateFeaturePenalties();
            Command search(Stats, CustomContext.get());
            if (FeatureNoise) {
                auto noise = Pending.Bootstrap; noise.Rows = P.features; noise.Stream = level + 1; noise.NoiseScale = Pending.NoiseScale;
                search.Dispatch("GenerateScoreFeatureNoise", {FeatureNoise}, noise, P.features);
                search.Dispatch("OrderedSessionFeatureNoise", {FeatureNoise, FeatureOptions}, P, P.features);
            }
            Histogram->Prepare(search, step.Leaves);
            if (!Histogram->CanHistogram()) {
                for (uint32_t begin = 0; begin < P.candidates; begin += BatchSize) {
                    hist.Candidates = std::min(BatchSize, P.candidates - begin);
                    Histogram->DirectCandidates(search, FeatureBanks[BinBanks > 1 ? selected : 0], Sampled ? Sampled : Derivatives,
                        Binding(CandidatePairs, begin * 8ull), hist.Candidates, Statistics);
                    search.Dispatch("ScoreOrderedCandidates", {Statistics, {CandidatePairs, begin * 8ull},
                        FeatureOptions, {Scores, begin * 8ull}}, hist, hist.Candidates);
                }
            } else for (const auto& tile : Histogram->Plan.Tiles) {
                Histogram->Compute(search, tile, FeatureBanks[BinBanks > 1 ? selected : 0], Sampled ? Sampled : Derivatives);
                for (uint32_t begin = 0; begin < tile.Candidates; begin += BatchSize) {
                    hist.Candidates = std::min(BatchSize, tile.Candidates - begin);
                    Histogram->Extract(search, tile, begin, hist.Candidates, Statistics);
                    search.Dispatch("ScoreOrderedCandidates", {Statistics,
                        {Histogram->CandidatePairs, uint64_t(tile.FirstCandidate + begin) * 8}, FeatureOptions, TileScores}, hist, hist.Candidates);
                    Histogram->ScatterScores(search, tile, begin, hist.Candidates, TileScores, Scores);
                }
            }
            if (Dynamic) search.Dispatch("OrderedSessionMaskCandidates", {Scores, CandidatePairs, FeatureActivity}, P.candidates, P.candidates);
            search.Dispatch("OrderedSessionFindWinner", {Scores, CandidatePairs, Winner}, P, 1, true); search.Wait();
            Histogram->Check();
            Histogram->Plan.AllowReuse = true;
            const auto winner = *static_cast<const SplitState*>(Winner.contents);
            Require(!winner.InvalidScore, "Ordered split score became nonfinite");
            bool duplicate = false;
            for (const auto& previous : Pending.Selected) duplicate |= previous.Index == winner.Index;
            if (!winner.Valid || duplicate) {
                Pending.Finished = true; Pending.Exhausted = !winner.Valid;
                StructureInfo(*info); return;
            }
            Pending.Selected.push_back(winner); Pending.LastWinner = winner;
            Pending.ScoreBefore = winner.Score;
            if (Dynamic && (FeatureFlags[winner.Feature] & 1) && CtrCounts[winner.Feature]) {
                UsedFeatures[winner.Feature] = 1; FeatureFlags[winner.Feature] |= 2;
            }
            Command partition(Stats, CustomContext.get());
            partition.Dispatch("OrderedSessionUpdateLeafIds", {Bins, Winner, LeafIds}, P, P.rows, false, BinBanks, 1, &step);
            if (level + 1 < P.depth) Histogram->Partition(partition, Binding(LeafIds, uint64_t(selected) * P.reserved0 * 4), step.Leaves * 2);
            partition.Wait();
            Pending.HasSplit = true; Pending.Finished = Pending.Selected.size() == P.depth;
            StructureInfo(*info);
        } catch (...) { Failed = true; throw; }
    }
    void FinishTree(CBMStepInfo* info, uint32_t* depth, uint32_t* splitFeatures,
        uint32_t* splitBins, uint8_t* splitTypes, float* values, float* weights) {
        Require(!Failed && Pending.Active, "Ordered finish requires an active tree");
        Require(info && depth && values && weights && (!P.depth || (splitFeatures && splitBins && splitTypes)),
                "Ordered step output buffers are required");
        Require(!(CombinationYetiCount && BacktrackingBytes) || Combination->HasYetiSeedCallback(),
                "Ordered Combination YetiRank backtracking requires a seed callback");
        if (!LangevinEnabled && (Yeti || (CombinationYetiCount && !Combination->HasYetiSeedCallback()))) {
            const uint32_t components = Combination ? CombinationYetiCount : 1;
            const uint32_t expected = YetiLeafBlocks.size() * (P.leaf_iterations == 1 ? 1 : P.leaf_iterations + 1) * components;
            Require(YetiLeafSeedPosition == 0 && YetiLeafSeeds.size() == expected,
                    "Ordered YetiRank leaf seed packet is missing or has the wrong size");
        }
        try {
            if (P.depth) {
                std::fill(splitFeatures, splitFeatures + P.depth, 0); std::fill(splitBins, splitBins + P.depth, 0);
                std::fill(splitTypes, splitTypes + P.depth, 0);
                for (uint32_t level = 0; level < Pending.Selected.size(); ++level) {
                    splitFeatures[level] = Pending.Selected[level].Feature; splitBins[level] = Pending.Selected[level].Bin;
                    splitTypes[level] = Pending.Selected[level].Type;
                }
            }
            std::fill(values, values + MaxLeaves, 0); std::fill(weights, weights + MaxLeaves, 0);
            const auto step = StepConfiguration(1u << Pending.Selected.size(), Pending.SelectedPermutation);
            if (P.leaf_method == 2) EstimateExact(step.Leaves);
            else if (LangevinEnabled) EstimateLangevin(step);
            else if (BacktrackingBytes && Combination) EstimateCombinationBacktracking(step);
            else if (BacktrackingBytes) EstimateBacktracking(step);
            else for (uint32_t iteration = 0; iteration < P.leaf_iterations; ++iteration) {
                Command estimate(Stats, CustomContext.get());
                if (IsQuery) {
                    EncodeQueryDerivatives(estimate, RawValues, 1, step.SelectedPermutation, false);
                    estimate.Dispatch("OrderedQueryEstimateLeaves", {QueryGradient, QueryHessian, QueryWeights, Permutations,
                        LeafIds, EstimationTasks, RawValues, LeafWeights, Status}, P, step.Leaves, true, Tasks, 1, &step);
                } else estimate.Dispatch("OrderedSessionEstimateLeaves", {Targets, Weights, Cursor, Permutations, LeafIds, TaskBuffer,
                    RawValues, LeafWeights, Status}, P, step.Leaves, true, Tasks, 1, &step);
                estimate.Wait(); CheckStatus();
            }
            if (!LangevinEnabled && (P.objective == 17 || (CombinationYetiCount && !BacktrackingBytes)) && P.leaf_iterations > 1) {
                Command finalEvaluation(Stats, CustomContext.get());
                EncodeQueryDerivatives(finalEvaluation, RawValues, 1, step.SelectedPermutation, false);
                finalEvaluation.Wait(); CheckStatus();
            }
            Command update(Stats, CustomContext.get());
            if (P.objective == 14 || P.objective == 17)
                update.Dispatch("OrderedQueryCenterLeaves", {RawValues, Status}, P, Tasks, true, 1, 1, &step);
            update.Dispatch("OrderedSessionApplyValues", {Cursor, Permutations, LeafIds, TaskBuffer, RawValues,
                NextCursor, NextPublished, Status}, P, P.rows, false, Tasks, 1, &step);
            update.Wait(); CheckStatus();
            const float nextLoss = ReadLoss(NextPublished);
            const float* raw = static_cast<const float*>(RawValues.contents) + uint64_t(Tasks - 1) * MaxLeaves;
            const float* leafWeights = static_cast<const float*>(LeafWeights.contents) + uint64_t(Tasks - 1) * MaxLeaves;
            for (uint32_t leaf = 0; leaf < step.Leaves; ++leaf) { values[leaf] = raw[leaf] * P.learning_rate; weights[leaf] = leafWeights[leaf]; }
            if (Bootstrap.bootstrap_type == 4 && !Bootstrap.mvs_reg_is_set) {
                double mean = 0; for (uint32_t leaf = 0; leaf < step.Leaves; ++leaf) mean += std::abs(values[leaf]);
                mean /= step.Leaves; MvsLambda = mean * mean; HasMvsLambda = true;
                Require(std::isfinite(MvsLambda), "Ordered MVS leaf regularization became nonfinite");
            }
            std::swap(Cursor, NextCursor); std::swap(Published, NextPublished);
            ++Completed; Loss = nextLoss; *depth = Pending.Selected.size(); Pending = {};
            YetiSeeds.clear(); YetiLeafSeeds.clear(); YetiSeedPosition = YetiLeafSeedPosition = 0;
            if (Combination && !LangevinEnabled) { Combination->SetYetiSeedCallback(nullptr, nullptr); CombinationSeedCallback = nullptr; CombinationSeedContext = nullptr; }
            Info(*info);
        } catch (...) { Failed = true; throw; }
    }
    void Step(uint32_t selected, CBMStepInfo* info, uint32_t* depth, uint32_t* splitFeatures,
        uint32_t* splitBins, uint8_t* splitTypes, float* values, float* weights) {
        Require(info && depth && values && weights && (!P.depth || (splitFeatures && splitBins && splitTypes)),
                "Ordered step output buffers are required");
        BeginTree(selected);
        while (!Pending.Finished) { CBMStructureInfo structure; GrowTree(&structure); }
        FinishTree(info, depth, splitFeatures, splitBins, splitTypes, values, weights);
    }
    void EnableDynamic() {
        if (Dynamic) return;
        const uint64_t extra = uint64_t(P.features) * (PenaltiesConfigured ? 4 : 12);
        Require(WorkingBytes + BootstrapBytes + NoiseBytes + BacktrackingBytes + extra <= MemoryLimit,
                "Ordered dynamic feature state exceeds 1 GiB");
        if (!PenaltiesConfigured) { CtrCounts.assign(P.features, 0); FeatureWeights.assign(P.features, 1); }
        FeatureFlags.assign(P.features, 2); UsedFeatures.assign(P.features, 0); ActiveFeatures.assign(P.features, 1);
        FeatureActivity = Context().Buffer(P.features, ActiveFeatures.data());
        Dynamic = true; WorkingBytes += extra; UpdateFeaturePenalties();
    }
    void AppendFeatures(const CBMAppendFeatureOptions* options, const uint8_t* const* matrices,
        const uint32_t* features, const uint32_t* borders, const uint8_t* types,
        const uint32_t* counts, const float* weights, const uint8_t* flags, const uint8_t* used, uint32_t* first) {
        Require(!Failed && options && first, "Ordered append options and output are required");
        Require((options->permutation_count == 1 || options->permutation_count == P.permutations) &&
                options->permutation_count >= BinBanks, "Ordered appended banks must retain every permutation");
        for (const auto reserved : options->reserved) Require(!reserved, "Ordered append reserved fields must be zero");
        Require(options->bins_per_feature && options->bins_per_feature <= 256, "Ordered appended bin capacity must be in [1,256]");
        Require(options->features || !options->candidates, "Ordered appended candidates need new features");
        if (!options->features) { EnableDynamic(); *first = P.features; ReopenCandidateExhaustion(); return; }
        const uint64_t newF64 = uint64_t(P.features) + options->features, newC64 = uint64_t(P.candidates) + options->candidates;
        const uint32_t banks = options->permutation_count;
        Require(newF64 < UINT32_MAX && newC64 <= UINT32_MAX && newF64 * P.rows <= UINT32_MAX,
                "Ordered appended dimensions exceed GPU indexing");
        Require(newF64 * P.rows * banks + newF64 * 28 + newC64 * 20 <= MemoryLimit,
                "Ordered appended feature banks exceed 1 GiB");
        Require(matrices && (!options->candidates || (features && borders)), "Ordered appended input vectors are required");
        Require(options->candidates <= uint64_t(options->features) * options->bins_per_feature,
                "Ordered appended candidate count exceeds feature grid");
        const uint32_t newF = newF64, newC = newC64;
        const uint64_t addedCells = uint64_t(options->features) * P.rows, oldCells = uint64_t(P.features) * P.rows;
        for (uint32_t bank = 0; bank < banks; ++bank) {
            Require(matrices[bank], "Ordered appended permutation matrix is required");
            for (uint64_t cell = 0; cell < addedCells; ++cell)
                Require(matrices[bank][cell] < options->bins_per_feature, "Ordered appended bin exceeds capacity");
        }
        const uint8_t* oldBins = static_cast<const uint8_t*>(Bins.contents);
        for (uint64_t cell = 0; cell < oldCells * BinBanks; ++cell)
            Require(oldBins[cell] < options->bins_per_feature, "Ordered appended capacity must cover the existing grid");
        std::vector<int8_t> kinds(options->features, -1);
        for (uint32_t c = 0; c < options->candidates; ++c) {
            const uint32_t type = types ? types[c] : 0;
            Require(features[c] < options->features && type <= 1 && borders[c] < options->bins_per_feature &&
                    borders[c] <= 255 - (type == 0), "Invalid Ordered appended numeric/one-hot candidate");
            auto& kind = kinds[features[c]];
            Require(kind < 0 || kind == type, "Ordered appended feature cannot mix numeric and one-hot candidates"); kind = type;
        }
        for (uint32_t f = 0; f < options->features; ++f) {
            Require(!weights || (std::isfinite(weights[f]) && weights[f] >= 0), "Ordered appended feature weights must be finite and nonnegative");
            Require(!flags || flags[f] <= 3, "Ordered appended flags must use only dynamic and registered bits");
            Require(!used || used[f] <= 1, "Ordered appended used flags must be boolean");
            Require(!counts || !counts[f] || !used || !used[f] || !flags || (flags[f] & 2),
                    "Ordered previously used CTRs must be globally registered");
        }
        std::vector<uint32_t> allFeatures(newC), allBorders(newC), allPairs(uint64_t(newC) * 2), allTypes(newC);
        std::vector<uint8_t> allTypes8(newC);
        const uint32_t* oldPairs = static_cast<const uint32_t*>(CandidatePairs.contents);
        for (uint32_t c = 0; c < newC; ++c) {
            allFeatures[c] = c < P.candidates ? oldPairs[2 * c] : P.features + features[c - P.candidates];
            allBorders[c] = c < P.candidates ? oldPairs[2 * c + 1] & 255 : borders[c - P.candidates];
            allTypes[c] = c < P.candidates ? oldPairs[2 * c + 1] >> 31 : (types ? types[c - P.candidates] : 0);
            allTypes8[c] = allTypes[c]; allPairs[2 * c] = allFeatures[c]; allPairs[2 * c + 1] = allBorders[c] | (allTypes[c] << 31);
        }
        std::unique_ptr<CBMOrderedHistogramPlan> plan;
        if (P.depth && newC) {
            uint32_t foldSlots = 1, span = 2, cacheLeaves = MaxLeaves / 2;
            while (foldSlots < FoldCount) foldSlots <<= 1;
            for (const auto border : allBorders) span = std::max(span, std::min(256u, border + 2));
            while (cacheLeaves > 1 && uint64_t(cacheLeaves) * foldSlots * span * 32 > (uint64_t(128) << 20)) cacheLeaves >>= 1;
            plan = std::make_unique<CBMOrderedHistogramPlan>(P.rows, PermutationCursorCount, FoldCount,
                MaxLeaves / 2, newF, newC, allFeatures.data(), allBorders.data(), uint64_t(128) << 20,
                !Pending.Active || Pending.Selected.empty(), cacheLeaves, allTypes8.data());
        }
        const uint64_t perCandidate = uint64_t(MaxLeaves) * FoldCount * 32;
        const uint32_t newBatch = newC ? std::max<uint64_t>(1, std::min<uint64_t>(newC, HistogramLimit / perCandidate)) : 1;
        const uint64_t oldLayout = oldCells * BinBanks + uint64_t(P.rows) * BinBanks * 4 +
            uint64_t(P.features) * (16 + ((PenaltiesConfigured || Dynamic) ? 8 : 0) + (Dynamic ? 4 : 0)) +
            uint64_t(P.candidates) * 20 + perCandidate * BatchSize +
            (Histogram ? Histogram->Plan.Bytes + oldCells * BinBanks + uint64_t(BatchSize) * 8 : 0);
        const uint64_t cells = newF64 * P.rows;
        const uint64_t newLayout = cells * banks + uint64_t(P.rows) * banks * 4 + newF64 * 28 + newC64 * 20 +
            perCandidate * newBatch + (plan ? plan->Bytes + cells * banks + uint64_t(newBatch) * 8 : 0);
        const uint64_t newNoiseBytes = FeatureNoise ? newF64 * 4 + FoldCount * 8ull : 0;
        Require(WorkingBytes >= oldLayout && WorkingBytes - oldLayout + newLayout + BootstrapBytes + newNoiseBytes + BacktrackingBytes <= MemoryLimit,
                "Ordered appended working set exceeds 1 GiB");
        Require(WorkingBytes + BootstrapBytes + NoiseBytes + BacktrackingBytes + newLayout + (FeatureNoise ? newF64 * 4 : 0) <= MemoryLimit,
                "Ordered append peak working set exceeds 1 GiB");
        auto nextCounts = CtrCounts; auto nextWeights = FeatureWeights;
        auto nextFlags = FeatureFlags, nextUsed = UsedFeatures, nextActive = ActiveFeatures;
        nextCounts.resize(newF, 0); nextWeights.resize(newF, 1); nextFlags.resize(newF, 2);
        nextUsed.resize(newF, 0); nextActive.resize(newF, 1);
        for (uint32_t f = 0; f < options->features; ++f) {
            const uint32_t global = P.features + f;
            nextCounts[global] = counts ? counts[f] : 0; nextWeights[global] = weights ? weights[f] : 1;
            nextFlags[global] = flags ? flags[f] : 3; nextUsed[global] = counts && counts[f] && used ? used[f] : 0;
        }
        std::vector<uint8_t> rowBins(cells * banks), columnBins(cells);
        std::vector<id<MTLBuffer>> newBanks;
        auto& context = Context();
        for (uint32_t bank = 0; bank < banks; ++bank) {
            const uint8_t* oldBank = oldBins + (BinBanks > 1 ? bank * oldCells : 0);
            for (uint32_t row = 0; row < P.rows; ++row) for (uint32_t f = 0; f < newF; ++f) {
                const uint8_t value = f < P.features ? oldBank[uint64_t(row) * P.features + f]
                    : matrices[bank][uint64_t(f - P.features) * P.rows + row];
                rowBins[bank * cells + uint64_t(row) * newF + f] = value;
                columnBins[uint64_t(f) * P.rows + row] = value;
            }
            if (plan) newBanks.push_back(context.Buffer(cells, columnBins.data()));
        }
        auto newBins = context.Buffer(cells * banks, rowBins.data());
        auto newLeafIds = context.Buffer(uint64_t(P.rows) * banks * 4);
        for (uint32_t bank = 0; bank < banks; ++bank)
            std::memcpy(static_cast<uint8_t*>(newLeafIds.contents) + uint64_t(bank) * P.rows * 4,
                static_cast<const uint8_t*>(LeafIds.contents) + (BinBanks > 1 ? uint64_t(bank) * P.rows * 4 : 0), P.rows * 4ull);
        auto candidatePairs = context.Buffer(newC64 * 8, allPairs.data());
        auto candidateTypes = context.Buffer(newC64 * 4, allTypes.data());
        std::vector<float> featureOptions(newF64 * 4, 0);
        std::memcpy(featureOptions.data(), FeatureOptions.contents, P.features * 16ull);
        auto featureOptionsBuffer = context.Buffer(newF64 * 16, featureOptions.data());
        auto activity = context.Buffer(newF, nextActive.data());
        auto statistics = context.Buffer(perCandidate * newBatch), scores = context.Buffer(newC64 * 8);
        id<MTLBuffer> tileScores = plan ? context.Buffer(uint64_t(newBatch) * 8) : nil;
        id<MTLBuffer> noise = FeatureNoise ? context.Buffer(newF64 * 4) : nil;
        std::unique_ptr<CBMOrderedHistogramWorkspace> histogram;
        if (plan) histogram = std::make_unique<CBMOrderedHistogramWorkspace>(context, std::move(*plan));
        if (Pending.Active && histogram && Pending.Selected.size() < P.depth) {
            const uint32_t selected = Pending.SelectedPermutation, offset = PermutationTaskOffsets[selected];
            Command reconstruct(Stats, CustomContext.get());
            histogram->Initialize(reconstruct, Binding(Permutations, uint64_t(selected) * P.rows * 4),
                Binding(TaskBuffer, uint64_t(offset) * 16), Descriptors[offset].CursorOffset,
                PermutationFoldCounts[selected], PermutationPackedRows[selected]);
            for (uint32_t level = 1; level <= Pending.Selected.size(); ++level)
                histogram->Partition(reconstruct, Binding(newLeafIds, banks > 1 ? uint64_t(selected) * P.rows * 4 : 0), 1u << level);
            reconstruct.Wait();
        }
        *first = P.features;
        P.features = newF; P.candidates = newC; P.reserved0 = banks > 1 ? P.rows : 0;
        BinBanks = banks; BatchSize = newBatch; WorkingBytes = WorkingBytes - oldLayout + newLayout; NoiseBytes = newNoiseBytes;
        Bins = newBins; LeafIds = newLeafIds; FeatureBanks = std::move(newBanks); Histogram = std::move(histogram);
        CandidatePairs = candidatePairs; CandidateTypes = candidateTypes; FeatureOptions = featureOptionsBuffer;
        FeatureActivity = activity; Statistics = statistics; Scores = scores; TileScores = tileScores; FeatureNoise = noise;
        CtrCounts = std::move(nextCounts); FeatureWeights = std::move(nextWeights); FeatureFlags = std::move(nextFlags);
        UsedFeatures = std::move(nextUsed); ActiveFeatures = std::move(nextActive); Dynamic = true;
        UpdateFeaturePenalties(); ReopenCandidateExhaustion();
    }
    void SetFeatureActivity(uint32_t count, const uint8_t* active) {
        Require(!Failed && count == P.features && active, "Ordered feature activity must cover every feature");
        for (uint32_t f = 0; f < count; ++f) Require(active[f] <= 1, "Ordered feature activity must be boolean");
        EnableDynamic(); std::copy_n(active, count, ActiveFeatures.begin());
        std::memcpy(FeatureActivity.contents, active, count); UpdateFeaturePenalties(); ReopenCandidateExhaustion();
    }
    void RestoreFeatureMetadata(uint32_t count, const uint8_t* flags, const uint8_t* used, const uint8_t* active) {
        Require(!Failed && !Completed && !Pending.Active && count == P.features && flags && used && active,
                "Restore Ordered feature metadata before the first tree with every feature");
        for (uint32_t f = 0; f < count; ++f) {
            Require(flags[f] <= 3 && used[f] <= 1 && active[f] <= 1, "Invalid Ordered restored feature flags");
            Require(!used[f] || ((PenaltiesConfigured || Dynamic) && CtrCounts[f] && (flags[f] & 2)),
                    "Ordered used features must be globally registered CTRs");
        }
        EnableDynamic(); std::copy_n(flags, count, FeatureFlags.begin()); std::copy_n(used, count, UsedFeatures.begin());
        std::copy_n(active, count, ActiveFeatures.begin()); std::memcpy(FeatureActivity.contents, active, count); UpdateFeaturePenalties();
    }
    void CopyFeatureMetadata(uint32_t capacity, uint32_t* counts, float* weights, uint8_t* flags,
        uint8_t* used, uint8_t* active) const {
        Require(!Failed && !Pending.Active && capacity >= P.features && counts && weights && flags && used && active,
                "Ordered feature metadata requires between-tree state and output capacity");
        for (uint32_t f = 0; f < P.features; ++f) {
            counts[f] = (PenaltiesConfigured || Dynamic) ? CtrCounts[f] : 0;
            weights[f] = (PenaltiesConfigured || Dynamic) ? FeatureWeights[f] : 1;
            flags[f] = Dynamic ? FeatureFlags[f] : 2; used[f] = Dynamic ? UsedFeatures[f] : 0; active[f] = Dynamic ? ActiveFeatures[f] : 1;
        }
    }
    void ConfigureFeaturePenalties(const CBMFeaturePenaltyOptions* options, const uint32_t* counts,
                                   const float* weights) {
        Require(!Completed && !Pending.Active && !Failed && !PenaltiesConfigured, "Ordered feature penalties can be configured once before training");
        Require(options && counts && !options->reserved0 && !options->reserved1 && !options->reserved2 &&
                std::isfinite(options->model_size_reg) && options->model_size_reg >= 0,
                "Invalid Ordered feature penalty options");
        for (uint32_t f = 0; f < P.features; ++f) {
            Require(!weights || (std::isfinite(weights[f]) && weights[f] >= 0), "Ordered feature weights must be finite and nonnegative");
        }
        const uint64_t extra = Dynamic ? 0 : uint64_t(P.features) * 8;
        Require(WorkingBytes + BootstrapBytes + NoiseBytes + BacktrackingBytes + extra <= MemoryLimit,
                "Ordered feature penalty state exceeds 1 GiB");
        CtrCounts.assign(counts, counts + P.features); FeatureWeights.resize(P.features, 1);
        if (weights) std::copy_n(weights, P.features, FeatureWeights.begin());
        ModelSizeReg = options->model_size_reg; PenaltiesConfigured = true;
        WorkingBytes += extra; UpdateFeaturePenalties();
    }
    void CopyPredictions(float* output) const {
        Require(!Pending.Active && !Failed, "Ordered predictions are available only between completed trees");
        Require(output != nullptr, "Ordered prediction output is required");
        std::memcpy(output, Published.contents, P.rows * 4ull);
    }
    void CopyState(uint32_t tasks, uint32_t cursors, uint32_t* descriptors, float* output) const {
        Require(!Pending.Active && !Failed, "Ordered state is available only between completed trees");
        Require(tasks == Tasks && cursors == CursorCount && descriptors && output, "Ordered state shape mismatch");
        std::memcpy(descriptors, Descriptors.data(), Tasks * 16ull); std::memcpy(output, Cursor.contents, CursorCount * 4ull);
    }
    void Restore(uint32_t count, const float* input) {
        Require(!Completed && !Pending.Active && !Failed && count == CursorCount && input, "Ordered state can only be restored before the first step with exact cursor count");
        for (uint32_t i = 0; i < count; ++i) Require(std::isfinite(input[i]), "Ordered restored cursors must be finite");
        auto restored = Context().Buffer(count * 4ull, input);
        const auto step = StepConfiguration();
        Command publish(Stats, CustomContext.get());
        publish.Dispatch("OrderedSessionPublish", {restored, Permutations, TaskBuffer, NextPublished}, P, P.rows, false, 1, 1, &step);
        publish.Wait(); const float nextLoss = ReadLoss(NextPublished);
        Cursor = restored; std::swap(Published, NextPublished); Loss = nextLoss;
    }
private:
    bool LangevinEnabled = false, LangevinStructureCall = false;
    float LangevinTemperature = 0;
    uint32_t LangevinActiveTasks = 0;
    CBMLangevinNoiseCallback LangevinNoiseCallback = nullptr;
    CBMLangevinSeedCallback LangevinSeedCallback = nullptr;
    void* LangevinContext = nullptr;
    id<MTLBuffer> LangevinTaskIds, LangevinStatistics, LangevinGradientNoise, LangevinHessianNoise;
    id<MTLBuffer> LangevinTrialValues, LangevinDirections, LangevinDirectionDot, LangevinTaskMass, LangevinLoss;
    std::vector<double> LangevinHostGradientNoise, LangevinHostHessianNoise;
    std::unique_ptr<Runtime> CustomContext;
    bool IsQuery = false;
    CBMQueryOptions QueryOptions = {};
    CBMYetiRankOptions YetiOptions = {};
    CBMCombinationOptions CombinationOptions = {};
    std::unique_ptr<CBMOrderedCombinationWorkspace> Combination;
    uint32_t CombinationYetiCount = 0;
    CBMCombinationYetiSeedCallback CombinationSeedCallback = nullptr;
    void* CombinationSeedContext = nullptr;
    id<MTLBuffer> QueryGradientWeights;
    uint32_t QueryCount = 0, FlatPairCount = 0;
    id<MTLBuffer> QueryTargets, QueryWeights, QueryPoint, QueryOffsets, QueryStatistics, QueryGradient, QueryHessian;
    id<MTLBuffer> QueryTaskRanges, EstimationTasks;
    id<MTLBuffer> PairWinners, PairLosers, PairWeights, PairRowOffsets, PairIncidence, PairSigns, PairEdges, PairQueryRanges;
    std::vector<std::array<uint32_t, 4>> HostQueryTaskRanges;
    std::vector<std::vector<uint32_t>> YetiWeakBlocks;
    std::vector<uint32_t> YetiLeafBlocks;
    std::unique_ptr<CBMOrderedYetiWorkspace> Yeti;
    std::vector<uint64_t> YetiSeeds, YetiLeafSeeds;
    uint32_t YetiSeedPosition = 0, YetiLeafSeedPosition = 0;
    struct PendingTree {
        bool Active = false, Finished = false, HasSplit = false, Exhausted = false;
        uint32_t SelectedPermutation = 0;
        float ScoreBefore = 0, NoiseScale = 0;
        BootstrapParams Bootstrap = {};
        SplitState LastWinner = {};
        std::vector<SplitState> Selected;
    } Pending;
    bool Failed = false, PenaltiesConfigured = false, Dynamic = false;
    std::vector<uint8_t> FeatureFlags, UsedFeatures, ActiveFeatures;
    id<MTLBuffer> FeatureActivity;
    float ModelSizeReg = 0;
    std::vector<uint32_t> CtrCounts;
    std::vector<float> FeatureWeights;
    void UpdateFeaturePenalties() {
        if (!PenaltiesConfigured && !Dynamic) return;
        if (Dynamic) {
            uint32_t dynamicMaximum = 1;
            for (uint32_t f = 0; f < P.features; ++f)
                if ((FeatureFlags[f] & 1) && ActiveFeatures[f] && !UsedFeatures[f])
                    dynamicMaximum = std::max(dynamicMaximum, CtrCounts[f]);
            uint32_t staticMaximum = dynamicMaximum;
            for (uint32_t f = 0; f < P.features; ++f)
                if ((FeatureFlags[f] & 2) && !UsedFeatures[f]) staticMaximum = std::max(staticMaximum, CtrCounts[f]);
            float* values = static_cast<float*>(FeatureOptions.contents);
            for (uint32_t f = 0; f < P.features; ++f) {
                const bool dynamic = FeatureFlags[f] & 1;
                const uint32_t maximum = dynamic ? dynamicMaximum : staticMaximum;
                values[4 * f] = CtrCounts[f] && (dynamic || !UsedFeatures[f])
                    ? static_cast<float>(std::pow(1.0f + float(CtrCounts[f]) / float(maximum), -double(ModelSizeReg))) : 1;
                values[4 * f + 1] = FeatureWeights[f];
            }
            return;
        }
        // FeatureParallel marks only dynamic tree CTRs used. With simple
        // projections (max_ctr_complexity=1), their penalty persists even
        // after selection; DocParallel's used-feature exemption does not apply.
        uint32_t maximum = 1;
        for (uint32_t f = 0; f < P.features; ++f) maximum = std::max(maximum, CtrCounts[f]);
        float* values = static_cast<float*>(FeatureOptions.contents);
        for (uint32_t f = 0; f < P.features; ++f) {
            values[4 * f] = CtrCounts[f]
                ? static_cast<float>(std::pow(1.0f + float(CtrCounts[f]) / float(maximum), -double(ModelSizeReg))) : 1;
            values[4 * f + 1] = FeatureWeights[f];
        }
    }
    bool HasActiveCandidates() const {
        if (!Dynamic) return P.candidates != 0;
        const uint32_t* pairs = static_cast<const uint32_t*>(CandidatePairs.contents);
        for (uint32_t c = 0; c < P.candidates; ++c) if (ActiveFeatures[pairs[2 * c]]) return true;
        return false;
    }
    void ReopenCandidateExhaustion() {
        if (Pending.Active && Pending.Exhausted && Pending.Selected.size() < P.depth && HasActiveCandidates())
            Pending.Finished = Pending.Exhausted = false;
    }
    uint32_t BatchSize = 1, LossGroups = 1, BinBanks = 1;
    uint32_t PermutationCursorCount = 0, BootstrapTestOnly = 1;
    uint32_t MvsGroups = 1;
    uint32_t ExactTiles = 1;
    uint64_t WorkingBytes = 0, BootstrapBytes = 0, NoiseBytes = 0, BacktrackingBytes = 0;
    uint32_t BacktrackingType = 0;
    CBMBootstrapOptions Bootstrap = {};
    float RandomStrength = 0;
    float MvsLambda = 0;
    bool HasMvsLambda = false;
    KernelParams K = {};
    id<MTLBuffer> Bins, Targets, Weights, Permutations, TaskBuffer, Cursor, NextCursor, Published, NextPublished;
    id<MTLBuffer> Derivatives, LeafIds, RawValues, LeafWeights, CandidatePairs, CandidateTypes, FeatureOptions;
    id<MTLBuffer> Statistics, Scores, Winner, Status, LossPartials;
    id<MTLBuffer> Multipliers, Sampled, FeatureNoise, QualityStatistics;
    id<MTLBuffer> MvsInput, MvsThresholds, MvsStatistics;
    id<MTLBuffer> TrialValues, Directions, DirectionDot, TaskMass, BacktrackingLoss;
    std::unique_ptr<CBMOrderedHistogramWorkspace> Histogram;
    std::vector<id<MTLBuffer>> FeatureBanks;
    id<MTLBuffer> TileScores;
    id<MTLBuffer> ExactTargets, ExactWeights, ExactPredictions, ExactLeafIds, ExactResiduals, ExactEffective;
    id<MTLBuffer> ExactKeysA, ExactRowsA, ExactKeysB, ExactRowsB, ExactOffsets, ExactPartials, ExactTileOffsets, ExactTotals, ExactSelected;
};

// Registry handles make repeated close and accidental stale handles safe.
std::mutex RegistryMutex;
std::unordered_map<uintptr_t, std::shared_ptr<Session>> Registry;
uintptr_t NextHandle = 1;
std::shared_ptr<Session> Get(void* handle) {
    std::lock_guard<std::mutex> lock(RegistryMutex);
    const auto entry = Registry.find(reinterpret_cast<uintptr_t>(handle));
    Require(entry != Registry.end(), "Ordered session is closed or invalid"); return entry->second;
}
template<class F> int Guard(char* error, size_t capacity, F&& work) {
    @autoreleasepool {
        try { work(); return 0; }
        catch (const std::exception& exception) { Text(error, capacity, exception.what()); return 1; }
    }
}
}

extern "C" int cbm_ordered_session_create(const CBMOrderedParams* params, const uint8_t* bins,
    const float* targets, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint32_t* permutations, void** output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        Require(output != nullptr, "Ordered session output is required"); *output = nullptr;
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial, features, borders, permutations);
        std::lock_guard<std::mutex> lock(RegistryMutex); const uintptr_t handle = NextHandle++;
        Registry.emplace(handle, std::move(session)); *output = reinterpret_cast<void*>(handle);
    });
}
extern "C" int cbm_ordered_session_create_typed(const CBMOrderedParams* params, const uint8_t* bins,
    const float* targets, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint8_t* types, const uint32_t* permutations,
    void** output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        Require(output != nullptr, "Ordered session output is required"); *output = nullptr;
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial, features, borders, permutations, types);
        std::lock_guard<std::mutex> lock(RegistryMutex); const uintptr_t handle = NextHandle++;
        Registry.emplace(handle, std::move(session)); *output = reinterpret_cast<void*>(handle);
    });
}
extern "C" int cbm_ordered_session_create_grouped(const CBMOrderedParams* params, const uint8_t* bins,
    const float* targets, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint8_t* types, const uint32_t* permutations,
    uint32_t groups, const uint32_t* offsets, double growth, void** output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        Require(output != nullptr, "Ordered session output is required"); *output = nullptr;
        Require(groups && offsets, "Grouped Ordered creation requires group offsets");
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial, features, borders,
                                                 permutations, types, groups, offsets, growth);
        std::lock_guard<std::mutex> lock(RegistryMutex); const uintptr_t handle = NextHandle++;
        Registry.emplace(handle, std::move(session)); *output = reinterpret_cast<void*>(handle);
    });
}
extern "C" int cbm_ordered_session_create_banked(const CBMOrderedParams* params, uint32_t banks, uint64_t cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint8_t* types, const uint32_t* permutations,
    uint32_t groups, const uint32_t* offsets, double growth, void** output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        Require(output != nullptr, "Ordered session output is required"); *output = nullptr;
        Require(params && banks && banks <= 64 && params->rows && params->features,
                "Ordered bank dimensions are required");
        Require(uint64_t(params->rows) * params->features <= MemoryLimit / banks &&
                cells == uint64_t(params->rows) * params->features * banks,
                "Ordered feature bank cell count is invalid or exceeds 1 GiB");
        Require(bool(groups) == bool(offsets), "Ordered bank group offsets and count must be supplied together");
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial, features, borders,
                                                 permutations, types, groups, offsets, growth, banks);
        std::lock_guard<std::mutex> lock(RegistryMutex); const uintptr_t handle = NextHandle++;
        Registry.emplace(handle, std::move(session)); *output = reinterpret_cast<void*>(handle);
    });
}
extern "C" int cbm_ordered_session_create_query_banked(const CBMOrderedParams* params, uint32_t banks, uint64_t cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint8_t* types, const uint32_t* permutations, const CBMQueryOptions* query,
    const uint32_t* offsets, double growth, void** output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        Require(output, "Ordered session output is required"); *output = nullptr;
        Require(params && query && (params->objective == 12 || params->objective == 13), "Ordered query objective/options mismatch");
        Require(banks && banks <= 64 && params->rows && params->features &&
                uint64_t(params->rows) * params->features <= MemoryLimit / banks &&
                cells == uint64_t(params->rows) * params->features * banks, "Invalid Ordered query feature bank geometry");
        OrderedTargetOptions target; target.Query = query;
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial, features, borders, permutations,
            types, query->group_count, offsets, growth, banks, &target);
        std::lock_guard<std::mutex> lock(RegistryMutex); const uintptr_t handle = NextHandle++;
        Registry.emplace(handle, std::move(session)); *output = reinterpret_cast<void*>(handle);
    });
}
extern "C" int cbm_ordered_session_create_pair_banked(const CBMOrderedParams* params, uint32_t banks, uint64_t cells,
    const uint8_t* bins, const float* initial, const uint32_t* features, const uint32_t* borders, const uint8_t* types,
    const uint32_t* permutations, const CBMPairOptions* pair, const uint32_t* winners, const uint32_t* losers,
    const float* weights, const uint32_t* offsets, double growth, void** output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        Require(output, "Ordered session output is required"); *output = nullptr;
        Require(params && pair && params->objective == 14, "Ordered PairLogit objective/options mismatch");
        Require(banks && banks <= 64 && params->rows && params->features &&
                uint64_t(params->rows) * params->features <= MemoryLimit / banks &&
                cells == uint64_t(params->rows) * params->features * banks, "Invalid Ordered pair feature bank geometry");
        OrderedTargetOptions target; target.Pair = pair; target.Winners = winners; target.Losers = losers; target.PairWeights = weights;
        auto session = std::make_shared<Session>(params, bins, nullptr, nullptr, initial, features, borders, permutations,
            types, pair->group_count, offsets, growth, banks, &target);
        std::lock_guard<std::mutex> lock(RegistryMutex); const uintptr_t handle = NextHandle++;
        Registry.emplace(handle, std::move(session)); *output = reinterpret_cast<void*>(handle);
    });
}
extern "C" int cbm_ordered_session_create_yeti_banked(const CBMOrderedParams* params, uint32_t banks, uint64_t cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint8_t* types, const uint32_t* permutations, const CBMYetiRankOptions* yeti,
    const uint32_t* offsets, double growth, void** output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        Require(output, "Ordered session output is required"); *output = nullptr;
        Require(params && yeti && params->objective == 17, "Ordered YetiRank objective/options mismatch");
        Require(banks && banks <= 64 && params->rows && params->features &&
                uint64_t(params->rows) * params->features <= MemoryLimit / banks &&
                cells == uint64_t(params->rows) * params->features * banks, "Invalid Ordered Yeti feature bank geometry");
        OrderedTargetOptions target; target.Yeti = yeti;
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial, features, borders, permutations,
            types, yeti->group_count, offsets, growth, banks, &target);
        std::lock_guard<std::mutex> lock(RegistryMutex); const uintptr_t handle = NextHandle++;
        Registry.emplace(handle, std::move(session)); *output = reinterpret_cast<void*>(handle);
    });
}
extern "C" int cbm_ordered_session_yeti_seed_shape(void* handle, uint32_t selected, uint32_t* weakCount,
    uint32_t* leafCount, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->YetiSeedShape(selected, weakCount, leafCount); });
}
extern "C" int cbm_ordered_session_set_yeti_oracle_seeds(void* handle, uint32_t count,
    const uint64_t* seeds, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->SetYetiSeeds(count, seeds, false); });
}
extern "C" int cbm_ordered_session_set_yeti_leaf_seeds(void* handle, uint32_t count,
    const uint64_t* seeds, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->SetYetiSeeds(count, seeds, true); });
}
extern "C" int cbm_ordered_session_create_custom_banked(const CBMOrderedParams* params, uint32_t banks, uint64_t cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint8_t* types, const uint32_t* permutations, const char* source,
    uint32_t groups, const uint32_t* offsets, double growth, void** output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        Require(output, "Ordered session output is required"); *output = nullptr;
        Require(params && source && params->objective == 20, "Ordered Custom objective/source mismatch");
        Require(banks && banks <= 64 && params->rows && params->features &&
                uint64_t(params->rows) * params->features <= MemoryLimit / banks &&
                cells == uint64_t(params->rows) * params->features * banks, "Invalid Ordered custom feature bank geometry");
        Require(bool(groups) == bool(offsets), "Ordered Custom grouping count and offsets must be supplied together");
        OrderedTargetOptions target; target.CustomSource = source;
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial, features, borders, permutations,
            types, groups, offsets, growth, banks, &target);
        std::lock_guard<std::mutex> lock(RegistryMutex); const uintptr_t handle = NextHandle++;
        Registry.emplace(handle, std::move(session)); *output = reinterpret_cast<void*>(handle);
    });
}
extern "C" int cbm_ordered_session_create_combination_banked(const CBMOrderedParams* params, uint32_t banks, uint64_t cells,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint8_t* types, const uint32_t* permutations, const CBMCombinationOptions* combination,
    const CBMCombinationComponent* components, uint32_t groups, const uint32_t* offsets, const uint32_t* winners,
    const uint32_t* losers, const float* pairWeights, double growth, void** output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        Require(output, "Ordered session output is required"); *output = nullptr;
        Require(params && combination && components && params->objective == 19, "Ordered Combination objective/options mismatch");
        Require(banks && banks <= 64 && params->rows && params->features &&
                uint64_t(params->rows) * params->features <= MemoryLimit / banks &&
                cells == uint64_t(params->rows) * params->features * banks, "Invalid Ordered Combination feature bank geometry");
        Require(bool(groups) == bool(offsets), "Ordered Combination grouping count and offsets must be supplied together");
        OrderedTargetOptions target; target.Combination = combination; target.Components = components;
        target.Winners = winners; target.Losers = losers; target.PairWeights = pairWeights;
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial, features, borders, permutations,
            types, groups, offsets, growth, banks, &target);
        std::lock_guard<std::mutex> lock(RegistryMutex); const uintptr_t handle = NextHandle++;
        Registry.emplace(handle, std::move(session)); *output = reinterpret_cast<void*>(handle);
    });
}
extern "C" int cbm_ordered_session_set_combination_yeti_seed_callback(void* handle,
    CBMCombinationYetiSeedCallback callback, void* context, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->SetCombinationSeedCallback(callback, context); });
}
extern "C" int cbm_ordered_session_step(void* handle, uint32_t selected, CBMStepInfo* info, uint32_t* depth,
    uint32_t* features, uint32_t* borders, uint8_t* types, float* values, float* weights, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->Step(selected, info, depth, features, borders, types, values, weights); });
}
extern "C" int cbm_ordered_session_begin_tree(void* handle, uint32_t selected, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->BeginTree(selected); });
}
extern "C" int cbm_ordered_session_grow_tree(void* handle, CBMStructureInfo* info, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->GrowTree(info); });
}
extern "C" int cbm_ordered_session_finish_tree(void* handle, CBMStepInfo* info, uint32_t* depth,
    uint32_t* features, uint32_t* borders, uint8_t* types, float* values, float* weights, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->FinishTree(info, depth, features, borders, types, values, weights); });
}
extern "C" int cbm_ordered_session_append_features(void* handle, const CBMAppendFeatureOptions* options,
    const uint8_t* const* matrices, const uint32_t* features, const uint32_t* borders, const uint8_t* types,
    const uint32_t* counts, const float* weights, const uint8_t* flags, const uint8_t* used,
    uint32_t* first, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->AppendFeatures(options, matrices, features, borders, types, counts, weights, flags, used, first); });
}
extern "C" int cbm_ordered_session_set_feature_activity(void* handle, uint32_t count, const uint8_t* active,
    char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->SetFeatureActivity(count, active); });
}
extern "C" int cbm_ordered_session_restore_feature_metadata(void* handle, uint32_t count, const uint8_t* flags,
    const uint8_t* used, const uint8_t* active, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->RestoreFeatureMetadata(count, flags, used, active); });
}
extern "C" int cbm_ordered_session_copy_feature_metadata(void* handle, uint32_t featureCapacity,
    uint32_t* counts, float* weights, uint8_t* flags, uint8_t* used, uint8_t* active, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->CopyFeatureMetadata(featureCapacity, counts, weights, flags, used, active); });
}
extern "C" int cbm_ordered_session_set_feature_penalties(void* handle, const CBMFeaturePenaltyOptions* options,
    const uint32_t* counts, const float* weights, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->ConfigureFeaturePenalties(options, counts, weights); });
}
extern "C" int cbm_ordered_session_info(void* handle, CBMStepInfo* info, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        Require(info != nullptr, "Ordered info output is required"); session->Info(*info); });
}
extern "C" int cbm_ordered_session_copy_predictions(void* handle, float* output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex); session->CopyPredictions(output); });
}
extern "C" int cbm_ordered_session_set_bootstrap(void* handle, const CBMBootstrapOptions* options,
    uint32_t testOnly, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->ConfigureBootstrap(options, testOnly); });
}
extern "C" int cbm_ordered_session_set_score_noise(void* handle, const CBMScoreNoiseOptions* options, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex); session->ConfigureNoise(options); });
}
extern "C" int cbm_ordered_set_add_ridge_to_target_function(void* handle, uint32_t enabled,
    char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->ConfigureRidge(enabled); });
}
extern "C" int cbm_ordered_session_set_langevin(void* handle, float temperature,
    CBMLangevinNoiseCallback noise, CBMLangevinSeedCallback seed, void* context, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->ConfigureLangevin(temperature, noise, seed, context); });
}
extern "C" int cbm_ordered_session_set_backtracking(void* handle, uint32_t type, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex); session->ConfigureBacktracking(type); });
}
extern "C" int cbm_ordered_session_get_bootstrap_state(void* handle, uint32_t* iteration,
    float* lambda, uint32_t* valid, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->BootstrapState(iteration, lambda, valid); });
}
extern "C" int cbm_ordered_session_state_shape(void* handle, uint32_t* tasks, uint32_t* cursors, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        Require(tasks && cursors, "Ordered state shape outputs are required"); *tasks = session->Tasks; *cursors = session->CursorCount; });
}
extern "C" int cbm_ordered_session_copy_state(void* handle, uint32_t tasks, uint32_t cursors,
    uint32_t* descriptors, float* output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->CopyState(tasks, cursors, descriptors, output); });
}
extern "C" int cbm_ordered_session_restore_cursors(void* handle, uint32_t count, const float* cursors, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex); session->Restore(count, cursors); });
}
extern "C" void cbm_ordered_session_close(void* handle) {
    @autoreleasepool { std::lock_guard<std::mutex> lock(RegistryMutex); Registry.erase(reinterpret_cast<uintptr_t>(handle)); }
}
