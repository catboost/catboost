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
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
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
struct BootstrapParams {
    uint32_t Rows, Type, SeedLow, SeedHigh, Iteration, Stream, Reserved0, Reserved1;
    float Temperature, Subsample, MVSLambda, NoiseScale;
};
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
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device && Device.hasUnifiedMemory && [Device supportsFamily:MTLGPUFamilyApple7],
                "Ordered training requires an Apple Silicon Metal GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Ordered command queue allocation failed");
        MTLCompileOptions* options = [MTLCompileOptions new];
        if (@available(macOS 13.0, *)) options.languageVersion = MTLLanguageVersion3_0;
        else throw std::runtime_error("Ordered training requires macOS 13 or newer");
        options.fastMathEnabled = NO;
        NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s",
            CBMMetalBootstrapSource, CBMMetalScoreNoiseSource, CBMMetalSource,
            CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
            CBMMetalBacktrackingSource, CBMMetalExactLeafSource, CBMMetalOrderedSource,
            CBMMetalOrderedSessionSource, CBMMetalOrderedBacktrackingSource,
            CBMMetalDeepPartitionSource, CBMMetalOrderedHistogramSource];
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
            "ClearOrderedPartitionCandidateStatistics", "ComputeOrderedPartitionCandidates"};
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
    explicit Command(CBMTrainStats& stats) : Stats(stats), Buffer([Context().Queue commandBuffer]) {
        Require(Buffer != nil, "Ordered command buffer allocation failed");
    }
    template<class P> void Dispatch(const char* name, std::initializer_list<Binding> inputs, const P& params,
        uint64_t width, bool grouped = false, uint64_t height = 1, uint64_t depth = 1,
        const StepParams* extra = nullptr, const void* final = nullptr, size_t finalSize = 0) {
        Require(width && width <= UINT32_MAX && height && depth, "Invalid Ordered dispatch");
        id<MTLComputeCommandEncoder> encoder = [Buffer computeCommandEncoder];
        Require(encoder != nil, "Ordered command encoder allocation failed");
        [encoder setComputePipelineState:Context().Pipelines.at(name)];
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
        [encoder setComputePipelineState:Context().Pipelines.at(name)];
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
        const uint8_t* candidateTypes = nullptr, uint32_t groupCount = 0, const uint32_t* groupOffsets = nullptr, double groupGrowth = 0, uint32_t binBanks = 1) {
        Require(params != nullptr, "Ordered parameters are required");
        P = *params;
        Require(P.rows >= 4 && P.rows <= (1u << 24), "Ordered rows must be in [4,16777216]");
        Require(P.features && uint64_t(P.rows) * P.features <= UINT32_MAX, "Invalid Ordered feature dimensions");
        Require(P.candidates <= uint64_t(P.features) * (candidateTypes ? 256 : 255) && P.iterations && P.iterations <= 100000,
                "Invalid Ordered candidate or iteration count");
        Require(P.depth <= 16 && P.objective <= 11 && P.score_function <= 1 && P.leaf_method <= 2,
                "Unsupported Ordered depth, objective, score or leaf method");
        Require(P.leaf_iterations && P.leaf_iterations <= 1000 && P.permutations && P.permutations <= 64 && P.normalize <= 1,
                "Invalid Ordered leaf iteration, permutation or normalization option");
        Require(!P.reserved0 && !P.reserved1 && !P.reserved2, "Ordered reserved parameters must be zero");
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
        Require(P.leaf_method != 0 || (P.objective < 8 && (P.objective != 6 || P.objective_param >= 2)),
                "Newton leaf estimation is unsupported for this Ordered objective");
        Require(P.leaf_method != 2 || P.objective >= 9, "Ordered Exact supports Quantile, MAE and MAPE only");
        Require(bins && targets && permutations && (!P.candidates || (features && borders)), "Ordered input buffers are required");
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
        std::vector<float> unitWeights(P.rows, 1);
        if (!weights) weights = unitWeights.data();
        double total = 0;
        for (uint32_t row = 0; row < P.rows; ++row) {
            Require(std::isfinite(targets[row]) && std::isfinite(weights[row]) && weights[row] >= 0 &&
                    (!initial || std::isfinite(initial[row])), "Ordered inputs must be finite with nonnegative weights");
            if (P.objective == 1) Require(targets[row] == 0 || targets[row] == 1, "Logloss targets must be zero or one");
            if (P.objective == 2) Require(targets[row] >= 0 && targets[row] <= 1, "CrossEntropy targets must be in [0,1]");
            if (P.objective == 3 || P.objective == 7) Require(targets[row] >= 0, "Poisson/Tweedie targets must be nonnegative");
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
        Loss = ReadLoss(Published);
    }
    StepParams StepConfiguration(uint32_t leaves = 1, uint32_t selected = 0) const { return {Tasks, leaves, CursorCount, selected}; }
    void Info(CBMStepInfo& info) const {
        info = {}; info.completed_iterations = Completed; info.finished = Completed == P.iterations;
        info.loss = Loss; info.stats = Stats;
    }
    float ReadLoss(id<MTLBuffer> predictions) {
        Command command(Stats);
        command.Dispatch("ReduceObjectiveLoss", {Targets, Weights, predictions, LossPartials}, K, LossGroups, true);
        command.Wait();
        const float* partials = static_cast<const float*>(LossPartials.contents);
        double sum = 0; for (uint32_t group = 0; group < LossGroups; ++group) sum += partials[group];
        if (P.objective == 0) sum = std::sqrt(sum);
        Require(std::isfinite(sum), "Ordered objective became nonfinite"); return sum;
    }
    void CheckStatus() const { Require(!*static_cast<const uint32_t*>(Status.contents), "Ordered GPU arithmetic became nonfinite"); }
    void ConfigureBootstrap(const CBMBootstrapOptions* options, uint32_t testOnly) {
        Require(!Completed && !Failed && options, "Configure Ordered bootstrap before the first step");
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
        Require(!Completed && !Failed && options, "Configure Ordered score noise before the first step");
        Require(std::isfinite(options->random_strength) && options->random_strength >= 0 &&
                !options->reserved0 && !options->reserved1 && !options->reserved2, "Invalid Ordered random_strength");
        const uint64_t extra = options->random_strength > 0 ? P.features * 4ull + FoldCount * 8ull : 0;
        Require(WorkingBytes + BootstrapBytes + extra + BacktrackingBytes <= MemoryLimit, "Ordered score noise exceeds 1 GiB workspace");
        RandomStrength = options->random_strength; NoiseBytes = extra;
        if (extra) { FeatureNoise = Context().Buffer(P.features * 4ull); QualityStatistics = Context().Buffer(FoldCount * 8ull); }
        else { FeatureNoise = nil; QualityStatistics = nil; }
    }
    void BootstrapState(uint32_t* iteration, float* lambda, uint32_t* valid) const {
        Require(iteration && lambda && valid, "Ordered bootstrap state outputs are required");
        *iteration = Bootstrap.iteration_offset + Completed; *valid = HasMvsLambda; *lambda = HasMvsLambda ? MvsLambda : 0;
    }
    void ConfigureBacktracking(uint32_t type) {
        Require(!Completed && !Failed && type <= 2, "Configure valid Ordered backtracking before the first step");
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
            command.Dispatch("OrderedBacktrackingDirections", {Targets, Weights, Cursor, Permutations, LeafIds, TaskBuffer,
                RawValues, Directions, LeafWeights, DirectionDot, TaskMass, Status}, P, step.Leaves, true, Tasks, 1, &step);
        };
        auto objective = [&](Command& command, id<MTLBuffer> values) {
            command.Dispatch("OrderedBacktrackingObjective", {Targets, Weights, Cursor, Permutations, LeafIds, TaskBuffer,
                values, TaskMass, BacktrackingLoss}, P, Tasks, true, 1, 1, &step);
        };
        Command initialize(Stats); direction(initialize); objective(initialize, RawValues); initialize.Wait(); CheckStatus();
        double current = ReadBacktrackingValue(), dot = ReadDirectionDot(step.Leaves);
        Require(std::isfinite(current), "Ordered initial backtracking objective became nonfinite");
        BacktrackingParams backtracking = {1, BacktrackingType, 0, P.normalize};
        bool updated = false, newDirection = false;
        // CUDA counts rejected trials against the budget, but extends up to
        // 100 attempts until the first successful update.
        for (uint32_t attempt = 0; attempt < P.leaf_iterations || (!updated && attempt < 100); ++attempt) {
            Command trial(Stats);
            if (newDirection) direction(trial);
            trial.Dispatch("OrderedBacktrackingCandidate", {RawValues, Directions, LeafWeights, TrialValues},
                P, uint64_t(Tasks) * step.Leaves, false, 1, 1, &step, &backtracking, sizeof(backtracking));
            objective(trial, TrialValues); trial.Wait(); CheckStatus();
            if (newDirection) dot = ReadDirectionDot(step.Leaves);
            const double candidate = ReadBacktrackingValue();
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
            Command command(Stats);
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
    void Step(uint32_t selected, CBMStepInfo* info, uint32_t* depth, uint32_t* splitFeatures,
        uint32_t* splitBins, uint8_t* splitTypes, float* values, float* weights) {
        Require(!Failed && Completed < P.iterations, "Ordered session is failed or has no remaining iterations");
        Require(selected < LearnPermutations, "Invalid Ordered search permutation");
        Require(info && depth && values && weights && (!P.depth || (splitFeatures && splitBins && splitTypes)),
                "Ordered step output buffers are required");
        try {
            if (P.depth) {
                std::fill(splitFeatures, splitFeatures + P.depth, 0); std::fill(splitBins, splitBins + P.depth, 0);
                std::fill(splitTypes, splitTypes + P.depth, 0);
            }
            std::fill(values, values + MaxLeaves, 0); std::fill(weights, weights + MaxLeaves, 0);
            const uint32_t foldCount = PermutationFoldCounts[selected];
            const uint32_t packedRows = PermutationPackedRows[selected];
            const uint32_t taskOffset = PermutationTaskOffsets[selected];
            StepParams step = StepConfiguration(1, selected);
            Command initialize(Stats); initialize.Zero(LeafIds); initialize.Zero(RawValues); initialize.Zero(LeafWeights); initialize.Zero(Status);
            initialize.Dispatch("OrderedSessionDerivatives", {Targets, Weights, Cursor, Permutations, TaskBuffer, Derivatives, Status},
                P, P.rows, false, Tasks, 1, &step);
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
                Command noise(Stats);
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
                    Command prepare(Stats);
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
                Command sample(Stats);
                if (MvsInput) {
                    sample.Dispatch("ComputeMvsThresholds", {MvsInput, MvsThresholds}, bootstrap, (packedRows + 8191) / 8192, true);
                    sample.Dispatch("GenerateMvsBootstrapWeights", {Multipliers, MvsInput, MvsThresholds}, bootstrap, packedRows);
                } else sample.Dispatch("GenerateBootstrapWeights", {Multipliers, Derivatives}, bootstrap, packedRows);
                sample.Dispatch("ApplyOrderedBootstrap", {Derivatives, Multipliers,
                    {TaskBuffer, uint64_t(taskOffset) * 16}, Sampled}, sampling, P.rows, false, foldCount);
                sample.Wait();
            }
            std::vector<uint32_t> chosen;
            float scoreBefore = 0;
            for (uint32_t level = 0; level < P.depth && P.candidates; ++level) {
                OrderedParams hist = {P.rows, P.features, foldCount, step.Leaves, 0, CursorCount, 1, 0,
                    P.l2, P.normalize, scoreBefore, P.learning_rate};
                UpdateFeaturePenalties();
                Command search(Stats);
                if (FeatureNoise) {
                    auto noise = bootstrap; noise.Rows = P.features; noise.Stream = level + 1; noise.NoiseScale = noiseScale;
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
                search.Dispatch("OrderedSessionFindWinner", {Scores, CandidatePairs, Winner}, P, 1, true); search.Wait();
                Histogram->Check();
                const auto winner = *static_cast<const SplitState*>(Winner.contents);
                Require(!winner.InvalidScore, "Ordered split score became nonfinite");
                if (!winner.Valid || std::find(chosen.begin(), chosen.end(), winner.Index) != chosen.end()) break;
                chosen.push_back(winner.Index); splitFeatures[level] = winner.Feature; splitBins[level] = winner.Bin;
                splitTypes[level] = winner.Type;
                scoreBefore = winner.Score;
                Command partition(Stats);
                partition.Dispatch("OrderedSessionUpdateLeafIds", {Bins, Winner, LeafIds}, P, P.rows, false, BinBanks, 1, &step);
                if (level + 1 < P.depth) Histogram->Partition(partition, Binding(LeafIds, uint64_t(selected) * P.reserved0 * 4), step.Leaves * 2);
                partition.Wait(); step.Leaves *= 2;
            }
            if (P.leaf_method == 2) EstimateExact(step.Leaves);
            else if (BacktrackingBytes) EstimateBacktracking(step);
            else for (uint32_t iteration = 0; iteration < P.leaf_iterations; ++iteration) {
                Command estimate(Stats);
                estimate.Dispatch("OrderedSessionEstimateLeaves", {Targets, Weights, Cursor, Permutations, LeafIds, TaskBuffer,
                    RawValues, LeafWeights, Status}, P, step.Leaves, true, Tasks, 1, &step);
                estimate.Wait(); CheckStatus();
            }
            Command update(Stats);
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
            ++Completed; Loss = nextLoss; *depth = chosen.size(); Info(*info);
        } catch (...) { Failed = true; throw; }
    }
    void ConfigureFeaturePenalties(const CBMFeaturePenaltyOptions* options, const uint32_t* counts,
                                   const float* weights) {
        Require(!Completed && !Failed && !PenaltiesConfigured, "Ordered feature penalties can be configured once before training");
        Require(options && counts && !options->reserved0 && !options->reserved1 && !options->reserved2 &&
                std::isfinite(options->model_size_reg) && options->model_size_reg >= 0,
                "Invalid Ordered feature penalty options");
        for (uint32_t f = 0; f < P.features; ++f) {
            Require(!weights || (std::isfinite(weights[f]) && weights[f] >= 0), "Ordered feature weights must be finite and nonnegative");
        }
        Require(WorkingBytes + BootstrapBytes + NoiseBytes + BacktrackingBytes + uint64_t(P.features) * 8 <= MemoryLimit,
                "Ordered feature penalty state exceeds 1 GiB");
        CtrCounts.assign(counts, counts + P.features); FeatureWeights.resize(P.features, 1);
        if (weights) std::copy_n(weights, P.features, FeatureWeights.begin());
        ModelSizeReg = options->model_size_reg; PenaltiesConfigured = true;
        WorkingBytes += uint64_t(P.features) * 8; UpdateFeaturePenalties();
    }
    void CopyPredictions(float* output) const {
        Require(output != nullptr, "Ordered prediction output is required");
        std::memcpy(output, Published.contents, P.rows * 4ull);
    }
    void CopyState(uint32_t tasks, uint32_t cursors, uint32_t* descriptors, float* output) const {
        Require(tasks == Tasks && cursors == CursorCount && descriptors && output, "Ordered state shape mismatch");
        std::memcpy(descriptors, Descriptors.data(), Tasks * 16ull); std::memcpy(output, Cursor.contents, CursorCount * 4ull);
    }
    void Restore(uint32_t count, const float* input) {
        Require(!Completed && !Failed && count == CursorCount && input, "Ordered state can only be restored before the first step with exact cursor count");
        for (uint32_t i = 0; i < count; ++i) Require(std::isfinite(input[i]), "Ordered restored cursors must be finite");
        auto restored = Context().Buffer(count * 4ull, input);
        const auto step = StepConfiguration();
        Command publish(Stats);
        publish.Dispatch("OrderedSessionPublish", {restored, Permutations, TaskBuffer, NextPublished}, P, P.rows, false, 1, 1, &step);
        publish.Wait(); const float nextLoss = ReadLoss(NextPublished);
        Cursor = restored; std::swap(Published, NextPublished); Loss = nextLoss;
    }
private:
    bool Failed = false, PenaltiesConfigured = false;
    float ModelSizeReg = 0;
    std::vector<uint32_t> CtrCounts;
    std::vector<float> FeatureWeights;
    void UpdateFeaturePenalties() {
        if (!PenaltiesConfigured) return;
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
extern "C" int cbm_ordered_session_step(void* handle, uint32_t selected, CBMStepInfo* info, uint32_t* depth,
    uint32_t* features, uint32_t* borders, uint8_t* types, float* values, float* weights, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto session = Get(handle); std::lock_guard<std::mutex> lock(session->Mutex);
        session->Step(selected, info, depth, features, borders, types, values, weights); });
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
