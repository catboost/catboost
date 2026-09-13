#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_greedy_trainer.h"
#include "metal_greedy_kernels.h"
#include "metal_greedy_bootstrap_kernels.h"
#include "metal_bootstrap_kernels.h"
#include "metal_score_noise_kernels.h"
#include "metal_kernels.h"
#include "metal_additional_objective_kernels.h"
#include "metal_objective_kernels.h"
#include "metal_backtracking_kernels.h"
#include "metal_exact_leaf_kernels.h"
#include "metal_sort.h"
#include "metal_incremental_partition_kernels.h"
#include "metal_compact_histogram_kernels.h"
#include "metal_querywise_kernels.h"
#include "metal_pairwise_runtime.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr uint64_t MemoryLimit = 1ull << 30;
constexpr uint32_t Missing = std::numeric_limits<uint32_t>::max();
static_assert(CBMMetalKernelAbiVersion == 2, "Review shared Metal objective parameter ABI changes");
using KernelParams = CBMMetalKernelParams;
struct CompactParams { uint32_t Rows, Features, Leaves, TotalBins, FeatureBegin, TileRows, JobCapacity, Reuse; };
struct EvaluationParams { uint32_t Rows, Features, MaxDepth, Reserved; };
struct QueryParams { uint32_t Rows, Groups, Objective, ApplyLeafValues; float Beta, Lambda; uint32_t Leaves, Reserved; };
struct QueryProjectionParams { uint32_t Rows, Leaves, Tiles, LeafMethod; };
static_assert(sizeof(QueryParams) == 32 && sizeof(QueryProjectionParams) == 16 && sizeof(CBMQueryOptions) == 16);
struct BacktrackingParams { float Step; uint32_t Type, AddRidge, Normalize; };
struct BootstrapParams {
    uint32_t Rows, Type, SeedLow, SeedHigh, Iteration, Stream, Reserved0, Reserved1;
    float Temperature, Subsample, MVSLambda, NoiseScale;
};
static_assert(sizeof(BootstrapParams) == 48);
static_assert(sizeof(KernelParams) == 96 && sizeof(CompactParams) == 32);
static_assert(sizeof(CBMGreedyTrainParams) == 80 && sizeof(CBMGreedyNode) == 24);

static const char* GreedyEvaluationSource = R"METAL(
struct GreedyTreeNode { uint feature, bin, type, left, right, leaf; };
struct GreedyEvaluationParams { uint rows, features, max_depth, reserved; };
kernel void RouteGreedyFixedTree(const device uchar* bins [[buffer(0)]],
    const device GreedyTreeNode* nodes [[buffer(1)]], device uint* leaves [[buffer(2)]],
    constant GreedyEvaluationParams& p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    uint index = 0;
    for (uint depth = 0; depth <= p.max_depth; ++depth) {
        const GreedyTreeNode node = nodes[index];
        if (node.leaf != 0xffffffffu) { leaves[row] = node.leaf; return; }
        const uint value = uint(bins[ulong(node.feature) * p.rows + row]);
        index = (node.type ? value == node.bin : value > node.bin) ? node.right : node.left;
    }
}
kernel void AddGreedyEvaluationTree(const device uchar* bins [[buffer(0)]],
    const device GreedyTreeNode* nodes [[buffer(1)]], const device float* values [[buffer(2)]],
    device float* predictions [[buffer(3)]], constant GreedyEvaluationParams& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    uint index = 0;
    // The host validates every node and path before upload. The bounded walk
    // includes the terminal node at max_depth and matches training additions.
    for (uint depth = 0; depth <= p.max_depth; ++depth) {
        const GreedyTreeNode node = nodes[index];
        if (node.leaf != 0xffffffffu) {
            const uint dimensions = max(p.reserved, 1u);
            for (uint k = 0; k < dimensions; ++k)
                predictions[ulong(row) * dimensions + k] += values[ulong(node.leaf) * dimensions + k];
            return;
        }
        const uint value = uint(bins[ulong(node.feature) * p.rows + row]);
        index = (node.type ? value == node.bin : value > node.bin) ? node.right : node.left;
    }
}
)METAL";

void Require(bool condition, const std::string& text) {
    if (!condition) throw std::runtime_error(text);
}
// Literal validation messages allocate only when a check fails.
void Require(bool condition, const char* text) {
    if (!condition) throw std::runtime_error(text);
}
void CopyText(char* output, size_t capacity, const char* text) {
    if (!output || !capacity) return;
    const size_t count = std::min(capacity - 1, std::strlen(text));
    std::memcpy(output, text, count);
    output[count] = '\0';
}
std::string ErrorText(NSError* error) {
    return error ? [[error localizedDescription] UTF8String] : "Unknown Metal error";
}

// This separately compiled session reuses the established GPU arithmetic,
// histogram and loss kernels. It does not link or copy the symmetric trainer.
struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device != nil && Device.hasUnifiedMemory && [Device supportsFamily:MTLGPUFamilyApple7],
            "Greedy Metal training requires an Apple Silicon GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Could not create Metal command queue");
        MTLCompileOptions* options = [MTLCompileOptions new];
        if (@available(macOS 13.0, *)) options.languageVersion = MTLLanguageVersion3_0;
        else throw std::runtime_error("Greedy Metal training requires macOS 13 or newer");
        options.fastMathEnabled = NO;
        NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s",
            CBMMetalSource, CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
            CBMMetalIncrementalPartitionSource, CBMMetalCompactHistogramSource, CBMMetalGreedySource,
            GreedyEvaluationSource, CBMMetalBacktrackingSource, CBMMetalExactLeafSource,
            CBMMetalBootstrapSource, CBMMetalScoreNoiseSource, CBMMetalGreedyBootstrapSource, CBMMetalQuerywiseSource];
        NSError* error = nil;
        id<MTLLibrary> library = [Device newLibraryWithSource:source options:options error:&error];
        Require(library != nil, "Greedy Metal shader compilation failed: " + ErrorText(error));
        const char* names[] = {
            "InitializeObjectivePredictions", "ObjectiveDerivatives", "InitializeRootPartition",
            "ReduceStructurePartials", "CollectPartitionStatistics", "InitializeLeafValues",
            "ReduceLeafObjectivePartials", "EstimateNewtonLeafValues", "FinalizeLeafValues",
            "AddObjectiveBinModelValue", "ReduceObjectiveLoss", "ResetCompactHistogramWorkState",
            "BuildCompactHistogramJobs", "BuildCompactHistogramDispatchArguments", "ClearCompactHistograms",
            "ComputeCompactHistograms", "ScanCompactHistograms", "FindGreedySplitWinners",
            "ReduceGreedySplitWinners", "SelectGreedyLeaves", "RouteGreedySplitRows",
            "UpdateGreedyLeafDepths", "CountGreedyPartitionBits", "ScanGreedyPartitionTiles",
            "BuildGreedyPartitionOffsets", "ScatterGreedyPartitionRows", "AddGreedyEvaluationTree", "RouteGreedyFixedTree",
            "PrepareBacktrackingDirection", "BuildBacktrackingCandidate", "ReduceBacktrackingObjective",
            "PrepareExactResiduals", "MakeExactLeafKeys", "BuildExactLeafOffsets", "ReduceExactLeafPartials",
            "PrefixExactLeafTiles", "SelectExactLeafQuantile", "FinalizeExactLeafValues",
            "GenerateBootstrapWeights", "ApplyBootstrapWeights", "GenerateScoreFeatureNoise",
            "CountGreedyBootstrapRows", "PrefixGreedyBootstrapOffsets", "ReduceGreedyScoreNoiseStatistics",
            "PrepareQuerywisePoint", "QueryRmseDerivatives", "QuerySoftMaxDerivatives",
            "ReduceQuerywiseLeafPartials", "ReduceQuerywiseObjective", "ResetQuerywiseLeafIds",
            "ValidateQuerywiseStructureCurvature"
        };
        for (const char* name : names) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing Metal kernel: ") + name);
            id<MTLComputePipelineState> pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline != nil, std::string("Could not create Metal pipeline ") + name + ": " + ErrorText(error));
            Require(pipeline.maxTotalThreadsPerThreadgroup >= 256, "Metal pipeline requires 256-thread groups");
            Pipelines.emplace(name, pipeline);
        }
    }
    id<MTLBuffer> Buffer(uint64_t bytes, const void* data = nullptr) {
        const uint64_t length = std::max<uint64_t>(bytes, 1);
        Require(length <= MemoryLimit && length <= Device.maxBufferLength, "Greedy Metal buffer exceeds memory limit");
        id<MTLBuffer> output = data && bytes
            ? [Device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:length options:MTLResourceStorageModeShared];
        Require(output != nil, "Greedy Metal buffer allocation failed");
        return output;
    }
};
Runtime& GetRuntime() { static Runtime runtime; return runtime; }

struct Command {
    Runtime& Context;
    CBMTrainStats& Stats;
    id<MTLCommandBuffer> Buffer;
    Command(Runtime& context, CBMTrainStats& stats) : Context(context), Stats(stats) { Reset(); }
    void Reset() {
        Buffer = [Context.Queue commandBuffer];
        Require(Buffer != nil, "Could not create Metal command buffer");
    }
    template <class Params>
    void Dispatch(const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                  const Params& p, uint64_t width, bool groups = false, uint64_t height = 1,
                  const void* extra = nullptr, size_t extraBytes = 0) {
        Require(width && width <= Missing && height && height <= Missing, "Invalid greedy Metal dispatch shape");
        id<MTLComputeCommandEncoder> encoder = [Buffer computeCommandEncoder];
        Require(encoder != nil, "Could not create Metal encoder");
        [encoder setComputePipelineState:Context.Pipelines.at(name)];
        NSUInteger index = 0;
        for (id<MTLBuffer> value : buffers) [encoder setBuffer:value offset:0 atIndex:index++];
        [encoder setBytes:&p length:sizeof(p) atIndex:index++];
        if (extra) [encoder setBytes:extra length:extraBytes atIndex:index];
        if (groups) [encoder dispatchThreadgroups:MTLSizeMake(width, height, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [encoder dispatchThreads:MTLSizeMake(width, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        ++Stats.kernel_dispatches;
    }
    void Indirect(const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                  const CompactParams& p, id<MTLBuffer> arguments, NSUInteger offset) {
        id<MTLComputeCommandEncoder> encoder = [Buffer computeCommandEncoder];
        Require(encoder != nil, "Could not create indirect Metal encoder");
        [encoder setComputePipelineState:Context.Pipelines.at(name)];
        NSUInteger index = 0;
        for (id<MTLBuffer> value : buffers) [encoder setBuffer:value offset:0 atIndex:index++];
        [encoder setBytes:&p length:sizeof(p) atIndex:index];
        [encoder dispatchThreadgroupsWithIndirectBuffer:arguments indirectBufferOffset:offset
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        ++Stats.kernel_dispatches;
    }
    void Wait() {
        [Buffer commit];
        [Buffer waitUntilCompleted];
        Require(Buffer.status == MTLCommandBufferStatusCompleted, "Greedy Metal command failed: " + ErrorText(Buffer.error));
        const double start = Buffer.GPUStartTime, end = Buffer.GPUEndTime;
        if (std::isfinite(start) && std::isfinite(end) && start > 0 && end >= start) Stats.gpu_seconds += end - start;
    }
};

class Session {
public:
    CBMGreedyTrainParams Options;
    CBMGreedyStepInfo Info = {};
    std::mutex Mutex;
    bool Failed = false;
    std::vector<CBMGreedyNode> Nodes;

    Session(const CBMGreedyTrainParams* input, const uint8_t* bins, const float* targets,
            const float* weights, const float* initial, const uint32_t* features,
            const uint32_t* candidateBins, const uint8_t* types,
            const CBMObjectiveOptions* objectiveOptions = nullptr,
            const CBMQueryOptions* queryOptions = nullptr, const uint32_t* groupOffsets = nullptr,
            const CBMPairOptions* pairOptions = nullptr, const uint32_t* pairWinners = nullptr,
            const uint32_t* pairLosers = nullptr, const float* pairWeights = nullptr) {
        Require(input && bins && targets, "Greedy parameters, bins and targets are required");
        Options = *input;
        const auto& p = Options;
        Require(p.rows && p.rows <= (1u << 24) && p.features && p.features <= 65536, "Invalid row or feature count");
        Require(p.candidates <= (1u << 24) && (!p.candidates || (features && candidateBins)), "Invalid candidate buffers");
        Require(p.bins_per_feature >= 1 && p.bins_per_feature <= 256, "bins_per_feature must be in [1,256]");
        Require(p.iterations && p.iterations <= 100000 &&
            (p.policy == 1 || p.depth <= (p.policy == 2 ? 65535u : 16u)),
            "Invalid iteration count or depth");
        Require(p.max_leaves && p.max_leaves <= 65536 && p.policy <= 2, "Invalid grow policy or max_leaves");
        Require(p.min_data_in_leaf && p.min_data_in_leaf <= (1u << 24), "Invalid min_data_in_leaf");
        Require(p.objective <= 14 && p.score_function <= 6 && p.leaf_method <= 2, "Unsupported greedy objective, score or leaf method");
        Require((p.objective == 12 || p.objective == 13) == bool(queryOptions),
            "QueryRMSE/QuerySoftMax require the query-aware greedy constructor");
        Require((p.objective == 14) == bool(pairOptions), "PairLogit requires the supplied-pair greedy constructor");
        if (pairOptions) {
            Require(pairOptions->pair_count && pairOptions->pair_count <= Missing / 2 &&
                pairWinners && pairLosers && pairWeights && !pairOptions->reserved0 && !pairOptions->reserved1,
                "Greedy PairLogit requires nonempty pair arrays and zero reserved fields");
            Require(pairOptions->group_count <= p.rows && bool(pairOptions->group_count) == bool(groupOffsets),
                "PairLogit group count and offsets must be supplied together");
        }
        if (queryOptions) {
            Require(groupOffsets && queryOptions->group_count && queryOptions->group_count <= p.rows,
                "Query group offsets and a valid group count are required");
            Require(std::isfinite(queryOptions->beta) && std::isfinite(queryOptions->lambda) && !queryOptions->reserved,
                "Query beta and lambda must be finite, with reserved options zero");
            Require(groupOffsets[0] == 0 && groupOffsets[queryOptions->group_count] == p.rows,
                "Query group offsets must span all training rows");
            for (uint32_t group = 0; group < queryOptions->group_count; ++group)
                Require(groupOffsets[group] < groupOffsets[group + 1], "Query group offsets must increase strictly");
            QueryOptions = *queryOptions;
        }
        const float objectiveParam = objectiveOptions ? objectiveOptions->objective_param :
            (p.objective >= 8 && p.objective <= 10 ? .5f : 1.f);
        Require(!objectiveOptions || (objectiveOptions->objective == p.objective &&
            objectiveOptions->leaf_estimation_method == p.leaf_method && !objectiveOptions->reserved),
            "Configured greedy objective/method must match training parameters, with zero reserved field");
        Require(std::isfinite(objectiveParam), "Objective parameter must be finite");
        if (p.objective == 4) Require(objectiveParam >= 0, "Huber delta must be nonnegative");
        if (p.objective == 5 || p.objective == 8 || p.objective == 9)
            Require(objectiveParam >= 0 && objectiveParam <= 1, "Objective alpha must be in [0,1]");
        if (p.objective == 6) Require(objectiveParam >= 1, "Lq q must be at least 1");
        if (p.objective == 7) Require(objectiveParam > 1 && objectiveParam < 2, "Tweedie power must be in (1,2)");
        if (p.leaf_method == 0) {
            Require(p.objective < 8 || p.objective >= 12, "This objective does not support Newton leaf estimation");
            Require(p.objective != 6 || objectiveParam >= 2, "Lq Newton requires q >= 2");
        }
        if (p.leaf_method == 2) Require(p.objective >= 9 && p.objective <= 11, "Exact supports Quantile, MAE and MAPE only");
        Require(p.leaf_iterations && p.leaf_iterations <= 1000, "leaf_iterations must be in [1,1000]");
        Require(!p.reserved0 && !p.reserved1 && !p.reserved2 && !p.reserved3, "Reserved parameters must be zero");
        Require(std::isfinite(p.learning_rate) && p.learning_rate > 0 && p.learning_rate <= 1 &&
            std::isfinite(p.l2_leaf_reg) && p.l2_leaf_reg >= 0 && std::isfinite(p.bias), "Invalid finite learning parameters");
        // Lossguide depth is a path bound, while its workspace is bounded by
        // max_leaves. Avoid shifting by an unrestricted requested depth.
        MaxLeaves = std::min(p.max_leaves, p.policy == 2 ? p.depth + 1 : (1u << std::min(p.depth, 16u)));
        std::vector<uint32_t> featureOffsets(p.features + 1, 0);
        std::vector<uint8_t> featureTypes(p.features, 255), candidateTypes(p.candidates, 0);
        for (uint32_t i = 0; i < p.candidates; ++i) {
            Require(features[i] < p.features && candidateBins[i] < p.bins_per_feature, "Candidate feature or bin is out of range");
            const uint8_t type = types ? types[i] : 0;
            Require(type <= 1, "Candidate type must be numeric or one-hot");
            Require(featureTypes[features[i]] == 255 || featureTypes[features[i]] == type,
                "Candidates of one feature must share their split type");
            featureTypes[features[i]] = candidateTypes[i] = type;
            featureOffsets[features[i] + 1] = std::max(featureOffsets[features[i] + 1], candidateBins[i] + 1);
        }
        for (uint32_t f = 0; f < p.features; ++f) {
            featureOffsets[f + 1] += featureOffsets[f];
            if (featureTypes[f] == 255) featureTypes[f] = 0;
        }
        TotalBins = featureOffsets.back();
        const uint64_t histogramCells = uint64_t(MaxLeaves) * std::max(TotalBins, 1u);
        const uint32_t objectiveTiles = std::min(32u, (p.rows - 1) / 8192 + 1);
        JobCapacity = (p.rows - 1) / 8192 + 1 + std::min(p.rows, MaxLeaves);
        const uint32_t scoreGroups = std::min(64u, std::max(1u, (p.candidates + 255) / 256));
        PartitionTiles = (p.rows - 1) / 256 + 1;
        LossGroups = std::min(256u, PartitionTiles);
        const uint64_t bytes = uint64_t(p.rows) * (p.features + 40ull) + 4ull * PartitionTiles +
            8ull * histogramCells + uint64_t(MaxLeaves) * (88ull + 32ull * objectiveTiles + 32ull * scoreGroups) +
            16ull * JobCapacity + 18ull * p.features + 9ull * p.candidates + 4096 +
            (queryOptions ? 4ull * p.rows + 12ull * QueryOptions.group_count + 8ull * LossGroups + 8 : 0);
        WorkingBytes = bytes;
        Require(bytes <= MemoryLimit && histogramCells <= Missing,
            "Greedy Metal workspace exceeds 1 GiB; reduce max_leaves, features, bins or rows");
        if (pairOptions) {
            const uint64_t pairBytes = 44ull * pairOptions->pair_count + 8ull * p.rows + 4 +
                8ull * LossGroups + 8ull * ((MaxLeaves + 255) / 256) + 8;
            Require(WorkingBytes + pairBytes <= MemoryLimit, "Greedy PairLogit combined workspace exceeds 1 GiB");
            Context = &GetRuntime();
            Pairwise = std::make_unique<CBMPairwiseRuntime>(Context->Device, p.rows, pairOptions->pair_count,
                pairWinners, pairLosers, pairWeights, MaxLeaves, LossGroups, pairOptions->group_count, groupOffsets);
            WorkingBytes += Pairwise->AllocatedBytes();
            Require(WorkingBytes <= MemoryLimit, "Greedy PairLogit combined workspace exceeds 1 GiB");
            weights = Pairwise->IncidentWeights().data();
        }
        std::vector<float> unitWeights;
        if (!weights) { unitWeights.assign(p.rows, 1); weights = unitWeights.data(); }
        TotalWeight = 0;
        double queryTargetMass = 0;
        for (uint32_t row = 0; row < p.rows; ++row) {
            Require(std::isfinite(targets[row]) && std::isfinite(weights[row]) && weights[row] >= 0,
                "Targets and nonnegative weights must be finite");
            Require(p.objective != 1 || targets[row] == 0 || targets[row] == 1, "Logloss targets must be zero or one");
            Require(p.objective != 2 || (targets[row] >= 0 && targets[row] <= 1), "CrossEntropy targets must be in [0,1]");
            Require((p.objective != 3 && p.objective != 7) || targets[row] >= 0,
                "Poisson/Tweedie targets must be nonnegative");
            Require(!initial || std::isfinite(initial[row]), "Initial predictions must be finite");
            Require(p.objective != 13 || targets[row] >= 0, "QuerySoftMax targets must be nonnegative");
            if (p.objective == 13) queryTargetMass += double(weights[row]) * targets[row];
            TotalWeight += weights[row];
        }
        Require(p.objective != 13 || (queryTargetMass > 0 && queryTargetMass <= std::numeric_limits<float>::max()),
            "QuerySoftMax requires positive finite target mass in float32");
        Require(TotalWeight > 0 && TotalWeight <= std::numeric_limits<float>::max(), "Total sample weight must be positive and finite in float32");
        for (uint64_t cell = 0; cell < uint64_t(p.rows) * p.features; ++cell)
            Require(bins[cell] < p.bins_per_feature, "Input quantized bin is out of range");
        Context = &GetRuntime();
        CopyText(Info.stats.device_name, sizeof(Info.stats.device_name), [Context->Device.name UTF8String]);
        K = {p.rows, p.features, p.bins_per_feature, 1, p.candidates, 0, 0, 0,
            p.bias, p.learning_rate, p.l2_leaf_reg, p.score_function, p.objective, 0, p.leaf_iterations,
            8192, static_cast<float>(TotalWeight), objectiveTiles, PartitionTiles, scoreGroups, objectiveParam, p.leaf_method, 0, 0};
        if (p.leaf_method == 2) {
            // Includes both resident radix-sort scratch sets retained by one
            // command, in addition to our residual/keys/row and tile buffers.
            ExactBytes = 50ull * p.rows + 8ull * MaxLeaves * objectiveTiles + 12ull * MaxLeaves;
            Require(WorkingBytes + ExactBytes <= MemoryLimit, "Greedy Exact workspace exceeds 1 GiB");
            ExactResiduals = Context->Buffer(4ull * p.rows); ExactEffectiveWeights = Context->Buffer(4ull * p.rows);
            ExactKeysA = Context->Buffer(4ull * p.rows); ExactRowsA = Context->Buffer(4ull * p.rows);
            ExactKeysB = Context->Buffer(4ull * p.rows); ExactRowsB = Context->Buffer(4ull * p.rows);
            ExactTileOffsets = Context->Buffer(8ull * MaxLeaves * objectiveTiles);
            ExactTotals = Context->Buffer(8ull * MaxLeaves); ExactSelected = Context->Buffer(4ull * MaxLeaves);
        }
        G = {p.rows, p.features, 1, p.candidates, TotalBins, scoreGroups, p.score_function, p.min_data_in_leaf,
            std::min(p.depth, MaxLeaves - 1), MaxLeaves, p.policy, 1, static_cast<uint32_t>(histogramCells), MaxLeaves, 0, 0, p.l2_leaf_reg, 0, 0, 0};
        Data = Context->Buffer(uint64_t(p.rows) * p.features, bins);
        Target = Context->Buffer(p.rows * 4ull, targets); Weight = Context->Buffer(p.rows * 4ull, weights);
        Prediction = Context->Buffer(p.rows * 4ull, initial);
        Gradient = Context->Buffer(p.rows * 4ull); Hessian = Context->Buffer(p.rows * 4ull);
        LeafIds = Context->Buffer(p.rows * 4ull);
        Rows = Context->Buffer(p.rows * 4ull); NextRows = Context->Buffer(p.rows * 4ull);
        RowPrefix = Context->Buffer(p.rows * 4ull); TilePrefix = Context->Buffer(PartitionTiles * 4ull);
        Offsets = Context->Buffer((MaxLeaves + 1) * 4ull); NextOffsets = Context->Buffer((MaxLeaves + 1) * 4ull);
        Depths = Context->Buffer(MaxLeaves * 4ull); NextDepths = Context->Buffer(MaxLeaves * 4ull);
        CandidateFeatures = Context->Buffer(p.candidates * 4ull, features);
        CandidateBins = Context->Buffer(p.candidates * 4ull, candidateBins);
        CandidateTypes = Context->Buffer(p.candidates, candidateTypes.data());
        FeatureTypes = Context->Buffer(p.features, featureTypes.data());
        FeatureOffsets = Context->Buffer((p.features + 1) * 4ull, featureOffsets.data());
        std::vector<float> featureWeights(p.features, 1), noise(p.features, 0);
        FeatureWeights = Context->Buffer(p.features * 4ull, featureWeights.data());
        FeatureNoise = Context->Buffer(p.features * 4ull, noise.data());
        HistSums = Context->Buffer(histogramCells * 4); HistWeights = Context->Buffer(histogramCells * 4);
        LeafSums = Context->Buffer(MaxLeaves * 4ull); LeafWeights = Context->Buffer(MaxLeaves * 4ull);
        RawValues = Context->Buffer(MaxLeaves * 4ull); Values = Context->Buffer(MaxLeaves * 4ull);
        OutputWeights = Context->Buffer(MaxLeaves * 4ull);
        ObjectivePartials = Context->Buffer(uint64_t(MaxLeaves) * objectiveTiles * 32);
        LossPartials = Context->Buffer(LossGroups * 4ull);
        if (queryOptions) {
            QueryOffsets = Context->Buffer(4ull * (QueryOptions.group_count + 1), groupOffsets);
            QueryPoint = Context->Buffer(4ull * p.rows);
            QueryStatistics = Context->Buffer(8ull * QueryOptions.group_count);
            QueryLossPartials = Context->Buffer(8ull * LossGroups);
            QueryValidation = Context->Buffer(4);
        }
        WinnerPartials = Context->Buffer(uint64_t(MaxLeaves) * scoreGroups * sizeof(CBMGreedySplit));
        Winners = Context->Buffer(MaxLeaves * sizeof(CBMGreedySplit));
        Selected = Context->Buffer(MaxLeaves * 4ull); RightIds = Context->Buffer(MaxLeaves * 4ull);
        Frontier = Context->Buffer(sizeof(CBMGreedyFrontier));
        Jobs = Context->Buffer(JobCapacity * 16ull); Active = Context->Buffer(MaxLeaves * 4ull);
        WorkState = Context->Buffer(16); Arguments = Context->Buffer(36);
        Command command(*Context, Info.stats);
        if (!initial) command.Dispatch("InitializeObjectivePredictions", {Prediction}, K, p.rows);
        EncodeLoss(command);
        command.Wait();
        Info.loss = ReadLoss();
        DatasetBins = {Data}; DatasetPredictions = {Prediction};
    }

    void Step() {
        Require(!Failed, "Greedy session failed and must be closed");
        Require(Info.completed_iterations < Options.iterations, "Greedy session is already complete");
        try { StepImpl(); }
        catch (...) { Failed = true; throw; }
    }
    void Copy(CBMGreedyStepInfo* info, CBMGreedyNode* nodes, float* values, float* weights) const {
        *info = Info;
        std::copy(Nodes.begin(), Nodes.end(), nodes);
        std::memcpy(values, Values.contents, Info.leaf_count * 4ull);
        std::memcpy(weights, OutputWeights.contents, Info.leaf_count * 4ull);
    }
    void Predictions(float* output) const {
        Require(!Failed, "Greedy session failed and must be closed");
        std::memcpy(output, DatasetPredictions.back().contents, Options.rows * 4ull);
    }
    void SetPermutations(uint32_t count, const uint8_t* const* bins, const float* const* predictions,
                         const float* lambdas, const uint8_t* valid) {
        Require(!Failed && !Info.completed_iterations && !PermutationsConfigured,
            "Greedy permutations can be configured once before training");
        Require(count >= 1 && count <= 64 && bins && bool(lambdas) == bool(valid),
            "Greedy permutation count must be in [1,64] with matching MVS arrays");
        const uint64_t cells = uint64_t(K.Rows) * K.Features;
        // All bank inputs are copied, including dataset zero; account for the
        // transient original bank as well as persistent sort scratch and nodes.
        const uint64_t bytes = uint64_t(count) * (cells + 4ull * K.Rows) +
            (count > 1 ? 16ull * K.Rows + 4096 + (2ull * MaxLeaves - 1) * sizeof(CBMGreedyNode) : 0);
        Require(WorkingBytes + ExactBytes + BacktrackingBytes + BootstrapBytes + NoiseBytes + bytes <= MemoryLimit,
            "Greedy permutation workspace exceeds 1 GiB");
        for (uint32_t p = 0; p < count; ++p) {
            Require(bins[p] && (!predictions || predictions[p]), "Greedy permutation inputs are required");
            Require(!lambdas || (lambdas[p] == 0 && valid[p] == 0), "Greedy permutations do not support MVS state");
            for (uint64_t i = 0; i < cells; ++i)
                Require(bins[p][i] < K.Bins, "Greedy permutation bin is outside the shared grid");
            if (predictions) for (uint32_t row = 0; row < K.Rows; ++row)
                Require(std::isfinite(predictions[p][row]), "Greedy permutation cursors must be finite");
        }
        std::vector<id<MTLBuffer>> dataBanks, cursors;
        for (uint32_t p = 0; p < count; ++p) {
            dataBanks.push_back(Context->Buffer(cells, bins[p]));
            cursors.push_back(Context->Buffer(4ull * K.Rows,
                predictions ? predictions[p] : DatasetPredictions[0].contents));
        }
        auto sort = count > 1 ? std::make_unique<CBMSortU32Workspace>(Context->Device, K.Rows) : nullptr;
        id<MTLBuffer> nodes = count > 1 ? Context->Buffer((2ull * MaxLeaves - 1) * sizeof(CBMGreedyNode)) : nil;
        DatasetBins = std::move(dataBanks); DatasetPredictions = std::move(cursors);
        PermutationSort = std::move(sort); PermutationNodes = nodes;
        PermutationBytes = bytes; PermutationsConfigured = true; SearchPermutation = 0;
        Data = DatasetBins.back(); Prediction = DatasetPredictions.back();
        Command command(*Context, Info.stats); EncodeLoss(command); command.Wait(); Info.loss = ReadLoss();
    }
    void SelectPermutation(uint32_t index) {
        Require(!Failed && index < DatasetBins.size(), "Greedy search permutation is out of range");
        SearchPermutation = index;
    }
    void CopyPermutationState(uint32_t capacity, float* predictions, float* lambdas, uint8_t* valid) const {
        Require(!Failed && capacity >= DatasetPredictions.size() && predictions && lambdas && valid,
            "Greedy permutation state output buffers are too small or missing");
        for (uint32_t p = 0; p < DatasetPredictions.size(); ++p) {
            std::memcpy(predictions + uint64_t(p) * K.Rows, DatasetPredictions[p].contents, 4ull * K.Rows);
            lambdas[p] = 0; valid[p] = 0;
        }
    }
    void SetBacktracking(uint32_t type) {
        Require(!Failed && !Info.completed_iterations && type <= 2,
            "Backtracking must be No/AnyImprovement/Armijo and configured before the first tree");
        if (type && Options.leaf_iterations > 1 && Options.leaf_method != 2 && !Directions) {
            const uint64_t bytes = MaxLeaves * 16ull + LossGroups * 8ull;
            Require(WorkingBytes + ExactBytes + BootstrapBytes + NoiseBytes + PermutationBytes + bytes <= MemoryLimit,
                "Greedy backtracking workspace exceeds 1 GiB");
            Directions = Context->Buffer(MaxLeaves * 4ull); TrialValues = Context->Buffer(MaxLeaves * 4ull);
            DirectionDot = Context->Buffer(MaxLeaves * 8ull); BacktrackingLoss = Context->Buffer(LossGroups * 8ull);
            BacktrackingBytes = bytes;
        }
        BacktrackingType = type;
    }
    void SetBootstrap(const CBMBootstrapOptions* options) {
        Require(options && !Failed && !Info.completed_iterations,
            "Bootstrap must be configured before the first tree");
        Require(options->bootstrap_type <= 3 && !options->reserved0 && !options->reserved1 &&
            options->mvs_reg_is_set <= 1 && options->initial_mvs_lambda_is_set <= 1,
            "Greedy bootstrap supports No, Bayesian, Bernoulli and Poisson; MVS is unsupported");
        Require(uint64_t(options->iteration_offset) + Options.iterations <= Missing,
            "Bootstrap absolute iteration count exceeds uint32");
        Require(std::isfinite(options->bagging_temperature) && options->bagging_temperature >= 0,
            "bagging_temperature must be finite and nonnegative");
        Require(std::isfinite(options->subsample) && options->subsample > 0 && options->subsample <= 1 &&
            (options->bootstrap_type != 3 || options->subsample < 1),
            "subsample must be in (0,1], and strictly below 1 for Poisson");
        Require((!options->mvs_reg_is_set || (std::isfinite(options->mvs_reg) && options->mvs_reg >= 0)) &&
            !options->initial_mvs_lambda_is_set, "MVS state is unsupported for greedy growth");
        if (options->bootstrap_type && !BootstrapMultipliers) {
            const uint64_t bytes = 8ull * K.Rows + 4ull * (MaxLeaves + 1);
            Require(WorkingBytes + ExactBytes + BacktrackingBytes + NoiseBytes + PermutationBytes + bytes <= MemoryLimit,
                "Greedy bootstrap workspace exceeds 1 GiB");
            BootstrapMultipliers = Context->Buffer(4ull * K.Rows);
            StructureWeightStorage = Context->Buffer(4ull * K.Rows);
            SampledOffsets = Context->Buffer(4ull * (MaxLeaves + 1));
            BootstrapBytes = bytes;
        }
        Sampling = *options;
    }
    void SetScoreNoise(const CBMScoreNoiseOptions* options) {
        Require(options && !Failed && !Info.completed_iterations && !options->reserved0 &&
            !options->reserved1 && !options->reserved2, "Score noise must be configured before the first tree");
        Require(std::isfinite(options->random_strength) && options->random_strength >= 0,
            "random_strength must be finite and nonnegative");
        if (options->random_strength > 0 && (K.ScoreFunction == 1 || K.ScoreFunction == 3) && !NoiseStatistics) {
            const uint64_t bytes = 16ull * LossGroups;
            Require(WorkingBytes + ExactBytes + BacktrackingBytes + BootstrapBytes + PermutationBytes + bytes <= MemoryLimit,
                "Greedy score noise workspace exceeds 1 GiB");
            NoiseStatistics = Context->Buffer(bytes); NoiseBytes = bytes;
        }
        RandomStrength = options->random_strength;
    }

private:
    Runtime* Context;
    KernelParams K;
    CBMGreedyParams G;
    uint32_t MaxLeaves, TotalBins, JobCapacity, PartitionTiles, LossGroups;
    double TotalWeight;
    uint64_t WorkingBytes = 0, BacktrackingBytes = 0, ExactBytes = 0, BootstrapBytes = 0, NoiseBytes = 0;
    uint64_t PermutationBytes = 0;
    uint32_t SearchPermutation = 0;
    bool PermutationsConfigured = false;
    std::vector<id<MTLBuffer>> DatasetBins, DatasetPredictions;
    id<MTLBuffer> PermutationNodes;
    std::unique_ptr<CBMSortU32Workspace> PermutationSort;
    uint32_t BacktrackingType = 0;
    float RandomStrength = 0;
    CBMBootstrapOptions Sampling = {0, 0, 0, 0, 1.f, 1.f, 0.f, 0, 0.f, 0, 0, 0};
    id<MTLBuffer> Data, Target, Weight, Prediction, Gradient, Hessian, LeafIds;
    id<MTLBuffer> Rows, NextRows, RowPrefix, TilePrefix, Offsets, NextOffsets, Depths, NextDepths;
    id<MTLBuffer> CandidateFeatures, CandidateBins, CandidateTypes, FeatureTypes, FeatureOffsets, FeatureWeights, FeatureNoise;
    id<MTLBuffer> HistSums, HistWeights, LeafSums, LeafWeights, RawValues, Values, OutputWeights, ObjectivePartials, LossPartials;
    id<MTLBuffer> WinnerPartials, Winners, Selected, RightIds, Frontier, Jobs, Active, WorkState, Arguments;
    std::unique_ptr<CBMPairwiseRuntime> Pairwise;
    CBMQueryOptions QueryOptions = {};
    id<MTLBuffer> QueryOffsets, QueryPoint, QueryStatistics, QueryLossPartials, QueryValidation;
    id<MTLBuffer> Directions, TrialValues, DirectionDot, BacktrackingLoss;
    id<MTLBuffer> ExactResiduals, ExactEffectiveWeights, ExactKeysA, ExactRowsA, ExactKeysB, ExactRowsB;
    id<MTLBuffer> ExactTileOffsets, ExactTotals, ExactSelected;
    id<MTLBuffer> BootstrapMultipliers, StructureWeightStorage, StructureWeight, SampledOffsets, NoiseStatistics;

    BootstrapParams SamplingParams() const {
        return {K.Rows, Sampling.bootstrap_type, Sampling.random_seed_low, Sampling.random_seed_high,
            Sampling.iteration_offset + Info.completed_iterations, 0, 0, 0,
            Sampling.bagging_temperature, Sampling.subsample, 0.f, 0.f};
    }
    void EncodeBootstrap(Command& command) {
        const id<MTLBuffer> denominator = (K.ScoreFunction == 2 || K.ScoreFunction == 3) ? Hessian : Weight;
        StructureWeight = denominator;
        if (Sampling.bootstrap_type) {
            const auto p = SamplingParams();
            StructureWeight = StructureWeightStorage;
            command.Dispatch("GenerateBootstrapWeights", {BootstrapMultipliers, Gradient}, p, K.Rows);
            command.Dispatch("ApplyBootstrapWeights", {Gradient, denominator, BootstrapMultipliers, StructureWeight}, p, K.Rows);
        }
    }
    float ReadScoreNoiseScale() const {
        const float* parts = static_cast<const float*>(NoiseStatistics.contents);
        double numerator = 0, denominator = 0;
        for (uint32_t group = 0; group < LossGroups; ++group) {
            numerator += double(parts[4 * group]) + parts[4 * group + 2];
            denominator += double(parts[4 * group + 1]) + parts[4 * group + 3];
        }
        Require(std::isfinite(numerator) && std::isfinite(denominator) && numerator >= 0 && denominator >= 0,
            "Invalid post-bootstrap greedy score variance");
        const double remaining = std::log(double(K.Rows)) -
            double(Sampling.iteration_offset + Info.completed_iterations) * Options.learning_rate;
        const double modelMultiplier = remaining >= 0 ? 1 / (1 + std::exp(-remaining)) :
            std::exp(remaining) / (1 + std::exp(remaining));
        const float scale = denominator > 0 ? static_cast<float>(std::sqrt(numerator / denominator) *
            modelMultiplier * RandomStrength) : 0.f;
        Require(std::isfinite(scale), "Greedy score noise scale exceeds float32");
        return scale;
    }

    void EncodeExactLeaves(Command& command) {
        const CBMExactLeafParams e = {K.Rows, K.Leaves, uint32_t(K.Objective == 11), K.HistogramTiles,
            K.Objective == 9 ? K.ObjectiveParam : .5f, 0, 0, 0};
        command.Dispatch("PrepareExactResiduals", {Target, Weight, Prediction, ExactResiduals,
            ExactEffectiveWeights, ExactKeysA, ExactRowsA}, e, K.Rows);
        CBMEncodeSortU32(command.Buffer, ExactKeysA, ExactRowsA, K.Rows, ExactKeysB, ExactRowsB, &Info.stats.kernel_dispatches);
        command.Dispatch("MakeExactLeafKeys", {ExactRowsB, LeafIds, ExactKeysA}, e, K.Rows);
        CBMEncodeSortU32(command.Buffer, ExactKeysA, ExactRowsB, K.Rows, ExactKeysB, ExactRowsA, &Info.stats.kernel_dispatches);
        command.Dispatch("BuildExactLeafOffsets", {ExactKeysB, Offsets}, e, uint64_t(K.Leaves) + 1);
        command.Dispatch("ReduceExactLeafPartials", {ExactRowsA, ExactEffectiveWeights, Weight,
            Offsets, ObjectivePartials}, e, K.HistogramTiles, true, K.Leaves);
        command.Dispatch("PrefixExactLeafTiles", {ObjectivePartials, ExactTileOffsets, ExactTotals,
            OutputWeights, ExactSelected}, e, K.Leaves, true);
        command.Dispatch("SelectExactLeafQuantile", {ExactRowsA, ExactEffectiveWeights, Offsets,
            ExactTileOffsets, ExactTotals, ExactSelected}, e, K.HistogramTiles, true, K.Leaves);
        command.Dispatch("FinalizeExactLeafValues", {ExactRowsA, ExactResiduals, Offsets,
            ExactTotals, ExactSelected, RawValues}, e, K.Leaves);
    }

    double ReadExpandedScalar(id<MTLBuffer> buffer, uint32_t count) const {
        const float* parts = static_cast<const float*>(buffer.contents);
        double value = 0;
        for (uint32_t i = 0; i < count; ++i) value += double(parts[2 * i]) + parts[2 * i + 1];
        return value;
    }
    QueryParams MakeQueryParams(bool apply, bool structure = false) const {
        return {K.Rows, QueryOptions.group_count, K.Objective, uint32_t(apply), QueryOptions.beta,
            QueryOptions.lambda, K.Leaves, uint32_t(structure && (K.ScoreFunction == 2 || K.ScoreFunction == 3))};
    }
    void EncodeQueryPoint(Command& command, id<MTLBuffer> values, bool apply, bool structure = false) {
        const auto q = MakeQueryParams(apply, structure);
        command.Dispatch("PrepareQuerywisePoint", {Prediction, values, LeafIds, QueryPoint}, q, K.Rows);
        command.Dispatch(K.Objective == 12 ? "QueryRmseDerivatives" : "QuerySoftMaxDerivatives",
            {Target, Weight, QueryPoint, QueryOffsets, Gradient, Hessian, QueryStatistics}, q, q.Groups, true);
    }
    void EncodeQueryLoss(Command& command) {
        command.Dispatch("ReduceQuerywiseObjective", {QueryStatistics, QueryLossPartials},
            MakeQueryParams(false), LossGroups, true);
    }
    void EncodeObjectivePartials(Command& command) {
        if (QueryOffsets || Pairwise) {
            // Normalize the entire query at the current point, then project its
            // original-row derivatives through this variable leaf partition.
            if (Pairwise) Pairwise->EncodePointDerivatives(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves,
                true, Gradient, Hessian, Weight, &Info.stats.kernel_dispatches);
            else EncodeQueryPoint(command, RawValues, true);
            const QueryProjectionParams q = {K.Rows, K.Leaves, K.HistogramTiles, K.LeafMethod};
            command.Dispatch("ReduceQuerywiseLeafPartials", {Gradient, Hessian, Weight, Rows, Offsets, ObjectivePartials},
                q, K.HistogramTiles, true, K.Leaves);
        } else command.Dispatch("ReduceLeafObjectivePartials", {Target, Weight, Prediction, RawValues, Rows,
            Offsets, ObjectivePartials}, K, K.HistogramTiles, true, K.Leaves);
    }
    void EncodeBacktrackingObjective(Command& command, id<MTLBuffer> values, const BacktrackingParams& b, bool trial = false) {
        if (Pairwise) {
            Pairwise->EncodePointDerivatives(command.Buffer, Prediction, values, LeafIds, K.Leaves,
                true, Gradient, Hessian, Weight, &Info.stats.kernel_dispatches, trial);
            Pairwise->EncodeLossReduction(command.Buffer, &Info.stats.kernel_dispatches);
        } else if (QueryOffsets) {
            EncodeQueryPoint(command, values, true);
            EncodeQueryLoss(command);
        } else command.Dispatch("ReduceBacktrackingObjective", {Target, Weight, Prediction, LeafIds, values, BacktrackingLoss},
            K, LossGroups, true, 1, &b, sizeof(b));
    }
    double ReadBacktrackingObjective() const {
        if (Pairwise) return -Pairwise->ReadLossPartials(true)[0];
        if (!QueryOffsets) return ReadExpandedScalar(BacktrackingLoss, LossGroups);
        const auto* parts = static_cast<const float*>(QueryLossPartials.contents);
        double value = 0;
        for (uint32_t i = 0; i < LossGroups; ++i) value -= parts[2 * i];
        return value;
    }
    void EstimateBacktrackingLeaves() {
        BacktrackingParams b = {1.f, BacktrackingType, 0, 0};
        Command initial(*Context, Info.stats);
        EncodeBacktrackingObjective(initial, RawValues, b);
        initial.Wait();
        double currentValue = ReadBacktrackingObjective(), directionDot = 0;
        Require(std::isfinite(currentValue), "Nonfinite GPU leaf objective before backtracking");
        bool updated = false, newDirection = true;
        // CUDA's walker counts rejected trials too and allows at most 100
        // attempts until the first accepted step, even with a smaller budget.
        for (uint32_t attempt = 0; attempt < Options.leaf_iterations || (!updated && attempt < 100); ++attempt) {
            Command trial(*Context, Info.stats);
            if (newDirection) {
                EncodeObjectivePartials(trial);
                trial.Dispatch("PrepareBacktrackingDirection", {ObjectivePartials, RawValues, Directions,
                    OutputWeights, DirectionDot}, K, K.Leaves, true, 1, &b, sizeof(b));
            }
            trial.Dispatch("BuildBacktrackingCandidate", {RawValues, Directions, OutputWeights, TrialValues},
                K, K.Leaves, false, 1, &b, sizeof(b));
            EncodeBacktrackingObjective(trial, TrialValues, b, true);
            trial.Wait();
            if (newDirection) {
                directionDot = ReadExpandedScalar(DirectionDot, K.Leaves);
                Require(std::isfinite(directionDot), "Nonfinite GPU leaf direction during backtracking");
            }
            const double trialValue = ReadBacktrackingObjective();
            const double threshold = currentValue + (b.Type == 2 ? 1e-5 * b.Step * directionDot : 0.);
            if (std::isfinite(trialValue) && trialValue >= threshold) {
                std::swap(RawValues, TrialValues); currentValue = trialValue;
                updated = true; newDirection = true; b.Step = 1.f;
            } else { b.Step *= .5f; newDirection = false; }
        }
    }

    void EncodeLoss(Command& command) {
        if (Pairwise) {
            Pairwise->EncodePointDerivatives(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves,
                false, Gradient, Hessian, Weight, &Info.stats.kernel_dispatches);
            Pairwise->EncodeLossReduction(command.Buffer, &Info.stats.kernel_dispatches);
            return;
        }
        if (QueryOffsets) { EncodeQueryPoint(command, RawValues, false); EncodeQueryLoss(command); return; }
        command.Dispatch("ReduceObjectiveLoss", {Target, Weight, Prediction, LossPartials}, K, LossGroups, true);
    }
    float ReadLoss() const {
        if (Pairwise) {
            const auto parts = Pairwise->ReadLossPartials();
            const float value = static_cast<float>(parts[0] / parts[1]);
            Require(std::isfinite(value), "Nonfinite greedy PairLogit loss");
            return value;
        }
        if (QueryOffsets) {
            const auto* parts = static_cast<const float*>(QueryLossPartials.contents);
            double numerator = 0, denominator = 0;
            for (uint32_t i = 0; i < LossGroups; ++i) {
                Require(std::isfinite(parts[2 * i]) && std::isfinite(parts[2 * i + 1]) && parts[2 * i + 1] >= 0,
                    "Nonfinite GPU greedy query objective");
                numerator += parts[2 * i]; denominator += parts[2 * i + 1];
            }
            Require(denominator > 0 && std::isfinite(numerator), "Query objective requires positive metric mass and finite loss");
            const double mean = numerator / denominator;
            const float value = static_cast<float>(K.Objective == 12 ? std::sqrt(mean) : mean);
            Require(std::isfinite(value), "Nonfinite GPU greedy query loss");
            return value;
        }
        const auto* partials = static_cast<const float*>(LossPartials.contents);
        double sum = 0;
        for (uint32_t i = 0; i < LossGroups; ++i) {
            Require(std::isfinite(partials[i]) && (Options.objective == 3 || partials[i] >= 0),
                "Invalid GPU objective loss");
            sum += partials[i];
        }
        const float loss = static_cast<float>(Options.objective == 0 ? std::sqrt(sum) : sum);
        Require(std::isfinite(loss), "Nonfinite GPU objective loss");
        return loss;
    }
    void EncodeHistograms(Command& command, id<MTLBuffer> scoreWeights) {
        if (!TotalBins) return;
        // Full compact builds have no paired-child assumptions. Only the
        // reuse=1 path requires symmetric halves; greedy always uses reuse=0.
        const CompactParams p = {K.Rows, K.Features, K.Leaves, TotalBins, 0, 8192, JobCapacity, 0};
        command.Dispatch("ResetCompactHistogramWorkState", {WorkState}, p, 1);
        command.Dispatch("BuildCompactHistogramJobs", {Offsets, Jobs, Active, WorkState}, p, K.Leaves);
        command.Dispatch("BuildCompactHistogramDispatchArguments", {WorkState, Arguments}, p, 1);
        command.Dispatch("ClearCompactHistograms", {HistSums, HistWeights}, p, uint64_t(K.Leaves) * TotalBins);
        command.Indirect("ComputeCompactHistograms", {Data, Gradient, scoreWeights, Rows, FeatureOffsets,
            Jobs, WorkState, HistSums, HistWeights}, p, Arguments, 0);
        command.Indirect("ScanCompactHistograms", {HistSums, HistWeights, FeatureTypes, FeatureOffsets,
            Active, WorkState}, p, Arguments, 12);
    }
    void StepImpl() {
        Data = DatasetBins[SearchPermutation]; Prediction = DatasetPredictions[SearchPermutation];
        K.Leaves = G.Leaves = 1;
        *static_cast<uint32_t*>(Depths.contents) = 0;
        Nodes = {{0, 0, 0, 0, 0, 0}};
        std::vector<uint32_t> leafNodes = {0};
        Command initial(*Context, Info.stats);
        if (Pairwise) {
            Pairwise->ClearStatus();
            initial.Dispatch("ResetQuerywiseLeafIds", {LeafIds}, MakeQueryParams(false), K.Rows);
            Pairwise->EncodePointDerivatives(initial.Buffer, Prediction, RawValues, LeafIds, K.Leaves,
                false, Gradient, Hessian, Weight, &Info.stats.kernel_dispatches);
        } else if (QueryOffsets) {
            const auto q = MakeQueryParams(false, true);
            initial.Dispatch("ResetQuerywiseLeafIds", {LeafIds}, q, K.Rows);
            EncodeQueryPoint(initial, RawValues, false, true);
            if (q.Reserved) {
                *static_cast<uint32_t*>(QueryValidation.contents) = 0;
                initial.Dispatch("ValidateQuerywiseStructureCurvature", {Hessian, QueryValidation}, q, K.Rows);
            }
        } else initial.Dispatch("ObjectiveDerivatives", {Target, Weight, Prediction, Gradient, Hessian, LeafIds}, K, K.Rows);
        initial.Dispatch("InitializeRootPartition", {Rows, Offsets}, K, K.Rows);
        EncodeBootstrap(initial);
        const bool noisy = RandomStrength > 0 && (K.ScoreFunction == 1 || K.ScoreFunction == 3);
        if (noisy) {
            const CBMGreedyBootstrapParams p = {K.Rows, K.Leaves, Sampling.bootstrap_type, 0};
            initial.Dispatch("ReduceGreedyScoreNoiseStatistics", {Gradient, StructureWeight, NoiseStatistics},
                p, LossGroups, true);
        }
        initial.Wait();
        if (Pairwise) Pairwise->CheckStatus();
        if (QueryOffsets && (K.ScoreFunction == 2 || K.ScoreFunction == 3))
            Require(*static_cast<const uint32_t*>(QueryValidation.contents) == 0,
                "Invalid GPU query Newton split score: row curvatures must be finite and nonnegative");
        const float noiseScale = noisy ? ReadScoreNoiseScale() : 0.f;
        uint32_t growthRound = 0;
        while (K.Leaves < MaxLeaves && Options.depth && Options.candidates) {
            id<MTLBuffer> scoreWeights = StructureWeight;
            Command search(*Context, Info.stats);
            if (noisy) {
                auto p = SamplingParams(); p.Rows = K.Features; p.Stream = ++growthRound; p.NoiseScale = noiseScale;
                search.Dispatch("GenerateScoreFeatureNoise", {FeatureNoise}, p, K.Features);
            }
            search.Dispatch("ReduceStructurePartials", {Gradient, scoreWeights, Rows, Offsets, ObjectivePartials},
                K, K.HistogramTiles, true, K.Leaves);
            search.Dispatch("CollectPartitionStatistics", {ObjectivePartials, LeafSums, LeafWeights}, K, K.Leaves, true);
            EncodeHistograms(search, scoreWeights);
            search.Dispatch("FindGreedySplitWinners", {HistSums, HistWeights, LeafSums, LeafWeights,
                CandidateFeatures, CandidateBins, CandidateTypes, FeatureOffsets, FeatureWeights,
                FeatureNoise, WinnerPartials}, G, G.ScoreGroups, true, G.Leaves);
            search.Dispatch("ReduceGreedySplitWinners", {WinnerPartials, Winners}, G, G.Leaves, true);
            id<MTLBuffer> terminalOffsets = Offsets;
            if (Sampling.bootstrap_type >= 2) {
                const CBMGreedyBootstrapParams p = {K.Rows, K.Leaves, Sampling.bootstrap_type, 0};
                search.Dispatch("CountGreedyBootstrapRows", {BootstrapMultipliers, Rows, Offsets, SampledOffsets},
                    p, K.Leaves, true);
                search.Dispatch("PrefixGreedyBootstrapOffsets", {SampledOffsets}, p, 1, true);
                terminalOffsets = SampledOffsets;
            }
            search.Dispatch("SelectGreedyLeaves", {Winners, terminalOffsets, Depths, Selected, RightIds, Frontier}, G, 1);
            search.Wait();
            Require(!TotalBins || static_cast<const uint32_t*>(WorkState.contents)[2] == 0,
                "GPU greedy histogram work capacity exceeded");
            const auto frontier = *static_cast<const CBMGreedyFrontier*>(Frontier.contents);
            Require(!frontier.Error, "Nonfinite GPU greedy split score");
            if (!frontier.Selected) break;
            const auto* selected = static_cast<const uint32_t*>(Selected.contents);
            const auto* rightIds = static_cast<const uint32_t*>(RightIds.contents);
            const auto* winners = static_cast<const CBMGreedySplit*>(Winners.contents);
            leafNodes.resize(frontier.NewLeaves);
            for (uint32_t i = 0; i < frontier.Selected; ++i) {
                const uint32_t parent = selected[i], rightLeaf = rightIds[parent];
                const auto split = winners[parent];
                Require(parent < K.Leaves && rightLeaf < frontier.NewLeaves && split.Valid,
                    "Invalid GPU greedy frontier metadata");
                const uint32_t parentNode = leafNodes[parent];
                const uint32_t leftNode = static_cast<uint32_t>(Nodes.size()), rightNode = leftNode + 1;
                Nodes[parentNode] = {split.Feature, split.Bin, split.Type, leftNode, rightNode, Missing};
                Nodes.push_back({0, 0, 0, 0, 0, parent});
                Nodes.push_back({0, 0, 0, 0, 0, rightLeaf});
                leafNodes[parent] = leftNode;
                leafNodes[rightLeaf] = rightNode;
            }
            Command partition(*Context, Info.stats);
            partition.Dispatch("RouteGreedySplitRows", {Data, LeafIds, Winners, RightIds}, G, G.Rows);
            partition.Dispatch("UpdateGreedyLeafDepths", {Depths, RightIds, NextDepths}, G, G.Leaves);
            partition.Dispatch("CountGreedyPartitionBits", {Rows, LeafIds, RowPrefix, TilePrefix}, G, PartitionTiles, true);
            partition.Dispatch("ScanGreedyPartitionTiles", {TilePrefix}, G, 1, true);
            partition.Dispatch("BuildGreedyPartitionOffsets", {Offsets, RightIds, RowPrefix, TilePrefix,
                Frontier, NextOffsets}, G, G.Leaves);
            partition.Dispatch("ScatterGreedyPartitionRows", {Rows, LeafIds, RowPrefix, TilePrefix, NextRows}, G, G.Rows);
            partition.Wait();
            std::swap(Rows, NextRows); std::swap(Offsets, NextOffsets); std::swap(Depths, NextDepths);
            K.Leaves = G.Leaves = frontier.NewLeaves;
        }
        EstimateLeaves();
        const uint32_t last = DatasetBins.size() - 1;
        if (last) {
            const float searchLoss = Info.loss;
            // Search histograms are dead now. Preserve the exported estimate
            // there if callers selected the final dataset for structure search.
            if (SearchPermutation == last) {
                std::memcpy(LeafSums.contents, Values.contents, 4ull * K.Leaves);
                std::memcpy(LeafWeights.contents, OutputWeights.contents, 4ull * K.Leaves);
            }
            std::memcpy(PermutationNodes.contents, Nodes.data(), Nodes.size() * sizeof(CBMGreedyNode));
            for (uint32_t p = 0; p <= last; ++p) {
                if (p == SearchPermutation) continue;
                Data = DatasetBins[p]; Prediction = DatasetPredictions[p];
                Command route(*Context, Info.stats);
                const EvaluationParams e = {K.Rows, K.Features, G.MaxDepth, 0};
                route.Dispatch("RouteGreedyFixedTree", {Data, PermutationNodes, LeafIds}, e, K.Rows);
                route.Dispatch("InitializeRootPartition", {Rows, Offsets}, K, K.Rows);
                // A stable leaf-ID sort recreates original row order inside
                // each leaf without repeating the structure search or sampling.
                PermutationSort->Encode(route.Buffer, LeafIds, Rows, K.Rows, RowPrefix, NextRows,
                    &Info.stats.kernel_dispatches, 16);
                const CBMExactLeafParams offsets = {K.Rows, K.Leaves, 0, K.HistogramTiles, 0, 0, 0, 0};
                route.Dispatch("BuildExactLeafOffsets", {RowPrefix, Offsets}, offsets, uint64_t(K.Leaves) + 1);
                route.Wait(); std::swap(Rows, NextRows);
                EstimateLeaves();
            }
            if (SearchPermutation == last) {
                std::memcpy(Values.contents, LeafSums.contents, 4ull * K.Leaves);
                std::memcpy(OutputWeights.contents, LeafWeights.contents, 4ull * K.Leaves);
                Info.loss = searchLoss;
            }
            Data = DatasetBins[last]; Prediction = DatasetPredictions[last];
        }
        Info.node_count = static_cast<uint32_t>(Nodes.size());
        Info.leaf_count = K.Leaves;
        ++Info.completed_iterations;
        Info.finished = Info.completed_iterations == Options.iterations;
    }
    void EstimateLeaves() {
        Command estimate(*Context, Info.stats);
        estimate.Dispatch("InitializeLeafValues", {RawValues, OutputWeights}, K, K.Leaves);
        if (Options.leaf_method == 2) {
            EncodeExactLeaves(estimate);
        } else if (BacktrackingType && Options.leaf_iterations > 1) {
            estimate.Wait();
            EstimateBacktrackingLeaves();
            estimate.Reset();
        } else for (uint32_t iteration = 0; iteration < Options.leaf_iterations; ++iteration) {
            K.LeafIteration = iteration;
            EncodeObjectivePartials(estimate);
            estimate.Dispatch("EstimateNewtonLeafValues", {ObjectivePartials, RawValues, OutputWeights}, K, K.Leaves, true);
        }
        if (Pairwise) Pairwise->EncodeCenterLeafValues(estimate.Buffer, RawValues, K.Leaves, &Info.stats.kernel_dispatches);
        estimate.Dispatch("FinalizeLeafValues", {RawValues, Values}, K, K.Leaves);
        estimate.Dispatch("AddObjectiveBinModelValue", {LeafIds, Values, Prediction}, K, K.Rows);
        EncodeLoss(estimate);
        estimate.Wait();
        const auto* values = static_cast<const float*>(Values.contents);
        const auto* leafWeights = static_cast<const float*>(OutputWeights.contents);
        double estimatedWeight = 0;
        for (uint32_t leaf = 0; leaf < K.Leaves; ++leaf) {
            Require(std::isfinite(values[leaf]) && std::isfinite(leafWeights[leaf]) && leafWeights[leaf] >= 0,
                "Invalid GPU greedy leaf estimate");
            estimatedWeight += leafWeights[leaf];
        }
        Require(std::abs(estimatedWeight - TotalWeight) <= std::max(1e-6, TotalWeight * 2e-5),
            "GPU greedy leaf weights do not cover the training observations");
        const auto* predictions = static_cast<const float*>(Prediction.contents);
        for (uint32_t row = 0; row < K.Rows; ++row)
            Require(std::isfinite(predictions[row]), "Nonfinite greedy model prediction");
        Info.loss = ReadLoss();
    }
};

class Evaluation {
public:
    std::mutex Mutex;
    CBMGreedyEvaluationStats Stats = {};
    Evaluation(uint32_t rows, uint32_t features, uint32_t maxDepth, const uint8_t* bins,
               uint64_t binsCount, float bias, const float* initial, uint64_t initialCount, uint32_t dimensions = 1)
        : Params{rows, features, maxDepth, dimensions == 1 ? 0 : dimensions}, Dimensions(dimensions) {
        Require(rows <= (1u << 27) && features && features <= 65536 && maxDepth <= 65535 &&
            dimensions >= 1 && dimensions <= 64 && uint64_t(rows) * dimensions <= UINT32_MAX,
            "Invalid greedy evaluation dimensions");
        const uint64_t cells = uint64_t(rows) * features;
        const uint64_t predictionsCount = uint64_t(rows) * dimensions;
        Require(binsCount == cells && (!cells || bins), "Greedy evaluation bins count does not match dimensions");
        Require(std::isfinite(bias) && ((!initial && !initialCount) || (initial && initialCount == predictionsCount)),
            "Greedy evaluation initial prediction count or bias is invalid");
        BaseBytes = std::max<uint64_t>(1, cells) + std::max<uint64_t>(1, predictionsCount * 4);
        Require(BaseBytes + 28 <= MemoryLimit, "Greedy evaluation exceeds 1 GiB resident limit");
        if (initial) for (uint64_t row = 0; row < predictionsCount; ++row)
            Require(std::isfinite(initial[row]), "Greedy evaluation initial predictions must be finite");
        Context = &GetRuntime();
        Bins = Context->Buffer(cells, bins);
        Predictions = Context->Buffer(predictionsCount * 4, initial);
        Stats.dataset_uploads = 1; Stats.bins_upload_bytes = cells; Stats.resident_bytes = BaseBytes;
        CopyText(Stats.compute.device_name, sizeof(Stats.compute.device_name), [Context->Device.name UTF8String]);
        if (rows && !initial) {
            KernelParams p = {}; p.Rows = predictionsCount; p.Bias = bias;
            Command command(*Context, Stats.compute);
            command.Dispatch("InitializeObjectivePredictions", {Predictions}, p, predictionsCount);
            command.Wait();
        }
    }
    uint32_t GetDimensions() const noexcept { return Dimensions; }
    void Add(const CBMGreedyNode* nodes, uint64_t nodeCount, const float* values, uint64_t leafCount) {
        Require(!Failed, "Greedy evaluation cursor failed and must be closed");
        Require(nodes && values && leafCount && leafCount <= 65536 && nodeCount == 2 * leafCount - 1,
            "Greedy evaluation needs a full binary tree with matching node/leaf counts");
        const uint64_t treeBytes = nodeCount * sizeof(CBMGreedyNode) + leafCount * Dimensions * 4;
        // Replacing a tree briefly retains both uploads. Bound that peak, and
        // publish the new buffers together after both allocations succeed.
        Require(BaseBytes + Nodes.length + Values.length + treeBytes <= MemoryLimit,
            "Greedy evaluation tree upload exceeds 1 GiB resident limit");
        for (uint64_t leaf = 0; leaf < leafCount * Dimensions; ++leaf)
            Require(std::isfinite(values[leaf]), "Greedy evaluation leaf values must be finite");
        std::vector<uint8_t> visited(nodeCount, 0), leaves(leafCount, 0);
        std::vector<std::pair<uint32_t, uint32_t>> pending = {{0, 0}};
        uint64_t visitedCount = 0;
        while (!pending.empty()) {
            const auto [index, depth] = pending.back(); pending.pop_back();
            Require(index < nodeCount && !visited[index] && depth <= Params.MaxDepth,
                "Greedy evaluation graph has an invalid child, cycle, shared node or excessive depth");
            visited[index] = 1; ++visitedCount;
            const auto& node = nodes[index];
            if (node.leaf != Missing) {
                Require(node.leaf < leafCount && !leaves[node.leaf], "Greedy evaluation leaf references are invalid");
                leaves[node.leaf] = 1;
            } else {
                Require(node.feature < Params.Features && node.bin <= 255 && node.type <= 1,
                    "Greedy evaluation split is out of range");
                pending.emplace_back(node.left, depth + 1); pending.emplace_back(node.right, depth + 1);
            }
        }
        Require(visitedCount == nodeCount && std::all_of(leaves.begin(), leaves.end(), [](uint8_t value) { return value; }),
            "Greedy evaluation graph has unreachable nodes or missing leaves");
        id<MTLBuffer> nextNodes = Context->Buffer(nodeCount * sizeof(CBMGreedyNode), nodes);
        id<MTLBuffer> nextValues = Context->Buffer(leafCount * Dimensions * 4, values);
        Nodes = nextNodes;
        Values = nextValues;
        Stats.tree_upload_bytes += treeBytes;
        Stats.resident_bytes = BaseBytes + treeBytes;
        try {
            if (Params.Rows) {
                Command command(*Context, Stats.compute);
                command.Dispatch("AddGreedyEvaluationTree", {Bins, Nodes, Values, Predictions}, Params, Params.Rows);
                command.Wait();
                const auto* predictions = static_cast<const float*>(Predictions.contents);
                for (uint64_t row = 0; row < uint64_t(Params.Rows) * Dimensions; ++row)
                    Require(std::isfinite(predictions[row]), "Greedy evaluation prediction overflowed float32");
            }
        } catch (...) { Failed = true; throw; }
    }
    void Copy(float* output, uint64_t count) const {
        Require(!Failed, "Greedy evaluation cursor failed and must be closed");
        Require(count == uint64_t(Params.Rows) * Dimensions && (!count || output), "Greedy evaluation prediction output count is invalid");
        if (count) std::memcpy(output, Predictions.contents, count * 4);
    }
    void ValidateOutput(float* output, uint64_t count) const {
        Require(count == uint64_t(Params.Rows) * Dimensions && (!count || output), "Greedy evaluation prediction output count is invalid");
    }
private:
    Runtime* Context;
    EvaluationParams Params;
    uint32_t Dimensions;
    uint64_t BaseBytes;
    bool Failed = false;
    id<MTLBuffer> Bins, Nodes, Values, Predictions;
};

std::mutex RegistryMutex;
std::unordered_map<uintptr_t, std::shared_ptr<Session>> Sessions;
std::unordered_map<uintptr_t, std::shared_ptr<Evaluation>> Evaluations;
std::atomic<uintptr_t> NextHandle{1};
std::shared_ptr<Session> GetSession(void* handle) {
    std::lock_guard<std::mutex> guard(RegistryMutex);
    const auto found = Sessions.find(reinterpret_cast<uintptr_t>(handle));
    Require(found != Sessions.end(), "Greedy session handle is invalid or closed");
    return found->second;
}
std::shared_ptr<Evaluation> GetEvaluation(void* handle) {
    std::lock_guard<std::mutex> guard(RegistryMutex);
    const auto found = Evaluations.find(reinterpret_cast<uintptr_t>(handle));
    Require(found != Evaluations.end(), "Greedy evaluation handle is invalid or closed");
    return found->second;
}
template <class Callback>
int ApiCall(char* error, size_t capacity, Callback callback) {
    CopyText(error, capacity, "");
    @autoreleasepool {
        @try {
            try { callback(); return 0; }
            catch (const std::exception& exception) { CopyText(error, capacity, exception.what()); }
            catch (...) { CopyText(error, capacity, "Unexpected greedy Metal training failure"); }
        } @catch (NSException* exception) { CopyText(error, capacity, [[exception reason] UTF8String]); }
    }
    return 1;
}
}

extern "C" int cbm_greedy_session_create(const CBMGreedyTrainParams* params,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial,
    const uint32_t* features, const uint32_t* candidateBins, const uint8_t* types,
    void** handle, char* error, size_t capacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, capacity, [&] {
        Require(handle != nullptr, "Greedy session output handle is required");
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial, features, candidateBins, types);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Greedy session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_greedy_session_create_configured(const CBMGreedyTrainParams* params,
    const CBMObjectiveOptions* objectiveOptions, const uint8_t* bins, const float* targets,
    const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* candidateBins, const uint8_t* types, void** handle, char* error, size_t capacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, capacity, [&] {
        Require(handle && objectiveOptions, "Greedy session handle and objective configuration are required");
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial,
            features, candidateBins, types, objectiveOptions);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Greedy session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_greedy_session_create_query(const CBMGreedyTrainParams* params,
    const CBMObjectiveOptions* objectiveOptions, const CBMQueryOptions* queryOptions,
    const uint32_t* groupOffsets, uint64_t groupOffsetsCount,
    const uint8_t* bins, const float* targets, const float* weights, const float* initial,
    const uint32_t* features, const uint32_t* candidateBins, const uint8_t* types,
    void** handle, char* error, size_t capacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, capacity, [&] {
        Require(handle && params && objectiveOptions && queryOptions && groupOffsets,
            "Greedy query session requires a handle, parameters and group offsets");
        Require(groupOffsetsCount == uint64_t(queryOptions->group_count) + 1,
            "Greedy query offsets count must match group_count + 1");
        auto session = std::make_shared<Session>(params, bins, targets, weights, initial,
            features, candidateBins, types, objectiveOptions, queryOptions, groupOffsets);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Greedy session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_greedy_session_create_pair(const CBMGreedyTrainParams* params,
    const CBMObjectiveOptions* objectiveOptions, const CBMPairOptions* pairOptions,
    const uint32_t* winners, const uint32_t* losers, const float* pairWeights, uint64_t pairCount,
    const uint32_t* groupOffsets, uint64_t groupOffsetsCount,
    const uint8_t* bins, const float* targets, const float* initial,
    const uint32_t* features, const uint32_t* candidateBins, const uint8_t* types,
    void** handle, char* error, size_t capacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, capacity, [&] {
        Require(handle && params && objectiveOptions && pairOptions, "Greedy pair session parameters and output are required");
        Require(params->objective == 14 && objectiveOptions->objective == 14, "Greedy supplied-pair constructor requires PairLogit");
        Require(pairCount == pairOptions->pair_count && groupOffsetsCount ==
            (pairOptions->group_count ? uint64_t(pairOptions->group_count) + 1 : 0),
            "Greedy pair and group offset counts must match their declared geometry");
        auto session = std::make_shared<Session>(params, bins, targets, nullptr, initial, features, candidateBins,
            types, objectiveOptions, nullptr, groupOffsets, pairOptions, winners, losers, pairWeights);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Greedy session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_greedy_session_step(void* handle, CBMGreedyStepInfo* info,
    CBMGreedyNode* nodes, float* values, float* weights, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        Require(info && nodes && values && weights, "Greedy step output buffers are required");
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->Step();
        session->Copy(info, nodes, values, weights);
    });
}
extern "C" int cbm_greedy_session_copy_predictions(void* handle, float* predictions, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        Require(predictions != nullptr, "Greedy prediction output is required");
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->Predictions(predictions);
    });
}
extern "C" int cbm_greedy_session_info(void* handle, CBMGreedyStepInfo* info, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        Require(info != nullptr, "Greedy session info output is required");
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        *info = session->Info;
    });
}
extern "C" void cbm_greedy_session_close(void* handle) {
    std::lock_guard<std::mutex> guard(RegistryMutex);
    Sessions.erase(reinterpret_cast<uintptr_t>(handle));
}
extern "C" int cbm_greedy_session_set_backtracking(void* handle, uint32_t type, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetBacktracking(type);
    });
}
extern "C" int cbm_greedy_session_set_bootstrap(void* handle, const CBMBootstrapOptions* options,
    char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetBootstrap(options);
    });
}
extern "C" int cbm_greedy_session_set_score_noise(void* handle, const CBMScoreNoiseOptions* options,
    char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetScoreNoise(options);
    });
}

extern "C" int cbm_greedy_session_set_permutations(void* handle, uint32_t count,
    const uint8_t* const* bins, const float* const* predictions, const float* lambdas, const uint8_t* valid,
    char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetPermutations(count, bins, predictions, lambdas, valid);
    });
}
extern "C" int cbm_greedy_session_select_permutation(void* handle, uint32_t index, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->SelectPermutation(index);
    });
}
extern "C" int cbm_greedy_session_copy_permutation_state(void* handle, uint32_t count,
    float* predictions, float* lambdas, uint8_t* valid, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->CopyPermutationState(count, predictions, lambdas, valid);
    });
}

extern "C" int cbm_greedy_evaluation_create(uint32_t rows, uint32_t features, uint32_t maxDepth,
    const uint8_t* bins, uint64_t binsCount, float bias, const float* initial, uint64_t initialCount,
    void** handle, char* error, size_t capacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, capacity, [&] {
        Require(handle != nullptr, "Greedy evaluation output handle is required");
        auto evaluation = std::make_shared<Evaluation>(rows, features, maxDepth, bins, binsCount, bias, initial, initialCount);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Greedy session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Evaluations.emplace(id, std::move(evaluation));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_greedy_evaluation_create_vector(uint32_t rows, uint32_t features, uint32_t maxDepth,
    uint32_t dimensions, const uint8_t* bins, uint64_t binsCount, const float* initial, uint64_t initialCount,
    void** handle, char* error, size_t capacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, capacity, [&] {
        Require(handle != nullptr, "Greedy vector evaluation output handle is required");
        auto evaluation = std::make_shared<Evaluation>(rows, features, maxDepth, bins, binsCount, 0.f,
            initial, initialCount, dimensions);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Greedy session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Evaluations.emplace(id, std::move(evaluation));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_greedy_evaluation_add_vector_tree(void* handle, const CBMGreedyNode* nodes, uint64_t nodeCount,
    const float* values, uint64_t valueCount, float* predictions, uint64_t predictionCount, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto evaluation = GetEvaluation(handle);
        std::lock_guard<std::mutex> guard(evaluation->Mutex);
        evaluation->ValidateOutput(predictions, predictionCount);
        const uint32_t dimensions = evaluation->GetDimensions();
        Require(valueCount % dimensions == 0 && valueCount / dimensions <= 65536,
            "Greedy vector leaf value count does not match dimensions");
        evaluation->Add(nodes, nodeCount, values, valueCount / dimensions);
        evaluation->Copy(predictions, predictionCount);
    });
}
extern "C" int cbm_greedy_evaluation_add_tree(void* handle, const CBMGreedyNode* nodes, uint64_t nodeCount,
    const float* values, uint64_t leafCount, float* predictions, uint64_t predictionCount, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto evaluation = GetEvaluation(handle);
        std::lock_guard<std::mutex> guard(evaluation->Mutex);
        evaluation->ValidateOutput(predictions, predictionCount);
        Require(evaluation->GetDimensions() == 1, "Use add_vector_tree for vector greedy evaluation");
        evaluation->Add(nodes, nodeCount, values, leafCount);
        evaluation->Copy(predictions, predictionCount);
    });
}
extern "C" int cbm_greedy_evaluation_predictions(void* handle, float* predictions, uint64_t count,
    char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto evaluation = GetEvaluation(handle);
        std::lock_guard<std::mutex> guard(evaluation->Mutex);
        evaluation->Copy(predictions, count);
    });
}
extern "C" int cbm_greedy_evaluation_stats(void* handle, CBMGreedyEvaluationStats* stats,
    char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        Require(stats != nullptr, "Greedy evaluation statistics output is required");
        auto evaluation = GetEvaluation(handle);
        std::lock_guard<std::mutex> guard(evaluation->Mutex);
        *stats = evaluation->Stats;
    });
}
extern "C" void cbm_greedy_evaluation_close(void* handle) {
    std::lock_guard<std::mutex> guard(RegistryMutex);
    Evaluations.erase(reinterpret_cast<uintptr_t>(handle));
}
