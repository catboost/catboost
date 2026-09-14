#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_trainer.h"
#include "metal_exception.h"
#include "metal_kernels.h"
#include "metal_objective_kernels.h"
#include "metal_histogram_kernels.h"
#include "metal_incremental_partition_kernels.h"
#include "metal_bootstrap_kernels.h"
#include "metal_histogram_reuse_kernels.h"
#include "metal_score_noise_kernels.h"
#include "metal_backtracking_kernels.h"
#include "metal_regularization_kernels.h"
#include "metal_langevin_kernels.h"
#include "metal_langevin_leaf_kernels.h"
#include "metal_additional_objective_kernels.h"
#include "metal_deep_partition_kernels.h"
#include "metal_compact_histogram_kernels.h"
#include "metal_exact_leaf_kernels.h"
#include "metal_sort.h"
#include "metal_streaming_score_kernels.h"
#include "metal_dynamic_score_kernels.h"
#include "metal_custom_objective.h"
#include "metal_combination_runtime.h"
#include "metal_querywise_kernels.h"
#include "metal_pairwise_runtime.h"
#include "metal_yeti_rank_runtime.h"
#include "metal_pairwise_matrix_runtime.h"
#include "metal_pfound_pair_runtime.h"
#include "metal_query_cross_entropy_runtime.h"

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
constexpr uint64_t MaxWorkingBytes = uint64_t(1) << 30;
constexpr uint64_t MaxOutputBytes = uint64_t(1) << 29;
constexpr uint32_t MaxRows = uint32_t(1) << 24;
using KernelParams = CBMMetalKernelParams;
static_assert(CBMMetalKernelAbiVersion == 2, "Review shared Metal kernel bindings and winner semantics");
struct NativeBootstrapParams {
    uint32_t Rows, Type, SeedLow, SeedHigh;
    uint32_t Iteration, Stream, Reserved0, Reserved1;
    float Temperature, Subsample, MVSLambda, NoiseScale;
};
struct NativeScoreTileParams { uint32_t Begin, End, Bins, Reserved[5]; };
struct NativeCompactParams { uint32_t Rows, Features, Leaves, TotalBins, FeatureBegin, TileRows, JobCapacity, Reuse; };
struct NativeScoreRegularizationParams { uint32_t Normalize; float MetaExponent; uint32_t PerFeature, Reserved; };
struct NativeBacktrackingParams { float Step; uint32_t Type, AddRidge, Normalize; };
struct NativeQueryParams {
    uint32_t Rows, Groups, Objective, ApplyLeafValues;
    float Beta, Lambda;
    uint32_t Leaves, Reserved;
};
struct NativeQueryProjectionParams { uint32_t Rows, Leaves, Tiles, LeafMethod; };
using SplitState = CBMMetalSplitState;
static_assert(sizeof(KernelParams) == 96, "Metal parameter ABI mismatch");
static_assert(sizeof(NativeBootstrapParams) == 48, "Metal bootstrap ABI mismatch");
static_assert(sizeof(SplitState) == 32, "Metal split ABI mismatch");
static_assert(sizeof(CBMStructureInfo) == 32 && sizeof(CBMAppendFeatureOptions) == 32,
              "Metal incremental feature API mismatch");
static_assert(sizeof(CBMQueryOptions) == 16 && sizeof(NativeQueryParams) == 32
              && sizeof(NativeQueryProjectionParams) == 16, "Metal grouped objective ABI mismatch");

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
// Literal validation messages allocate only when a check fails.
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void CopyText(char* destination, size_t capacity, const char* source) {
    if (!destination || !capacity) return;
    if (!source) source = "Unknown Metal error";
    const size_t length = std::min(capacity - 1, std::strlen(source));
    std::memcpy(destination, source, length);
    destination[length] = '\0';
}
std::string ErrorText(NSError* error) {
    const char* text = error ? [[error localizedDescription] UTF8String] : nullptr;
    return text ? text : "Unknown Metal error";
}
uint64_t CheckedProduct(uint64_t a, uint64_t b, const char* name) {
    Require(b == 0 || a <= std::numeric_limits<uint64_t>::max() / b,
            std::string(name) + " size overflows");
    return a * b;
}

// Full-matrix targets own their derivatives and projections. Reserve only the
// shared topology/cursor buffers before choosing the target's matrix tile size.
uint64_t FullMatrixCoreBytes(const CBMSessionParams& options) {
    const auto& p = options.train;
    Require(p.rows && p.rows <= MaxRows && p.features && p.depth <= 8,
            "Invalid full-matrix core dimensions");
    const uint64_t leaves = 1ull << p.depth;
    const uint64_t partitionTiles = (uint64_t(p.rows) + 4095) / 4096;
    const uint64_t scoreGroups = std::max<uint64_t>(1, std::min<uint64_t>((uint64_t(p.candidates) + 255) / 256, 64));
    const uint64_t lossGroups = std::min<uint64_t>((uint64_t(p.rows) + 255) / 256, 4096);
    uint64_t bytes = uint64_t(p.features) * p.rows + 28ull * p.rows + 10ull * p.candidates
        + 13ull * p.features + 16 * leaves + 60 + 4 * leaves * partitionTiles
        + 32 * (scoreGroups + 1) + 4 * lossGroups + 4ull * (uint64_t(p.features) + 1);
    if (options.leaf_estimation_backtracking && options.leaf_estimation_iterations > 1)
        bytes += 16 * leaves + 8 * lossGroups;
    return bytes;
}

uint64_t FullMatrixTargetBudget(const CBMSessionParams& options) {
    const uint64_t coreBytes = FullMatrixCoreBytes(options);
    Require(coreBytes < MaxWorkingBytes, "Full-matrix core exceeds the 1 GiB GPU memory limit");
    return MaxWorkingBytes - coreBytes;
}

static const char* SimpleLeafSource = R"METAL(
struct SimpleLeafParams { uint leaves; uint bootstrap; float l2; uint reserved; };
kernel void EstimateSimpleWeakLeaves(const device float* sums [[buffer(0)]],
    const device float* masses [[buffer(1)]], const device uint* offsets [[buffer(2)]],
    const device uint* rows [[buffer(3)]], const device float* multipliers [[buffer(4)]],
    device float* values [[buffer(5)]], device float* weights [[buffer(6)]],
    constant SimpleLeafParams& p [[buffer(7)]], uint leaf [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    if (leaf >= p.leaves) return;
    threadgroup uint counts[256];
    uint count = 0;
    for (uint i = offsets[leaf] + tid; i < offsets[leaf + 1]; i += 256)
        count += p.bootstrap == 0 || multipliers[rows[i]] != 0.0f;
    counts[tid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128; width; width >>= 1) {
        if (tid < width) counts[tid] += counts[tid + width];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        values[leaf] = counts[0] ? sums[leaf] / (masses[leaf] + p.l2) : 0.0f;
        weights[leaf] = masses[leaf];
    }
}
)METAL";

void ValidateObjectiveOptions(const CBMObjectiveOptions* options) {
    Require(options != nullptr, "Objective configuration is required");
        Require((options->objective <= 20) && options->leaf_estimation_method <= 3 && options->reserved == 0,
                "Unsupported objective or leaf estimation method");
        Require(std::isfinite(options->objective_param), "Objective parameter must be finite");
        if (options->objective == 4) Require(options->objective_param >= 0, "Huber delta must be nonnegative");
        if (options->objective == 5) Require(options->objective_param >= 0 && options->objective_param <= 1,
                                          "Expectile alpha must be in [0, 1]");
        if (options->objective == 6) Require(options->objective_param >= 1, "Lq q must be at least 1");
        if (options->objective == 7) Require(options->objective_param > 1 && options->objective_param < 2,
                                          "Tweedie variance_power must be in (1, 2)");
        if (options->objective == 8 || options->objective == 9)
            Require(options->objective_param >= 0 && options->objective_param <= 1, "Quantile alpha must be in [0, 1]");
        if (options->leaf_estimation_method == 0) {
            Require(options->objective < 8 || options->objective >= 12, "This objective does not support Newton leaf estimation");
            Require(options->objective != 6 || options->objective_param >= 2, "Lq Newton requires q >= 2");
        }
    if (options->leaf_estimation_method == 2)
        Require(options->objective >= 9 && options->objective <= 11, "Exact supports Quantile, MAE and MAPE");
    if (options->leaf_estimation_method == 3)
        Require(options->objective != 17, "YetiRank requires Newton leaves");
    if (options->objective == 16) {
        Require(options->leaf_estimation_method == 0 || options->leaf_estimation_method == 3,
                "QueryCrossEntropy requires Newton or Simple leaves like CUDA");
        Require(options->objective_param >= 0 && options->objective_param <= 1, "QueryCrossEntropy alpha must be in [0, 1]");
    }
    if (options->objective == 17)
        Require(options->leaf_estimation_method == 0, "YetiRank requires Newton leaf estimation like CUDA");
}

struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    explicit Runtime(const char* customSource = nullptr) {
        const auto customPrefix = CBMCustomObjectivePrefix(customSource);
        Device = MTLCreateSystemDefaultDevice();
        Require(Device != nil, "No Metal GPU device is available");
        Require([Device hasUnifiedMemory] && [Device supportsFamily:MTLGPUFamilyApple7],
                "This backend requires an Apple Silicon GPU (Apple family 7 or newer)");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Could not create a Metal command queue");
        MTLCompileOptions* options = [MTLCompileOptions new];
        if (@available(macOS 13.0, *)) options.languageVersion = MTLLanguageVersion3_0;
        else throw std::runtime_error("Metal training requires macOS 13 or newer (Metal 3)");
        options.fastMathEnabled = NO;
        NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s",
            CBMMetalBootstrapSource, CBMMetalSource, CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
            CBMMetalHistogramSource, CBMMetalIncrementalPartitionSource, CBMMetalHistogramReuseSource,
            CBMMetalScoreNoiseSource, CBMMetalBacktrackingSource,
            CBMMetalDeepPartitionSource, CBMMetalCompactHistogramSource, CBMMetalExactLeafSource,
            CBMMetalStreamingScoreSource, CBMMetalQuerywiseSource, CBMMetalDynamicScoreSource];
        if (!customPrefix.empty()) source = [NSString stringWithFormat:@"%s\n%@", customPrefix.c_str(), source];
        source = [NSString stringWithFormat:@"%@\n%s\n%s\n%s\n%s", source, SimpleLeafSource,
            CBMMetalRegularizationSource, CBMMetalLangevinSource, CBMMetalLangevinLeafSource];
        NSError* error = nil;
        id<MTLLibrary> library = [Device newLibraryWithSource:source options:options error:&error];
        Require(library != nil, "Metal shader compilation failed: " + ErrorText(error));
        const char* names[] = {
            "InitializeObjectivePredictions", "ObjectiveDerivatives", "InitializeLeafValues",
            "EstimateSimpleWeakLeaves",
            "ReduceLeafObjectivePartials", "EstimateNewtonLeafValues", "FinalizeLeafValues",
            "AddObjectiveBinModelValue", "ReduceObjectiveLoss", "CollectPartitionStatistics",
            "CountPartitionTiles", "PrefixPartitionTiles", "ScatterPartitionRows",
            "ClearHistograms", "ComputeHistograms", "ScanHistograms", "UpdateLeafBins",
            "FindSplitWinners", "ReduceSplitWinners", "ReduceStructurePartials",
            "InitializeRootPartition", "CountIncrementalPartitionTiles", "ScatterIncrementalPartitionRows",
            "GenerateBootstrapWeights", "ApplyBootstrapWeights", "ComputeMvsThresholds",
            "GenerateMvsBootstrapWeights", "ReduceBootstrapStatistics",
            "ClearChildHistograms", "ComputeSmallerChildHistograms", "ScanChildHistograms",
            "SubtractSiblingHistograms", "ReduceScoreNoiseStatistics", "GenerateScoreFeatureNoise",
            "PrepareBacktrackingDirection", "BuildBacktrackingCandidate", "ReduceBacktrackingObjective",
            "CountDeepPartitionBits", "ScanDeepPartitionTiles", "ScanDeepPartitionBlocks",
            "BuildDeepPartitionOffsets", "ScatterDeepPartitionRows", "ResetCompactHistogramWorkState",
            "BuildCompactHistogramJobs", "BuildCompactHistogramDispatchArguments", "ClearCompactHistograms",
            "ComputeCompactHistograms", "ScanCompactHistograms", "SubtractCompactSiblingHistograms",
            "PrepareExactResiduals", "MakeExactLeafKeys", "BuildExactLeafOffsets", "ReduceExactLeafPartials",
            "PrefixExactLeafTiles", "SelectExactLeafQuantile", "FinalizeExactLeafValues", "UpdateFixedPermutationSplit", "FindTileSplitWinners", "MergeTileSplitWinner",
            "PrepareQuerywisePoint", "QueryRmseDerivatives", "QuerySoftMaxDerivatives",
            "ReduceQuerywiseLeafPartials", "ReduceQuerywiseObjective", "ResetQuerywiseLeafIds",
            "ValidateQuerywiseStructureCurvature", "FindDynamicTileSplitWinners",
            "FindSplitWinnersRegularized", "FindTileSplitWinnersRegularized",
            "FindDynamicTileSplitWinnersRegularized", "EstimateRegularizedNewtonLeafValues",
            "PrepareLangevinBacktrackingDirection", "AddLangevinWeakNoise"
        };
        for (const char* name : names) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing Metal kernel: ") + name);
            error = nil;
            id<MTLComputePipelineState> pipeline =
                [Device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline != nil, std::string("Could not create Metal pipeline ") + name
                    + ": " + ErrorText(error));
            Require(pipeline.maxTotalThreadsPerThreadgroup >= 256,
                    std::string("Metal kernel cannot run 256 threads: ") + name);
            Pipelines.emplace(name, pipeline);
        }
    }
    id<MTLBuffer> Buffer(uint64_t bytes, const void* source = nullptr) {
        const uint64_t length = std::max<uint64_t>(bytes, 1);
        Require(length <= Device.maxBufferLength && length <= MaxWorkingBytes,
                "Requested buffer exceeds the Metal memory limit");
        id<MTLBuffer> result = source && bytes
            ? [Device newBufferWithBytes:source length:static_cast<NSUInteger>(length)
                                 options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:static_cast<NSUInteger>(length)
                                  options:MTLResourceStorageModeShared];
        Require(result != nil, "Metal buffer allocation failed");
        return result;
    }
};
Runtime& GetRuntime() { static Runtime runtime; return runtime; }

struct Command {
    Runtime& Context;
    CBMTrainStats& Stats;
    id<MTLCommandBuffer> Buffer;
    Command(Runtime& context, CBMTrainStats& stats) : Context(context), Stats(stats) { Restart(); }
    void Restart() {
        Buffer = [Context.Queue commandBuffer];
        Require(Buffer != nil, "Could not allocate a Metal command buffer");
    }
    template <class Parameters>
    void Dispatch(const char* kernel, std::initializer_list<id<MTLBuffer>> inputs,
                  const Parameters& params, uint64_t count, bool wholeGroups = false,
                  uint64_t height = 1, uint64_t depth = 1,
                  const void* extraParams = nullptr, size_t extraBytes = 0) {
        Require(count > 0 && count <= std::numeric_limits<uint32_t>::max()
                && height > 0 && depth > 0, "Invalid Metal dispatch size");
        id<MTLComputeCommandEncoder> encoder = [Buffer computeCommandEncoder];
        Require(encoder != nil, "Could not allocate a Metal command encoder");
        [encoder setComputePipelineState:Context.Pipelines.at(kernel)];
        NSUInteger index = 0;
        for (id<MTLBuffer> input : inputs) [encoder setBuffer:input offset:0 atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index++];
        if (extraParams) [encoder setBytes:extraParams length:extraBytes atIndex:index];
        if (wholeGroups) {
            [encoder dispatchThreadgroups:MTLSizeMake(count, height, depth)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [encoder endEncoding];
        ++Stats.kernel_dispatches;
    }
    template <class Parameters>
    void DispatchIndirect(const char* kernel, std::initializer_list<id<MTLBuffer>> inputs,
                          const Parameters& params, id<MTLBuffer> arguments, NSUInteger offset) {
        id<MTLComputeCommandEncoder> encoder = [Buffer computeCommandEncoder];
        Require(encoder != nil, "Could not allocate an indirect Metal encoder");
        [encoder setComputePipelineState:Context.Pipelines.at(kernel)];
        NSUInteger index = 0;
        for (id<MTLBuffer> input : inputs) [encoder setBuffer:input offset:0 atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index];
        [encoder dispatchThreadgroupsWithIndirectBuffer:arguments indirectBufferOffset:offset
                                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        ++Stats.kernel_dispatches;
    }
    void Wait() {
        [Buffer commit];
        [Buffer waitUntilCompleted];
        Require(Buffer.status == MTLCommandBufferStatusCompleted,
                "Metal command failed: " + ErrorText(Buffer.error));
        const double start = Buffer.GPUStartTime, end = Buffer.GPUEndTime;
        if (std::isfinite(start) && std::isfinite(end) && start > 0 && end >= start)
            Stats.gpu_seconds += end - start;
    }
};

class Session {
public:
    CBMSessionParams Options;
    CBMTrainStats Stats = {};
    uint32_t Completed = 0;
    std::mutex Mutex;

    Session(const CBMSessionParams* input, const uint8_t* bins, const float* targets,
            const float* sampleWeights, const float* initialPredictions,
            const uint32_t* candidateFeatures, const uint32_t* candidateBins,
            const uint8_t* candidateTypes, const CBMObjectiveOptions* objectiveOptions = nullptr,
            const CBMQueryOptions* queryOptions = nullptr, const uint32_t* groupOffsets = nullptr,
            std::shared_ptr<CBMPairwiseRuntime> pairwise = nullptr,
            std::shared_ptr<CBMYetiRankRuntime> yeti = nullptr,
            std::shared_ptr<CBMFullMatrixRuntime> coupled = nullptr, float coupledNonDiag = 0.1f,
            std::shared_ptr<CBMCombinationRuntime> combination = nullptr, const char* customSource = nullptr) {
        Require(input != nullptr, "Training parameters are required");
        Options = *input;
        Pairwise = std::move(pairwise);
        Yeti = std::move(yeti);
        Coupled = std::move(coupled); CoupledNonDiag = coupledNonDiag;
        Combination = std::move(combination);
        Require(std::isfinite(coupledNonDiag) && coupledNonDiag >= 0, "Invalid pairwise non-diagonal regularization");
        if (objectiveOptions) {
            ValidateObjectiveOptions(objectiveOptions);
            Require(objectiveOptions->objective == Options.objective, "Configured objective must match session objective");
        }
        // private/libs/options/catboost_options.cpp normalizes exactly-zero L2.
        if (Options.train.l2_leaf_reg == 0) Options.train.l2_leaf_reg = 1e-20f;
        const auto& p = Options.train;
        Require(p.rows > 0 && p.rows <= MaxRows, "rows must be between 1 and 16777216");
        Require(p.features > 0, "At least one feature is required");
        Require(p.bins_per_feature >= 1 && p.bins_per_feature <= 256,
                "bins_per_feature must be between 1 and 256");
        Require(p.iterations > 0 && p.iterations <= 100000,
                "iterations must be between 1 and 100000");
        Require(p.depth <= 16, "This Metal port supports tree depth from 0 through 16");
        Deep = Compact = p.depth > 8;
        Require(p.score_function <= 6, "Unsupported Metal split score function");
        Require(Options.objective <= 20, "objective is not implemented by the Metal backend");
        Require((Options.objective == 19) == bool(Combination), "Combination requires its component-aware constructor");
        Require((Options.objective == 20) == bool(customSource), "Custom objectives require a Metal source constructor");
        Require((Options.objective == 15 || Options.objective == 16 || Options.objective == 18) == bool(Coupled), "This objective requires its full-matrix constructor");
        Require(Options.objective != 16 || p.score_function != 0, "QueryCrossEntropy does not support L2 structure score like CUDA");
        Require(!Coupled || p.depth <= 8, "Full-matrix pairwise training requires depth <= 8 like CUDA");
        Require((Options.objective == 17) == bool(Yeti), "YetiRank requires the resident YetiRank constructor");
        Require((Options.objective == 12 || Options.objective == 13) == (queryOptions != nullptr),
                "Grouped objectives require the query-aware constructor");
        Require((Options.objective == 14) == bool(Pairwise), "PairLogit requires the supplied-pair constructor");
        if (queryOptions) {
            Require(groupOffsets && queryOptions->group_count > 0 && queryOptions->group_count <= p.rows,
                    "Query group offsets and a valid group count are required");
            Require(std::isfinite(queryOptions->beta) && std::isfinite(queryOptions->lambda) && !queryOptions->reserved,
                    "Query beta and lambda must be finite, with reserved options zero");
            Require(groupOffsets[0] == 0 && groupOffsets[queryOptions->group_count] == p.rows,
                    "Query group offsets must span all training rows");
            for (uint32_t group = 0; group < queryOptions->group_count; ++group)
                Require(groupOffsets[group] < groupOffsets[group + 1], "Query group offsets must increase strictly");
            QueryOptions = *queryOptions;
        }
        Require(Options.leaf_estimation_iterations >= 1 && Options.leaf_estimation_iterations <= 1000,
                "leaf_estimation_iterations must be between 1 and 1000");
        Require(Options.leaf_estimation_backtracking <= 2,
                "leaf_estimation_backtracking must be No, AnyImprovement or Armijo");
        Require((!Yeti && Options.objective != 18) || Options.leaf_estimation_backtracking == 0, "YetiRank objectives do not support leaf backtracking");
        Require(Options.reserved == 0, "Reserved session options must be zero");
        Require(!objectiveOptions || objectiveOptions->leaf_estimation_method != 3 ||
                (Options.leaf_estimation_iterations == 1 && (!Coupled || (p.depth > 0 && p.candidates > 0))),
                "Simple leaves require one estimation iteration; full-matrix Simple also requires split candidates");
        Require(std::isfinite(p.learning_rate) && p.learning_rate > 0 && p.learning_rate <= 1,
                "learning_rate must be finite and in (0, 1]");
        Require(std::isfinite(p.l2_leaf_reg) && p.l2_leaf_reg >= 0,
                "l2_leaf_reg must be finite and nonnegative");
        Require(std::isfinite(p.bias), "bias must be finite");
        Require(bins && targets, "Training input buffers must not be null");
        Require(p.candidates == 0 || (candidateFeatures && candidateBins),
                "Candidate input buffers are required");
        DataCells = CheckedProduct(p.rows, p.features, "Input data");
        MaxLeaves = uint32_t(1) << p.depth;
        std::vector<uint32_t> featureOffsets(p.features + 1, 0);
        for (uint32_t candidate = 0; candidate < p.candidates; ++candidate) {
            Require(candidateFeatures[candidate] < p.features && candidateBins[candidate] < p.bins_per_feature,
                    "Candidate feature or bin exceeds its declared dimension");
            auto& span = featureOffsets[candidateFeatures[candidate] + 1];
            span = std::max(span, candidateBins[candidate] + 1);
        }
        uint64_t totalBins = 0;
        for (uint32_t feature = 0; feature < p.features; ++feature) {
            totalBins += featureOffsets[feature + 1];
            Require(totalBins <= std::numeric_limits<uint32_t>::max(), "Compact feature offsets overflow");
            featureOffsets[feature + 1] = static_cast<uint32_t>(totalBins);
        }
        CompactBins = static_cast<uint32_t>(totalBins);
        HistogramCells = Coupled ? 1 : Deep ? CheckedProduct(MaxLeaves, std::max<uint32_t>(CompactBins, 1), "Compact histogram")
            : CheckedProduct(CheckedProduct(MaxLeaves, p.features, "Histogram"), p.bins_per_feature, "Histogram");
        Require(p.candidates <= CheckedProduct(p.features, p.bins_per_feature, "Candidate count"),
                "Candidate count exceeds feature/border combinations");
        Require(DataCells <= std::numeric_limits<uint32_t>::max()
                && (Deep || HistogramCells <= std::numeric_limits<uint32_t>::max()),
                "Dataset or histogram exceeds the GPU indexing limit");
        const uint32_t partitionTiles = (p.rows + 4095) / 4096;
        const uint32_t histogramTiles = std::min<uint32_t>(partitionTiles, 32);
        const uint32_t scoreGroups = std::max<uint32_t>(1, std::min<uint32_t>((p.candidates + 255) / 256, 64));
        LossGroups = std::min<uint32_t>((p.rows + 255) / 256, 4096);
        DeepTiles = (p.rows + 255) / 256;
        DeepBlocks = (p.rows + 65535) / 65536;
        HistogramJobCapacity = (p.rows + 8191) / 8192 + std::min(p.rows, MaxLeaves);
        WorkingBytes = DataCells + 36ull * p.rows + 9ull * p.candidates + 13ull * p.features
            + 8ull * HistogramCells + 24ull * MaxLeaves + 4
            + (Deep ? 4ull * DeepTiles + 4ull * DeepBlocks + 4ull * (MaxLeaves + 1)
                    : 4ull * MaxLeaves * partitionTiles) + 32ull * MaxLeaves * histogramTiles
            + 32ull * (scoreGroups + 1) + 4ull * LossGroups;
        WorkingBytes += 4ull * (p.features + 1);
        if (queryOptions) WorkingBytes += 4ull * p.rows + 12ull * queryOptions->group_count + 8 + 8ull * LossGroups;
        if (Pairwise) WorkingBytes += Pairwise->AllocatedBytes();
        if (Yeti) WorkingBytes += Yeti->AllocatedBytes();
        if (Combination) WorkingBytes += Combination->AllocatedBytes() + 4ull * p.rows;
        if (Coupled) {
            WorkingBytes -= 8ull * (p.rows - 1) + 8ull * (MaxLeaves - 1)
                + 32ull * (uint64_t(MaxLeaves) * histogramTiles - 1);
            WorkingBytes += Coupled->AllocatedBytes() + p.candidates;
        }
        if (Deep) WorkingBytes += 16ull * HistogramJobCapacity + 4ull * MaxLeaves + 52;
        if (UsesBacktracking()) WorkingBytes += 16ull * MaxLeaves + 8ull * LossGroups;
        if (Coupled) Require(WorkingBytes == FullMatrixCoreBytes(Options) + Coupled->AllocatedBytes(),
                             "Full-matrix core allocation accounting mismatch");
        if (Deep) {
            // Reserve tile metadata and bound the histogram pair to 256 MiB.
            // Wider feature grids stream tiles, rebuilding at each tree depth.
            const uint64_t baseBytes = WorkingBytes - 8ull * HistogramCells + 8ull * p.features + 32;
            const uint64_t exactReserve = objectiveOptions && objectiveOptions->leaf_estimation_method == 2
                ? 50ull * p.rows + 8ull * MaxLeaves * histogramTiles + 12ull * MaxLeaves : 0;
            Require(baseBytes + exactReserve < MaxWorkingBytes, "Non-histogram workspace exceeds the GPU memory limit");
            const uint64_t budget = std::min<uint64_t>(uint64_t(256) << 20, MaxWorkingBytes - baseBytes - exactReserve);
            const uint64_t capacityBins = budget / (8ull * MaxLeaves);
            uint32_t begin = 0, tileBins = 0, maximumTileBins = 0;
            for (uint32_t feature = 0; feature < p.features; ++feature) {
                const uint32_t span = featureOffsets[feature + 1] - featureOffsets[feature];
                Require(span <= capacityBins, "One feature histogram exceeds the available memory budget");
                if (tileBins && uint64_t(tileBins) + span > capacityBins) {
                    HistogramTiles.push_back({begin, feature, tileBins, nil});
                    maximumTileBins = std::max(maximumTileBins, tileBins);
                    begin = feature; tileBins = 0;
                }
                tileBins += span;
            }
            HistogramTiles.push_back({begin, p.features, tileBins, nil});
            maximumTileBins = std::max(maximumTileBins, tileBins);
            HistogramCells = CheckedProduct(MaxLeaves, std::max<uint32_t>(maximumTileBins, 1), "Tiled histograms");
            WorkingBytes = baseBytes + 8ull * HistogramCells;
        }
        Require(WorkingBytes <= MaxWorkingBytes,
                "Training exceeds the 1 GiB experimental GPU memory limit");
        const uint64_t splitCount = CheckedProduct(p.iterations, p.depth, "Tree splits");
        const uint64_t leafCount = CheckedProduct(p.iterations, MaxLeaves, "Tree leaves");
        const uint64_t outputBytes = 4ull * p.iterations + 9ull * splitCount + 8ull * leafCount
            + 4ull * p.rows + 4ull * (uint64_t(p.iterations) + 1);
        Require(Deep || outputBytes <= MaxOutputBytes, "Training outputs exceed the 512 MiB limit");
        for (uint64_t i = 0; i < DataCells; ++i)
            Require(bins[i] < p.bins_per_feature, "A quantized value exceeds bins_per_feature");
        double totalWeight = 0;
        for (uint32_t row = 0; row < p.rows; ++row) {
            Require(std::isfinite(targets[row]), "Targets must all be finite");
            if (Options.objective == 1)
                Require(targets[row] == 0 || targets[row] == 1, "Logloss targets must be 0 or 1");
            if (Options.objective == 2)
                Require(targets[row] >= 0 && targets[row] <= 1, "CrossEntropy targets must be in [0, 1]");
            if (Options.objective == 3 || Options.objective == 13)
                Require(targets[row] >= 0, "Poisson/QuerySoftMax targets must be nonnegative");
            const float weight = sampleWeights ? sampleWeights[row] : 1;
            Require(std::isfinite(weight) && weight >= 0, "Sample weights must be finite and nonnegative");
            totalWeight += weight;
            Require(!initialPredictions || std::isfinite(initialPredictions[row]),
                    "Initial raw predictions must all be finite");
        }
        Require(totalWeight > 0 && totalWeight <= std::numeric_limits<float>::max(),
                "Total sample weight must be positive and representable as float32");
        TotalWeight = totalWeight;
        if (Options.objective == 13) {
            double targetMass = 0;
            for (uint32_t row = 0; row < p.rows; ++row)
                targetMass += double(sampleWeights ? sampleWeights[row] : 1.0f) * targets[row];
            Require(targetMass > 0 && targetMass <= std::numeric_limits<float>::max(),
                    "QuerySoftMax requires positive finite float32 total weighted target mass");
        }
        std::vector<uint8_t> featureTypes(p.features, 0), seen(p.features, 0);
        std::vector<uint8_t> types(p.candidates, 0);
        for (uint32_t i = 0; i < p.candidates; ++i) {
            const uint32_t feature = candidateFeatures[i], bin = candidateBins[i];
            const uint8_t type = candidateTypes ? candidateTypes[i] : 0;
            Require(feature < p.features, "A split candidate references an invalid feature");
            Require(type <= 1, "Candidate split types must be numeric (0) or one-hot (1)");
            Require(bin < p.bins_per_feature && (type == 1 || bin + 1 < p.bins_per_feature),
                    "A split candidate references an invalid border bin");
            Require(!seen[feature] || featureTypes[feature] == type,
                    "A feature cannot mix numeric and one-hot candidate types");
            featureTypes[feature] = type; seen[feature] = 1; types[i] = type;
        }
        if (!candidateTypes)
            Require(p.candidates <= CheckedProduct(p.features, p.bins_per_feature - 1, "Candidate count"),
                    "Candidate count exceeds feature/border combinations");
        if (customSource) OwnedContext = std::make_unique<Runtime>(customSource);
        Context = OwnedContext ? OwnedContext.get() : &GetRuntime();
        CopyText(Stats.device_name, sizeof(Stats.device_name), [Context->Device.name UTF8String]);
        Data = Context->Buffer(DataCells, bins);
        Target = Context->Buffer(4ull * p.rows, targets);
        Prediction = Context->Buffer(4ull * p.rows, initialPredictions);
        SampleWeight = Context->Buffer(4ull * p.rows, sampleWeights);
        if (!sampleWeights) std::fill_n(static_cast<float*>(SampleWeight.contents), p.rows, 1.0f);
        StructureWeight = SampleWeight;
        Gradient = Context->Buffer(Coupled ? 4 : 4ull * p.rows);
        Hessian = Context->Buffer(Coupled ? 4 : 4ull * p.rows);
        if (Combination) CombinationGradientWeights = Context->Buffer(4ull * p.rows);
        LeafIds = Context->Buffer(4ull * p.rows);
        RowIndices = Context->Buffer(4ull * p.rows);
        NextRowIndices = Context->Buffer(4ull * p.rows);
        RowRanks = Context->Buffer(4ull * p.rows);
        PartitionTiles = Context->Buffer(Deep ? 4ull * DeepTiles : 4ull * MaxLeaves * partitionTiles);
        if (Deep) {
            DeepBlockPrefix = Context->Buffer(4ull * DeepBlocks);
            NextPartitionOffsets = Context->Buffer(4ull * (MaxLeaves + 1));
            HistogramJobs = Context->Buffer(16ull * HistogramJobCapacity);
            HistogramActive = Context->Buffer(4ull * MaxLeaves);
            HistogramState = Context->Buffer(16);
            HistogramArguments = Context->Buffer(36);
        }
        PartitionOffsets = Context->Buffer(4ull * (MaxLeaves + 1));
        CandidateFeature = Context->Buffer(4ull * p.candidates, candidateFeatures);
        CandidateBin = Context->Buffer(4ull * p.candidates, candidateBins);
        CandidateType = Context->Buffer(p.candidates, types.data());
        if (Coupled && p.candidates) {
            CandidateActive = Context->Buffer(p.candidates);
            std::fill_n(static_cast<uint8_t*>(CandidateActive.contents), p.candidates, uint8_t(1));
            Coupled->SetCandidateMask(CandidateActive);
        }
        FeatureType = Context->Buffer(p.features, featureTypes.data());
        FeatureOffsets = Context->Buffer(4ull * (p.features + 1), featureOffsets.data());
        if (Deep) {
            for (auto& tile : HistogramTiles) {
                std::vector<uint32_t> offsets(tile.End - tile.Begin + 1);
                for (uint32_t feature = tile.Begin; feature <= tile.End; ++feature)
                    offsets[feature - tile.Begin] = featureOffsets[feature] - featureOffsets[tile.Begin];
                tile.Offsets = Context->Buffer(4ull * offsets.size(), offsets.data());
            }
            TileWinner = Context->Buffer(sizeof(SplitState));
        }
        FeaturePenaltyWeights = Context->Buffer(8ull * p.features);
        std::fill_n(static_cast<float*>(FeaturePenaltyWeights.contents), 2ull * p.features, 1.0f);
        CtrUniqueValues.resize(p.features, 0); UsedFeatures.resize(p.features, 0); BinFeatureWeights.resize(p.features, 1.0f);
        FeatureFlags.resize(p.features, 2); ActiveFeatures.resize(p.features, 1);
        FeatureNoise = Context->Buffer(4ull * p.features);
        std::fill_n(static_cast<float*>(FeatureNoise.contents), p.features, 0.0f);
        HistogramSums = Context->Buffer(4ull * HistogramCells);
        HistogramWeights = Context->Buffer(4ull * HistogramCells);
        LeafSums = Context->Buffer(Coupled ? 4 : 4ull * MaxLeaves);
        LeafWeights = Context->Buffer(Coupled ? 4 : 4ull * MaxLeaves);
        RawValues = Context->Buffer(4ull * MaxLeaves);
        Values = Context->Buffer(4ull * MaxLeaves);
        Weights = Context->Buffer(4ull * MaxLeaves);
        ObjectivePartials = Context->Buffer(Coupled ? 32 : 32ull * MaxLeaves * histogramTiles);
        WinnerPartials = Context->Buffer(32ull * scoreGroups);
        Winner = Context->Buffer(sizeof(SplitState));
        LossPartials = Context->Buffer(4ull * LossGroups);
        if (queryOptions) {
            QueryOffsets = Context->Buffer(4ull * (QueryOptions.group_count + 1), groupOffsets);
            QueryPoint = Context->Buffer(4ull * p.rows);
            QueryStatistics = Context->Buffer(8ull * QueryOptions.group_count);
            QueryLossPartials = Context->Buffer(8ull * LossGroups);
            QueryValidation = Context->Buffer(4);
        }
        if (UsesBacktracking()) {
            TrialValues = Context->Buffer(4ull * MaxLeaves);
            Directions = Context->Buffer(4ull * MaxLeaves);
            DirectionDot = Context->Buffer(8ull * MaxLeaves);
            BacktrackingLoss = Context->Buffer(8ull * LossGroups);
        }
        K = {p.rows, p.features, p.bins_per_feature, 1, p.candidates, 0, 0, 0,
             p.bias, p.learning_rate, p.l2_leaf_reg, p.score_function,
             Options.objective, 0, Options.leaf_estimation_iterations, 4096,
             static_cast<float>(totalWeight), histogramTiles, partitionTiles, scoreGroups,
             objectiveOptions ? objectiveOptions->objective_param :
                 (Options.objective == 5 || Options.objective >= 8 ? 0.5f :
                  Options.objective == 6 ? 2.0f : Options.objective == 7 ? 1.5f : 1.0f),
             objectiveOptions ? objectiveOptions->leaf_estimation_method : 0, uint32_t(Deep), 0};
        SimpleLeavesRequested = K.LeafMethod == 3;
        TreeDepths.resize(p.iterations, 0);
        SplitFeatures.resize(splitCount, 0); SplitBins.resize(splitCount, 0); SplitTypes.resize(splitCount, 0);
        if (Deep) { DeepTreeValues.resize(p.iterations); DeepTreeWeights.resize(p.iterations); }
        else { TreeValues.resize(leafCount, 0); TreeWeights.resize(leafCount, 0); }
        Losses.resize(p.iterations + 1, 0);
        Command command(*Context, Stats);
        if (!initialPredictions) command.Dispatch("InitializeObjectivePredictions", {Prediction}, K, p.rows);
        EncodeLoss(command);
        command.Wait();
        Losses[0] = ReadLoss();
    }

    void Step() {
        BeginTree();
        while (!Pending.Finished) GrowTree();
        FinishTree();
    }
    void BeginTree() {
        RequireCompletedState();
        Require(!Failed, "This training session failed and must be closed");
        Require(Completed < Options.train.iterations, "Training session has no remaining iterations");
        try { BeginTreeImpl(); } catch (...) { Failed = true; throw; }
    }
    void GrowTree() {
        Require(Pending.Active, "No tree is open; begin_tree must be called first");
        Require(!Failed, "This training session failed and must be closed");
        try { GrowTreeImpl(); } catch (...) { Failed = true; throw; }
    }
    void FinishTree() {
        Require(Pending.Active, "No tree is open; begin_tree must be called first");
        Require(!Failed, "This training session failed and must be closed");
        try { FinishTreeImpl(); Pending.Active = false; } catch (...) { Failed = true; throw; }
    }
    void CopyStructureInfo(CBMStructureInfo* info) const {
        Require(info != nullptr, "Structure step output is required");
        const auto& winner = Pending.LastWinner;
        *info = {Pending.Depth, uint32_t(Pending.Finished), uint32_t(Pending.HasSplit),
                 winner.Feature, winner.Bin, winner.Type, winner.Score, winner.Gain};
    }
    void RequireCompletedState() const {
        Require(!Pending.Active, "A tree is open; finish_tree is required before this operation");
    }
    void ConfigureObjective(const CBMObjectiveOptions* options) {
        RequireCompletedState();
        Require(options != nullptr, "Objective configuration is required");
        Require(Completed == 0 && !Failed, "Objective configuration must precede the first training step");
        ValidateObjectiveOptions(options);
        Require((options->objective == 12 || options->objective == 13) == (QueryOffsets != nil),
                "Objective configuration cannot change between grouped and scalar sessions");
        Require((options->objective == 14) == bool(Pairwise), "Objective configuration cannot change supplied-pair sessions");
        Require((options->objective == 15 || options->objective == 16 || options->objective == 18) == bool(Coupled), "Objective configuration cannot change full-matrix pairwise sessions");
        Require((options->objective == 17) == bool(Yeti), "Objective configuration cannot change YetiRank sessions");
        Require((options->objective == 19) == bool(Combination), "Objective configuration cannot change Combination sessions");
        Require((options->objective == 20) == bool(OwnedContext), "Objective configuration cannot change custom Metal sessions");
        Require(options->leaf_estimation_method != 3 ||
                (Options.leaf_estimation_iterations == 1 && (!Coupled || (Options.train.depth > 0 && K.Candidates > 0))),
                "Simple leaves require one estimation iteration; full-matrix Simple also requires split candidates");
        Require(!Coupled || (options->objective == K.Objective && options->objective_param == K.ObjectiveParam),
                "Changing a full-matrix target requires a new session");
        Require(!QueryOffsets || options->objective == K.Objective,
                "Changing the grouped objective requires a new query session");
        const float* targets = static_cast<const float*>(Target.contents);
        for (uint32_t row = 0; row < K.Rows; ++row) {
            if (options->objective == 1) Require(targets[row] == 0 || targets[row] == 1, "Logloss targets must be 0 or 1");
            if (options->objective == 2) Require(targets[row] >= 0 && targets[row] <= 1, "CrossEntropy targets must be in [0, 1]");
            if (options->objective == 3 || options->objective == 7)
                Require(targets[row] >= 0, "Poisson/Tweedie targets must be nonnegative");
        }
        if (options->leaf_estimation_method == 2)
            Require(options->objective >= 9 && options->objective <= 11, "Exact supports Quantile, MAE and MAPE");
        if (options->leaf_estimation_method == 2 && !ExactResiduals) {
            // The two resident radix sorts retain temporary buffers until command
            // completion. Reserve a conservative 26 bytes/row for both scratch sets.
            const uint64_t exactBytes = 50ull * K.Rows + 8ull * MaxLeaves * K.HistogramTiles + 12ull * MaxLeaves;
            const uint64_t bootstrapBytes = BootstrapOptions.bootstrap_type
                ? 8ull * K.Rows + 8ull * LossGroups + 4ull * ((K.Rows + 8191) / 8192) : 0;
            Require(WorkingBytes + LangevinBytes + PermutationBytes + exactBytes + bootstrapBytes + (NoiseStatistics ? 4ull * LossGroups : 0) <= MaxWorkingBytes,
                    "Exact leaf workspace exceeds the 1 GiB GPU memory limit");
            // Publish the workspace only after every allocation succeeds. A
            // rejected configuration must leave this session usable.
            auto residuals = Context->Buffer(4ull * K.Rows);
            auto effectiveWeights = Context->Buffer(4ull * K.Rows);
            auto keysA = Context->Buffer(4ull * K.Rows), rowsA = Context->Buffer(4ull * K.Rows);
            auto keysB = Context->Buffer(4ull * K.Rows), rowsB = Context->Buffer(4ull * K.Rows);
            auto tileOffsets = Context->Buffer(8ull * MaxLeaves * K.HistogramTiles);
            auto totals = Context->Buffer(8ull * MaxLeaves);
            auto selected = Context->Buffer(4ull * MaxLeaves);
            ExactResiduals = residuals; ExactEffectiveWeights = effectiveWeights;
            ExactKeysA = keysA; ExactRowsA = rowsA; ExactKeysB = keysB; ExactRowsB = rowsB;
            ExactTileOffsets = tileOffsets; ExactTotals = totals; ExactSelected = selected;
            ExactBytes = exactBytes;
        }
        Options.objective = K.Objective = options->objective;
        K.ObjectiveParam = options->objective_param;
        K.LeafMethod = Dynamic && options->leaf_estimation_method == 3 ? 1 : options->leaf_estimation_method;
        SimpleLeavesRequested = options->leaf_estimation_method == 3;
        Command command(*Context, Stats);
        EncodeLoss(command); command.Wait(); Losses[0] = ReadLoss();
    }
    void SetYetiOracleSeeds(uint32_t count, const uint64_t* seeds) {
        RequireCompletedState();
        Require(Yeti && !Failed, "A valid YetiRank session is required for oracle seeds");
        Require((count == 1 || count == YetiLeafSeedCount() + 1) && seeds,
            "YetiRank needs its weak seed alone or the complete weak/leaf oracle seed schedule");
        YetiSeeds.assign(seeds, seeds + count);
        if (Dynamic && count > 1) ReorderFeatureParallelYetiSeeds();
        YetiSeedPosition = 0;
    }
    void SetYetiLeafSeeds(uint32_t count, const uint64_t* seeds) {
        Require(Yeti && !Failed && Pending.Active && Pending.Finished,
            "YetiRank leaf seeds must follow completed structure search");
        // A depth-zero/empty-candidate tree still prepares its weak target in
        // finish_tree. Its seed remains first and must not be replaced.
        Require(YetiSeeds.size() == 1 && YetiSeedPosition <= 1 && count == YetiLeafSeedCount() && seeds,
            "YetiRank leaf seed schedule is invalid or already supplied");
        YetiSeeds.insert(YetiSeeds.end(), seeds, seeds + count);
        if (Dynamic) ReorderFeatureParallelYetiSeeds();
    }
    void SetCombinationYetiSeeds(uint32_t count, const uint64_t* seeds) {
        Require(Combination && Combination->HasYeti() && !Failed,
            "A valid Combination with YetiRank is required for oracle seeds");
        Require(!Pending.Active || Pending.Finished,
            "Combination YetiRank seeds can only change at a tree or leaf-estimation boundary");
        Combination->SetYetiSeeds(count, seeds);
    }
    void SetCombinationYetiSeedCallback(CBMCombinationYetiSeedCallback callback, void* context) {
        Require(Combination && Combination->HasYeti() && !Failed,
            "A valid Combination with YetiRank is required for an oracle seed callback");
        Require(!Pending.Active || Pending.Finished,
            "Combination YetiRank seed callbacks can only change at a tree or leaf-estimation boundary");
        Combination->SetYetiSeedCallback(callback, context);
    }
    void ConfigureBootstrap(const CBMBootstrapOptions* options) {
        RequireCompletedState();
        Require(options != nullptr, "Bootstrap configuration is required");
        Require(Completed == 0 && !Failed, "Bootstrap configuration must precede the first training step");
        Require(options->bootstrap_type <= 4 && !options->reserved0 && !options->reserved1,
                "Unsupported bootstrap type or reserved options");
        Require(options->mvs_reg_is_set <= 1 && options->initial_mvs_lambda_is_set <= 1,
                "Invalid bootstrap state flags");
        Require(uint64_t(options->iteration_offset) + Options.train.iterations <= std::numeric_limits<uint32_t>::max(),
                "Absolute iteration count exceeds the bootstrap RNG index limit");
        Require(std::isfinite(options->bagging_temperature) && options->bagging_temperature >= 0,
                "bagging_temperature must be finite and nonnegative");
        Require(std::isfinite(options->subsample) && options->subsample > 0 && options->subsample <= 1,
                "subsample must be finite and in (0, 1]");
        Require(options->bootstrap_type != 3 || options->subsample < 1,
                "Poisson bootstrap requires float32 subsample less than 1");
        Require(!options->mvs_reg_is_set || (std::isfinite(options->mvs_reg) && options->mvs_reg >= 0),
                "mvs_reg must be finite and nonnegative");
        Require(!options->initial_mvs_lambda_is_set ||
                (std::isfinite(options->initial_mvs_lambda) && options->initial_mvs_lambda >= 0),
                "Initial MVS state must be finite and nonnegative");
        Require(!Coupled || options->bootstrap_type != 4, "Full-matrix pairwise search does not support MVS");
        Require(Options.objective != 18 || options->bootstrap_type <= 2, "YetiRankPairwise supports No, Bayesian and Bernoulli bootstrap");
        Require(Options.objective != 16 || options->bootstrap_type == 0 || options->bootstrap_type == 2,
                "QueryCrossEntropy supports No or Bernoulli query bootstrap only");
        if (options->bootstrap_type != 0 && !Coupled) {
            const uint64_t extra = 8ull * K.Rows + 8ull * LossGroups + 4ull * ((K.Rows + 8191) / 8192);
            Require(WorkingBytes + LangevinBytes + PermutationBytes + ExactBytes + extra + (NoiseStatistics ? 4ull * LossGroups : 0) <= MaxWorkingBytes,
                    "Bootstrap workspace exceeds the 1 GiB experimental GPU memory limit");
            auto structureWeight = Context->Buffer(4ull * K.Rows);
            auto multipliers = Context->Buffer(4ull * K.Rows);
            auto statistics = Context->Buffer(8ull * LossGroups);
            auto thresholds = Context->Buffer(4ull * ((K.Rows + 8191) / 8192));
            StructureWeight = structureWeight; BootstrapMultipliers = multipliers;
            BootstrapStatistics = statistics; MVSThresholds = thresholds;
        } else {
            StructureWeight = SampleWeight;
            BootstrapMultipliers = nil; BootstrapStatistics = nil; MVSThresholds = nil;
        }
        BootstrapOptions = *options;
        HasMVSLambda = options->initial_mvs_lambda_is_set != 0;
        MVSLambda = options->initial_mvs_lambda;
    }
    void ConfigureLangevin(float temperature, uint32_t weakNoise, CBMLangevinNoiseCallback noise,
                           CBMLangevinSeedCallback seed, void* context) {
        RequireCompletedState();
        Require(!Completed && !Failed && !Coupled && weakNoise <= 1 && noise && seed,
                "Configure scalar Langevin with valid callbacks before the first tree");
        Require(std::isfinite(temperature) && temperature >= 0,
                "Langevin diffusion temperature must be finite and nonnegative");
        if (!BacktrackingLoss) {
            Require(WorkspaceBytes() + 8ull * LossGroups <= MaxWorkingBytes,
                    "Langevin objective workspace exceeds the GPU memory limit");
            BacktrackingLoss = Context->Buffer(8ull * LossGroups);
            LangevinBytes = 8ull * LossGroups;
        }
        if (!weakNoise && K.LeafMethod == 3) K.LeafMethod = 1;
        Langevin = true; WeakLangevin = weakNoise; LangevinTemperature = temperature;
        LangevinNoise = noise; LangevinSeed = seed; LangevinContext = context;
    }
    void ConfigureRegularization(const CBMRegularizationOptions* options) {
        RequireCompletedState();
        Require(options && !options->reserved && options->normalize_score <= 1 &&
                options->normalize_leaf <= 1 && options->add_ridge <= 1,
                "Regularization options require boolean flags and zero reserved fields");
        Require(Completed == 0 && !Failed, "Regularization must be configured before the first tree");
        Require(std::isfinite(options->meta_l2_exponent) && std::isfinite(options->meta_l2_frequency),
                "MetaL2 exponent and frequency must be finite");
        Regularization = *options;
    }
    void SetMetaL2Callback(CBMMetaL2ExponentCallback callback, void* context) {
        RequireCompletedState();
        Require(Completed == 0 && !Failed, "MetaL2 callback must precede the first tree");
        MetaL2Callback = callback; MetaL2Context = context;
    }
    void ConfigureScoreNoise(const CBMScoreNoiseOptions* options) {
        RequireCompletedState();
        Require(options && !options->reserved0 && !options->reserved1 && !options->reserved2,
                "Valid score noise configuration is required");
        Require(Completed == 0 && !Failed, "Score noise configuration must precede the first training step");
        Require(std::isfinite(options->random_strength) && options->random_strength >= 0,
                "random_strength must be finite and nonnegative");
        if (!Coupled && options->random_strength > 0 && (K.ScoreFunction == 1 || K.ScoreFunction == 3)) {
            const uint64_t extra = BootstrapOptions.bootstrap_type
                ? 8ull * K.Rows + 8ull * LossGroups + 4ull * ((K.Rows + 8191) / 8192) : 0;
            Require(WorkingBytes + LangevinBytes + PermutationBytes + ExactBytes + extra + 4ull * LossGroups <= MaxWorkingBytes,
                    "Score noise workspace exceeds the 1 GiB GPU memory limit");
            NoiseStatistics = Context->Buffer(4ull * LossGroups);
        } else {
            NoiseStatistics = nil;
            std::fill_n(static_cast<float*>(FeatureNoise.contents), K.Features, 0.0f);
        }
        RandomStrength = options->random_strength;
    }
    void ConfigurePermutations(uint32_t count, const uint8_t* const* bins,
                               const float* const* cursors, const float* lambdas, const uint8_t* valid) {
        RequireCompletedState();
        Require(Completed == 0 && !Failed && Permutations.empty(), "Permutations can be configured once before training");
        Require(count >= 1 && count <= 64 && bins, "Permutation count must be in [1, 64] with matrices supplied");
        Require(!Yeti || YetiSeeds.empty(), "Configure YetiRank permutations before supplying oracle seeds");
        Require((lambdas == nullptr) == (valid == nullptr), "MVS permutation values and validity flags must be supplied together");
        for (uint32_t permutation = 0; permutation < count; ++permutation) {
            Require(bins[permutation] != nullptr, "Every permutation matrix is required");
            for (uint64_t cell = 0; cell < DataCells; ++cell)
                Require(bins[permutation][cell] < K.Bins, "Permutation bin exceeds the original bin dimension");
            if (cursors && cursors[permutation]) for (uint32_t row = 0; row < K.Rows; ++row)
                Require(std::isfinite(cursors[permutation][row]), "Permutation raw cursors must be finite");
            if (valid) Require(valid[permutation] <= 1 && (!valid[permutation] ||
                (std::isfinite(lambdas[permutation]) && lambdas[permutation] >= 0)), "Invalid permutation MVS state");
        }
        const uint64_t extra = uint64_t(count - 1) * (DataCells + 4ull * K.Rows) + 32ull * Options.train.depth;
        const uint64_t bootstrapBytes = !Coupled && BootstrapOptions.bootstrap_type
            ? 8ull * K.Rows + 8ull * LossGroups + 4ull * ((K.Rows + 8191) / 8192) : 0;
        Require(WorkingBytes + LangevinBytes + ExactBytes + extra + bootstrapBytes + (NoiseStatistics ? 4ull * LossGroups : 0) <= MaxWorkingBytes,
                "Permutation matrices and cursors exceed the 1 GiB GPU memory limit");
        std::vector<PermutationData> states(count);
        for (uint32_t permutation = 0; permutation < count; ++permutation) {
            states[permutation].Bins = permutation == 0 ? Data : Context->Buffer(DataCells, bins[permutation]);
            states[permutation].Cursor = permutation == 0 ? Prediction : Context->Buffer(4ull * K.Rows,
                cursors && cursors[permutation] ? cursors[permutation] : Prediction.contents);
            states[permutation].Lambda = lambdas ? lambdas[permutation] : MVSLambda;
            states[permutation].HasLambda = valid ? valid[permutation] != 0 : HasMVSLambda;
        }
        std::memcpy(Data.contents, bins[0], DataCells);
        if (cursors && cursors[0]) std::memcpy(Prediction.contents, cursors[0], 4ull * K.Rows);
        FixedPermutationSplits = Context->Buffer(32ull * Options.train.depth);
        Permutations = std::move(states);
        PermutationBytes = extra;
        Data = Permutations.back().Bins; Prediction = Permutations.back().Cursor;
        MVSLambda = Permutations.back().Lambda; HasMVSLambda = Permutations.back().HasLambda;
        Command command(*Context, Stats); EncodeLoss(command); command.Wait(); Losses[0] = ReadLoss();
    }
    uint32_t PermutationCount() const { return Permutations.empty() ? 1 : static_cast<uint32_t>(Permutations.size()); }
    void SelectPermutation(uint32_t index) {
        RequireCompletedState();
        Require(!Failed && index < PermutationCount(), "Search permutation index is out of range");
        SearchPermutation = index;
    }
    void CopyPermutationState(uint32_t capacity, float* cursors, float* lambdas, uint8_t* valid) const {
        RequireCompletedState();
        Require(!Failed && capacity >= PermutationCount() && cursors && lambdas && valid,
                "Permutation state output capacity or buffers are invalid");
        for (uint32_t permutation = 0; permutation < PermutationCount(); ++permutation) {
            const auto cursor = Permutations.empty() ? Prediction : Permutations[permutation].Cursor;
            const float* values = static_cast<const float*>(cursor.contents);
            for (uint32_t row = 0; row < K.Rows; ++row) Require(std::isfinite(values[row]), "Non-finite permutation cursor");
            std::memcpy(cursors + uint64_t(permutation) * K.Rows, values, 4ull * K.Rows);
            valid[permutation] = Permutations.empty() ? HasMVSLambda : Permutations[permutation].HasLambda;
            lambdas[permutation] = valid[permutation]
                ? (Permutations.empty() ? MVSLambda : Permutations[permutation].Lambda) : 0;
        }
    }
    void CopyBootstrapState(uint32_t* absolute, float* lambda, uint32_t* valid) const {
        RequireCompletedState();
        Require(absolute && lambda && valid, "Bootstrap state output buffers are required");
        *absolute = BootstrapOptions.iteration_offset + Completed;
        *valid = HasMVSLambda;
        *lambda = HasMVSLambda ? MVSLambda : 0;
    }
    void ConfigureFeaturePenalties(const CBMFeaturePenaltyOptions* options, const uint32_t* counts,
                                   const float* weights, const uint8_t* used) {
        RequireCompletedState();
        Require(Completed == 0 && !Failed && !FeaturePenaltiesConfigured,
                "Feature penalties can be configured once before training");
        Require(options && counts && !options->reserved0 && !options->reserved1 && !options->reserved2,
                "Feature penalty options and CTR unique counts are required");
        Require(std::isfinite(options->model_size_reg) && options->model_size_reg >= 0,
                "model_size_reg must be finite and nonnegative");
        for (uint32_t feature = 0; feature < K.Features; ++feature) {
            Require(!weights || (std::isfinite(weights[feature]) && weights[feature] >= 0),
                    "Feature weights must be finite and nonnegative");
            Require(!used || used[feature] <= 1, "Used-feature flags must be boolean");
        }
        ModelSizeReg = options->model_size_reg;
        std::copy_n(counts, K.Features, CtrUniqueValues.begin());
        for (uint32_t feature = 0; feature < K.Features; ++feature) {
            BinFeatureWeights[feature] = weights ? weights[feature] : 1.0f;
            UsedFeatures[feature] = counts[feature] && used ? used[feature] : 0;
            if (UsedFeatures[feature]) FeatureFlags[feature] |= 2;
        }
        FeaturePenaltiesConfigured = true;
        UpdateFeaturePenalties();
    }
    void CopyFeaturePenaltyState(uint8_t* used) const {
        RequireCompletedState();
        Require(!Failed && used, "Feature penalty state output is required");
        std::copy(UsedFeatures.begin(), UsedFeatures.end(), used);
    }
    void SetFeatureSamplingMask(uint32_t featureCount, const uint8_t* active) {
        Require(!Failed && Coupled, "Feature sampling masks require a valid full-matrix training session");
        Require(!Pending.Active || (Pending.Depth == 0 && !Pending.SplitPending),
                "Feature sampling masks must be set before the first split of a tree");
        Require(featureCount == K.Features && active, "Feature sampling mask size must match the training features");
        for (uint32_t feature = 0; feature < featureCount; ++feature)
            Require(active[feature] <= 1, "Feature sampling mask entries must be boolean");
        const auto* features = static_cast<const uint32_t*>(CandidateFeature.contents);
        bool any = false;
        for (uint32_t candidate = 0; candidate < K.Candidates; ++candidate) any |= active[features[candidate]] != 0;
        Require(!K.Candidates || any, "Feature sampling must retain at least one split candidate");
        auto* mask = static_cast<uint8_t*>(CandidateActive.contents);
        for (uint32_t candidate = 0; candidate < K.Candidates; ++candidate) mask[candidate] = active[features[candidate]];
    }
    void AppendFeatures(const CBMAppendFeatureOptions* options, const uint8_t* const* matrices,
                        const uint32_t* features, const uint32_t* bins, const uint8_t* types,
                        const uint32_t* counts, const float* weights, const uint8_t* flags,
                        const uint8_t* used, uint32_t* first, bool enableOnly = false) {
        Require(!Coupled, "Full-matrix pairwise dynamic feature banks are not yet connected");
        Require(!Failed, "This training session failed and must be closed");
        Require(options && first && options->permutation_count == PermutationCount(),
                "Append options, output and matching permutation count are required");
        for (uint32_t value : options->reserved) Require(value == 0, "Reserved append options must be zero");
        Require(enableOnly || options->features > 0, "At least one new feature is required");
        Require(options->bins_per_feature >= K.Bins && options->bins_per_feature <= 256,
                "Appended global bin capacity must cover the existing grid and not exceed 256");
        const uint64_t fullFeatures = uint64_t(K.Features) + options->features;
        const uint64_t fullCandidates = uint64_t(K.Candidates) + options->candidates;
        Require(fullFeatures < std::numeric_limits<uint32_t>::max()
                && fullCandidates <= std::numeric_limits<uint32_t>::max(), "Appended feature/candidate counts overflow");
        const uint64_t cells = CheckedProduct(fullFeatures, K.Rows, "Appended feature bank");
        Require(cells <= std::numeric_limits<uint32_t>::max() && cells <= MaxWorkingBytes,
                "Appended feature bank exceeds the GPU index or memory limit");
        Require(uint64_t(PermutationCount()) * cells + 10ull * fullCandidates + 21ull * fullFeatures <= MaxWorkingBytes,
                "Appended data and metadata exceed the 1 GiB GPU memory limit");
        Require(!options->features || matrices, "Every appended permutation bank is required");
        Require(!options->candidates || (features && bins), "Appended candidate vectors are required");
        Require(options->candidates <= uint64_t(options->features) * options->bins_per_feature,
                "Appended candidate count exceeds feature/bin combinations");
        const uint64_t addedCells = uint64_t(options->features) * K.Rows;
        for (uint32_t permutation = 0; options->features && permutation < PermutationCount(); ++permutation) {
            Require(matrices[permutation] != nullptr, "Every appended permutation bank is required");
            for (uint64_t cell = 0; cell < addedCells; ++cell)
                Require(matrices[permutation][cell] < options->bins_per_feature, "Appended bin exceeds the new grid");
        }
        const uint32_t newF = static_cast<uint32_t>(fullFeatures), newC = static_cast<uint32_t>(fullCandidates);
        std::vector<uint8_t> newFeatureTypes(newF, 0), seen(options->features, 0);
        std::memcpy(newFeatureTypes.data(), FeatureType.contents, K.Features);
        for (uint32_t candidate = 0; candidate < options->candidates; ++candidate) {
            const uint32_t feature = features[candidate], bin = bins[candidate];
            const uint8_t type = types ? types[candidate] : 0;
            Require(feature < options->features && type <= 1, "Invalid appended candidate feature or type");
            Require(bin < options->bins_per_feature && (type == 1 || bin + 1 < options->bins_per_feature),
                    "Invalid appended candidate border");
            Require(!seen[feature] || newFeatureTypes[K.Features + feature] == type,
                    "A feature cannot mix numeric and one-hot candidate types");
            newFeatureTypes[K.Features + feature] = type; seen[feature] = 1;
        }
        for (uint32_t feature = 0; feature < options->features; ++feature) {
            Require(!weights || (std::isfinite(weights[feature]) && weights[feature] >= 0),
                    "Appended feature weights must be finite and nonnegative");
            Require(!flags || flags[feature] <= 3, "Appended feature flags must use only dynamic/registered bits");
            Require(!used || used[feature] <= 1, "Appended used-feature flags must be boolean");
            Require(!counts || !counts[feature] || !used || !used[feature] || !flags || (flags[feature] & 2),
                    "A previously used CTR must be globally registered");
        }
        auto newCounts = CtrUniqueValues;
        auto newFeatureWeights = BinFeatureWeights;
        auto newUsed = UsedFeatures, newFlags = FeatureFlags, newFeatureActivity = ActiveFeatures;
        newCounts.resize(newF, 0); newFeatureWeights.resize(newF, 1);
        newUsed.resize(newF, 0); newFlags.resize(newF, 3); newFeatureActivity.resize(newF, 1);
        for (uint32_t feature = 0; feature < options->features; ++feature) {
            const uint32_t global = K.Features + feature;
            newCounts[global] = counts ? counts[feature] : 0;
            newFeatureWeights[global] = weights ? weights[feature] : 1;
            newFlags[global] = flags ? flags[feature] : 3;
            newUsed[global] = counts && counts[feature] && used ? used[feature] : 0;
        }
        std::vector<uint32_t> newFeatures(newC), newBins(newC), offsets(newF + 1, 0);
        std::vector<uint8_t> newTypes(newC), newActive(newC, 1);
        if (K.Candidates) {
            std::memcpy(newFeatures.data(), CandidateFeature.contents, 4ull * K.Candidates);
            std::memcpy(newBins.data(), CandidateBin.contents, 4ull * K.Candidates);
            std::memcpy(newTypes.data(), CandidateType.contents, K.Candidates);
        }
        for (uint32_t candidate = 0; candidate < options->candidates; ++candidate) {
            newFeatures[K.Candidates + candidate] = K.Features + features[candidate];
            newBins[K.Candidates + candidate] = bins[candidate];
            newTypes[K.Candidates + candidate] = types ? types[candidate] : 0;
        }
        for (uint32_t candidate = 0; candidate < newC; ++candidate) {
            auto& span = offsets[newFeatures[candidate] + 1];
            span = std::max(span, newBins[candidate] + 1);
            if (newFeatures[candidate] < K.Features) newActive[candidate] = ActiveFeatures[newFeatures[candidate]];
        }
        uint64_t totalBins = 0;
        for (uint32_t feature = 0; feature < newF; ++feature) {
            totalBins += offsets[feature + 1];
            Require(totalBins <= std::numeric_limits<uint32_t>::max(), "Appended compact grid overflows");
            offsets[feature + 1] = static_cast<uint32_t>(totalBins);
        }
        const uint32_t scoreGroups = std::max<uint32_t>(1, std::min<uint32_t>((uint64_t(newC) + 255) / 256, 64));
        const uint64_t jobBytes = Compact ? 0 : 16ull * HistogramJobCapacity + 4ull * MaxLeaves + 52;
        const uint64_t replaceData = options->features ? uint64_t(PermutationCount()) * cells : 0;
        // Replacements are allocated transactionally while the previous banks,
        // histograms and metadata still exist. Include that overlap, not just
        // the steady-state footprint, before allocating anything on the GPU.
        const uint64_t newNonHistogram = replaceData + 10ull * newC + 13ull * newF
            + 4ull * (newF + 1) + 8ull * newF + 32 + 32ull * scoreGroups + jobBytes;
        const uint64_t currentPeak = WorkspaceBytes();
        Require(currentPeak + newNonHistogram + 8ull * MaxLeaves <= MaxWorkingBytes,
                "Dynamic append peak exceeds the 1 GiB GPU memory limit");
        const uint64_t histogramBudget = std::min<uint64_t>(uint64_t(256) << 20,
            MaxWorkingBytes - currentPeak - newNonHistogram);
        const uint64_t capacityBins = histogramBudget / (8ull * MaxLeaves);
        std::vector<HistogramTile> tiles;
        uint32_t begin = 0, tileBins = 0, maxTileBins = 0;
        for (uint32_t feature = 0; feature < newF; ++feature) {
            const uint32_t span = offsets[feature + 1] - offsets[feature];
            Require(span <= capacityBins, "One appended feature cannot fit the bounded histogram workspace");
            if (tileBins && uint64_t(tileBins) + span > capacityBins) {
                tiles.push_back({begin, feature, tileBins, nil});
                maxTileBins = std::max(maxTileBins, tileBins); begin = feature; tileBins = 0;
            }
            tileBins += span;
        }
        tiles.push_back({begin, newF, tileBins, nil}); maxTileBins = std::max(maxTileBins, tileBins);
        const uint64_t histogramCells = uint64_t(MaxLeaves) * std::max<uint32_t>(maxTileBins, 1);
        const uint64_t oldLayout = DataCells + 9ull * K.Candidates + 13ull * K.Features
            + 4ull * (K.Features + 1) + 8ull * HistogramCells + 32ull * K.ScoreGroups
            + (Compact ? 8ull * K.Features + 32 : 0) + (Dynamic ? K.Candidates : 0);
        const uint64_t newLayout = cells + 10ull * newC + 13ull * newF + 4ull * (newF + 1)
            + 8ull * histogramCells + 32ull * scoreGroups + 8ull * newF + 32;
        std::vector<id<MTLBuffer>> newData;
        if (options->features) {
            newData.resize(PermutationCount());
            for (uint32_t permutation = 0; permutation < PermutationCount(); ++permutation) {
                newData[permutation] = Context->Buffer(cells);
                std::memcpy(static_cast<uint8_t*>(newData[permutation].contents) + DataCells,
                            matrices[permutation], addedCells);
            }
        }
        auto candidateFeature = Context->Buffer(4ull * newC, newFeatures.data());
        auto candidateBin = Context->Buffer(4ull * newC, newBins.data());
        auto candidateType = Context->Buffer(newC, newTypes.data());
        auto candidateActive = Context->Buffer(newC, newActive.data());
        auto featureType = Context->Buffer(newF, newFeatureTypes.data());
        auto featureOffsets = Context->Buffer(4ull * (newF + 1), offsets.data());
        auto penalties = Context->Buffer(8ull * newF);
        auto noise = Context->Buffer(4ull * newF);
        std::fill_n(static_cast<float*>(noise.contents), newF, 0.0f);
        auto winnerPartials = Context->Buffer(32ull * scoreGroups);
        auto tileWinner = Context->Buffer(sizeof(SplitState));
        auto histogramSums = Context->Buffer(4ull * histogramCells);
        auto histogramWeights = Context->Buffer(4ull * histogramCells);
        auto jobs = Compact ? HistogramJobs : Context->Buffer(16ull * HistogramJobCapacity);
        auto active = Compact ? HistogramActive : Context->Buffer(4ull * MaxLeaves);
        auto state = Compact ? HistogramState : Context->Buffer(16);
        auto arguments = Compact ? HistogramArguments : Context->Buffer(36);
        for (auto& tile : tiles) {
            std::vector<uint32_t> local(tile.End - tile.Begin + 1);
            for (uint32_t feature = tile.Begin; feature <= tile.End; ++feature)
                local[feature - tile.Begin] = offsets[feature] - offsets[tile.Begin];
            tile.Offsets = Context->Buffer(4ull * local.size(), local.data());
        }
        if (!newData.empty()) {
            Command command(*Context, Stats);
            id<MTLBlitCommandEncoder> encoder = [command.Buffer blitCommandEncoder];
            Require(encoder != nil, "Could not create feature-bank copy encoder");
            for (uint32_t permutation = 0; permutation < PermutationCount(); ++permutation)
                [encoder copyFromBuffer:Permutations.empty() ? Data : Permutations[permutation].Bins
                           sourceOffset:0 toBuffer:newData[permutation] destinationOffset:0 size:DataCells];
            [encoder endEncoding];
            try { command.Wait(); } catch (...) { Failed = true; throw; }
        }
        // Only completed copies are published; caller mistakes and allocation
        // failures above leave all existing tree/search state usable.
        *first = K.Features;
        if (!newData.empty()) {
            if (Permutations.empty()) Data = newData[0];
            else {
                for (uint32_t permutation = 0; permutation < PermutationCount(); ++permutation)
                    Permutations[permutation].Bins = newData[permutation];
                Data = Permutations[Pending.Active ? SearchPermutation : PermutationCount() - 1].Bins;
            }
        }
        WorkingBytes = WorkingBytes - oldLayout + newLayout + jobBytes;
        DynamicPeakBytes = std::max(DynamicPeakBytes, currentPeak + newNonHistogram + 8ull * histogramCells);
        PermutationBytes += uint64_t(PermutationCount() - 1) * (cells - DataCells);
        DataCells = cells; HistogramCells = histogramCells; CompactBins = static_cast<uint32_t>(totalBins);
        K.Features = Options.train.features = newF; K.Candidates = Options.train.candidates = newC;
        K.Bins = Options.train.bins_per_feature = options->bins_per_feature; K.ScoreGroups = scoreGroups;
        CandidateFeature = candidateFeature; CandidateBin = candidateBin; CandidateType = candidateType;
        CandidateActive = candidateActive; FeatureType = featureType; FeatureOffsets = featureOffsets;
        FeaturePenaltyWeights = penalties; FeatureNoise = noise; WinnerPartials = winnerPartials; TileWinner = tileWinner;
        HistogramSums = histogramSums; HistogramWeights = histogramWeights; HistogramTiles = std::move(tiles);
        HistogramJobs = jobs; HistogramActive = active; HistogramState = state; HistogramArguments = arguments;
        CtrUniqueValues = std::move(newCounts); BinFeatureWeights = std::move(newFeatureWeights);
        UsedFeatures = std::move(newUsed); FeatureFlags = std::move(newFlags); ActiveFeatures = std::move(newFeatureActivity);
        Dynamic = Compact = true;
        // FeatureParallel still estimates Simple leaves through its one-step
        // Gradient walker; DocParallel exports sampled weak statistics.
        if (K.LeafMethod == 3) K.LeafMethod = 1;
        Pending.HistogramValid = false;
        UpdateFeaturePenalties();
        ReopenCandidateExhaustion();
    }
    void SetFeatureActivity(uint32_t count, const uint8_t* active) {
        Require(!Failed && count == K.Features && active, "Activity must cover every current feature");
        for (uint32_t feature = 0; feature < count; ++feature)
            Require(active[feature] <= 1, "Feature activity must be boolean");
        if (!Dynamic) {
            const CBMAppendFeatureOptions options = {PermutationCount(), 0, 0, K.Bins, {0, 0, 0, 0}};
            uint32_t first;
            AppendFeatures(&options, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &first, true);
        }
        std::copy_n(active, count, ActiveFeatures.begin());
        uint8_t* enabled = static_cast<uint8_t*>(CandidateActive.contents);
        const uint32_t* features = static_cast<const uint32_t*>(CandidateFeature.contents);
        for (uint32_t candidate = 0; candidate < K.Candidates; ++candidate) enabled[candidate] = active[features[candidate]];
        UpdateFeaturePenalties(); ReopenCandidateExhaustion();
    }
    void CopyFeatureMetadata(uint32_t capacity, uint32_t* counts, float* weights,
                             uint8_t* flags, uint8_t* used, uint8_t* active) const {
        RequireCompletedState();
        Require(!Failed && capacity >= K.Features && counts && weights && flags && used && active,
                "Feature metadata output capacity and vectors are required");
        std::copy(CtrUniqueValues.begin(), CtrUniqueValues.end(), counts);
        std::copy(BinFeatureWeights.begin(), BinFeatureWeights.end(), weights);
        std::copy(FeatureFlags.begin(), FeatureFlags.end(), flags);
        std::copy(UsedFeatures.begin(), UsedFeatures.end(), used);
        std::copy(ActiveFeatures.begin(), ActiveFeatures.end(), active);
    }
    void RestoreFeatureMetadata(uint32_t count, const uint8_t* flags,
                                const uint8_t* used, const uint8_t* active) {
        RequireCompletedState();
        Require(Completed == 0 && !Failed, "Feature metadata restoration must precede the first training step");
        Require(count == K.Features && flags && used && active,
                "Feature metadata restoration must cover every current feature");
        for (uint32_t feature = 0; feature < count; ++feature) {
            Require(flags[feature] <= 3, "Feature flags must use only dynamic/registered bits");
            Require(used[feature] <= 1 && active[feature] <= 1, "Used-feature and activity flags must be boolean");
            Require(!used[feature] || (CtrUniqueValues[feature] && (flags[feature] & 2)),
                    "A previously used feature must be a globally registered CTR");
        }
        // Validate all metadata before enabling the replacement compact layout.
        // Allocation failures leave the existing metadata and banks untouched.
        SetFeatureActivity(count, active);
        std::copy_n(flags, count, FeatureFlags.begin());
        std::copy_n(used, count, UsedFeatures.begin());
        UpdateFeaturePenalties();
    }

    void CopyWorkspaceInfo(uint32_t* tiles, uint64_t* histogramBytes, uint64_t* peakBytes) const {
        Require(tiles && histogramBytes && peakBytes, "Workspace output pointers are required");
        *tiles = Compact ? static_cast<uint32_t>(HistogramTiles.size()) : 1;
        *histogramBytes = 8ull * HistogramCells;
        *peakBytes = std::max(WorkspaceBytes(), DynamicPeakBytes);
    }
    void CopyPredictions(float* output) const {
        RequireCompletedState();
        Require(!Failed && output, "An open training session and prediction output buffer are required");
        const float* cursor = static_cast<const float*>(Prediction.contents);
        for (uint32_t row = 0; row < K.Rows; ++row)
            Require(std::isfinite(cursor[row]), "Non-finite GPU predictions");
        std::memcpy(output, cursor, 4ull * K.Rows);
    }
    void CopyStep(CBMStepInfo* info, uint32_t* depth, uint32_t* features, uint32_t* bins,
                  uint8_t* types, float* values, float* weights) const {
        const auto& p = Options.train;
        const uint32_t tree = Completed - 1;
        *depth = TreeDepths[tree];
        *info = {Completed, uint32_t(Completed == p.iterations), Losses[Completed], 0, Stats};
        if (p.depth) {
            std::memcpy(features, SplitFeatures.data() + uint64_t(tree) * p.depth, 4ull * p.depth);
            std::memcpy(bins, SplitBins.data() + uint64_t(tree) * p.depth, 4ull * p.depth);
            if (types) std::memcpy(types, SplitTypes.data() + uint64_t(tree) * p.depth, p.depth);
        }
        if (Deep) {
            std::fill_n(values, MaxLeaves, 0.0f); std::fill_n(weights, MaxLeaves, 0.0f);
            std::memcpy(values, DeepTreeValues[tree].data(), 4ull * DeepTreeValues[tree].size());
            std::memcpy(weights, DeepTreeWeights[tree].data(), 4ull * DeepTreeWeights[tree].size());
        } else {
            std::memcpy(values, TreeValues.data() + uint64_t(tree) * MaxLeaves, 4ull * MaxLeaves);
            std::memcpy(weights, TreeWeights.data() + uint64_t(tree) * MaxLeaves, 4ull * MaxLeaves);
        }
    }
    void CopyResult(uint32_t capacity, uint32_t* completed, uint32_t* depths,
                    uint32_t* features, uint32_t* bins, uint8_t* types,
                    float* values, float* weights, float* predictions, float* loss,
                    CBMTrainStats* stats) const {
        RequireCompletedState();
        Require(!Failed, "This training session failed and must be closed");
        Require(completed && predictions && loss, "Result count, predictions and loss buffers are required");
        Require(capacity >= Completed, "Result capacity is smaller than the completed tree count");
        const auto& p = Options.train;
        Require(Completed == 0 || (depths && values && weights), "Tree result buffers are required");
        Require(Completed == 0 || p.depth == 0 || (features && bins), "Split result buffers are required");
        const float* cursor = static_cast<const float*>(Prediction.contents);
        for (uint32_t row = 0; row < p.rows; ++row)
            Require(std::isfinite(cursor[row]), "Non-finite GPU predictions");
        if (Completed) {
            std::memcpy(depths, TreeDepths.data(), 4ull * Completed);
            if (p.depth) {
                std::memcpy(features, SplitFeatures.data(), 4ull * Completed * p.depth);
                std::memcpy(bins, SplitBins.data(), 4ull * Completed * p.depth);
                if (types) std::memcpy(types, SplitTypes.data(), uint64_t(Completed) * p.depth);
            }
            Require(8ull * Completed * MaxLeaves <= MaxOutputBytes, "Padded result exceeds the 512 MiB limit; use step outputs");
            if (Deep) {
                std::fill_n(values, uint64_t(Completed) * MaxLeaves, 0.0f);
                std::fill_n(weights, uint64_t(Completed) * MaxLeaves, 0.0f);
                for (uint32_t tree = 0; tree < Completed; ++tree) {
                    std::memcpy(values + uint64_t(tree) * MaxLeaves, DeepTreeValues[tree].data(), 4ull * DeepTreeValues[tree].size());
                    std::memcpy(weights + uint64_t(tree) * MaxLeaves, DeepTreeWeights[tree].data(), 4ull * DeepTreeWeights[tree].size());
                }
            } else {
                std::memcpy(values, TreeValues.data(), 4ull * Completed * MaxLeaves);
                std::memcpy(weights, TreeWeights.data(), 4ull * Completed * MaxLeaves);
            }
        }
        std::memcpy(predictions, cursor, 4ull * p.rows);
        std::memcpy(loss, Losses.data(), 4ull * (Completed + 1));
        *completed = Completed;
        if (stats) *stats = Stats;
    }
private:
    std::unique_ptr<Runtime> OwnedContext;
    Runtime* Context = nullptr;
    KernelParams K = {};
    uint64_t DataCells = 0, HistogramCells = 0;
    uint64_t WorkingBytes = 0;
    uint32_t MaxLeaves = 0, LossGroups = 0;
    double TotalWeight = 0;
    bool Failed = false, Deep = false, Compact = false, Dynamic = false;
    bool SimpleLeavesRequested = false;
    uint64_t DynamicPeakBytes = 0;
    uint64_t WorkspaceBytes() const {
        const uint64_t bootstrapBytes = !Coupled && BootstrapOptions.bootstrap_type
            ? 8ull * K.Rows + 8ull * LossGroups + 4ull * ((K.Rows + 8191) / 8192) : 0;
        return WorkingBytes + ExactBytes + PermutationBytes + LangevinBytes + bootstrapBytes + (NoiseStatistics ? 4ull * LossGroups : 0);
    }
    uint32_t CompactBins = 0, DeepTiles = 0, DeepBlocks = 0, HistogramJobCapacity = 0;
    struct HistogramTile { uint32_t Begin, End, Bins; id<MTLBuffer> Offsets; };
    std::vector<HistogramTile> HistogramTiles;
    id<MTLBuffer> TileWinner;
    uint64_t DeepModelBytes = 0;
    std::vector<std::vector<float>> DeepTreeValues, DeepTreeWeights;
    id<MTLBuffer> FeatureOffsets, DeepBlockPrefix, NextPartitionOffsets;
    id<MTLBuffer> HistogramJobs, HistogramActive, HistogramState, HistogramArguments;
    id<MTLBuffer> Data, Target, Prediction, SampleWeight, Gradient, Hessian, LeafIds;
    id<MTLBuffer> RowIndices, NextRowIndices, RowRanks, PartitionTiles, PartitionOffsets;
    id<MTLBuffer> CandidateFeature, CandidateBin, CandidateType, FeatureType, FeatureNoise, NoiseStatistics;
    float RandomStrength = 0;
    bool Langevin = false, WeakLangevin = false;
    float LangevinTemperature = 0;
    uint64_t LangevinBytes = 0;
    CBMLangevinNoiseCallback LangevinNoise = nullptr;
    CBMLangevinSeedCallback LangevinSeed = nullptr;
    void* LangevinContext = nullptr;
    uint64_t ReadLangevinSeed(uint32_t event) {
        uint64_t seed = 0;
        Require(LangevinSeed && LangevinSeed(LangevinContext, event, &seed) == 0,
                "Langevin seed callback failed");
        return seed;
    }
    bool LangevinLeaves() const { return Langevin && K.LeafMethod != 2 && K.LeafMethod != 3; }
    void EncodeLangevinWeak(Command& command) {
        if (!Langevin || !WeakLangevin) return;
        struct WeakParams { NativeBootstrapParams Random; uint32_t Offset, Stride, Filter, Reserved; } p = {};
        p.Random = BootstrapParams(); p.Random.Stream = 0x4c470001u;
        p.Random.NoiseScale = LangevinTemperature > 0
            ? static_cast<float>(std::sqrt(2.0 / double(K.LearningRate) / double(LangevinTemperature))) : 0;
        p.Stride = 1; p.Filter = BootstrapOptions.bootstrap_type == 2 || BootstrapOptions.bootstrap_type == 3;
        command.Dispatch("AddLangevinWeakNoise", {Gradient, p.Filter ? BootstrapMultipliers : SampleWeight}, p, K.Rows);
    }
    CBMRegularizationOptions Regularization = {0, 0, 0, 0, 1, 0};
    CBMMetaL2ExponentCallback MetaL2Callback = nullptr;
    void* MetaL2Context = nullptr;
    bool RegularizedLeaves() const { return Regularization.normalize_leaf || Regularization.add_ridge; }
    NativeBacktrackingParams BacktrackingOptions() const {
        return {1, Options.leaf_estimation_backtracking, Regularization.add_ridge, Regularization.normalize_leaf};
    }
    void EncodeNewtonLeaves(Command& command) {
        if (!RegularizedLeaves()) {
            command.Dispatch("EstimateNewtonLeafValues", {ObjectivePartials, RawValues, Weights}, K, K.Leaves, true);
        } else {
            const auto b = BacktrackingOptions();
            command.Dispatch("EstimateRegularizedNewtonLeafValues", {ObjectivePartials, RawValues, Weights},
                K, K.Leaves, true, 1, 1, &b, sizeof(b));
        }
    }
    double RegularizedObjective(double value, id<MTLBuffer> values) const {
        if (Regularization.normalize_leaf) value /= TotalWeight;
        if (Regularization.add_ridge) {
            const float* point = static_cast<const float*>(values.contents);
            for (uint32_t leaf = 0; leaf < K.Leaves; ++leaf)
                value -= 0.5 * double(K.L2) * double(point[leaf]) * point[leaf];
        }
        return value;
    }
    NativeScoreRegularizationParams ScoreRegularization() const {
        float exponent = 1; uint32_t perFeature = 0;
        if ((K.ScoreFunction == 0 || K.ScoreFunction == 2) &&
            Regularization.meta_l2_exponent != 1 && Regularization.meta_l2_frequency > 0) {
            if (Regularization.meta_l2_frequency > 1) exponent = Regularization.meta_l2_exponent;
            else {
                Require(MetaL2Callback, "Fractional MetaL2 requires a per-feature exponent callback");
                std::vector<uint8_t> choices(K.Features, 1);
                Require(MetaL2Callback(MetaL2Context, K.Features, choices.data()) == 0,
                        "MetaL2 exponent callback failed");
                auto* output = static_cast<float*>(FeatureNoise.contents);
                for (uint32_t feature = 0; feature < K.Features; ++feature) {
                    Require(choices[feature] >= 1 && choices[feature] <= 3,
                            "MetaL2 callback returned invalid exponent choices");
                    output[feature] = choices[feature];
                }
                // Scalar L2 scores never use feature noise. Reuse that buffer
                // for the union of choices from CUDA dataset/policy seeds.
                exponent = Regularization.meta_l2_exponent;
                perFeature = 1;
            }
        }
        return {Regularization.normalize_score, exponent, perFeature, 0};
    }

    id<MTLBuffer> FeaturePenaltyWeights, CandidateActive;
    std::vector<uint32_t> CtrUniqueValues;
    std::vector<uint8_t> UsedFeatures, FeatureFlags, ActiveFeatures;
    std::vector<float> BinFeatureWeights;
    float ModelSizeReg = 0;
    bool FeaturePenaltiesConfigured = false;
    void UpdateFeaturePenalties() {
        if (!FeaturePenaltiesConfigured && !Dynamic) return;
        // Static CUDA update_feature_weights.cpp scans globally registered,
        // unused CTRs. Tree-CTR weights instead use active unused dynamic CTRs
        // and still penalize a dynamic feature after it has been selected.
        uint32_t dynamicMaximum = 1, staticMaximum = 1;
        if (Dynamic) for (uint32_t feature = 0; feature < K.Features; ++feature)
            if ((FeatureFlags[feature] & 1) && ActiveFeatures[feature] && !UsedFeatures[feature])
                dynamicMaximum = std::max(dynamicMaximum, CtrUniqueValues[feature]);
        staticMaximum = dynamicMaximum;
        for (uint32_t feature = 0; feature < K.Features; ++feature)
            if ((!Dynamic || (FeatureFlags[feature] & 2)) && !UsedFeatures[feature])
                staticMaximum = std::max(staticMaximum, CtrUniqueValues[feature]);
        float* output = static_cast<float*>(FeaturePenaltyWeights.contents);
        for (uint32_t feature = 0; feature < K.Features; ++feature) {
            const bool dynamic = Dynamic && (FeatureFlags[feature] & 1);
            const uint32_t maximum = dynamic ? dynamicMaximum : staticMaximum;
            output[2 * feature] = CtrUniqueValues[feature] && (dynamic || !UsedFeatures[feature])
                ? static_cast<float>(std::pow(1.0f + float(CtrUniqueValues[feature]) / float(maximum), -double(ModelSizeReg)))
                : 1.0f;
            output[2 * feature + 1] = BinFeatureWeights[feature];
        }
    }
    bool HasActiveCandidates() const {
        if (!Dynamic) return K.Candidates != 0;
        const uint8_t* active = static_cast<const uint8_t*>(CandidateActive.contents);
        for (uint32_t candidate = 0; candidate < K.Candidates; ++candidate) if (active[candidate]) return true;
        return false;
    }
    void ReopenCandidateExhaustion() {
        if (Pending.Active && Pending.Exhausted && Pending.Depth < Options.train.depth && HasActiveCandidates()) {
            Pending.Finished = Pending.Exhausted = false;
        }
    }

    id<MTLBuffer> HistogramSums, HistogramWeights, LeafSums, LeafWeights;
    id<MTLBuffer> RawValues, Values, Weights, ObjectivePartials, WinnerPartials, Winner, LossPartials;
    id<MTLBuffer> StructureWeight, BootstrapMultipliers, BootstrapStatistics, MVSThresholds;
    id<MTLBuffer> TrialValues, Directions, DirectionDot, BacktrackingLoss;
    CBMQueryOptions QueryOptions = {};
    std::shared_ptr<CBMPairwiseRuntime> Pairwise;
    std::shared_ptr<CBMYetiRankRuntime> Yeti;
    std::shared_ptr<CBMCombinationRuntime> Combination;
    id<MTLBuffer> CombinationGradientWeights;
    std::shared_ptr<CBMFullMatrixRuntime> Coupled;
    float CoupledNonDiag = 0.1f;
    std::vector<uint64_t> YetiSeeds;
    uint32_t YetiSeedPosition = 0;
    id<MTLBuffer> QueryOffsets, QueryPoint, QueryStatistics, QueryLossPartials, QueryValidation;
    id<MTLBuffer> ExactResiduals, ExactEffectiveWeights, ExactKeysA, ExactRowsA, ExactKeysB, ExactRowsB;
    id<MTLBuffer> ExactTileOffsets, ExactTotals, ExactSelected;
    uint64_t ExactBytes = 0, PermutationBytes = 0;
    struct PendingTree {
        bool Active = false, Finished = false, HasSplit = false, Exhausted = false, HistogramValid = false;
        bool Initialize = true, SplitPending = false, PartitionsValid = false, ScoreNoise = false;
        bool DerivativesReady = false;
        uint32_t Depth = 0;
        float NoiseScale = 0;
        SplitState LastWinner = {0xffffffffu, 0xffffffffu, 0, 0, 0, 0, 0, 0};
        std::vector<SplitState> Selected;
    } Pending;
    struct PermutationData { id<MTLBuffer> Bins, Cursor; float Lambda = 0; bool HasLambda = false; };
    struct PermutationTree { std::vector<float> Values, Weights; float Loss; };
    std::vector<PermutationData> Permutations;
    uint32_t SearchPermutation = 0;
    id<MTLBuffer> FixedPermutationSplits;
    CBMBootstrapOptions BootstrapOptions = {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0};
    float MVSLambda = 0;
    bool HasMVSLambda = false;
    std::vector<uint32_t> TreeDepths, SplitFeatures, SplitBins;
    std::vector<uint8_t> SplitTypes;
    std::vector<float> TreeValues, TreeWeights, Losses;

    bool UsesBacktracking() const {
        return Options.leaf_estimation_backtracking != 0 && Options.leaf_estimation_iterations > 1 && K.LeafMethod != 2;
    }
    uint32_t YetiLeafSeedCount() const {
        return PermutationCount() * YetiPermutationLeafSeedCount();
    }
    uint32_t YetiPermutationLeafSeedCount() const {
        return Options.leaf_estimation_iterations + uint32_t(Options.leaf_estimation_iterations > 1);
    }
    void ReorderFeatureParallelYetiSeeds() {
        // FeatureParallel's batch estimator evaluates each history once per
        // walker iteration. DocParallel retains its dataset-major packet ABI.
        const auto source = YetiSeeds;
        const uint32_t evaluations = YetiPermutationLeafSeedCount(), histories = PermutationCount();
        for (uint32_t history = 0; history < histories; ++history)
            for (uint32_t evaluation = 0; evaluation < evaluations; ++evaluation)
                YetiSeeds[1 + history * evaluations + evaluation] = source[1 + evaluation * histories + history];
    }
    double ReadBacktrackingScalar(id<MTLBuffer> buffer, uint32_t count) const {
        const float* parts = static_cast<const float*>(buffer.contents);
        double value = 0;
        for (uint32_t i = 0; i < count; ++i) value += double(parts[2 * i]) + parts[2 * i + 1];
        return value;
    }
    NativeQueryParams QueryParams(bool applyLeafValues, bool structure = false) const {
        return {K.Rows, QueryOptions.group_count, K.Objective, uint32_t(applyLeafValues),
                QueryOptions.beta, QueryOptions.lambda, K.Leaves,
                uint32_t(structure && !SimpleLeavesRequested &&
                    (K.ScoreFunction == 2 || K.ScoreFunction == 3))};
    }
    void EncodeQueryPoint(Command& command, id<MTLBuffer> leafValues, bool applyLeafValues,
                          bool structure = false) {
        const auto q = QueryParams(applyLeafValues, structure);
        command.Dispatch("PrepareQuerywisePoint", {Prediction, leafValues, LeafIds, QueryPoint}, q, K.Rows);
        command.Dispatch(K.Objective == 12 ? "QueryRmseDerivatives" : "QuerySoftMaxDerivatives",
            {Target, SampleWeight, QueryPoint, QueryOffsets, Gradient, Hessian, QueryStatistics},
            q, q.Groups, true);
    }
    void EncodeDerivatives(Command& command) {
        if (Combination) {
            command.Dispatch("ResetQuerywiseLeafIds", {LeafIds}, QueryParams(false), K.Rows);
            Combination->EncodePointDerivatives(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves,
                false, Target, SampleWeight, Gradient, Hessian, CombinationGradientWeights, &Stats.kernel_dispatches);
            return;
        }
        if (Coupled) {
            command.Dispatch("ResetQuerywiseLeafIds", {LeafIds}, QueryParams(false), K.Rows);
            Coupled->SetTargetPoint(Prediction, BootstrapOptions, BootstrapOptions.iteration_offset + Completed, SearchPermutation);
            Coupled->EncodeEdges(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves, false,
                &BootstrapOptions, BootstrapOptions.iteration_offset + Completed, &Stats.kernel_dispatches);
            Stats.gpu_seconds += Coupled->TakeAuxiliaryGPUSeconds();
            return;
        }
        if (Yeti) {
            command.Dispatch("ResetQuerywiseLeafIds", {LeafIds}, QueryParams(false), K.Rows);
            EncodeYetiPoint(command, false);
            return;
        }
        if (Pairwise) {
            const auto q = QueryParams(false);
            command.Dispatch("ResetQuerywiseLeafIds", {LeafIds}, q, K.Rows);
            Pairwise->EncodePointDerivatives(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves,
                false, Gradient, Hessian, SampleWeight, &Stats.kernel_dispatches);
            return;
        }
        if (!QueryOffsets) {
            command.Dispatch("ObjectiveDerivatives", {Target, SampleWeight, Prediction, Gradient, Hessian, LeafIds}, K, K.Rows);
            return;
        }
        const auto q = QueryParams(false, true);
        command.Dispatch("ResetQuerywiseLeafIds", {LeafIds}, q, K.Rows);
        EncodeQueryPoint(command, RawValues, false, true);
        if (q.Reserved) {
            id<MTLBlitCommandEncoder> encoder = [command.Buffer blitCommandEncoder];
            Require(encoder != nil, "Could not create query validation reset encoder");
            [encoder fillBuffer:QueryValidation range:NSMakeRange(0, 4) value:0];
            [encoder endEncoding];
            command.Dispatch("ValidateQuerywiseStructureCurvature", {Hessian, QueryValidation}, q, K.Rows);
        }
    }
    void ValidateQueryStructure() const {
        if (Combination) Combination->CheckStatus();
        if (Coupled) Coupled->CheckStatus();
        if (Pairwise) Pairwise->CheckStatus();
        if (Yeti) Yeti->CheckStatus();
        if (QueryOffsets && !SimpleLeavesRequested && (K.ScoreFunction == 2 || K.ScoreFunction == 3))
            Require(*static_cast<const uint32_t*>(QueryValidation.contents) == 0,
                    "Invalid GPU query Newton split score: row curvatures must be finite and nonnegative");
    }
    void EncodeYetiPoint(Command& command, bool applyShift) {
        if (!Langevin) Require(YetiSeedPosition < YetiSeeds.size(), "YetiRank oracle seed schedule is missing or exhausted");
        const uint64_t seed = Langevin ? ReadLangevinSeed(applyShift ? CBM_LANGEVIN_YETI_LEAF : CBM_LANGEVIN_YETI_WEAK)
            : YetiSeeds[YetiSeedPosition++];
        Yeti->EncodePointDerivatives(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves,
            applyShift, Target, SampleWeight, Gradient, Hessian, seed, &Stats.kernel_dispatches);
    }
    void EncodeQueryLossReduction(Command& command) {
        command.Dispatch("ReduceQuerywiseObjective", {QueryStatistics, QueryLossPartials}, QueryParams(false), LossGroups, true);
    }
    double QueryLossNumerator() const {
        const float* parts = static_cast<const float*>(QueryLossPartials.contents);
        double value = 0;
        for (uint32_t i = 0; i < LossGroups; ++i) value += parts[2 * i];
        return value;
    }
    void EncodeBacktrackingObjective(Command& command, id<MTLBuffer> leafValues,
                                    const NativeBacktrackingParams& b, bool trial = false) {
        if (Combination) {
            Combination->EncodeLoss(command.Buffer, Prediction, leafValues, LeafIds, K.Leaves,
                true, Target, SampleWeight, &Stats.kernel_dispatches, trial);
        } else if (Pairwise) {
            Pairwise->EncodePointDerivatives(command.Buffer, Prediction, leafValues, LeafIds, K.Leaves,
                true, Gradient, Hessian, SampleWeight, &Stats.kernel_dispatches, trial);
            Pairwise->EncodeLossReduction(command.Buffer, &Stats.kernel_dispatches);
        } else if (QueryOffsets) {
            EncodeQueryPoint(command, leafValues, true);
            EncodeQueryLossReduction(command);
        } else {
            // CUDA reduces each unnormalized task value, then applies task
            // normalization and its ridge penalty in the host oracle.
            const NativeBacktrackingParams valueOptions = {b.Step, b.Type, 0, 0};
            command.Dispatch("ReduceBacktrackingObjective",
                {Target, SampleWeight, Prediction, LeafIds, leafValues, BacktrackingLoss},
                K, LossGroups, true, 1, 1, &valueOptions, sizeof(valueOptions));
        }
    }
    double ReadCurrentBacktrackingObjective(id<MTLBuffer> values) const {
        if (Combination) return RegularizedObjective(Combination->ReadObjective(true), values);
        if (Pairwise) return RegularizedObjective(-Pairwise->ReadLossPartials(true)[0], values);
        return QueryOffsets ? RegularizedObjective(-QueryLossNumerator(), values)
            : RegularizedObjective(ReadBacktrackingScalar(BacktrackingLoss, LossGroups), values);
    }
    void EstimateCoupledBacktrackingLeaves() {
        Command initial(*Context, Stats);
        Coupled->EncodeEdges(initial.Buffer, Prediction, RawValues, LeafIds, K.Leaves, true,
            nullptr, 0, &Stats.kernel_dispatches);
        Coupled->EncodeLoss(initial.Buffer, LeafIds, K.Leaves, true, &Stats.kernel_dispatches);
        initial.Wait();
        double currentValue = RegularizedObjective(-Coupled->ReadLoss().first, RawValues);
        bool updated = false, newDirection = true;
        float step = 1; double directionDot = 0;
        for (uint32_t attempt = 0; attempt < Options.leaf_estimation_iterations || (!updated && attempt < 100); ++attempt) {
            Command trial(*Context, Stats);
            if (newDirection) {
                Coupled->EncodeEdges(trial.Buffer, Prediction, RawValues, LeafIds, K.Leaves, true,
                    nullptr, 0, &Stats.kernel_dispatches);
                Coupled->EncodeLeafProjection(trial.Buffer, K.Leaves, K.LeafMethod == 1, &Stats.kernel_dispatches);
                if (Regularization.add_ridge) Coupled->EncodeLeafRidge(trial.Buffer, RawValues, K.Leaves, K.L2, &Stats.kernel_dispatches);
                Coupled->EncodeLeafDirection(trial.Buffer, K.Leaves, K.L2, CoupledNonDiag, &Stats.kernel_dispatches);
                Coupled->EncodeDirectionDot(trial.Buffer, DirectionDot, K.Leaves, &Stats.kernel_dispatches);
            }
            Coupled->EncodeBeginTrial(trial.Buffer);
            Coupled->EncodeLeafUpdate(trial.Buffer, RawValues, Weights, TrialValues, K.Leaves, step, &Stats.kernel_dispatches, true);
            Coupled->EncodeEdges(trial.Buffer, Prediction, TrialValues, LeafIds, K.Leaves, true,
                nullptr, 0, &Stats.kernel_dispatches, true);
            Coupled->EncodeLoss(trial.Buffer, LeafIds, K.Leaves, true, &Stats.kernel_dispatches);
            trial.Wait();
            if (newDirection) {
                directionDot = ReadBacktrackingScalar(DirectionDot, 1);
                Require(std::isfinite(directionDot), "Non-finite coupled pairwise direction");
            }
            const double trialValue = RegularizedObjective(-Coupled->ReadTrialLoss(), TrialValues);
            const double threshold = currentValue +
                (Options.leaf_estimation_backtracking == 2 ? 1e-5 * double(step) * directionDot : 0);
            if (std::isfinite(trialValue) && trialValue >= threshold) {
                std::swap(RawValues, TrialValues); currentValue = trialValue;
                updated = true; newDirection = true; step = 1;
            } else { step *= 0.5f; newDirection = false; }
        }
    }
    void EstimateBacktrackingLeaves() {
        NativeBacktrackingParams b = BacktrackingOptions();
        auto Dispatch = [&](Command& command, const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                            uint64_t count, bool groups) {
            command.Dispatch(name, buffers, K, count, groups, 1, 1, &b, sizeof(b));
        };
        Command initial(*Context, Stats);
        EncodeBacktrackingObjective(initial, RawValues, b);
        initial.Wait();
        double currentValue = ReadCurrentBacktrackingObjective(RawValues);
        Require(std::isfinite(currentValue), "Non-finite current GPU leaf objective");
        bool updated = false, newDirection = true;
        double directionDot = 0;
        // CUDA leaves_estimation/walker.cpp counts every trial against the
        // iteration budget, but permits up to 100 attempts until first acceptance.
        for (uint32_t attempt = 0; attempt < Options.leaf_estimation_iterations || (!updated && attempt < 100);
             ++attempt) {
            Command trial(*Context, Stats);
            if (newDirection) {
                EncodeObjectivePartials(trial);
                Dispatch(trial, "PrepareBacktrackingDirection",
                         {ObjectivePartials, RawValues, Directions, Weights, DirectionDot}, K.Leaves, true);
            }
            Dispatch(trial, "BuildBacktrackingCandidate", {RawValues, Directions, Weights, TrialValues}, K.Leaves, false);
            EncodeBacktrackingObjective(trial, TrialValues, b, true);
            trial.Wait();
            if (newDirection) {
                directionDot = ReadBacktrackingScalar(DirectionDot, K.Leaves);
                Require(std::isfinite(directionDot), QueryOffsets
                    ? "Invalid GPU query leaf direction: Newton requires a finite positive regularized Hessian"
                    : "Non-finite GPU leaf direction");
            }
            const double trialValue = ReadCurrentBacktrackingObjective(TrialValues);
            const double threshold = currentValue + (b.Type == 2 ? 1e-5 * double(b.Step) * directionDot : 0);
            if (std::isfinite(trialValue) && trialValue >= threshold) {
                std::swap(RawValues, TrialValues);
                currentValue = trialValue;
                updated = true;
                newDirection = true;
                b.Step = 1.0f;
            } else {
                b.Step *= 0.5f;
                newDirection = false;
            }
        }
    }
    void EstimateCombinationYetiLeaves() {
        NativeBacktrackingParams b = BacktrackingOptions();
        auto oracle = [&](Command& command, id<MTLBuffer> values, bool trial = false) {
            Combination->EncodeOracle(command.Buffer, Prediction, values, LeafIds, K.Leaves, true,
                Target, SampleWeight, Gradient, Hessian, CombinationGradientWeights, &Stats.kernel_dispatches, trial);
        };
        auto project = [&](Command& command) {
            const NativeQueryProjectionParams q = {K.Rows, K.Leaves, K.HistogramTiles, K.LeafMethod};
            command.Dispatch("ReduceQuerywiseLeafPartials",
                {Gradient, Hessian, SampleWeight, RowIndices, PartitionOffsets, ObjectivePartials},
                q, K.HistogramTiles, true, K.Leaves);
        };
        Command initial(*Context, Stats);
        oracle(initial, RawValues); initial.Wait();
        double currentValue = RegularizedObjective(Combination->ReadObjective(), RawValues);
        // CUDA performs one initial evaluation for I=1. Longer walks jointly
        // evaluate value/gradient/Hessian at EVERY trial, including the last.
        if (!UsesBacktracking()) {
            for (uint32_t iteration = 0; iteration < Options.leaf_estimation_iterations; ++iteration) {
                Command update(*Context, Stats);
                project(update);
                EncodeNewtonLeaves(update);
                if (Options.leaf_estimation_iterations > 1) oracle(update, RawValues);
                update.Wait(); Combination->CheckStatus();
            }
            return;
        }
        bool updated = false, newDirection = true;
        double directionDot = 0;
        for (uint32_t attempt = 0; attempt < Options.leaf_estimation_iterations || (!updated && attempt < 100); ++attempt) {
            Command trial(*Context, Stats);
            if (newDirection) {
                // Accepted-trial derivatives already reside in these buffers;
                // rejected trials keep the preceding direction until accepted.
                project(trial);
                trial.Dispatch("PrepareBacktrackingDirection", {ObjectivePartials, RawValues, Directions, Weights, DirectionDot},
                    K, K.Leaves, true, 1, 1, &b, sizeof(b));
            }
            trial.Dispatch("BuildBacktrackingCandidate", {RawValues, Directions, Weights, TrialValues},
                K, K.Leaves, false, 1, 1, &b, sizeof(b));
            oracle(trial, TrialValues, true);
            trial.Wait();
            if (newDirection) {
                directionDot = ReadBacktrackingScalar(DirectionDot, K.Leaves);
                Require(std::isfinite(directionDot), "Non-finite Combination leaf direction");
            }
            const double value = RegularizedObjective(Combination->ReadObjective(true), TrialValues);
            const double threshold = currentValue + (b.Type == 2 ? 1e-5 * b.Step * directionDot : 0);
            if (std::isfinite(value) && value >= threshold) {
                std::swap(RawValues, TrialValues);
                currentValue = value; updated = true; newDirection = true; b.Step = 1;
            } else {
                b.Step *= 0.5f; newDirection = false;
            }
        }
    }
    void EncodeExactLeaves(Command& command) {
        const CBMExactLeafParams e = {K.Rows, K.Leaves, uint32_t(K.Objective == 11), K.HistogramTiles,
                                     K.Objective == 9 ? K.ObjectiveParam : 0.5f, 0, 0, 0};
        command.Dispatch("PrepareExactResiduals", {Target, SampleWeight, Prediction, ExactResiduals,
            ExactEffectiveWeights, ExactKeysA, ExactRowsA}, e, K.Rows);
        CBMEncodeSortU32(command.Buffer, ExactKeysA, ExactRowsA, K.Rows, ExactKeysB, ExactRowsB, &Stats.kernel_dispatches);
        command.Dispatch("MakeExactLeafKeys", {ExactRowsB, LeafIds, ExactKeysA}, e, K.Rows);
        CBMEncodeSortU32(command.Buffer, ExactKeysA, ExactRowsB, K.Rows, ExactKeysB, ExactRowsA, &Stats.kernel_dispatches);
        command.Dispatch("BuildExactLeafOffsets", {ExactKeysB, PartitionOffsets}, e, uint64_t(K.Leaves) + 1);
        // Exact partials need one float4, fitting the existing two-float4 workspace.
        command.Dispatch("ReduceExactLeafPartials", {ExactRowsA, ExactEffectiveWeights, SampleWeight,
            PartitionOffsets, ObjectivePartials}, e, K.HistogramTiles, true, K.Leaves);
        command.Dispatch("PrefixExactLeafTiles", {ObjectivePartials, ExactTileOffsets, ExactTotals,
            Weights, ExactSelected}, e, K.Leaves, true);
        command.Dispatch("SelectExactLeafQuantile", {ExactRowsA, ExactEffectiveWeights, PartitionOffsets,
            ExactTileOffsets, ExactTotals, ExactSelected}, e, K.HistogramTiles, true, K.Leaves);
        command.Dispatch("FinalizeExactLeafValues", {ExactRowsA, ExactResiduals, PartitionOffsets,
            ExactTotals, ExactSelected, RawValues}, e, K.Leaves);
    }
    void EncodePartitions(Command& command) {
        if (K.Leaves == 1) {
            command.Dispatch("InitializeRootPartition", {RowIndices, PartitionOffsets}, K, K.Rows);
            return;
        }
        if (Deep) {
            command.Dispatch("CountDeepPartitionBits", {RowIndices, LeafIds, RowRanks, PartitionTiles}, K, DeepTiles, true);
            command.Dispatch("ScanDeepPartitionTiles", {PartitionTiles, DeepBlockPrefix}, K, DeepBlocks, true);
            command.Dispatch("ScanDeepPartitionBlocks", {DeepBlockPrefix}, K, 1, true);
            command.Dispatch("BuildDeepPartitionOffsets", {PartitionOffsets, RowRanks, PartitionTiles, DeepBlockPrefix,
                                                           NextPartitionOffsets}, K, K.Leaves / 2);
            command.Dispatch("ScatterDeepPartitionRows", {RowIndices, LeafIds, RowRanks, PartitionTiles,
                                                          DeepBlockPrefix, NextRowIndices}, K, K.Rows);
            std::swap(RowIndices, NextRowIndices);
            std::swap(PartitionOffsets, NextPartitionOffsets);
            return;
        }
        command.Dispatch("CountIncrementalPartitionTiles", {RowIndices, LeafIds, RowRanks, PartitionTiles},
                         K, K.PartitionTiles, true);
        command.Dispatch("PrefixPartitionTiles", {PartitionTiles, PartitionOffsets}, K, 1, true);
        command.Dispatch("ScatterIncrementalPartitionRows",
                         {RowIndices, LeafIds, RowRanks, PartitionTiles, PartitionOffsets, NextRowIndices},
                         K, K.Rows);
        std::swap(RowIndices, NextRowIndices);
    }
    void EncodeCompactHistograms(Command& command, bool reuse, const HistogramTile& tile) {
        NativeCompactParams h = {K.Rows, tile.End - tile.Begin, K.Leaves, tile.Bins, tile.Begin, 8192,
                                 HistogramJobCapacity, uint32_t(reuse)};
        command.Dispatch("ResetCompactHistogramWorkState", {HistogramState}, h, 1);
        command.Dispatch("BuildCompactHistogramJobs", {PartitionOffsets, HistogramJobs, HistogramActive, HistogramState},
                         h, reuse ? K.Leaves / 2 : K.Leaves);
        command.Dispatch("BuildCompactHistogramDispatchArguments", {HistogramState, HistogramArguments}, h, 1);
        command.Dispatch("ClearCompactHistograms", {HistogramSums, HistogramWeights}, h,
                         uint64_t(reuse ? K.Leaves / 2 : K.Leaves) * tile.Bins);
        command.DispatchIndirect("ComputeCompactHistograms", {Data, Gradient, StructureWeight, RowIndices,
            tile.Offsets, HistogramJobs, HistogramState, HistogramSums, HistogramWeights}, h, HistogramArguments, 0);
        command.DispatchIndirect("ScanCompactHistograms", {HistogramSums, HistogramWeights, FeatureType, tile.Offsets,
            HistogramActive, HistogramState}, h, HistogramArguments, 12);
        if (reuse) command.DispatchIndirect("SubtractCompactSiblingHistograms", {HistogramSums, HistogramWeights,
            PartitionOffsets, HistogramActive, HistogramState}, h, HistogramArguments, 24);
    }
    void EncodeObjectivePartials(Command& command) {
        if (QueryOffsets || Pairwise || Yeti || Combination) {
            if (Combination) Combination->EncodePointDerivatives(command.Buffer, Prediction, RawValues, LeafIds,
                K.Leaves, true, Target, SampleWeight, Gradient, Hessian, CombinationGradientWeights, &Stats.kernel_dispatches);
            else if (Yeti) EncodeYetiPoint(command, true);
            else if (Pairwise) Pairwise->EncodePointDerivatives(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves,
                true, Gradient, Hessian, SampleWeight, &Stats.kernel_dispatches);
            else EncodeQueryPoint(command, RawValues, true);
            const NativeQueryProjectionParams q = {K.Rows, K.Leaves, K.HistogramTiles, K.LeafMethod};
            command.Dispatch("ReduceQuerywiseLeafPartials",
                {Gradient, Hessian, SampleWeight, RowIndices, PartitionOffsets, ObjectivePartials},
                q, K.HistogramTiles, true, K.Leaves);
            return;
        }
        command.Dispatch("ReduceLeafObjectivePartials",
            {Target, SampleWeight, Prediction, RawValues, RowIndices, PartitionOffsets, ObjectivePartials},
            K, K.HistogramTiles, true, K.Leaves);
    }
    NativeBootstrapParams BootstrapParams() const {
        return {K.Rows, BootstrapOptions.bootstrap_type, BootstrapOptions.random_seed_low,
                BootstrapOptions.random_seed_high, BootstrapOptions.iteration_offset + Completed,
                0, 0, 0, BootstrapOptions.bagging_temperature, BootstrapOptions.subsample,
                BootstrapOptions.mvs_reg_is_set ? BootstrapOptions.mvs_reg : MVSLambda, 0};
    }
    void EncodeBootstrap(Command& command) {
        if (Coupled) return; // The frozen weak target samples edges before differentiation.

        const id<MTLBuffer> weakWeight = (Yeti || K.ScoreFunction == 2 || K.ScoreFunction == 3) ? Hessian :
            Combination ? CombinationGradientWeights : SampleWeight;
        if (!BootstrapOptions.bootstrap_type) { StructureWeight = weakWeight; return; }
        const auto params = BootstrapParams();
        if (BootstrapOptions.bootstrap_type == 4) {
            command.Dispatch("ComputeMvsThresholds", {Gradient, MVSThresholds}, params,
                             (K.Rows + 8191) / 8192, true);
            command.Dispatch("GenerateMvsBootstrapWeights", {BootstrapMultipliers, Gradient, MVSThresholds},
                             params, K.Rows);
        } else {
            command.Dispatch("GenerateBootstrapWeights", {BootstrapMultipliers, Gradient}, params, K.Rows);
        }
        command.Dispatch("ApplyBootstrapWeights", {Gradient, weakWeight, BootstrapMultipliers, StructureWeight},
                         params, K.Rows);
    }
    void EncodeLoss(Command& command) {
        if (Combination) {
            Combination->EncodeLoss(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves,
                false, Target, SampleWeight, &Stats.kernel_dispatches);
            return;
        }
        if (Coupled && !Coupled->HasObjectiveValue()) return;
        if (Coupled) {
            Coupled->EncodeEdges(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves, false,
                nullptr, 0, &Stats.kernel_dispatches);
            Coupled->EncodeLoss(command.Buffer, LeafIds, K.Leaves, false, &Stats.kernel_dispatches);
            return;
        }
        if (Yeti) return; // CUDA YetiRank's oracle value is zero; its tracker uses PFound.
        if (Pairwise) {
            Pairwise->EncodePointDerivatives(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves,
                false, Gradient, Hessian, SampleWeight, &Stats.kernel_dispatches);
            Pairwise->EncodeLossReduction(command.Buffer, &Stats.kernel_dispatches);
            return;
        }
        if (QueryOffsets) {
            EncodeQueryPoint(command, RawValues, false);
            EncodeQueryLossReduction(command);
            return;
        }
        command.Dispatch("ReduceObjectiveLoss", {Target, SampleWeight, Prediction, LossPartials}, K, LossGroups, true);
    }
    float ReadLoss() const {
        if (Combination) return Combination->ReadMetric();
        if (Coupled && !Coupled->HasObjectiveValue()) { Coupled->CheckStatus();return 0.f; }
        if (Coupled) {
            const auto parts = Coupled->ReadLoss();
            Require(parts.second > 0, "PairLogitPairwise metric requires positive edge mass");
            const float loss = static_cast<float>(parts.first / parts.second);
            Require(std::isfinite(loss), "Non-finite GPU PairLogitPairwise loss");
            return loss;
        }
        if (Yeti) { Yeti->CheckStatus(); return 0.0f; }
        if (Pairwise) {
            const auto parts = Pairwise->ReadLossPartials();
            const float loss = static_cast<float>(parts[0] / parts[1]);
            Require(std::isfinite(loss), "Non-finite GPU PairLogit objective loss");
            return loss;
        }
        if (QueryOffsets) {
            const float* parts = static_cast<const float*>(QueryLossPartials.contents);
            double loss = 0, denominator = 0;
            for (uint32_t i = 0; i < LossGroups; ++i) {
                Require(std::isfinite(parts[2 * i]) && std::isfinite(parts[2 * i + 1]) && parts[2 * i + 1] >= 0,
                        "Non-finite GPU grouped objective; rescale targets, predictions or weights");
                loss += parts[2 * i]; denominator += parts[2 * i + 1];
            }
            Require(denominator > 0 && std::isfinite(loss), "Grouped objective requires positive metric mass and finite loss");
            loss /= denominator;
            const float result = static_cast<float>(K.Objective == 12 ? std::sqrt(loss) : loss);
            Require(std::isfinite(result), "Non-finite GPU grouped objective loss");
            return result;
        }
        const float* partials = static_cast<const float*>(LossPartials.contents);
        double sum = 0;
        for (uint32_t i = 0; i < LossGroups; ++i) {
            Require(std::isfinite(partials[i]) && (Options.objective == 3 || Options.objective == 20 || partials[i] >= 0),
                    "Non-finite GPU loss; rescale targets, predictions or weights");
            sum += partials[i];
        }
        const float loss = static_cast<float>(Options.objective == 0 ? std::sqrt(sum) : sum);
        Require(std::isfinite(loss), "Non-finite GPU objective loss");
        return loss;
    }
    void BeginTreeImpl() {
        const auto& p = Options.train;
        if (Pairwise) Pairwise->ClearStatus();
        if (Coupled) Coupled->ClearStatus();
        if (Combination) Combination->ClearStatus();
        if (Yeti) {
            Require(Langevin || (YetiSeedPosition == 0 && (YetiSeeds.size() == 1 || YetiSeeds.size() == YetiLeafSeedCount() + 1)),
                "Supply a fresh YetiRank oracle seed schedule before every tree");
            Yeti->ClearStatus();
        }
        if (!Permutations.empty()) {
            Data = Permutations[SearchPermutation].Bins; Prediction = Permutations[SearchPermutation].Cursor;
            MVSLambda = Permutations[SearchPermutation].Lambda; HasMVSLambda = Permutations[SearchPermutation].HasLambda;
        }
        K.Leaves = 1;
        K.ScoreBeforeSplit = 0;
        const bool initializeMVS = BootstrapOptions.bootstrap_type == 4
            && !BootstrapOptions.mvs_reg_is_set && !HasMVSLambda;
        const bool scoreNoise = !Coupled && RandomStrength > 0 && (K.ScoreFunction == 1 || K.ScoreFunction == 3);
        // A staged Combination callback replaces the weak packet after search.
        // Consume that packet at begin even for a stump/no-candidate tree.
        const bool combinationWeak = Combination && Combination->HasYeti();
        float scoreNoiseScale = 0;
        if (initializeMVS || scoreNoise || combinationWeak) {
            Command statistics(*Context, Stats);
            EncodeDerivatives(statistics);
            if (initializeMVS) statistics.Dispatch("ReduceBootstrapStatistics", {Gradient, BootstrapStatistics},
                                                   BootstrapParams(), LossGroups, true);
            if (scoreNoise) statistics.Dispatch("ReduceScoreNoiseStatistics", {Gradient,
                (Yeti || K.ScoreFunction == 2 || K.ScoreFunction == 3) ? Hessian :
                    Combination ? CombinationGradientWeights : SampleWeight, NoiseStatistics},
                                                BootstrapParams(), LossGroups, true);
            statistics.Wait();
            ValidateQueryStructure();
            if (initializeMVS) {
                const float* partials = static_cast<const float*>(BootstrapStatistics.contents);
                double meanAbsoluteGradient = 0;
                for (uint32_t i = 0; i < LossGroups; ++i) {
                    Require(std::isfinite(partials[2 * i]) && partials[2 * i] >= 0,
                            "Non-finite gradient statistic for automatic MVS regularization");
                    meanAbsoluteGradient += partials[2 * i];
                }
                MVSLambda = static_cast<float>(meanAbsoluteGradient * meanAbsoluteGradient);
                Require(std::isfinite(MVSLambda), "Automatic MVS regularization overflowed float32");
                HasMVSLambda = true;
            }
            if (scoreNoise) {
                const float* partials = static_cast<const float*>(NoiseStatistics.contents);
                double variance = 0;
                for (uint32_t i = 0; i < LossGroups; ++i) {
                    Require(std::isfinite(partials[i]) &&
                            (Combination || (SimpleLeavesRequested && K.Objective == 13) || partials[i] >= 0),
                        "Non-finite GPU score noise statistic");
                    variance += partials[i];
                }
                Require(std::isfinite(variance) && variance >= 0,
                    "GPU score noise requires a finite nonnegative signed weighted variance");
                // CUDA random_score_helper.h and doc_parallel_boosting.h use
                // model size = absolute iteration * learning rate, before bootstrap.
                const double logRemaining = std::log(double(K.Rows))
                    - double(BootstrapOptions.iteration_offset + Completed) * p.learning_rate;
                const double modelMultiplier = logRemaining >= 0 ? 1 / (1 + std::exp(-logRemaining))
                    : std::exp(logRemaining) / (1 + std::exp(logRemaining));
                scoreNoiseScale = static_cast<float>(modelMultiplier * std::sqrt(variance) * RandomStrength);
                Require(std::isfinite(scoreNoiseScale), "Score noise scale exceeds float32");
            }
        }
        Pending = PendingTree{};
        // Stochastic targets must reuse the draw used for noise/MVS statistics,
        // instead of silently drawing a second weak target for the same tree.
        Pending.DerivativesReady = (Yeti || combinationWeak) && (initializeMVS || scoreNoise || combinationWeak);
        Pending.Active = true;
        Pending.Exhausted = !HasActiveCandidates();
        Pending.Finished = p.depth == 0 || Pending.Exhausted;
        Pending.ScoreNoise = scoreNoise;
        Pending.NoiseScale = scoreNoiseScale;
        Pending.Selected.reserve(p.depth);
    }
    void PrepareTree(Command& command) {
        const auto& p = Options.train;
        if (Pending.Initialize) {
            if (!Pending.DerivativesReady) EncodeDerivatives(command);
            KernelParams clear = K;
            clear.Leaves = MaxLeaves;
            command.Dispatch("InitializeLeafValues", {RawValues, Weights}, clear, MaxLeaves);
            if (Langevin) ReadLangevinSeed(CBM_LANGEVIN_WEAK_SEED_CACHE);
            EncodeBootstrap(command);
            EncodeLangevinWeak(command);
            Pending.Initialize = false;
            Pending.PartitionsValid = false;
        }
        if (Pending.SplitPending) {
            command.Dispatch("UpdateLeafBins", {Data, LeafIds, Winner}, K, p.rows);
            Pending.SplitPending = false;
            Pending.PartitionsValid = false;
        }
        if (!Pending.PartitionsValid) { EncodePartitions(command); Pending.PartitionsValid = true; }
    }
    void GrowCoupledTree() {
        const auto& p = Options.train;
        K.Leaves = uint32_t(1) << Pending.Depth;
        UpdateFeaturePenalties();
        Command command(*Context, Stats);
        PrepareTree(command);
        Coupled->EncodeCandidates(command.Buffer, Data, LeafIds, CandidateFeature, CandidateBin,
            CandidateType, K.Features, K.Leaves, K.ScoreFunction == 0, K.L2, CoupledNonDiag, &Stats.kernel_dispatches);
        Coupled->EncodeSelectWinner(command.Buffer, CandidateFeature, FeaturePenaltyWeights, K.Features,
            K.ScoreBeforeSplit, true, &Stats.kernel_dispatches);
        command.Wait();
        const auto selected = Coupled->ReadWinner();
        const SplitState winner = {selected.Index,
            static_cast<const uint32_t*>(CandidateFeature.contents)[selected.Index],
            static_cast<const uint32_t*>(CandidateBin.contents)[selected.Index],
            static_cast<const uint8_t*>(CandidateType.contents)[selected.Index],selected.Score,1,0,selected.Gain};
        if (K.LeafMethod == 3 && Pending.Depth + 1 == p.depth) {
            // The winning tile may no longer be resident. Reproject only that
            // candidate using the same frozen weak target and parent leaf IDs.
            Command solution(*Context, Stats);
            Coupled->EncodeCandidates(solution.Buffer, Data, LeafIds, CandidateFeature, CandidateBin,
                CandidateType, K.Features, K.Leaves, K.ScoreFunction == 0, K.L2, CoupledNonDiag,
                &Stats.kernel_dispatches, selected.Index, 1);
            Coupled->EncodeSimpleLeafValues(solution.Buffer, RawValues, Weights, 2 * K.Leaves,
                winner.Type != 0, &Stats.kernel_dispatches);
            solution.Wait(); Coupled->CheckStatus();
        }
        *static_cast<SplitState*>(Winner.contents) = winner;
        K.ScoreBeforeSplit = winner.Score;
        if (CtrUniqueValues[winner.Feature]) { UsedFeatures[winner.Feature] = 1; FeatureFlags[winner.Feature] |= 2; }
        // CUDA pairwise structure search retains repeated winning splits.
        Pending.LastWinner = winner; Pending.Selected.push_back(winner);
        K.SplitLevel = Pending.Depth; Pending.SplitPending = true;
        ++Pending.Depth; Pending.HasSplit = true; Pending.Finished = Pending.Depth == p.depth;
    }
    void GrowTreeImpl() {
        Pending.HasSplit = false;
        if (Pending.Finished) return;
        if (!HasActiveCandidates()) { Pending.Finished = Pending.Exhausted = true; return; }
        const auto& p = Options.train;
        if (Coupled) { GrowCoupledTree(); return; }
        const uint32_t level = Pending.Depth;
        K.Leaves = uint32_t(1) << Pending.Depth;
        UpdateFeaturePenalties();
        Command command(*Context, Stats);
        PrepareTree(command);
        command.Dispatch("ReduceStructurePartials",
            {Gradient, StructureWeight, RowIndices, PartitionOffsets, ObjectivePartials},
            K, K.HistogramTiles, true, K.Leaves);
        command.Dispatch("CollectPartitionStatistics", {ObjectivePartials, LeafSums, LeafWeights}, K, K.Leaves, true);
        if (Pending.ScoreNoise) {
            auto noiseParams = BootstrapParams();
            noiseParams.Rows = K.Features;
            noiseParams.Stream = level + 1;
            noiseParams.NoiseScale = Pending.NoiseScale;
            command.Dispatch("GenerateScoreFeatureNoise", {FeatureNoise}, noiseParams, K.Features);
        }
        const auto scoreRegularization = ScoreRegularization();
        const bool regularizedScore = scoreRegularization.Normalize || scoreRegularization.MetaExponent != 1 || scoreRegularization.PerFeature;
        uint32_t metaBits; std::memcpy(&metaBits, &scoreRegularization.MetaExponent, sizeof(metaBits));
        if (Compact) {
            for (size_t index = 0; index < HistogramTiles.size(); ++index) {
                const auto& tile = HistogramTiles[index];
                EncodeCompactHistograms(command, level != 0 && HistogramTiles.size() == 1 && (!Dynamic || Pending.HistogramValid), tile);
                const NativeScoreTileParams scoreTile = {tile.Begin, tile.End, tile.Bins,
                    {scoreRegularization.Normalize, metaBits, scoreRegularization.PerFeature, 0, 0}};
                if (Dynamic) command.Dispatch(regularizedScore ? "FindDynamicTileSplitWinnersRegularized" : "FindDynamicTileSplitWinners", {HistogramSums, HistogramWeights, LeafSums, LeafWeights,
                    CandidateFeature, CandidateBin, CandidateType, WinnerPartials, FeatureNoise, tile.Offsets,
                    FeaturePenaltyWeights, CandidateActive}, K, K.ScoreGroups, true, 1, 1, &scoreTile, sizeof(scoreTile));
                else command.Dispatch(regularizedScore ? "FindTileSplitWinnersRegularized" : "FindTileSplitWinners", {HistogramSums, HistogramWeights, LeafSums, LeafWeights,
                    CandidateFeature, CandidateBin, CandidateType, WinnerPartials, FeatureNoise, tile.Offsets, FeaturePenaltyWeights},
                    K, K.ScoreGroups, true, 1, 1, &scoreTile, sizeof(scoreTile));
                command.Dispatch("ReduceSplitWinners", {WinnerPartials, index == 0 ? Winner : TileWinner}, K, 1, true);
                if (index) command.Dispatch("MergeTileSplitWinner", {Winner, TileWinner}, K, 1);
            }
        } else if (level == 0) {
            command.Dispatch("ClearHistograms", {HistogramSums, HistogramWeights}, K,
                             uint64_t(K.Leaves) * p.features * p.bins_per_feature);
            command.Dispatch("ComputeHistograms",
                {Data, Gradient, StructureWeight, RowIndices, PartitionOffsets, HistogramSums, HistogramWeights},
                K, K.HistogramTiles, true, K.Features, K.Leaves);
            command.Dispatch("ScanHistograms", {HistogramSums, HistogramWeights, FeatureType},
                             K, uint64_t(K.Leaves) * K.Features, true);
        } else {
            const uint32_t parents = K.Leaves / 2;
            command.Dispatch("ClearChildHistograms", {HistogramSums, HistogramWeights}, K,
                             uint64_t(parents) * K.Features * K.Bins);
            command.Dispatch("ComputeSmallerChildHistograms",
                {Data, Gradient, StructureWeight, RowIndices, PartitionOffsets, HistogramSums, HistogramWeights},
                K, K.HistogramTiles, true, K.Features, parents);
            command.Dispatch("ScanChildHistograms", {HistogramSums, HistogramWeights, FeatureType},
                             K, uint64_t(parents) * K.Features, true);
            command.Dispatch("SubtractSiblingHistograms", {HistogramSums, HistogramWeights, PartitionOffsets},
                             K, uint64_t(parents) * K.Features * K.Bins);
        }
        if (!Compact) {
            command.Dispatch(regularizedScore ? "FindSplitWinnersRegularized" : "FindSplitWinners",
                {HistogramSums, HistogramWeights, LeafSums, LeafWeights,
                 CandidateFeature, CandidateBin, CandidateType, WinnerPartials, FeatureNoise, FeatureOffsets, FeaturePenaltyWeights}, K, K.ScoreGroups, true, 1, 1,
                 regularizedScore ? &scoreRegularization : nullptr, regularizedScore ? sizeof(scoreRegularization) : 0);
            command.Dispatch("ReduceSplitWinners", {WinnerPartials, Winner}, K, 1, true);
        }
        command.Wait();
        ValidateQueryStructure();
        if (Compact) Require(static_cast<const uint32_t*>(HistogramState.contents)[2] == 0, "GPU compact histogram work capacity exceeded");
        Pending.HistogramValid = true;
        const SplitState winner = *static_cast<const SplitState*>(Winner.contents);
        Require(winner.Valid && !winner.InvalidScore && std::isfinite(winner.Score) && std::isfinite(winner.Gain),
                QueryOffsets && (K.ScoreFunction == 2 || K.ScoreFunction == 3)
                ? "Invalid GPU query Newton split score: row curvatures must be finite and nonnegative"
                : "Non-finite GPU split score; rescale targets, predictions or weights");
        K.ScoreBeforeSplit = winner.Score;
        const bool treeCtr = Dynamic && CtrUniqueValues[winner.Feature] && (FeatureFlags[winner.Feature] & 1);
        if (treeCtr) FeatureFlags[winner.Feature] |= 2;
        // DocParallel tracks all selected CTRs. FeatureParallel registers tree
        // CTR winners and marks only an installed tree-CTR split as used;
        // selecting a static/simple CTR does not update CUDA's used-CTR set.
        if (!Dynamic && CtrUniqueValues[winner.Feature]) {
            UsedFeatures[winner.Feature] = 1;
            FeatureFlags[winner.Feature] |= 2;
        }
        bool duplicate = false;
        for (const auto& previous : Pending.Selected)
            duplicate |= previous.Feature == winner.Feature && previous.Bin == winner.Bin && previous.Type == winner.Type;
        // CUDA doc-parallel symmetric search evaluates all candidates and
        // stops only when its winning split is already in the tree.
        Pending.LastWinner = winner;
        if (duplicate) { Pending.Finished = true; return; }
        if (treeCtr) UsedFeatures[winner.Feature] = 1;
        Pending.Selected.push_back(winner);
        K.SplitLevel = Pending.Depth;
        Pending.SplitPending = true;
        ++Pending.Depth;
        Pending.HasSplit = true;
        Pending.Finished = Pending.Depth == p.depth;
    }
    PermutationTree EstimateFeatureParallelCombinationLeaves() {
        // CUDA symmetric scalar estimation has one walker over all tasks. Keep
        // each task's accepted point and derivatives on the GPU, and reduce
        // only scalar objective/direction values to make one shared decision.
        struct Task {
            id<MTLBuffer> Ids, Rows, Offsets, Raw, Weights, Gradient, Hessian, Trial, Direction, Dot, NoiseG, NoiseH;
        };
        const uint32_t count = PermutationCount(), leaves = K.Leaves;
        const uint64_t extra = uint64_t(count) * (16ull * K.Rows + (LangevinLeaves() ? 44ull : 28ull) * leaves + 4)
            + 32ull * Pending.Depth;
        Require(WorkspaceBytes() + extra <= MaxWorkingBytes,
            "FeatureParallel Combination leaf tasks exceed the 1 GiB GPU memory limit");
        DynamicPeakBytes = std::max(DynamicPeakBytes, WorkspaceBytes() + extra);
        std::vector<Task> tasks(count);
        auto fixedSplits = Context->Buffer(32ull * Pending.Depth, Pending.Selected.data());
        Command layout(*Context, Stats);
        for (uint32_t i = 0; i < count; ++i) {
            if (!Permutations.empty()) { Data = Permutations[i].Bins; Prediction = Permutations[i].Cursor; }
            auto& t = tasks[i];
            t.Ids = Context->Buffer(4ull * K.Rows); t.Rows = Context->Buffer(4ull * K.Rows);
            t.Offsets = Context->Buffer(4ull * (leaves + 1));
            t.Gradient = Context->Buffer(4ull * K.Rows); t.Hessian = Context->Buffer(4ull * K.Rows);
            t.Raw = Context->Buffer(4ull * leaves); t.Weights = Context->Buffer(4ull * leaves);
            t.Trial = Context->Buffer(4ull * leaves); t.Direction = Context->Buffer(4ull * leaves);
            t.Dot = Context->Buffer(8ull * leaves);
            if (LangevinLeaves()) { t.NoiseG = Context->Buffer(8ull * leaves); t.NoiseH = Context->Buffer(8ull * leaves); }
            layout.Dispatch("ResetQuerywiseLeafIds", {LeafIds}, QueryParams(false), K.Rows);
            K.Leaves = 1; EncodePartitions(layout);
            for (uint32_t level = 0; level < Pending.Depth; ++level) {
                K.SplitLevel = level;
                layout.Dispatch("UpdateFixedPermutationSplit", {Data, LeafIds, fixedSplits}, K, K.Rows);
                K.Leaves = uint32_t(1) << (level + 1); EncodePartitions(layout);
            }
            layout.Dispatch("InitializeLeafValues", {t.Raw, t.Weights}, K, leaves);
            id<MTLBlitCommandEncoder> copy = [layout.Buffer blitCommandEncoder];
            Require(copy != nil, "Could not allocate FeatureParallel leaf layout encoder");
            [copy copyFromBuffer:LeafIds sourceOffset:0 toBuffer:t.Ids destinationOffset:0 size:4ull * K.Rows];
            [copy copyFromBuffer:RowIndices sourceOffset:0 toBuffer:t.Rows destinationOffset:0 size:4ull * K.Rows];
            [copy copyFromBuffer:PartitionOffsets sourceOffset:0 toBuffer:t.Offsets destinationOffset:0 size:4ull * (leaves + 1)];
            [copy endEncoding];
        }
        layout.Wait();
        const Task saved = {LeafIds, RowIndices, PartitionOffsets, RawValues, Weights, Gradient, Hessian,
                            TrialValues, Directions, DirectionDot};
        auto bind = [&](const Task& t) {
            LeafIds = t.Ids; RowIndices = t.Rows; PartitionOffsets = t.Offsets;
            RawValues = t.Raw; Weights = t.Weights; Gradient = t.Gradient; Hessian = t.Hessian;
            TrialValues = t.Trial; Directions = t.Direction; DirectionDot = t.Dot;
        };
        auto select = [&](uint32_t i) {
            bind(tasks[i]);
            if (!Permutations.empty()) { Data = Permutations[i].Bins; Prediction = Permutations[i].Cursor; }
        };
        auto project = [&](Command& command) {
            if (!Combination && !(LangevinLeaves() && (Yeti || Pairwise || QueryOffsets))) { EncodeObjectivePartials(command); return; }
            const NativeQueryProjectionParams q = {K.Rows, leaves, K.HistogramTiles, K.LeafMethod};
            command.Dispatch("ReduceQuerywiseLeafPartials",
                {Gradient, Hessian, SampleWeight, RowIndices, PartitionOffsets, ObjectivePartials},
                q, K.HistogramTiles, true, leaves);
        };
        auto evaluate = [&](bool trial) {
            double value = 0;
            for (uint32_t i = 0; i < count; ++i) {
                select(i); Command oracle(*Context, Stats);
                if (Combination) {
                    Combination->EncodeOracle(oracle.Buffer, Prediction, trial ? TrialValues : RawValues,
                        LeafIds, leaves, true, Target, SampleWeight, Gradient, Hessian,
                        CombinationGradientWeights, &Stats.kernel_dispatches, trial);
                    oracle.Wait(); value += RegularizedObjective(Combination->ReadObjective(trial), trial ? TrialValues : RawValues);
                } else if (Yeti) {
                    Yeti->EncodePointDerivatives(oracle.Buffer, Prediction, trial ? TrialValues : RawValues,
                        LeafIds, leaves, true, Target, SampleWeight, Gradient, Hessian,
                        ReadLangevinSeed(CBM_LANGEVIN_YETI_LEAF), &Stats.kernel_dispatches);
                    oracle.Wait(); Yeti->CheckStatus();
                    value += RegularizedObjective(0, trial ? TrialValues : RawValues);
                } else {
                    const auto b = BacktrackingOptions();
                    EncodeBacktrackingObjective(oracle, trial ? TrialValues : RawValues, b, trial);
                    oracle.Wait(); value += ReadCurrentBacktrackingObjective(trial ? TrialValues : RawValues);
                }
            }
            return value;
        };
        auto noise = [&](uint32_t event, bool diagonal, bool add) {
            std::vector<double> samples(uint64_t(count) * leaves);
            Require(LangevinNoise(LangevinContext, event, static_cast<uint32_t>(samples.size()), samples.data()) == 0,
                    "Langevin leaf noise callback failed");
            for (uint32_t task = 0; task < count; ++task) {
                float* output = static_cast<float*>((diagonal ? tasks[task].NoiseH : tasks[task].NoiseG).contents);
                for (uint32_t leaf = 0; leaf < leaves; ++leaf) {
                    double value = samples[uint64_t(task) * leaves + leaf];
                    if (add) value += double(output[2 * leaf]) + output[2 * leaf + 1];
                    Require(std::isfinite(value), "Langevin callback returned nonfinite leaf noise");
                    output[2 * leaf] = static_cast<float>(value);
                    output[2 * leaf + 1] = static_cast<float>(value - output[2 * leaf]);
                }
            }
        };
        double currentValue = evaluate(false);
        if (LangevinLeaves()) {
            noise(CBM_LANGEVIN_INITIAL_GRADIENT, false, false);
            noise(CBM_LANGEVIN_INITIAL_HESSIAN, true, false);
        }
        if (!UsesBacktracking() && !LangevinLeaves()) {
            for (uint32_t iteration = 0; iteration < Options.leaf_estimation_iterations; ++iteration) {
                for (uint32_t i = 0; i < count; ++i) {
                    select(i); Command update(*Context, Stats); project(update);
                    EncodeNewtonLeaves(update);
                    update.Wait();
                }
                if (Options.leaf_estimation_iterations > 1) evaluate(false);
            }
        } else {
            NativeBacktrackingParams b = BacktrackingOptions();
            bool updated = false, newDirection = true;
            double directionDot = 0;
            for (uint32_t attempt = 0; attempt < Options.leaf_estimation_iterations || (!updated && attempt < 100); ++attempt) {
                if (newDirection) directionDot = 0;
                for (uint32_t i = 0; i < count; ++i) {
                    select(i); Command trial(*Context, Stats);
                    if (newDirection) {
                        project(trial);
                        if (LangevinLeaves()) trial.Dispatch("PrepareLangevinBacktrackingDirection",
                            {ObjectivePartials, RawValues, Directions, Weights, DirectionDot, tasks[i].NoiseG, tasks[i].NoiseH},
                            K, leaves, true, 1, 1, &b, sizeof(b));
                        else trial.Dispatch("PrepareBacktrackingDirection", {ObjectivePartials, RawValues, Directions, Weights, DirectionDot},
                            K, leaves, true, 1, 1, &b, sizeof(b));
                    }
                    trial.Dispatch("BuildBacktrackingCandidate", {RawValues, Directions, Weights, TrialValues},
                        K, leaves, false, 1, 1, &b, sizeof(b));
                    trial.Wait();
                    if (newDirection) directionDot += ReadBacktrackingScalar(DirectionDot, leaves);
                }
                Require(std::isfinite(directionDot), "Non-finite FeatureParallel Combination direction");
                if (LangevinLeaves() && Options.leaf_estimation_iterations == 1) {
                    for (auto& t : tasks) std::swap(t.Raw, t.Trial);
                    break;
                }
                const double value = evaluate(true);
                if (LangevinLeaves()) noise(CBM_LANGEVIN_TRIAL_GRADIENT, false, false);
                const double threshold = currentValue + (b.Type == 2 ? 1e-5 * b.Step * directionDot : 0);
                if ((LangevinLeaves() && b.Type == 0) || (std::isfinite(value) && value >= threshold)) {
                    for (auto& t : tasks) std::swap(t.Raw, t.Trial);
                    if (LangevinLeaves()) {
                        noise(CBM_LANGEVIN_ACCEPTED_GRADIENT, false, true);
                        for (auto& t : tasks) std::memset(t.NoiseH.contents, 0, 8ull * leaves);
                    }
                    currentValue = value; updated = true; newDirection = true; b.Step = 1;
                } else { b.Step *= 0.5f; newDirection = false; }
            }
        }
        PermutationTree exported;
        for (uint32_t i = 0; i < count; ++i) {
            select(i); Command finish(*Context, Stats);
            if (Pairwise) Pairwise->EncodeCenterLeafValues(finish.Buffer, RawValues, leaves, &Stats.kernel_dispatches);
            if (Yeti) Yeti->EncodeCenterLeafValues(finish.Buffer, RawValues, leaves, &Stats.kernel_dispatches);
            finish.Dispatch("FinalizeLeafValues", {RawValues, Values}, K, leaves);
            finish.Dispatch("AddObjectiveBinModelValue", {LeafIds, Values, Prediction}, K, K.Rows);
            EncodeLoss(finish); finish.Wait();
            const auto* values = static_cast<const float*>(Values.contents);
            const auto* weights = static_cast<const float*>(Weights.contents);
            double mass = 0, meanAbsoluteLeaf = 0;
            for (uint32_t leaf = 0; leaf < leaves; ++leaf) {
                Require(std::isfinite(values[leaf]) && std::isfinite(weights[leaf]) && weights[leaf] >= 0,
                    "Invalid FeatureParallel Combination leaf statistics");
                mass += weights[leaf]; meanAbsoluteLeaf += std::abs(values[leaf]);
            }
            Require(std::abs(mass - TotalWeight) <= std::max(1e-6, TotalWeight * 2e-5),
                "FeatureParallel Combination leaf weights do not cover the training sample weights");
            if (!Permutations.empty()) { MVSLambda = Permutations[i].Lambda; HasMVSLambda = Permutations[i].HasLambda; }
            if (BootstrapOptions.bootstrap_type == 4 && !BootstrapOptions.mvs_reg_is_set) {
                meanAbsoluteLeaf /= leaves; MVSLambda = float(meanAbsoluteLeaf * meanAbsoluteLeaf); HasMVSLambda = true;
                Require(std::isfinite(MVSLambda), "Automatic MVS regularization overflowed float32");
            }
            if (!Permutations.empty()) { Permutations[i].Lambda = MVSLambda; Permutations[i].HasLambda = HasMVSLambda; }
            const float loss = ReadLoss();
            if (i + 1 == count) exported = {std::vector<float>(values, values + leaves),
                std::vector<float>(weights, weights + leaves), loss};
        }
        bind(saved);
        return exported;
    }
    void FinishTreeImpl() {
        const auto& p = Options.train;
        const uint32_t actualDepth = Pending.Depth;
        const auto& selected = Pending.Selected;

        K.Leaves = uint32_t(1) << actualDepth;
        Command command(*Context, Stats);
        PrepareTree(command);
        if (Yeti && !Langevin) Require(YetiSeedPosition == 1 && YetiSeeds.size() == YetiLeafSeedCount() + 1,
            "YetiRank requires one weak draw and every permutation's leaf seed schedule");
        auto EstimatePermutation = [&](Command& command, uint32_t permutation) -> PermutationTree {
        // CUDA estimates complete leaf walks in dataset order. Structure
        // search runs first here, but explicit seeds retain that host order.
        if (Yeti) YetiSeedPosition = 1 + permutation * YetiPermutationLeafSeedCount();
        if (!Coupled && K.LeafMethod == 3) {
            // DocParallel Simple exports the searched, bootstrapped weak
            // target statistics and copies those values to every history.
            command.Dispatch("ReduceStructurePartials",
                {Gradient, StructureWeight, RowIndices, PartitionOffsets, ObjectivePartials},
                K, K.HistogramTiles, true, K.Leaves);
            command.Dispatch("CollectPartitionStatistics", {ObjectivePartials, LeafSums, LeafWeights}, K, K.Leaves, true);
            const struct { uint32_t Leaves, Bootstrap; float L2; uint32_t Reserved; } simple =
                {K.Leaves, uint32_t(BootstrapOptions.bootstrap_type == 2 || BootstrapOptions.bootstrap_type == 3), K.L2, 0};
            command.Dispatch("EstimateSimpleWeakLeaves", {LeafSums, LeafWeights, PartitionOffsets, RowIndices,
                BootstrapOptions.bootstrap_type ? BootstrapMultipliers : SampleWeight, RawValues, Weights},
                simple, K.Leaves, true);
        } else if (Coupled && K.LeafMethod == 3) {
            Require(actualDepth == p.depth && !selected.empty(), "Simple leaves need a complete selected structure");
            // RawValues and Weights came from the final weak-target solution.
        } else if (Coupled) {
            // A generated target reuses weak buffers for its separately sampled
            // fixed leaf target. Complete pending topology/weak commands first.
            if (!Coupled->HasObjectiveValue()) { command.Wait();command.Restart(); }
            Coupled->SetTargetPoint(Prediction, BootstrapOptions, BootstrapOptions.iteration_offset + Completed, permutation);
            Coupled->EncodeLeafLayout(command.Buffer, LeafIds, K.Leaves, &Stats.kernel_dispatches);
            Stats.gpu_seconds += Coupled->TakeAuxiliaryGPUSeconds();
            Coupled->EncodeLeafWeights(command.Buffer, SampleWeight, RowIndices, PartitionOffsets, Weights,
                K.Leaves, &Stats.kernel_dispatches);
            if (UsesBacktracking()) {
                command.Wait(); EstimateCoupledBacktrackingLeaves(); command.Restart();
            } else for (uint32_t iteration = 0; iteration < Options.leaf_estimation_iterations; ++iteration) {
                Coupled->EncodeEdges(command.Buffer, Prediction, RawValues, LeafIds, K.Leaves, true,
                    nullptr, 0, &Stats.kernel_dispatches);
                Coupled->EncodeLeafProjection(command.Buffer, K.Leaves, K.LeafMethod == 1, &Stats.kernel_dispatches);
                if (Regularization.add_ridge) Coupled->EncodeLeafRidge(command.Buffer, RawValues, K.Leaves, K.L2, &Stats.kernel_dispatches);
                Coupled->EncodeLeafDirection(command.Buffer, K.Leaves, K.L2, CoupledNonDiag, &Stats.kernel_dispatches);
                Coupled->EncodeLeafUpdate(command.Buffer, RawValues, Weights, RawValues, K.Leaves, 1, &Stats.kernel_dispatches);
            }
            Coupled->EncodeCenterSolvedPoint(command.Buffer, RawValues, K.Leaves, &Stats.kernel_dispatches);
        } else if (Combination && Combination->HasYeti()) {
            command.Wait(); EstimateCombinationYetiLeaves(); command.Restart();
        } else if (K.LeafMethod == 2) {
            EncodeExactLeaves(command);
        } else if (UsesBacktracking()) {
            command.Wait();
            EstimateBacktrackingLeaves();
            command.Restart();
        } else {
            for (uint32_t iteration = 0; iteration < Options.leaf_estimation_iterations; ++iteration) {
                K.LeafIteration = iteration;
                EncodeObjectivePartials(command);
                EncodeNewtonLeaves(command);
            }
            if (Yeti && Options.leaf_estimation_iterations > 1) {
                // TNewtonLikeWalker evaluates the final point after its last
                // update. Preserve that RNG draw; its derivatives are unused.
                Require(YetiSeedPosition < YetiSeeds.size(), "Missing YetiRank final evaluation seed");
                ++YetiSeedPosition;
            }
        }
        // CUDA PairLogit makes the solved point zero-average across ALL leaves,
        // including empty leaves, before applying the learning rate.
        if (Pairwise && K.LeafMethod != 3) Pairwise->EncodeCenterLeafValues(command.Buffer, RawValues, K.Leaves, &Stats.kernel_dispatches);
        if (Yeti) Yeti->EncodeCenterLeafValues(command.Buffer, RawValues, K.Leaves, &Stats.kernel_dispatches);
        if (Yeti) Require(YetiSeedPosition == 1 + (permutation + 1) * YetiPermutationLeafSeedCount(),
            "YetiRank did not consume the permutation's complete leaf seed schedule");
        command.Dispatch("FinalizeLeafValues", {RawValues, Values}, K, K.Leaves);
        command.Dispatch("AddObjectiveBinModelValue", {LeafIds, Values, Prediction}, K, p.rows);
        EncodeLoss(command);
        command.Wait();
        const float* values = static_cast<const float*>(Values.contents);
        const float* weights = static_cast<const float*>(Weights.contents);
        double totalWeight = 0;
        for (uint32_t leaf = 0; leaf < K.Leaves; ++leaf) {
            Require(std::isfinite(values[leaf]) && std::isfinite(weights[leaf])
                    && (weights[leaf] >= 0 || ((Combination || K.Objective == 13) && K.LeafMethod == 3)),
                    QueryOffsets ? "Invalid GPU query leaf statistics: Newton requires a finite positive regularized Hessian"
                                 : "Invalid GPU leaf statistics");
            totalWeight += weights[leaf];
        }
        Require(K.LeafMethod == 3 || std::abs(totalWeight - TotalWeight) <= std::max(1e-6, TotalWeight * 2e-5),
                "GPU leaf weights do not cover the training sample weights");
        const float loss = ReadLoss();
        if (BootstrapOptions.bootstrap_type == 4 && !BootstrapOptions.mvs_reg_is_set) {
            double meanAbsoluteLeaf = 0;
            for (uint32_t leaf = 0; leaf < K.Leaves; ++leaf) meanAbsoluteLeaf += std::abs(values[leaf]);
            meanAbsoluteLeaf /= K.Leaves;
            MVSLambda = static_cast<float>(meanAbsoluteLeaf * meanAbsoluteLeaf);
            Require(std::isfinite(MVSLambda), "Automatic MVS regularization overflowed float32");
            HasMVSLambda = true;
        }
            return {std::vector<float>(values, values + K.Leaves), std::vector<float>(weights, weights + K.Leaves), loss};
        };
        PermutationTree exported;
        if (!Permutations.empty() && actualDepth)
            std::memcpy(FixedPermutationSplits.contents, selected.data(), sizeof(SplitState) * actualDepth);
        auto EstimateOtherPermutation = [&](uint32_t permutation) {
                Data = Permutations[permutation].Bins; Prediction = Permutations[permutation].Cursor;
                MVSLambda = Permutations[permutation].Lambda; HasMVSLambda = Permutations[permutation].HasLambda;
                Command other(*Context, Stats);
                const bool reuseSimple = K.LeafMethod == 3;
                if (Coupled || Yeti || Combination || reuseSimple) {
                    // CUDA only samples the searched weak target. Leaf
                    // estimation uses the original objective for each cursor.
                    other.Dispatch("ResetQuerywiseLeafIds", {LeafIds}, QueryParams(false), K.Rows);
                } else {
                    EncodeDerivatives(other);
                }
                if (!reuseSimple) {
                    KernelParams clear = K; clear.Leaves = MaxLeaves;
                    other.Dispatch("InitializeLeafValues", {RawValues, Weights}, clear, MaxLeaves);
                }
                K.Leaves = 1; EncodePartitions(other);
                for (uint32_t level = 0; level < actualDepth; ++level) {
                    K.SplitLevel = level;
                    other.Dispatch("UpdateFixedPermutationSplit", {Data, LeafIds, FixedPermutationSplits}, K, K.Rows);
                    K.Leaves = uint32_t(1) << (level + 1);
                    EncodePartitions(other);
                }
                PermutationTree estimate;
                if (reuseSimple) {
                    // doc_parallel_boosting copies the searched model into
                    // every dataset. NeedEstimation is false for Simple, so
                    // reuse the exact scaled values and weak Hessian weights.
                    other.Dispatch("AddObjectiveBinModelValue", {LeafIds, Values, Prediction}, K, p.rows);
                    EncodeLoss(other);
                    other.Wait();
                    estimate = {exported.Values, exported.Weights, ReadLoss()};
                } else {
                    estimate = EstimatePermutation(other, permutation);
                }
                Permutations[permutation].Lambda = MVSLambda;
                Permutations[permutation].HasLambda = HasMVSLambda;
                return estimate;
        };
        if (LangevinLeaves() || (Combination && Dynamic) || (RegularizedLeaves() && !Coupled && !Yeti &&
            PermutationCount() > 1 && UsesBacktracking())) {
            command.Wait();
            exported = EstimateFeatureParallelCombinationLeaves();
        } else {
        // The callback advances the actual shared host RNG. Complete each
        // DocParallel Combination task in dataset order even when structure
        // search chose a later history; legacy seed-packet targets keep their
        // preceding search-first execution order.
        const bool orderedCombinationTasks = Combination && Combination->HasYeti() && K.LeafMethod != 3;
        const uint32_t firstPermutation = orderedCombinationTasks ? 0 : SearchPermutation;
        if (firstPermutation != SearchPermutation) {
            command.Wait();
            exported = EstimateOtherPermutation(firstPermutation);
        } else {
            exported = EstimatePermutation(command, firstPermutation);
        }
        if (!Permutations.empty()) {
            Permutations[firstPermutation].Lambda = MVSLambda;
            Permutations[firstPermutation].HasLambda = HasMVSLambda;
            for (uint32_t permutation = 0; permutation < Permutations.size(); ++permutation) {
                if (permutation == firstPermutation) continue;
                auto estimate = EstimateOtherPermutation(permutation);
                if (permutation + 1 == Permutations.size()) exported = std::move(estimate);
            }
            Data = Permutations.back().Bins; Prediction = Permutations.back().Cursor;
            MVSLambda = Permutations.back().Lambda; HasMVSLambda = Permutations.back().HasLambda;
        }
        }
        if (Yeti) YetiSeeds.clear();
        const float* values = exported.Values.data();
        const float* weights = exported.Weights.data();
        const float loss = exported.Loss;
        TreeDepths[Completed] = actualDepth;
        for (uint32_t level = 0; level < actualDepth; ++level) {
            const uint64_t offset = uint64_t(Completed) * p.depth + level;
            SplitFeatures[offset] = selected[level].Feature;
            SplitBins[offset] = selected[level].Bin;
            SplitTypes[offset] = static_cast<uint8_t>(selected[level].Type);
        }
        if (Deep) {
            Require(DeepModelBytes + 8ull * K.Leaves <= MaxOutputBytes, "Completed compact model exceeds 512 MiB");
            DeepTreeValues[Completed].assign(values, values + K.Leaves);
            DeepTreeWeights[Completed].assign(weights, weights + K.Leaves);
            DeepModelBytes += 8ull * K.Leaves;
        } else {
            std::memcpy(TreeValues.data() + uint64_t(Completed) * MaxLeaves, values, 4ull * K.Leaves);
            std::memcpy(TreeWeights.data() + uint64_t(Completed) * MaxLeaves, weights, 4ull * K.Leaves);
        }
        Losses[++Completed] = loss;
    }
};

std::mutex RegistryMutex;
std::unordered_map<uintptr_t, std::shared_ptr<Session>> Sessions;
struct QueryCrossEntropyMetricSession {
    std::mutex Mutex;
    std::unique_ptr<CBMQueryCrossEntropyRuntime> Metric;
    uint32_t Rows=0;
    uint64_t Evaluations=0;
};
std::unordered_map<uintptr_t,std::shared_ptr<QueryCrossEntropyMetricSession>> QueryCrossEntropyMetrics;
std::atomic<uintptr_t> NextHandle{1};
std::shared_ptr<Session> GetSession(void* handle) {
    std::lock_guard<std::mutex> guard(RegistryMutex);
    const auto found = Sessions.find(reinterpret_cast<uintptr_t>(handle));
    Require(found != Sessions.end(), "Training session is closed or invalid");
    return found->second;
}

std::shared_ptr<QueryCrossEntropyMetricSession> GetQueryCrossEntropyMetric(void* handle) {
    std::lock_guard<std::mutex> guard(RegistryMutex);
    const auto found=QueryCrossEntropyMetrics.find(reinterpret_cast<uintptr_t>(handle));
    Require(found!=QueryCrossEntropyMetrics.end(),"QueryCrossEntropy metric is closed or invalid");
    return found->second;
}

template <class Callback>
int ApiCall(char* error, size_t capacity, Callback callback) {
    CopyText(error, capacity, "");
    struct Invocation { Callback& Function; char* Error; size_t Capacity; } invocation = {callback, error, capacity};
    auto invoke = [](void* context) -> int {
        auto& call = *static_cast<Invocation*>(context);
        // Catch Objective-C exceptions before they can reach the pure C++
        // catch-all, preserving Metal's actual exception reason.
        @try { call.Function(); return 0; }
        @catch (NSException* exception) { CopyText(call.Error, call.Capacity, [[exception reason] UTF8String]); }
        return 1;
    };
    @autoreleasepool {
        return CBMInvokeCppGuard(invoke, &invocation, error, capacity);
    }
}
} // namespace

extern "C" int cbm_device_info(char* name, size_t nameCapacity, char* error, size_t errorCapacity) {
    CopyText(name, nameCapacity, "");
    return ApiCall(error, errorCapacity, [&] {
        if (@available(macOS 13.0, *)) {}
        else throw std::runtime_error("Metal training requires macOS 13 or newer (Metal 3)");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        Require(device != nil, "No Metal GPU device is available");
        Require([device hasUnifiedMemory] && [device supportsFamily:MTLGPUFamilyApple7],
                "This backend requires an Apple Silicon GPU (Apple family 7 or newer)");
        CopyText(name, nameCapacity, [device.name UTF8String]);
    });
}
extern "C" int cbm_session_create(const CBMSessionParams* params, const uint8_t* bins,
    const float* targets, const float* weights, const float* initialPredictions,
    const uint32_t* candidateFeatures, const uint32_t* candidateBins,
    const uint8_t* candidateTypes, void** handle, char* error, size_t errorCapacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, errorCapacity, [&] {
        Require(handle != nullptr, "Session output pointer is required");
        auto session = std::make_shared<Session>(params, bins, targets, weights, initialPredictions,
                                                 candidateFeatures, candidateBins, candidateTypes);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_session_create_configured(const CBMSessionParams* params,
    const CBMObjectiveOptions* objectiveOptions, const uint8_t* bins, const float* targets,
    const float* weights, const float* initialPredictions, const uint32_t* candidateFeatures,
    const uint32_t* candidateBins, const uint8_t* candidateTypes, void** handle,
    char* error, size_t errorCapacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, errorCapacity, [&] {
        Require(handle != nullptr, "Session output pointer is required");
        ValidateObjectiveOptions(objectiveOptions);
        auto session = std::make_shared<Session>(params, bins, targets, weights, initialPredictions,
            candidateFeatures, candidateBins, candidateTypes, objectiveOptions);
        session->ConfigureObjective(objectiveOptions);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_session_create_custom(const CBMSessionParams* params,
    const CBMObjectiveOptions* objectiveOptions, const char* source,
    const uint8_t* bins, const float* targets, const float* weights, const float* initialPredictions,
    const uint32_t* candidateFeatures, const uint32_t* candidateBins, const uint8_t* candidateTypes,
    void** handle, char* error, size_t errorCapacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, errorCapacity, [&] {
        Require(params && objectiveOptions && handle && source,
            "Custom Metal objective requires parameters, source and output handle");
        ValidateObjectiveOptions(objectiveOptions);
        Require(params->objective == 20 && objectiveOptions->objective == 20,
            "Custom Metal constructor requires objective 20");
        auto session = std::make_shared<Session>(params, bins, targets, weights, initialPredictions,
            candidateFeatures, candidateBins, candidateTypes, objectiveOptions, nullptr, nullptr,
            nullptr, nullptr, nullptr, 0.1f, nullptr, source);
        session->ConfigureObjective(objectiveOptions);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_session_create_combination(const CBMSessionParams* params,
    const CBMObjectiveOptions* objectiveOptions, const CBMCombinationOptions* combinationOptions,
    const CBMCombinationComponent* components, const uint32_t* groupOffsets,
    const uint32_t* pairWinners, const uint32_t* pairLosers, const float* pairWeights,
    const uint8_t* bins, const float* targets, const float* weights, const float* initialPredictions,
    const uint32_t* candidateFeatures, const uint32_t* candidateBins, const uint8_t* candidateTypes,
    void** handle, char* error, size_t errorCapacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, errorCapacity, [&] {
        Require(params && objectiveOptions && combinationOptions && handle && !combinationOptions->reserved,
            "Combination requires parameters, component options and output handle");
        ValidateObjectiveOptions(objectiveOptions);
        Require(params->objective == 19 && objectiveOptions->objective == 19,
            "Combination constructor requires objective 19");
        Require(params->train.rows && params->train.rows <= MaxRows && params->train.depth <= 16,
            "Invalid Combination row or depth capacity");
        auto combination = std::make_shared<CBMCombinationRuntime>(GetRuntime().Device, params->train.rows,
            combinationOptions->component_count, components, combinationOptions->group_count, groupOffsets,
            combinationOptions->pair_count, pairWinners, pairLosers, pairWeights, uint32_t(1) << params->train.depth,
            std::min<uint32_t>((params->train.rows + 255) / 256, 4096));
        combination->ValidateTargets(targets, weights);
        auto session = std::make_shared<Session>(params, bins, targets, weights, initialPredictions,
            candidateFeatures, candidateBins, candidateTypes, objectiveOptions, nullptr, nullptr,
            nullptr, nullptr, nullptr, 0.1f, combination);
        session->ConfigureObjective(objectiveOptions);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_session_set_combination_yeti_seeds(void* handle, uint32_t count,
    const uint64_t* seeds, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetCombinationYetiSeeds(count, seeds);
    });
}
extern "C" int cbm_session_set_combination_yeti_seed_callback(void* handle,
    CBMCombinationYetiSeedCallback callback, void* context, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetCombinationYetiSeedCallback(callback, context);
    });
}
extern "C" int cbm_session_step(void* handle, CBMStepInfo* info, uint32_t* depth,
    uint32_t* features, uint32_t* bins, uint8_t* types, float* values, float* weights,
    char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        Require(info && depth && values && weights, "Step info, depth and leaf output buffers are required");
        Require(session->Options.train.depth == 0 || (features && bins), "Step split output buffers are required");
        session->Step();
        session->CopyStep(info, depth, features, bins, types, values, weights);
    });
}
extern "C" int cbm_session_create_query(const CBMSessionParams* params,
    const CBMObjectiveOptions* objectiveOptions, const CBMQueryOptions* queryOptions,
    const uint32_t* groupOffsets, const uint8_t* bins, const float* targets,
    const float* weights, const float* initialPredictions, const uint32_t* candidateFeatures,
    const uint32_t* candidateBins, const uint8_t* candidateTypes, void** handle,
    char* error, size_t errorCapacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, errorCapacity, [&] {
        Require(handle && queryOptions, "Query options and session output pointer are required");
        ValidateObjectiveOptions(objectiveOptions);
        Require(objectiveOptions->objective == 12 || objectiveOptions->objective == 13,
                "Query constructor supports QueryRMSE and QuerySoftMax");
        auto session = std::make_shared<Session>(params, bins, targets, weights, initialPredictions,
            candidateFeatures, candidateBins, candidateTypes, objectiveOptions, queryOptions, groupOffsets);
        session->ConfigureObjective(objectiveOptions);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_session_begin_tree(void* handle, char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->BeginTree();
    });
}
extern "C" int cbm_session_grow_tree(void* handle, CBMStructureInfo* info, char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        Require(info != nullptr, "Structure step output is required");
        session->GrowTree();
        session->CopyStructureInfo(info);
    });
}
extern "C" int cbm_session_append_features(void* handle, const CBMAppendFeatureOptions* options,
    const uint8_t* const* matrices, const uint32_t* features, const uint32_t* bins,
    const uint8_t* types, const uint32_t* counts, const float* weights,
    const uint8_t* flags, const uint8_t* used, uint32_t* first, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->AppendFeatures(options, matrices, features, bins, types, counts, weights, flags, used, first);
    });
}
extern "C" int cbm_session_set_feature_activity(void* handle, uint32_t count,
    const uint8_t* active, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetFeatureActivity(count, active);
    });
}
extern "C" int cbm_session_copy_feature_metadata(void* handle, uint32_t capacity,
    uint32_t* counts, float* weights, uint8_t* flags, uint8_t* used, uint8_t* active,
    char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->CopyFeatureMetadata(capacity, counts, weights, flags, used, active);
    });
}
extern "C" int cbm_session_restore_feature_metadata(void* handle, uint32_t count,
    const uint8_t* flags, const uint8_t* used, const uint8_t* active,
    char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->RestoreFeatureMetadata(count, flags, used, active);
    });
}
extern "C" int cbm_session_finish_tree(void* handle, CBMStepInfo* info, uint32_t* depth,
    uint32_t* features, uint32_t* bins, uint8_t* types, float* values, float* weights,
    char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        Require(info && depth && values && weights, "Step info, depth and leaf output buffers are required");
        Require(session->Options.train.depth == 0 || (features && bins), "Step split output buffers are required");
        session->FinishTree();
        session->CopyStep(info, depth, features, bins, types, values, weights);
    });
}
extern "C" int cbm_session_result(void* handle, uint32_t capacity, uint32_t* completed,
    uint32_t* depths, uint32_t* features, uint32_t* bins, uint8_t* types,
    float* values, float* weights, float* predictions, float* loss, CBMTrainStats* stats,
    char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->CopyResult(capacity, completed, depths, features, bins, types,
                            values, weights, predictions, loss, stats);
    });
}
extern "C" int cbm_session_create_pair(const CBMSessionParams* params,
    const CBMObjectiveOptions* objectiveOptions, const CBMPairOptions* pairOptions,
    const uint32_t* winners, const uint32_t* losers, const float* pairWeights,
    const uint32_t* groupOffsets, const uint8_t* bins, const float* initialPredictions,
    const uint32_t* candidateFeatures, const uint32_t* candidateBins, const uint8_t* candidateTypes,
    void** handle, char* error, size_t errorCapacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, errorCapacity, [&] {
        Require(params && pairOptions && handle, "Pair options, training parameters and output pointer are required");
        ValidateObjectiveOptions(objectiveOptions);
        Require(params->objective == 14 && objectiveOptions->objective == 14,
                "Supplied-pair constructor requires PairLogit");
        Require(params->train.rows > 0 && params->train.rows <= MaxRows && params->train.depth <= 16,
                "Invalid PairLogit row count or tree depth");
        Require(!pairOptions->reserved0 && !pairOptions->reserved1, "Reserved pair options must be zero");
        auto pairwise = std::make_shared<CBMPairwiseRuntime>(GetRuntime().Device, params->train.rows,
            pairOptions->pair_count, winners, losers, pairWeights, uint32_t(1) << params->train.depth,
            std::min<uint32_t>((params->train.rows + 255) / 256, 4096), pairOptions->group_count, groupOffsets);
        std::vector<float> targets(params->train.rows, 0.0f);
        auto session = std::make_shared<Session>(params, bins, targets.data(), pairwise->IncidentWeights().data(),
            initialPredictions, candidateFeatures, candidateBins, candidateTypes, objectiveOptions,
            nullptr, nullptr, pairwise);
        session->ConfigureObjective(objectiveOptions);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_session_create_pair_matrix(const CBMSessionParams* params,
    const CBMObjectiveOptions* objectiveOptions, const CBMPairOptions* pairOptions, float nonDiag,
    const uint32_t* winners, const uint32_t* losers, const float* pairWeights,
    const uint32_t* groupOffsets, const uint8_t* bins, const float* sampleWeights, const float* initialPredictions,
    const uint32_t* candidateFeatures, const uint32_t* candidateBins, const uint8_t* candidateTypes,
    void** handle, char* error, size_t errorCapacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, errorCapacity, [&] {
        Require(params && pairOptions && handle, "Pairwise matrix options, parameters and output pointer are required");
        ValidateObjectiveOptions(objectiveOptions);
        Require(params->objective == 15 && objectiveOptions->objective == 15,
                "Full-matrix pair constructor requires PairLogitPairwise objective 15");
        const uint32_t rows = params->train.rows;
        Require(rows > 0 && rows <= MaxRows && params->train.depth <= 8,
                "Invalid PairLogitPairwise row count or depth (maximum 8)");
        Require(!pairOptions->reserved0 && !pairOptions->reserved1, "Reserved pair options must be zero");
        Require(std::isfinite(nonDiag) && nonDiag >= 0, "Pairwise non-diagonal regularization must be finite and nonnegative");
        Require(pairOptions->pair_count > 0 && pairOptions->pair_count <= MaxRows && winners && losers && pairWeights,
                "Valid supplied pair arrays are required");
        const uint32_t groups = pairOptions->group_count;
        Require(groups <= rows && ((groups != 0) == (groupOffsets != nullptr)), "Invalid pairwise query metadata");
        if (groups) {
            Require(groupOffsets[0] == 0 && groupOffsets[groups] == rows, "Pairwise query offsets must span all rows");
            for (uint32_t group = 0; group < groups; ++group)
                Require(groupOffsets[group] < groupOffsets[group + 1], "Pairwise query offsets must increase strictly");
            for (uint32_t edge = 0; edge < pairOptions->pair_count; ++edge)
                Require(winners[edge] < rows && losers[edge] < rows &&
                    std::upper_bound(groupOffsets, groupOffsets + groups + 1, winners[edge]) ==
                    std::upper_bound(groupOffsets, groupOffsets + groups + 1, losers[edge]),
                    "Every pair must remain inside one query group");
        }
        auto coupled = std::make_shared<CBMPairwiseMatrixRuntime>(GetRuntime().Device, rows,
            pairOptions->pair_count, winners, losers, pairWeights, uint32_t(1) << params->train.depth,
            std::max<uint32_t>(params->train.candidates, 1), 32, FullMatrixTargetBudget(*params));
        std::vector<float> targets(rows, 0.0f);
        auto session = std::make_shared<Session>(params, bins, targets.data(), sampleWeights, initialPredictions,
            candidateFeatures, candidateBins, candidateTypes, objectiveOptions, nullptr, nullptr, nullptr, nullptr, coupled, nonDiag);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_session_create_query_cross_entropy(const CBMSessionParams* params,
    const CBMObjectiveOptions* objectiveOptions, const CBMQueryCrossEntropyOptions* queryOptions, float nonDiag,
    const uint32_t* groupOffsets, const float* queryScales, const uint8_t* bins, const float* targets,
    const float* weights, const float* initialPredictions, const uint32_t* candidateFeatures,
    const uint32_t* candidateBins, const uint8_t* candidateTypes, void** handle, char* error, size_t errorCapacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, errorCapacity, [&] {
        Require(params && queryOptions && handle, "QueryCrossEntropy options, parameters and output pointer are required");
        ValidateObjectiveOptions(objectiveOptions);
        Require(params->objective == 16 && objectiveOptions->objective == 16,
                "QueryCrossEntropy constructor requires objective 16");
        Require(!queryOptions->reserved0 && !queryOptions->reserved1 && !queryOptions->reserved2,
                "Reserved QueryCrossEntropy options must be zero");
        const uint32_t rows = params->train.rows;
        Require(rows > 0 && rows <= MaxRows && params->train.depth <= 8 && params->train.score_function != 0,
                "QueryCrossEntropy requires depth <= 8 and a Newton structure score");
        Require(std::isfinite(nonDiag) && nonDiag >= 0, "QueryCrossEntropy non-diagonal regularization must be finite and nonnegative");
        std::vector<float> unitWeights;
        if (!weights) { unitWeights.assign(rows, 1.0f); weights = unitWeights.data(); }
        auto coupled = std::make_shared<CBMQueryCrossEntropyRuntime>(GetRuntime().Device, rows, queryOptions->group_count,
            targets, weights, groupOffsets, queryScales, objectiveOptions->objective_param,
            uint32_t(1) << params->train.depth, std::max<uint32_t>(params->train.candidates, 1),
            8, 64, FullMatrixTargetBudget(*params));
        auto session = std::make_shared<Session>(params, bins, targets, weights, initialPredictions,
            candidateFeatures, candidateBins, candidateTypes, objectiveOptions, nullptr, nullptr, nullptr, nullptr, coupled, nonDiag);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_query_cross_entropy_metric(uint32_t rows,uint32_t groups,
    const float* targets,const float* weights,const float* predictions,
    const uint32_t* groupOffsets,const float* queryScales,float alpha,
    double* result,char* error,size_t errorCapacity) {
    return ApiCall(error,errorCapacity,[&] {
        Require(result && rows > 0 && rows <= MaxRows,"QueryCrossEntropy metric rows and output are required");
        std::vector<float> unitWeights;
        if (!weights) { unitWeights.assign(rows,1.f);weights=unitWeights.data(); }
        auto& context=GetRuntime();
        CBMQueryCrossEntropyRuntime metric(context.Device,rows,groups,targets,weights,groupOffsets,queryScales,alpha,1,1,8,64,1ull<<30,32,true);
        CBMTrainStats stats={};Command command(context,stats);
        metric.EncodeMetric(command.Buffer,predictions);
        command.Wait();const auto loss=metric.ReadLoss();*result=loss.first/loss.second;
    });
}
extern "C" int cbm_query_cross_entropy_metric_create(uint32_t rows,uint32_t groups,
    const float* targets,const float* weights,const uint32_t* groupOffsets,const float* queryScales,
    uint64_t budget,void** handle,uint64_t* allocatedBytes,char* error,size_t errorCapacity) {
    if(handle)*handle=nullptr;
    if(allocatedBytes)*allocatedBytes=0;
    return ApiCall(error,errorCapacity,[&] {
        Require(handle && allocatedBytes && rows && rows<=MaxRows,"QueryCrossEntropy metric dimensions and outputs are required");
        std::vector<float> unitWeights;
        if(!weights){unitWeights.assign(rows,1.f);weights=unitWeights.data();}
        auto session=std::make_shared<QueryCrossEntropyMetricSession>();session->Rows=rows;
        session->Metric=std::make_unique<CBMQueryCrossEntropyRuntime>(GetRuntime().Device,rows,groups,targets,weights,
            groupOffsets,queryScales,.95f,1,1,8,64,budget,32,true);
        const uintptr_t id=NextHandle.fetch_add(1);Require(id!=0,"Metric identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        QueryCrossEntropyMetrics.emplace(id,session);*handle=reinterpret_cast<void*>(id);*allocatedBytes=session->Metric->AllocatedBytes();
    });
}
extern "C" int cbm_query_cross_entropy_metric_evaluate(void* handle,uint32_t rows,
    const float* predictions,float alpha,double* result,uint64_t* evaluations,char* error,size_t errorCapacity) {
    return ApiCall(error,errorCapacity,[&] {
        auto session=GetQueryCrossEntropyMetric(handle);std::lock_guard<std::mutex> guard(session->Mutex);
        Require(rows==session->Rows && predictions && result && evaluations,"QueryCrossEntropy metric prediction shape and outputs are required");
        auto& context=GetRuntime();CBMTrainStats stats={};Command command(context,stats);
        session->Metric->ClearStatus();session->Metric->EncodeMetricWithAlpha(command.Buffer,predictions,alpha);
        command.Wait();const auto loss=session->Metric->ReadLoss();*result=loss.first/loss.second;
        *evaluations=++session->Evaluations;
    });
}
extern "C" void cbm_query_cross_entropy_metric_destroy(void* handle) {
    std::lock_guard<std::mutex> guard(RegistryMutex);QueryCrossEntropyMetrics.erase(reinterpret_cast<uintptr_t>(handle));
}
extern "C" int cbm_session_create_yeti_pairwise(const CBMSessionParams* params,
    const CBMObjectiveOptions* objectiveOptions,const CBMYetiRankPairwiseOptions* yetiOptions,float nonDiag,
    const uint32_t* groupOffsets,const uint8_t* bins,const float* targets,const float* weights,
    const float* initialPredictions,const uint32_t* candidateFeatures,const uint32_t* candidateBins,
    const uint8_t* candidateTypes,void** handle,char* error,size_t errorCapacity) {
    if(handle)*handle=nullptr;
    return ApiCall(error,errorCapacity,[&] {
        Require(params && yetiOptions && handle,"YetiRankPairwise parameters, options and output are required");
        ValidateObjectiveOptions(objectiveOptions);
        Require(params->objective==18 && objectiveOptions->objective==18,"YetiRankPairwise constructor requires objective 18");
        Require(yetiOptions->sampling_unit<=1,"YetiRankPairwise sampling unit must be Object or Group");
        Require(std::isfinite(nonDiag) && nonDiag>=0,"Invalid YetiRankPairwise non-diagonal regularization");
        const uint32_t rows=params->train.rows;
        Require(rows && rows<=MaxRows && params->train.depth<=8,"Invalid YetiRankPairwise row/depth capacity");
        std::vector<float> unitWeights;if(!weights){unitWeights.assign(rows,1.f);weights=unitWeights.data();}
        auto coupled=std::make_shared<CBMPFoundPairRuntime>(GetRuntime().Device,rows,yetiOptions->group_count,
            groupOffsets,targets,weights,yetiOptions->permutations,yetiOptions->decay,
            uint32_t(1)<<params->train.depth,std::max(params->train.candidates,1u),32,FullMatrixTargetBudget(*params));
        coupled->SetGroupSampling(yetiOptions->sampling_unit==1);
        auto session=std::make_shared<Session>(params,bins,targets,weights,initialPredictions,
            candidateFeatures,candidateBins,candidateTypes,objectiveOptions,nullptr,nullptr,nullptr,nullptr,coupled,nonDiag);
        const uintptr_t id=NextHandle.fetch_add(1);Require(id!=0,"Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);Sessions.emplace(id,std::move(session));*handle=reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_session_create_yeti(const CBMSessionParams* params,
    const CBMObjectiveOptions* objectiveOptions, const CBMYetiRankOptions* yetiOptions,
    const uint32_t* groupOffsets, const uint8_t* bins, const float* targets,
    const float* weights, const float* initialPredictions,
    const uint32_t* candidateFeatures, const uint32_t* candidateBins, const uint8_t* candidateTypes,
    void** handle, char* error, size_t errorCapacity) {
    if (handle) *handle = nullptr;
    return ApiCall(error, errorCapacity, [&] {
        Require(params && yetiOptions && handle, "YetiRank options, training parameters and output pointer are required");
        ValidateObjectiveOptions(objectiveOptions);
        Require(params->objective == 17 && objectiveOptions->objective == 17,
            "YetiRank constructor requires objective 17");
        Require(yetiOptions->legacy_prefix_centering <= 1, "Invalid YetiRank centering mode");
        auto yeti = std::make_shared<CBMYetiRankRuntime>(GetRuntime().Device, params->train.rows,
            yetiOptions->group_count, groupOffsets, yetiOptions->permutations, yetiOptions->decay,
            yetiOptions->legacy_prefix_centering);
        auto session = std::make_shared<Session>(params, bins, targets, weights, initialPredictions,
            candidateFeatures, candidateBins, candidateTypes, objectiveOptions, nullptr, nullptr, nullptr, yeti);
        session->ConfigureObjective(objectiveOptions);
        const uintptr_t id = NextHandle.fetch_add(1);
        Require(id != 0, "Session identifier capacity exhausted");
        std::lock_guard<std::mutex> guard(RegistryMutex);
        Sessions.emplace(id, std::move(session));
        *handle = reinterpret_cast<void*>(id);
    });
}
extern "C" int cbm_session_set_yeti_oracle_seeds(void* handle, uint32_t count,
    const uint64_t* seeds, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetYetiOracleSeeds(count, seeds);
    });
}
extern "C" int cbm_session_set_yeti_leaf_seeds(void* handle, uint32_t count,
    const uint64_t* seeds, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetYetiLeafSeeds(count, seeds);
    });
}
extern "C" void cbm_session_close(void* handle) {
    std::lock_guard<std::mutex> guard(RegistryMutex);
    Sessions.erase(reinterpret_cast<uintptr_t>(handle));
}
extern "C" int cbm_session_set_objective(void* handle, const CBMObjectiveOptions* options,
    char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->ConfigureObjective(options);
    });
}
extern "C" int cbm_session_set_bootstrap(void* handle, const CBMBootstrapOptions* options,
    char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->ConfigureBootstrap(options);
    });
}
extern "C" int cbm_session_set_score_noise(void* handle, const CBMScoreNoiseOptions* options,
    char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->ConfigureScoreNoise(options);
    });
}
extern "C" int cbm_session_get_bootstrap_state(void* handle, uint32_t* absolute, float* lambda,
    uint32_t* valid, char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->CopyBootstrapState(absolute, lambda, valid);
    });
}
extern "C" int cbm_session_set_permutations(void* handle, uint32_t count, const uint8_t* const* bins,
    const float* const* cursors, const float* lambdas, const uint8_t* valid, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->ConfigurePermutations(count, bins, cursors, lambdas, valid);
    });
}
extern "C" int cbm_session_select_permutation(void* handle, uint32_t index, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->SelectPermutation(index);
    });
}
extern "C" int cbm_session_copy_permutation_state(void* handle, uint32_t capacity, float* cursors,
    float* lambdas, uint8_t* valid, char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->CopyPermutationState(capacity, cursors, lambdas, valid);
    });
}
extern "C" int cbm_session_set_feature_penalties(void* handle, const CBMFeaturePenaltyOptions* options,
    const uint32_t* counts, const float* weights, const uint8_t* used, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->ConfigureFeaturePenalties(options, counts, weights, used);
    });
}
extern "C" int cbm_session_copy_feature_penalty_state(void* handle, uint8_t* used, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->CopyFeaturePenaltyState(used);
    });
}
extern "C" int cbm_session_set_feature_sampling_mask(void* handle, uint32_t featureCount,
    const uint8_t* active, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetFeatureSamplingMask(featureCount, active);
    });
}
extern "C" int cbm_session_set_langevin(void* handle, float temperature, uint32_t weakNoise,
    CBMLangevinNoiseCallback noise, CBMLangevinSeedCallback seed, void* context, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->ConfigureLangevin(temperature, weakNoise, noise, seed, context);
    });
}
extern "C" int cbm_session_set_regularization(void* handle, const CBMRegularizationOptions* options,
    char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->ConfigureRegularization(options);
    });
}
extern "C" int cbm_session_set_meta_l2_exponent_callback(void* handle, CBMMetaL2ExponentCallback callback,
    void* context, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->SetMetaL2Callback(callback, context);
    });
}
extern "C" int cbm_session_get_workspace_info(void* handle, uint32_t* tiles, uint64_t* histogramBytes,
    uint64_t* peakBytes, char* error, size_t capacity) {
    return ApiCall(error, capacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->CopyWorkspaceInfo(tiles, histogramBytes, peakBytes);
    });
}
extern "C" int cbm_session_copy_predictions(void* handle, float* predictions,
    char* error, size_t errorCapacity) {
    return ApiCall(error, errorCapacity, [&] {
        auto session = GetSession(handle); std::lock_guard<std::mutex> guard(session->Mutex);
        session->CopyPredictions(predictions);
    });
}
extern "C" int cbm_train(const CBMTrainParams* params, const uint8_t* bins,
    const float* targets, const uint32_t* candidateFeatures, const uint32_t* candidateBins,
    uint32_t* depths, uint32_t* features, uint32_t* splitBins, float* values,
    float* weights, float* predictions, float* rmse, CBMTrainStats* stats,
    char* error, size_t errorCapacity) {
    if (stats) *stats = {};
    return ApiCall(error, errorCapacity, [&] {
        Require(params != nullptr, "Training parameters are required");
        Require(depths && values && weights && predictions && rmse,
                "Training output buffers must not be null");
        Require(params->depth == 0 || (features && splitBins), "Split output buffers are required");
        CBMSessionParams options = {*params, 0, 1, 0, 0};
        Session session(&options, bins, targets, nullptr, nullptr, candidateFeatures, candidateBins, nullptr);
        while (session.Completed < params->iterations) { @autoreleasepool { session.Step(); } }
        uint32_t completed = 0;
        session.CopyResult(params->iterations, &completed, depths, features, splitBins, nullptr,
                           values, weights, predictions, rmse, stats);
    });
}
