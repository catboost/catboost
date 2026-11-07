#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_multiclass.h"
#include "metal_kernels.h"
#include "metal_additional_objective_kernels.h"
#include "metal_objective_kernels.h"
#include "metal_histogram_kernels.h"
#include "metal_incremental_partition_kernels.h"
#include "metal_deep_partition_kernels.h"
#include "metal_multiclass_math.h"
#include "metal_multiclass_scores.h"
#include "metal_greedy_kernels.h"
#include "metal_fixed_splits.h"
#include "metal_exception.h"
#include "metal_greedy_bootstrap_kernels.h"
#include "metal_greedy_vector_scores.h"
#include "metal_multioutput_math_kernels.h"
#include "metal_multiclass_kernels.h"
#include "metal_bootstrap_kernels.h"
#include "metal_score_noise_kernels.h"
#include "metal_multiclass_bootstrap.h"
#include "metal_multiclass_backtracking.h"
#include "metal_vector_langevin.h"
#include "metal_kernel_abi.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr uint64_t MemoryLimit = uint64_t(1) << 30;
using KernelParams = CBMMetalKernelParams;
struct MathParams {
    uint32_t Rows, Classes, Objective, Leaves;
    float L2, MinLeafWeight;
    uint32_t LeafMethod, Reserved;
};
struct TrainingParams { uint32_t Dimensions, HistogramStride, LeafStride, Reserved; };
struct BootstrapParams {
    uint32_t Rows, Type, SeedLow, SeedHigh, Iteration, Stream, Reserved0, Reserved1;
    float Temperature, Subsample, MVSLambda, NoiseScale;
};
struct MulticlassBootstrapParams { uint32_t Rows, Dimensions, MultiLogit, Reserved; };
struct BacktrackingParams { float Step; uint32_t Type, Reserved0, Reserved1; };
using SplitState = CBMMetalSplitState;
static_assert(CBMMetalKernelAbiVersion == 2, "Review multiclass bindings before adopting a new shared Metal ABI");
static_assert(sizeof(KernelParams) == 96 && sizeof(MathParams) == 32 &&
              sizeof(SplitState) == 32 && sizeof(CBMMulticlassParams) == 64, "Metal ABI mismatch");
void Require(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error(message);
}
// Literal validation messages allocate only when a check fails.
void Require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
void Text(char* output, size_t capacity, const char* text) {
    if (!output || !capacity) return;
    size_t length = std::min(capacity - 1, std::strlen(text));
    std::memcpy(output, text, length); output[length] = 0;
}
std::string Error(NSError* error) { return error ? error.localizedDescription.UTF8String : "unknown Metal error"; }
uint32_t Dimension(const MathParams& p) { return p.Classes - uint32_t(p.Objective == 0); }
uint32_t StatsWidth(const MathParams& p) {
    return 1 + p.Classes + (p.Objective == 0 ? p.Classes * (p.Classes + 1) / 2 : p.Classes);
}
void ValidateMath(const MathParams& p, const uint32_t* labels, const float* weights, const float* logits,
                  const float* targets = nullptr, bool allowSimple = false) {
    Require(p.Rows > 0 && p.Rows <= (1u << 24), "rows must be in [1,16777216]");
    Require(p.Classes >= 2 && p.Classes <= 64, "classes must be in [2,64]");
    Require(p.Objective <= 5 && (p.LeafMethod <= 1 || (allowSimple && p.LeafMethod == 3)),
            "Unsupported vector objective or leaf method");
    Require(p.Objective != 3 || p.Classes == 2, "RMSEWithUncertainty requires two outputs");
    Require(p.Leaves > 0 && p.Leaves <= 65536, "leaves must be in [1,65536]");
    Require(std::isfinite(p.L2) && p.L2 >= 0, "l2 must be finite and nonnegative");
    Require(p.Objective < 2 ? labels != nullptr : targets != nullptr, "Targets are required");
    if (p.Objective >= 2) for (uint64_t i = 0; i < uint64_t(p.Rows) * (p.Objective == 3 ? 1 : p.Classes); ++i) {
        Require(std::isfinite(targets[i]), "Targets must be finite");
        if (p.Objective >= 4) Require(targets[i] >= 0 && targets[i] <= 1, "Multilabel targets must be in [0,1]");
        if (p.Objective == 4) Require(targets[i] == 0 || targets[i] == 1, "MultiLogloss targets must be binary");
    }
    double total = 0;
    for (uint32_t row = 0; row < p.Rows; ++row) {
        if (p.Objective < 2) Require(labels[row] < p.Classes, "Class label is outside [0,classes)");
        const float weight = weights ? weights[row] : 1;
        Require(std::isfinite(weight) && weight >= 0, "weights must be finite and nonnegative");
        total += weight;
    }
    Require(std::isfinite(total) && total > 0 && total < 1e30, "Total weight must be positive and below 1e30");
    if (logits) for (uint64_t i = 0; i < uint64_t(p.Rows) * Dimension(p); ++i)
        Require(std::isfinite(logits[i]), "Initial logits must be finite");
}
struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device && Device.hasUnifiedMemory && [Device supportsFamily:MTLGPUFamilyApple7],
                "Multiclass training requires an Apple Silicon Metal GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Metal command queue allocation failed");
        MTLCompileOptions* options = [MTLCompileOptions new];
        if (@available(macOS 13.0, *)) options.languageVersion = MTLLanguageVersion3_0;
        else throw std::runtime_error("Multiclass training requires macOS 13 or newer");
        options.fastMathEnabled = NO;
        NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s",
            CBMMetalBootstrapSource, CBMMetalScoreNoiseSource,
            CBMMetalSource, CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
            CBMMetalHistogramSource, CBMMetalIncrementalPartitionSource, CBMMetalDeepPartitionSource,
            CBMMetalMulticlassMathSource, CBMMetalMulticlassScoresSource, CBMMetalMulticlassTrainingSource, CBMMetalMulticlassBootstrapSource,
            CBMMetalMulticlassBacktrackingSource, CBMMetalMultioutputMathSource];
        source = [source stringByAppendingFormat:@"\n%s\n%s\n%s", CBMMetalGreedySource,
            CBMMetalGreedyBootstrapSource, CBMMetalGreedyVectorScoresSource];
        source = [source stringByAppendingFormat:@"\n%s", CBMMetalVectorLangevinSource];
        NSError* error = nil;
        id<MTLLibrary> library = [Device newLibraryWithSource:source options:options error:&error];
        Require(library != nil, "Multiclass shader compilation failed: " + Error(error));
        const char* names[] = {"MulticlassDerivatives", "MulticlassReduceLeafStats", "MulticlassSolveLeaves",
            "MulticlassInitializeTree", "MulticlassAccumulateDirections", "MulticlassBuildCursor", "MulticlassBuildPublishedCursor",
            "MulticlassFindSplitWinners", "ReduceSplitWinners", "ReduceStructurePartials",
            "CollectPartitionStatistics", "MulticlassCollectPartitionStatistics", "ClearHistograms", "ComputeHistograms", "ScanHistograms",
            "UpdateLeafBins", "InitializeRootPartition", "CountDeepPartitionBits", "ScanDeepPartitionTiles",
            "ScanDeepPartitionBlocks", "BuildDeepPartitionOffsets", "ScatterDeepPartitionRows",
            "GenerateBootstrapWeights", "GenerateScoreFeatureNoise",
            "ApplyMulticlassBootstrap", "ReduceMulticlassScoreStatistics", "MulticlassBacktrackingDirectionDot",
            "MulticlassBacktrackingBuildCandidate", "MulticlassBacktrackingReduceObjective", "MulticlassBacktrackingBuildCursor",
            "MultioutputDerivatives", "MultioutputReduceLeafStats", "MultioutputSolveLeaves", "MultioutputBacktrackingReduceObjective",
            "FindGreedyVectorSplitWinners", "EstimateGreedyVectorSimpleLeaves", "ReduceGreedySplitWinners", "SelectGreedyLeaves",
            "RouteGreedySplitRows", "UpdateGreedyLeafDepths", "CountGreedyPartitionBits", "ScanGreedyPartitionTiles",
            "BuildGreedyPartitionOffsets", "ScatterGreedyPartitionRows", "CountGreedyBootstrapRows", "PrefixGreedyBootstrapOffsets",
            "VectorLangevinDirection", "VectorLangevinCandidate", "VectorLangevinGauge"};
        for (const char* name : names) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing multiclass kernel: ") + name);
            id<MTLComputePipelineState> state = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(state != nil, "Multiclass pipeline compilation failed: " + Error(error));
            Require(state.maxTotalThreadsPerThreadgroup >= 256, "Multiclass pipeline requires 256 threads");
            Pipelines.emplace(name, state);
        }
    }
    id<MTLBuffer> Buffer(uint64_t bytes, const void* source = nullptr) {
        Require(bytes <= MemoryLimit && bytes <= Device.maxBufferLength, "Multiclass buffer exceeds memory limit");
        NSUInteger length = std::max<uint64_t>(bytes, 1);
        id<MTLBuffer> result = source && bytes
            ? [Device newBufferWithBytes:source length:length options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:length options:MTLResourceStorageModeShared];
        Require(result != nil, "Multiclass Metal buffer allocation failed");
        return result;
    }
};
Runtime& Context() { static Runtime context; return context; }
struct Binding {
    id<MTLBuffer> Buffer;
    uint64_t Offset;
    Binding(id<MTLBuffer> buffer, uint64_t offset = 0) : Buffer(buffer), Offset(offset) {}
};
struct Command {
    CBMTrainStats& Stats;
    id<MTLCommandBuffer> Buffer;
    explicit Command(CBMTrainStats& stats) : Stats(stats), Buffer([Context().Queue commandBuffer]) {
        Require(Buffer != nil, "Metal command buffer allocation failed");
    }
    template <class P> void Dispatch(const char* name, std::initializer_list<Binding> inputs, const P& params,
                                     uint64_t count, bool groups = false, uint64_t height = 1,
                                     uint64_t depth = 1, const void* extra = nullptr, size_t extraSize = 0) {
        Require(count > 0 && count <= UINT32_MAX && height > 0 && depth > 0, "Invalid multiclass dispatch");
        id<MTLComputeCommandEncoder> encoder = [Buffer computeCommandEncoder];
        Require(encoder != nil, "Metal command encoder allocation failed");
        [encoder setComputePipelineState:Context().Pipelines.at(name)];
        NSUInteger index = 0;
        for (const auto& item : inputs) [encoder setBuffer:item.Buffer offset:item.Offset atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index++];
        if (extra) [encoder setBytes:extra length:extraSize atIndex:index];
        if (groups) [encoder dispatchThreadgroups:MTLSizeMake(count, height, depth) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding]; ++Stats.kernel_dispatches;
    }
    void Copy(id<MTLBuffer> source, id<MTLBuffer> target, uint64_t bytes) {
        id<MTLBlitCommandEncoder> encoder = [Buffer blitCommandEncoder];
        Require(encoder != nil, "Metal blit allocation failed");
        [encoder copyFromBuffer:source sourceOffset:0 toBuffer:target destinationOffset:0 size:bytes];
        [encoder endEncoding];
    }
    void Wait() {
        [Buffer commit]; [Buffer waitUntilCompleted];
        Require(Buffer.status == MTLCommandBufferStatusCompleted, "Multiclass command failed: " + Error(Buffer.error));
        double elapsed = Buffer.GPUEndTime - Buffer.GPUStartTime;
        if (std::isfinite(elapsed) && elapsed >= 0 && Buffer.GPUStartTime > 0) Stats.gpu_seconds += elapsed;
    }
};

class Session {
public:
    CBMMulticlassParams Options;
    CBMTrainStats Stats = {};
    std::mutex Mutex;
    uint32_t Completed = 0;
    float Loss = 0;
    Session(const CBMMulticlassParams* options, const uint8_t* bins, const uint32_t* labels,
            const float* weights, const float* initial, const uint32_t* features,
            const uint32_t* borders, const uint8_t* types, const float* targets = nullptr,
            const CBMVectorGreedyOptions* greedy = nullptr) {
        Require(options != nullptr, "Multiclass parameters are required");
        Options = *options; const auto& p = Options;
        Greedy = greedy != nullptr;
        if (Greedy) {
            GreedyOptions = *greedy;
            Require(greedy->policy <= 2 && !greedy->reserved && greedy->max_leaves >= 1 &&
                    greedy->max_leaves <= 65536, "Invalid vector greedy policy or leaf capacity");
            Require(p.objective == 0 || p.objective == 1 || p.objective == 3,
                    "CUDA greedy vector objectives are MultiClass, MultiClassOneVsAll and RMSEWithUncertainty");
            Require(greedy->policy != 0 || p.depth <= 16, "Depthwise depth must be in [0,16]");
            Require(greedy->policy != 2 || p.depth <= 65535, "Region depth must be in [0,65535]");
        } else Require(p.depth <= 16, "Multiclass tree depth must be in [0,16]");
        const uint32_t capacity = Greedy ? std::min(greedy->max_leaves,
            greedy->policy == 2 ? p.depth + 1 : (1u << std::min(p.depth, 16u))) : 1u << p.depth;
        // Public CUDA options normalize an exactly zero L2 before Simple
        // consumes the searched weak model. Preserve legacy private 0/1 math.
        if (p.leaf_method == 3 && Options.l2 == 0) Options.l2 = 1e-20f;
        M = {p.rows, p.classes, p.objective, capacity, p.l2, 1e-20f, p.leaf_method, 0};
        ValidateMath(M, labels, weights, nullptr, targets, true);
        Require(p.features > 0 && p.bins_per_feature > 0 && p.bins_per_feature <= 256, "Invalid feature or bin count");
        Require(p.iterations > 0 && p.iterations <= 100000, "iterations must be in [1,100000]");
        Require(p.leaf_iterations > 0 && p.leaf_iterations <= 1000, "leaf_iterations must be in [1,1000]");
        Require(p.leaf_method != 3 || p.leaf_iterations == 1, "Simple leaf estimation requires exactly one iteration");
        Require(p.score_function <= 1 || (p.score_function >= 4 && p.score_function <= 6),
                "Vector score_function must be L2, Cosine, SolarL2, LOOL2 or SatL2");
        Require(!p.reserved && !p.reserved1 && !p.reserved2, "Reserved parameters must be zero");
        Require(std::isfinite(p.learning_rate) && p.learning_rate > 0 && p.learning_rate <= 1, "learning_rate must be in (0,1]");
        Require(bins && (p.candidates == 0 || (features && borders)), "Training input buffers are required");
        const uint64_t dataCells = uint64_t(p.rows) * p.features;
        D = Dimension(M); MaxLeaves = M.Leaves;
        const uint64_t histogramCells = uint64_t(MaxLeaves) * p.features * p.bins_per_feature;
        Require(dataCells <= UINT32_MAX && histogramCells <= UINT32_MAX &&
                histogramCells * D <= UINT32_MAX, "Multiclass data or histogram index exceeds uint32");
        Require(uint64_t(p.candidates) <= uint64_t(p.features) * p.bins_per_feature, "Too many split candidates");
        HistCells = uint32_t(histogramCells);
        std::vector<uint8_t> featureTypes(p.features, 0), seen(p.features, 0), candidateTypes(p.candidates, 0);
        for (uint32_t i = 0; i < p.candidates; ++i) {
            uint8_t type = types ? types[i] : 0;
            Require(features[i] < p.features && borders[i] < p.bins_per_feature && type <= 1, "Invalid split candidate");
            Require(!seen[features[i]] || featureTypes[features[i]] == type, "Conflicting candidate types for one feature");
            featureTypes[features[i]] = type; seen[features[i]] = 1; candidateTypes[i] = type;
        }
        for (uint64_t i = 0; i < dataCells; ++i) Require(bins[i] < p.bins_per_feature, "Input bin exceeds bins_per_feature");
        // Account for all permanent GPU allocations before copying input arrays.
        const uint32_t tiles = std::min(256u, (p.rows + 255) / 256);
        const uint64_t bytes = dataCells + uint64_t(p.rows) * (36 + 4 * (3 * D + 3 * p.classes))
            + uint64_t(HistCells) * 4 * (D + 1) + uint64_t(MaxLeaves) *
                (4 * (2 * D + StatsWidth(M) + p.classes * p.classes + p.classes + 2 * D + 6) + 32 * tiles)
            + uint64_t(p.candidates) * 9 + uint64_t(p.features) * 9 + 16384;
        Require(bytes <= MemoryLimit, "Multiclass working set exceeds 1 GiB; reduce depth, classes, features or bins");
        const uint64_t targetExtra = p.objective >= 2 && p.objective != 3 ? uint64_t(p.rows) * (D - 1) * 4 : 0;
        Require(bytes + targetExtra <= MemoryLimit, "Vector target matrix exceeds 1 GiB working set");
        const uint64_t greedyExtra = Greedy ? uint64_t(MaxLeaves) *
            (32 * std::min(256u, std::max(1u, (p.candidates + 255) / 256)) + 32 + 5 * 4) +
            uint64_t(p.features + 1) * 4 + 20 : 0;
        Require(bytes + targetExtra + greedyExtra <= MemoryLimit, "Vector greedy working set exceeds 1 GiB");
        const uint64_t retainedLeafBytes = uint64_t(MaxLeaves) * p.classes * sizeof(float) * 2;
        Require(bytes + targetExtra + greedyExtra + retainedLeafBytes <= MemoryLimit,
                "Multiclass retained permutation leaves exceed 1 GiB working set");
        WorkingBytes = bytes + targetExtra + greedyExtra + retainedLeafBytes;
        LastPermutationLeaves.resize(uint64_t(MaxLeaves) * p.classes);
        PendingPermutationLeaves.resize(LastPermutationLeaves.size());
        std::vector<float> unitWeights;
        if (!weights) { unitWeights.assign(p.rows, 1); weights = unitWeights.data(); }
        TotalWeight = 0; for (uint32_t row = 0; row < p.rows; ++row) TotalWeight += weights[row];
        std::vector<float> logits(uint64_t(p.rows) * D, 0);
        std::vector<float> original(uint64_t(p.rows) * p.classes, 0);
        if (initial) for (uint32_t row = 0; row < p.rows; ++row) {
            for (uint32_t k = 0; k < p.classes; ++k)
                Require(std::isfinite(initial[uint64_t(row) * p.classes + k]), "Initial predictions must be finite");
            const float anchor = p.objective == 0 ? initial[uint64_t(row) * p.classes + p.classes - 1] : 0;
            std::copy(initial + uint64_t(row) * p.classes, initial + uint64_t(row + 1) * p.classes,
                      original.begin() + uint64_t(row) * p.classes);
            for (uint32_t k = 0; k < D; ++k) {
                const float value = initial[uint64_t(row) * p.classes + k] - anchor;
                Require(std::isfinite(value), "Initial prediction differences overflow float32");
                logits[uint64_t(k) * p.rows + row] = value;
            }
        }
        auto& r = Context();
        Text(Stats.device_name, sizeof(Stats.device_name), r.Device.name.UTF8String);
        Bins = r.Buffer(dataCells, bins);
        Labels = p.objective >= 2
            ? r.Buffer(uint64_t(p.rows) * (p.objective == 3 ? 1 : D) * 4, targets)
            : r.Buffer(uint64_t(p.rows) * 4, labels);
        Weights = r.Buffer(uint64_t(p.rows) * 4, weights);
        StructureWeights = Weights;
        Cursor = r.Buffer(logits.size() * 4, logits.data()); Base = r.Buffer(logits.size() * 4);
        Published = r.Buffer(original.size() * 4, original.data()); NextPublished = r.Buffer(original.size() * 4);
        Gradients = r.Buffer(logits.size() * 4); Probabilities = r.Buffer(uint64_t(p.rows) * p.classes * 4);
        Losses = r.Buffer(uint64_t(p.rows) * 4);
        RowIndices = r.Buffer(uint64_t(p.rows) * 4); NextRows = r.Buffer(uint64_t(p.rows) * 4);
        Offsets = r.Buffer(uint64_t(MaxLeaves + 1) * 4); NextOffsets = r.Buffer(uint64_t(MaxLeaves + 1) * 4);
        LeafIds = r.Buffer(uint64_t(p.rows) * 4); RowPrefix = r.Buffer(uint64_t(p.rows) * 4);
        TilePrefix = r.Buffer(uint64_t((p.rows + 255) / 256) * 4);
        BlockPrefix = r.Buffer(uint64_t((p.rows + 65535) / 65536) * 4);
        RawValues = r.Buffer(uint64_t(MaxLeaves) * D * 4); Directions = r.Buffer(uint64_t(MaxLeaves) * D * 4);
        LeafStats = r.Buffer(uint64_t(MaxLeaves) * StatsWidth(M) * 4);
        SolveWorkspace = r.Buffer(uint64_t(MaxLeaves) * (p.classes * p.classes + p.classes) * 4);
        Status = r.Buffer(uint64_t(MaxLeaves) * 4);
        HistSums = r.Buffer(uint64_t(HistCells) * D * 4); HistWeights = r.Buffer(uint64_t(HistCells) * 4);
        LeafSums = r.Buffer(uint64_t(MaxLeaves) * D * 8); LeafWeights = r.Buffer(uint64_t(MaxLeaves) * 8);
        Partials = r.Buffer(uint64_t(MaxLeaves) * tiles * 32);
        CandidateFeatures = r.Buffer(uint64_t(p.candidates) * 4, features);
        CandidateBins = r.Buffer(uint64_t(p.candidates) * 4, borders);
        CandidateTypes = r.Buffer(p.candidates, candidateTypes.data()); FeatureTypes = r.Buffer(p.features, featureTypes.data());
        FeatureNoise = r.Buffer(uint64_t(p.features) * 4);
        std::fill_n(static_cast<float*>(FeatureNoise.contents), p.features, 0.0f);
        FeatureWeights = r.Buffer(uint64_t(p.features) * 4);
        std::fill_n(static_cast<float*>(FeatureWeights.contents), p.features, 1.0f);
        CtrUniqueValues.assign(p.features, 0); UsedFeatures.assign(p.features, 0); UserFeatureWeights.assign(p.features, 1);
        Winner = r.Buffer(sizeof(SplitState)); Winners = r.Buffer(sizeof(SplitState) * 256);
        K = {}; K.Rows = p.rows; K.Features = p.features; K.Bins = p.bins_per_feature;
        K.Candidates = p.candidates; K.LearningRate = p.learning_rate; K.L2 = p.l2;
        K.ScoreFunction = p.score_function; K.Objective = p.objective; K.TileRows = 256;
        K.HistogramTiles = tiles; K.ScoreGroups = std::min(256u, std::max(1u, (p.candidates + 255) / 256));
        K.TotalWeight = float(TotalWeight);
        Command command(Stats); Derivatives(command); command.Wait(); UpdateLoss();
        if (Greedy) {
            G = {p.rows, p.features, 1, p.candidates, p.features * p.bins_per_feature,
                K.ScoreGroups, p.score_function, GreedyOptions.min_data_in_leaf,
                std::min(p.depth, 65535u), MaxLeaves, GreedyOptions.policy, D, HistCells, MaxLeaves,
                uint32_t(p.objective == 0), 0, p.l2, 0, 0, 0};
            GreedyDepths = r.Buffer(uint64_t(MaxLeaves) * 4); GreedyNextDepths = r.Buffer(uint64_t(MaxLeaves) * 4);
            GreedySelected = r.Buffer(uint64_t(MaxLeaves) * 4); GreedyRightIds = r.Buffer(uint64_t(MaxLeaves) * 4);
            GreedySampledOffsets = r.Buffer(uint64_t(MaxLeaves + 1) * 4);
            GreedyWinners = r.Buffer(uint64_t(MaxLeaves) * sizeof(CBMGreedySplit));
            GreedyPartials = r.Buffer(uint64_t(MaxLeaves) * K.ScoreGroups * sizeof(CBMGreedySplit));
            GreedyFrontier = r.Buffer(sizeof(CBMGreedyFrontier));
            std::vector<uint32_t> featureOffsets(p.features + 1);
            for (uint32_t f = 0; f <= p.features; ++f) featureOffsets[f] = f * p.bins_per_feature;
            GreedyFeatureOffsets = r.Buffer(featureOffsets.size() * 4, featureOffsets.data());
        }
    }
    void Info(CBMStepInfo& info) const {
        info = {}; info.completed_iterations = Completed; info.finished = Completed == Options.iterations;
        info.loss = Loss; info.stats = Stats;
    }
    void ConfigureBootstrap(const CBMBootstrapOptions* options) {
        Require(options != nullptr && Completed == 0, "Bootstrap must be configured before training");
        Require(options->bootstrap_type <= 3, "MVS bootstrap is unsupported for multiclass, matching CUDA");
        Require(!options->reserved0 && !options->reserved1 &&
                options->mvs_reg_is_set <= 1 && options->initial_mvs_lambda_is_set <= 1, "Invalid bootstrap options");
        Require(uint64_t(options->iteration_offset) + Options.iterations <= UINT32_MAX, "Bootstrap iteration index exceeds uint32");
        Require(std::isfinite(options->bagging_temperature) && options->bagging_temperature >= 0,
                "bagging_temperature must be finite and nonnegative");
        Require(std::isfinite(options->subsample) && options->subsample > 0 && options->subsample <= 1 &&
                (options->bootstrap_type != 3 || options->subsample < 1), "Invalid bootstrap subsample");
        Require(!options->mvs_reg_is_set || (std::isfinite(options->mvs_reg) && options->mvs_reg >= 0), "Invalid mvs_reg");
        Require(!options->initial_mvs_lambda_is_set, "Multiclass does not accept MVS continuation state");
        const uint64_t extra = options->bootstrap_type ? uint64_t(Options.rows) * 8 : 0;
        Require(WorkingBytes + PermutationBytes + extra + NoiseBytes + BacktrackingBytes + LangevinBytes <= MemoryLimit, "Multiclass bootstrap exceeds 1 GiB working set");
        if (extra) {
            auto& r = Context(); StructureWeights = r.Buffer(uint64_t(Options.rows) * 4);
            Multipliers = r.Buffer(uint64_t(Options.rows) * 4);
        } else {
            StructureWeights = Weights; Multipliers = nil;
        }
        BootstrapOptions = *options; BootstrapBytes = extra;
        MVSLambda = options->initial_mvs_lambda; HasMVSLambda = options->initial_mvs_lambda_is_set != 0;
    }
    void BootstrapState(uint32_t* iteration, float* lambda, uint32_t* valid) const {
        Require(iteration && lambda && valid, "Bootstrap state output buffers are required");
        *iteration = BootstrapOptions.iteration_offset + Completed;
        *lambda = HasMVSLambda ? MVSLambda : 0; *valid = HasMVSLambda;
    }
    void ConfigureFixedSplits(uint32_t count, const uint32_t* features) {
        Require(Greedy && Completed == 0 && !TreeActive,
            "Fixed splits require a greedy vector session before its first tree");
        FixedSplits.Configure(count, features, Options.features, Options.candidates,
            static_cast<const uint32_t*>(CandidateFeatures.contents),
            static_cast<const uint32_t*>(CandidateBins.contents),
            static_cast<const uint8_t*>(CandidateTypes.contents));
    }
    void ConfigureNoise(const CBMScoreNoiseOptions* options) {
        Require(options && !options->reserved0 && !options->reserved1 && !options->reserved2 && Completed == 0,
                "Score noise must be configured before training");
        Require(std::isfinite(options->random_strength) && options->random_strength >= 0, "Invalid random_strength");
        const uint64_t extra = options->random_strength > 0 && K.ScoreFunction == 1
            ? uint64_t(std::min(4096u, (Options.rows + 255) / 256)) * 8 : 0;
        Require(WorkingBytes + PermutationBytes + BootstrapBytes + extra + BacktrackingBytes + LangevinBytes <= MemoryLimit, "Multiclass score noise exceeds 1 GiB working set");
        NoiseStatistics = extra ? Context().Buffer(extra) : nil;
        RandomStrength = options->random_strength; NoiseBytes = extra;
        if (!extra) std::fill_n(static_cast<float*>(FeatureNoise.contents), Options.features, 0.0f);
    }
    void ConfigureBacktracking(uint32_t type) {
        Require(type <= 2 && Completed == 0, "Backtracking must be No, AnyImprovement or Armijo and configured before training");
        const uint64_t extra = ((type && Options.leaf_iterations > 1) || (Langevin && M.LeafMethod != 3)) ? uint64_t(MaxLeaves) * (D * 4 + 16) +
            uint64_t(std::min(256u, (Options.rows + 255) / 256)) * 8 : 0;
        Require(WorkingBytes + PermutationBytes + BootstrapBytes + NoiseBytes + extra + LangevinBytes <= MemoryLimit, "Multiclass backtracking exceeds 1 GiB working set");
        if (extra) {
            TrialValues = Context().Buffer(uint64_t(MaxLeaves) * D * 4);
            DirectionDots = Context().Buffer(uint64_t(MaxLeaves) * 16);
            TrialLoss = Context().Buffer(uint64_t(std::min(256u, (Options.rows + 255) / 256)) * 8);
        } else { TrialValues = nil; DirectionDots = nil; TrialLoss = nil; }
        BacktrackingType = type; BacktrackingBytes = extra;
    }
    void ConfigureLangevin(float temperature, CBMLangevinNoiseCallback noise,
                           CBMLangevinSeedCallback seed, void* context) {
        Require(!Completed && !Langevin && noise && seed,
            "Configure vector Langevin once with valid callbacks before the first tree");
        Require(std::isfinite(temperature) && temperature >= 0,
            "Vector Langevin temperature must be finite and nonnegative");
        const uint64_t gradientCells = uint64_t(MaxLeaves) * M.Classes;
        const uint64_t hessianCells = gradientCells * ((M.LeafMethod == 1 || M.Objective == 1) ? 1 : M.Classes);
        const uint64_t workspaceCells = uint64_t(MaxLeaves) * (M.Classes * M.Classes + 2 * M.Classes);
        const uint64_t bytes = M.LeafMethod == 3 ? 0 : 20 * gradientCells + 8 * (hessianCells + workspaceCells);
        const uint64_t backtracking = M.LeafMethod == 3 ? BacktrackingBytes : uint64_t(MaxLeaves) * (D * 4 + 16)
            + uint64_t(std::min(256u, (Options.rows + 255) / 256)) * 8;
        Require(WorkingBytes + PermutationBytes + BootstrapBytes + NoiseBytes + bytes + backtracking <= MemoryLimit,
            "Vector Langevin workspace exceeds 1 GiB");
        if (bytes) {
            auto gradient = Context().Buffer(8 * gradientCells);
            auto hessian = Context().Buffer(8 * hessianCells);
            auto workspace = Context().Buffer(8 * workspaceCells);
            auto point = Context().Buffer(4 * gradientCells), trial = Context().Buffer(4 * gradientCells);
            auto direction = Context().Buffer(4 * gradientCells);
            LangevinGradientNoise = gradient; LangevinHessianNoise = hessian; LangevinWorkspace = workspace;
            LangevinPoint = point; LangevinTrial = trial; LangevinDirection = direction;
        }
        Langevin = true; LangevinBytes = bytes;
        try { ConfigureBacktracking(BacktrackingType); }
        catch (...) { Langevin = false; LangevinBytes = 0;
            LangevinGradientNoise = nil; LangevinHessianNoise = nil; LangevinWorkspace = nil;
            LangevinPoint = nil; LangevinTrial = nil; LangevinDirection = nil; throw; }
        LangevinNoise = noise; LangevinSeed = seed; LangevinContext = context;
    }
    void ConfigureFeaturePenalties(const CBMFeaturePenaltyOptions* options, const uint32_t* counts,
                                   const float* weights, const uint8_t* used) {
        Require(!Greedy, "Dynamic feature penalties are not supported by CUDA greedy search");
        Require(options && counts && Completed == 0 && !options->reserved0 && !options->reserved1 && !options->reserved2,
                "Feature penalties must be configured before training");
        Require(std::isfinite(options->model_size_reg) && options->model_size_reg >= 0, "model_size_reg must be finite and nonnegative");
        for (uint32_t feature = 0; feature < Options.features; ++feature) {
            Require(!weights || (std::isfinite(weights[feature]) && weights[feature] >= 0), "Invalid feature weight");
            Require(!used || used[feature] <= 1, "Feature usage flags must be zero or one");
        }
        ModelSizeReg = options->model_size_reg;
        for (uint32_t feature = 0; feature < Options.features; ++feature) {
            CtrUniqueValues[feature] = counts[feature];
            UsedFeatures[feature] = used ? used[feature] : 0;
            UserFeatureWeights[feature] = weights ? weights[feature] : 1;
        }
    }
    void CopyFeaturePenaltyState(uint8_t* used) const {
        Require(used != nullptr, "Feature penalty state output is required");
        std::copy(UsedFeatures.begin(), UsedFeatures.end(), used);
    }
    void ConfigureGreedyFeatureWeights(uint32_t count, const float* weights) {
        Require(Greedy && Completed == 0 && count == Options.features && weights,
                "Greedy feature weights must match all features and precede training");
        for (uint32_t feature = 0; feature < count; ++feature)
            Require(std::isfinite(weights[feature]) && weights[feature] >= 0,
                    "Greedy feature weights must be finite and nonnegative");
        std::memcpy(FeatureWeights.contents, weights, uint64_t(count) * 4);
    }
    void Step(CBMStepInfo* info, uint32_t* depth, uint32_t* splitFeatures, uint32_t* splitBins,
              uint8_t* splitTypes, float* leafValues, float* leafWeights) {
        Require(!Greedy, "Use step_greedy for a vector greedy session");
        BeginLeafCapture();
        try {
            if (!Permutations.empty()) {
                StepPermutations(info, depth, splitFeatures, splitBins, splitTypes, leafValues, leafWeights);
            } else {
                StepOne(info, depth, splitFeatures, splitBins, splitTypes, leafValues, leafWeights);
                StagePermutationLeaves(0, leafValues);
            }
            CommitLeafCapture();
        } catch (...) {
            CapturingLeaves = false;
            throw;
        }
    }
    void StepOne(CBMStepInfo* info, uint32_t* depth, uint32_t* splitFeatures, uint32_t* splitBins,
                 uint8_t* splitTypes, float* leafValues, float* leafWeights) {
        TreeActive = false;
        const float previousLoss = Loss;
        const float previousLambda = MVSLambda;
        const bool previousHasLambda = HasMVSLambda;
        const auto previousUsed = UsedFeatures;
        try {
            BuildTree(info, depth, splitFeatures, splitBins, splitTypes, leafValues, leafWeights);
            TreeActive = false;
        } catch (...) {
            // A failed Newton iteration must not expose its unshrunk trial
            // cursor as a completed ensemble or corrupt a later retry.
            if (TreeActive) {
                LeafCaptureHealthy = false;
                Command restore(Stats);
                restore.Copy(Base, Cursor, uint64_t(Options.rows) * D * 4);
                restore.Wait();
                Loss = previousLoss;
                MVSLambda = previousLambda; HasMVSLambda = previousHasLambda;
                UsedFeatures = previousUsed;
                TreeActive = false;
                LeafCaptureHealthy = true;
            }
            throw;
        }
    }
    void BuildTree(CBMStepInfo* info, uint32_t* depth, uint32_t* splitFeatures, uint32_t* splitBins,
                   uint8_t* splitTypes, float* leafValues, float* leafWeights) {
        Require(info && depth && leafValues && leafWeights &&
                (!Options.depth || (splitFeatures && splitBins && splitTypes)), "Step output buffers are required");
        Require(Completed < Options.iterations, "Multiclass session is already complete");
        std::fill(leafValues, leafValues + uint64_t(MaxLeaves) * Options.classes, 0);
        std::fill(leafWeights, leafWeights + MaxLeaves, 0);
        if (Options.depth) {
            std::fill(splitFeatures, splitFeatures + Options.depth, 0);
            std::fill(splitBins, splitBins + Options.depth, 0);
            std::fill(splitTypes, splitTypes + Options.depth, 0);
        }
        M.Leaves = MaxLeaves; K.Leaves = 1;
        if (Langevin && !ForcedTree && BootstrapOptions.bootstrap_type) ReadLangevinSeed(CBM_LANGEVIN_WEAK_SEED_CACHE);
        {
            Command c(Stats); c.Copy(Cursor, Base, uint64_t(Options.rows) * D * 4);
            c.Dispatch("MulticlassInitializeTree", {LeafIds, RawValues}, M, std::max(Options.rows, MaxLeaves * D));
            c.Dispatch("InitializeRootPartition", {RowIndices, Offsets}, K, Options.rows);
            Derivatives(c); c.Wait(); TreeActive = true;
        }
        uint32_t actualDepth = 0;
        const float noiseScale = ForcedTree ? 0 : PrepareStructure();
        std::vector<uint32_t> chosen;
        uint32_t scoredBatches = 0;
        while (actualDepth < Options.depth && (ForcedTree ? actualDepth < ForcedTree->size() : Options.candidates != 0)) {
            SplitState winner;
            if (ForcedTree) {
                winner = (*ForcedTree)[actualDepth];
                std::memcpy(Winner.contents, &winner, sizeof(winner));
            } else {
            if (Langevin) { ReadLangevinSeed(CBM_LANGEVIN_SEARCH); ++scoredBatches; }
            UpdateFeatureWeights();
            Command c(Stats);
            for (uint32_t k = 0; k < D; ++k) {
                Binding gradient(Gradients, uint64_t(k) * Options.rows * 4);
                Binding histogram(HistSums, uint64_t(k) * HistCells * 4);
                Binding leafSums(LeafSums, uint64_t(k) * MaxLeaves * 8);
                c.Dispatch("ReduceStructurePartials", {gradient, StructureWeights, RowIndices, Offsets, Partials}, K,
                           K.HistogramTiles, true, K.Leaves);
                c.Dispatch("MulticlassCollectPartitionStatistics", {Partials, leafSums, LeafWeights}, K, K.Leaves, true);
                c.Dispatch("ClearHistograms", {histogram, HistWeights}, K, uint64_t(K.Leaves) * K.Features * K.Bins);
                c.Dispatch("ComputeHistograms", {Bins, gradient, StructureWeights, RowIndices, Offsets, histogram, HistWeights}, K,
                           K.HistogramTiles, true, K.Features, K.Leaves);
                c.Dispatch("ScanHistograms", {histogram, HistWeights, FeatureTypes}, K, uint64_t(K.Leaves) * K.Features, true);
            }
            TrainingParams training = {D, HistCells, MaxLeaves, 0};
            if (NoiseStatistics) {
                auto noise = BootstrapParameters(); noise.Rows = Options.features;
                noise.Stream = actualDepth + 1; noise.NoiseScale = noiseScale;
                c.Dispatch("GenerateScoreFeatureNoise", {FeatureNoise}, noise, Options.features);
            }
            c.Dispatch("MulticlassFindSplitWinners", {HistSums, HistWeights, LeafSums, LeafWeights,
                       CandidateFeatures, CandidateBins, CandidateTypes, Winners, FeatureNoise, FeatureWeights}, K, K.ScoreGroups, true,
                       1, 1, &training, sizeof(training));
            c.Dispatch("ReduceSplitWinners", {Winners, Winner}, K, 1, true); c.Wait();
            winner = *static_cast<const SplitState*>(Winner.contents);
            Require(!winner.InvalidScore, "Nonfinite multiclass split score");
            if (winner.Valid && CtrUniqueValues[winner.Feature]) UsedFeatures[winner.Feature] = 1;
            // CUDA's generic symmetric vector search keeps a repeated
            // negative winner, including empty child branches. New Simple
            // and Langevin modes follow that rule; retain the published
            // legacy stop rule for their existing disabled counterparts.
            if (!winner.Valid || winner.Score >= 0 ||
                ((!Langevin && M.LeafMethod != 3) &&
                 std::find(chosen.begin(), chosen.end(), winner.Index) != chosen.end())) break;
            chosen.push_back(winner.Index);
            }
            splitFeatures[actualDepth] = winner.Feature; splitBins[actualDepth] = winner.Bin;
            splitTypes[actualDepth] = uint8_t(winner.Type);
            K.SplitLevel = actualDepth; ++actualDepth;
            Command partition(Stats);
            partition.Dispatch("UpdateLeafBins", {Bins, LeafIds, Winner}, K, Options.rows);
            K.Leaves = 1u << actualDepth;
            partition.Dispatch("CountDeepPartitionBits", {RowIndices, LeafIds, RowPrefix, TilePrefix}, K,
                               (Options.rows + 255) / 256, true);
            partition.Dispatch("ScanDeepPartitionTiles", {TilePrefix, BlockPrefix}, K, (Options.rows + 65535) / 65536, true);
            partition.Dispatch("ScanDeepPartitionBlocks", {BlockPrefix}, K, 1, true);
            partition.Dispatch("BuildDeepPartitionOffsets", {Offsets, RowPrefix, TilePrefix, BlockPrefix, NextOffsets}, K, K.Leaves / 2);
            partition.Dispatch("ScatterDeepPartitionRows", {RowIndices, LeafIds, RowPrefix, TilePrefix, BlockPrefix, NextRows}, K, Options.rows);
            partition.Wait(); std::swap(RowIndices, NextRows); std::swap(Offsets, NextOffsets);
        }
        // Greedy vector CUDA registers the initial root score even for a
        // depth-zero model. Forced replay never owns search draws.
        if (Langevin && !ForcedTree && Options.candidates && !scoredBatches)
            ReadLangevinSeed(CBM_LANGEVIN_SEARCH);
        if (!DeferLeafEstimation) EstimateAndPublish(leafValues, leafWeights);
        *depth = actualDepth; Info(*info);
    }
    void EstimateAndPublish(float* leafValues, float* leafWeights) {
        M.Leaves = K.Leaves;
        const bool simple = M.LeafMethod == 3;
        if (simple) {
            EstimateSimple();
        } else if (Langevin) {
            EstimateLangevinLeaves();
        } else if (BacktrackingBytes) {
            EstimateBacktracking();
        } else for (uint32_t iteration = 0; iteration < Options.leaf_iterations; ++iteration) {
            Command c(Stats); Derivatives(c);
            c.Dispatch(M.Objective < 2 ? "MulticlassReduceLeafStats" : "MultioutputReduceLeafStats", {Gradients, Probabilities, Weights, RowIndices, Offsets, LeafStats},
                       M, uint64_t(M.Leaves) * StatsWidth(M), true);
            c.Dispatch(M.Objective < 2 ? "MulticlassSolveLeaves" : "MultioutputSolveLeaves", {LeafStats, SolveWorkspace, Directions, Status}, M, M.Leaves);
            c.Wait();
            const auto* status = static_cast<const uint32_t*>(Status.contents);
            for (uint32_t leaf = 0; leaf < M.Leaves; ++leaf)
                Require(status[leaf] == 0, "Multiclass leaf solve failed (status " + std::to_string(status[leaf]) + ")");
            Command update(Stats);
            update.Dispatch("MulticlassAccumulateDirections", {Directions, RawValues}, M, M.Leaves * D);
            const float step = iteration + 1 == Options.leaf_iterations ? Options.learning_rate : 1.0f;
            update.Dispatch("MulticlassBuildCursor", {Base, RawValues, LeafIds, Cursor}, M, uint64_t(M.Rows) * D,
                            false, 1, 1, &step, sizeof(step));
            if (iteration + 1 == Options.leaf_iterations) Derivatives(update);
            update.Wait();
        }
        const auto* values = static_cast<const float*>(RawValues.contents);
        const auto* statistics = static_cast<const float*>(LeafStats.contents);
        for (uint32_t leaf = 0; leaf < M.Leaves; ++leaf) {
            leafWeights[leaf] = statistics[simple ? leaf : uint64_t(leaf) * StatsWidth(M)];
            for (uint32_t k = 0; k < D; ++k) {
                float value = Options.learning_rate * values[uint64_t(leaf) * D + k];
                Require(std::isfinite(value), "Nonfinite multiclass leaf value");
                leafValues[uint64_t(leaf) * Options.classes + k] = value;
            }
        }
        UpdateLoss();
        Command publish(Stats);
        publish.Dispatch("MulticlassBuildPublishedCursor", {Published, RawValues, LeafIds, NextPublished},
                         M, uint64_t(M.Rows) * M.Classes, false, 1, 1, &Options.learning_rate, sizeof(float));
        publish.Wait(); std::swap(Published, NextPublished);
        ++Completed;
    }
    void EstimateSimple() {
        Command c(Stats);
        const bool copied = ForcedGreedyRounds || ForcedTree;
        if (copied) {
            // CUDA copies the searched weak model into every permutation;
            // these solver buffers are otherwise unused by Simple.
            c.Copy(Directions, RawValues, uint64_t(K.Leaves) * D * 4);
            c.Copy(SolveWorkspace, LeafStats, uint64_t(K.Leaves) * 4);
        } else {
            // The last split changes leaf membership after its statistics were
            // scored. Project the sampled weak target into the final leaves,
            // including root-only trees, before any fresh derivatives overwrite it.
            for (uint32_t k = 0; k < D; ++k) {
                Binding gradient(Gradients, uint64_t(k) * Options.rows * 4);
                Binding leafSums(LeafSums, uint64_t(k) * MaxLeaves * 8);
                c.Dispatch("ReduceStructurePartials", {gradient, StructureWeights, RowIndices, Offsets, Partials},
                    K, K.HistogramTiles, true, K.Leaves);
                c.Dispatch("MulticlassCollectPartitionStatistics", {Partials, leafSums, LeafWeights}, K, K.Leaves, true);
            }
            CBMGreedyParams simple = {};
            simple.Leaves = K.Leaves; simple.Dimensions = D; simple.LeafStride = MaxLeaves;
            simple.MulticlassOptimization = M.Objective == 0; simple.L2 = M.L2;
            c.Dispatch("EstimateGreedyVectorSimpleLeaves", {LeafSums, LeafWeights, RawValues, LeafStats, Status},
                simple, K.Leaves);
        }
        c.Wait();
        if (!copied) {
            const auto* status = static_cast<const uint32_t*>(Status.contents);
            for (uint32_t leaf = 0; leaf < K.Leaves; ++leaf)
                Require(status[leaf] == 0, "Nonfinite vector Simple leaf statistics or value");
        }
        Command update(Stats);
        update.Dispatch("MulticlassBuildCursor", {Base, RawValues, LeafIds, Cursor}, M, uint64_t(M.Rows) * D,
            false, 1, 1, &Options.learning_rate, sizeof(float));
        Derivatives(update); update.Wait();
    }
    void StepGreedy(CBMGreedyStepInfo* info, CBMGreedyNode* nodes, float* values, float* weights) {
        Require(Greedy && info && nodes && values && weights, "Vector greedy session and output buffers are required");
        Require(Completed < Options.iterations, "Vector greedy session is already complete");
        BeginLeafCapture();
        try {
            StepGreedyImpl(info, nodes, values, weights);
            CommitLeafCapture();
        } catch (...) {
            CapturingLeaves = false;
            throw;
        }
    }
    void StepGreedyImpl(CBMGreedyStepInfo* info, CBMGreedyNode* nodes, float* values, float* weights) {
        if (Permutations.empty()) {
            StepGreedyOne(info, nodes, values, weights);
            StagePermutationLeaves(0, values);
            return;
        }
        const uint32_t priorCompleted = Completed, count = Permutations.size();
        std::vector<float> priorLoss;
        Command backup(Stats);
        for (const auto& state : Permutations) {
            priorLoss.push_back(state.Loss);
            backup.Copy(state.Cursor, state.BackupCursor, uint64_t(Options.rows) * D * 4);
            backup.Copy(state.Published, state.BackupPublished, uint64_t(Options.rows) * Options.classes * 4);
        }
        backup.Wait();
        std::vector<CBMGreedyNode> temporaryNodes(2 * MaxLeaves - 1);
        std::vector<float> temporaryValues(uint64_t(MaxLeaves) * Options.classes), temporaryWeights(MaxLeaves);
        GreedyRounds selected;
        try {
            // Search the chosen weak history first, then evaluate every leaf
            // oracle in source history order. Structure-only work draws no leaf noise.
            const bool defer = Langevin && M.LeafMethod != 3;
            for (uint32_t order = 0; order < count + uint32_t(defer); ++order) {
                const uint32_t index = defer ? (order ? order - 1 : SearchPermutation)
                    : (order == 0 ? SearchPermutation : (order - 1 < SearchPermutation ? order - 1 : order));
                DeferLeafEstimation = defer && order == 0;
                LoadPermutation(index); Completed = priorCompleted;
                ForcedGreedyRounds = order ? &selected : nullptr;
                const bool exported = index + 1 == count;
                CBMGreedyStepInfo temporaryInfo;
                StepGreedyOne(exported ? info : &temporaryInfo, exported ? nodes : temporaryNodes.data(),
                    exported ? values : temporaryValues.data(), exported ? weights : temporaryWeights.data());
                if (!DeferLeafEstimation)
                    StagePermutationLeaves(index, exported ? values : temporaryValues.data());
                Permutations[index].Cursor = Cursor; Permutations[index].Published = Published;
                Permutations[index].Loss = Loss;
                if (!order) {
                    selected = SelectedGreedyRounds;
                    if (M.LeafMethod == 3 && count > 1) {
                        Command cache(Stats);
                        cache.Copy(RawValues, Directions, uint64_t(K.Leaves) * D * 4);
                        cache.Copy(LeafStats, SolveWorkspace, uint64_t(K.Leaves) * 4);
                        cache.Wait();
                    }
                }
            }
            ForcedGreedyRounds = nullptr; DeferLeafEstimation = false; Completed = priorCompleted + 1;
            LoadPermutation(count - 1); info->loss = Loss; info->stats = Stats;
        } catch (...) {
            ForcedGreedyRounds = nullptr; DeferLeafEstimation = false;
            LeafCaptureHealthy = false;
            Command restore(Stats);
            for (uint32_t index = 0; index < count; ++index) {
                auto& state = Permutations[index];
                restore.Copy(state.BackupCursor, state.Cursor, uint64_t(Options.rows) * D * 4);
                restore.Copy(state.BackupPublished, state.Published, uint64_t(Options.rows) * Options.classes * 4);
                state.Loss = priorLoss[index];
            }
            restore.Wait(); Completed = priorCompleted; LoadPermutation(count - 1);
            LeafCaptureHealthy = true;
            throw;
        }
    }
    void StepGreedyOne(CBMGreedyStepInfo* info, CBMGreedyNode* nodes, float* values, float* weights) {
        TreeActive = false;
        const float previousLoss = Loss;
        try {
            BuildGreedyTree(info, nodes, values, weights);
            TreeActive = false;
        } catch (...) {
            if (TreeActive) {
                LeafCaptureHealthy = false;
                Command restore(Stats); restore.Copy(Base, Cursor, uint64_t(Options.rows) * D * 4); restore.Wait();
                Loss = previousLoss; TreeActive = false;
                LeafCaptureHealthy = true;
            }
            throw;
        }
    }
    void BuildGreedyTree(CBMGreedyStepInfo* info, CBMGreedyNode* output, float* values, float* weights) {
        std::fill_n(values, uint64_t(MaxLeaves) * Options.classes, 0.f);
        std::fill_n(weights, MaxLeaves, 0.f);
        std::fill_n(output, 2 * MaxLeaves - 1, CBMGreedyNode{});
        M.Leaves = MaxLeaves; K.Leaves = G.Leaves = 1;
        *static_cast<uint32_t*>(GreedyDepths.contents) = 0;
        SelectedGreedyRounds.clear();
        std::vector<CBMGreedyNode> nodes = {{0, 0, 0, 0, 0, 0}};
        std::vector<uint32_t> leafNodes = {0};
        CBMFixedSplitSearch fixedSearch(FixedSplits);
        if (Langevin && !ForcedGreedyRounds && BootstrapOptions.bootstrap_type) ReadLangevinSeed(CBM_LANGEVIN_WEAK_SEED_CACHE);
        {
            Command c(Stats); c.Copy(Cursor, Base, uint64_t(Options.rows) * D * 4);
            c.Dispatch("MulticlassInitializeTree", {LeafIds, RawValues}, M, std::max(Options.rows, MaxLeaves * D));
            c.Dispatch("InitializeRootPartition", {RowIndices, Offsets}, K, Options.rows);
            Derivatives(c); c.Wait(); TreeActive = true;
        }
        const float noiseScale = ForcedGreedyRounds ? 0 : PrepareStructure();
        uint32_t round = 0;
        uint32_t scoredRound = 0;
        uint32_t scoredBatches = 0;
        std::vector<uint8_t> unscored(1, 1);
        std::vector<CBMGreedySplit> cachedWinners(1);
        while (K.Leaves < MaxLeaves && Options.depth &&
               (ForcedGreedyRounds ? round < ForcedGreedyRounds->size() : Options.candidates != 0)) {
            CBMGreedyFrontier frontier;
            if (ForcedGreedyRounds) {
                const auto& branches = (*ForcedGreedyRounds)[round];
                auto* winners = static_cast<CBMGreedySplit*>(GreedyWinners.contents);
                auto* selected = static_cast<uint32_t*>(GreedySelected.contents);
                auto* right = static_cast<uint32_t*>(GreedyRightIds.contents);
                std::fill_n(right, K.Leaves, UINT32_MAX);
                frontier = {uint32_t(branches.size()), K.Leaves + uint32_t(branches.size()), 0, 0};
                for (uint32_t i = 0; i < branches.size(); ++i) {
                    const auto& branch = branches[i];
                    selected[i] = branch.Parent; right[branch.Parent] = branch.Right; winners[branch.Parent] = branch.Split;
                }
                std::memcpy(GreedyFrontier.contents, &frontier, sizeof(frontier));
            } else {
                bool scoreNewLeaves = true;
                std::vector<uint8_t> scoreMask;
                if (FixedSplits.Enabled() || Langevin) {
                    id<MTLBuffer> eligibilityOffsets = Offsets;
                    if (BootstrapOptions.bootstrap_type >= 2) {
                        Command eligibility(Stats);
                        const CBMGreedyBootstrapParams p = {K.Rows, K.Leaves, BootstrapOptions.bootstrap_type, 0};
                        eligibility.Dispatch("CountGreedyBootstrapRows", {Multipliers, RowIndices, Offsets, GreedySampledOffsets}, p, K.Leaves, true);
                        eligibility.Dispatch("PrefixGreedyBootstrapOffsets", {GreedySampledOffsets}, p, 1, true);
                        eligibility.Wait(); eligibilityOffsets = GreedySampledOffsets;
                    }
                    const auto* depths = static_cast<const uint32_t*>(GreedyDepths.contents);
                    const auto* offsets = static_cast<const uint32_t*>(eligibilityOffsets.contents);
                    if (FixedSplits.Enabled()) scoreNewLeaves = fixedSearch.Begin(K.Leaves,
                        depths, offsets, G.MaxDepth, G.MinDataInLeaf);
                    else {
                        scoreMask.resize(K.Leaves, 0); scoreNewLeaves = false;
                        for (uint32_t leaf = 0; leaf < K.Leaves; ++leaf) {
                            const bool root = K.Leaves == 1 && depths[leaf] == 0;
                            scoreMask[leaf] = unscored[leaf] && (root ||
                                (depths[leaf] < G.MaxDepth && offsets[leaf + 1] - offsets[leaf] > G.MinDataInLeaf));
                            scoreNewLeaves |= scoreMask[leaf] != 0; unscored[leaf] = 0;
                        }
                    }
                }
                if (Langevin && scoreNewLeaves) { ReadLangevinSeed(CBM_LANGEVIN_SEARCH); ++scoredBatches; }
                Command c(Stats);
                for (uint32_t k = 0; k < D; ++k) {
                    Binding gradient(Gradients, uint64_t(k) * Options.rows * 4);
                    Binding histogram(HistSums, uint64_t(k) * HistCells * 4);
                    Binding leafSums(LeafSums, uint64_t(k) * MaxLeaves * 8);
                    c.Dispatch("ReduceStructurePartials", {gradient, StructureWeights, RowIndices, Offsets, Partials},
                        K, K.HistogramTiles, true, K.Leaves);
                    c.Dispatch("MulticlassCollectPartitionStatistics", {Partials, leafSums, LeafWeights}, K, K.Leaves, true);
                    c.Dispatch("ClearHistograms", {histogram, HistWeights}, K, uint64_t(K.Leaves) * K.Features * K.Bins);
                    c.Dispatch("ComputeHistograms", {Bins, gradient, StructureWeights, RowIndices, Offsets, histogram, HistWeights},
                        K, K.HistogramTiles, true, K.Features, K.Leaves);
                    c.Dispatch("ScanHistograms", {histogram, HistWeights, FeatureTypes}, K, uint64_t(K.Leaves) * K.Features, true);
                }
                if (NoiseStatistics && scoreNewLeaves) {
                    auto noise = BootstrapParameters(); noise.Rows = Options.features;
                    noise.Stream = (FixedSplits.Enabled() || Langevin) ? ++scoredRound : round + 1; noise.NoiseScale = noiseScale;
                    c.Dispatch("GenerateScoreFeatureNoise", {FeatureNoise}, noise, Options.features);
                }
                if (!fixedSearch.IsForced()) {
                c.Dispatch("FindGreedyVectorSplitWinners", {HistSums, HistWeights, LeafSums, LeafWeights,
                    CandidateFeatures, CandidateBins, CandidateTypes, GreedyFeatureOffsets, FeatureWeights,
                    FeatureNoise, GreedyPartials}, G, G.ScoreGroups, true, G.Leaves);
                c.Dispatch("ReduceGreedySplitWinners", {GreedyPartials, GreedyWinners}, G, G.Leaves, true);
                }
                id<MTLBuffer> terminalOffsets = Offsets;
                if (BootstrapOptions.bootstrap_type >= 2) {
                    const CBMGreedyBootstrapParams p = {K.Rows, K.Leaves, BootstrapOptions.bootstrap_type, 0};
                    c.Dispatch("CountGreedyBootstrapRows", {Multipliers, RowIndices, Offsets, GreedySampledOffsets}, p, K.Leaves, true);
                    c.Dispatch("PrefixGreedyBootstrapOffsets", {GreedySampledOffsets}, p, 1, true);
                    terminalOffsets = GreedySampledOffsets;
                }
                if (FixedSplits.Enabled() || Langevin) {
                    c.Wait();
                    auto* winners = static_cast<CBMGreedySplit*>(GreedyWinners.contents);
                    if (FixedSplits.Enabled()) fixedSearch.Merge(winners);
                    else for (uint32_t leaf = 0; leaf < K.Leaves; ++leaf) {
                        if (scoreMask[leaf]) cachedWinners[leaf] = winners[leaf];
                        else winners[leaf] = cachedWinners[leaf];
                    }
                    auto selection = G;
                    if (fixedSearch.ForceAll()) selection.Policy = 0;
                    Command select(Stats);
                    select.Dispatch("SelectGreedyLeaves", {GreedyWinners, terminalOffsets, GreedyDepths,
                        GreedySelected, GreedyRightIds, GreedyFrontier}, selection, 1);
                    select.Wait();
                } else {
                    c.Dispatch("SelectGreedyLeaves", {GreedyWinners, terminalOffsets, GreedyDepths,
                        GreedySelected, GreedyRightIds, GreedyFrontier}, G, 1);
                    c.Wait();
                }
                frontier = *static_cast<const CBMGreedyFrontier*>(GreedyFrontier.contents);
            }
            Require(!frontier.Error, "Nonfinite vector greedy split score");
            Require(!FixedSplits.Enabled() || G.Policy != 2 || frontier.Selected <= 1,
                "CUDA Region fixed splits cannot produce a branching prefix");
            if (!frontier.Selected) break;
            const auto* selected = static_cast<const uint32_t*>(GreedySelected.contents);
            const auto* rightIds = static_cast<const uint32_t*>(GreedyRightIds.contents);
            const auto* winners = static_cast<const CBMGreedySplit*>(GreedyWinners.contents);
            std::vector<GreedyBranch> branches;
            leafNodes.resize(frontier.NewLeaves);
            if (Langevin) { unscored.resize(frontier.NewLeaves, 0); cachedWinners.resize(frontier.NewLeaves); }
            for (uint32_t i = 0; i < frontier.Selected; ++i) {
                const uint32_t parent = selected[i], right = rightIds[parent];
                if (FixedSplits.Enabled() && !ForcedGreedyRounds) fixedSearch.Split(parent, right);
                if (Langevin) { unscored[parent] = unscored[right] = 1; cachedWinners[parent] = {}; }
                const auto split = winners[parent];
                Require(parent < K.Leaves && right < frontier.NewLeaves && split.Valid, "Invalid vector greedy frontier");
                const uint32_t leftNode = nodes.size(), rightNode = leftNode + 1;
                nodes[leafNodes[parent]] = {split.Feature, split.Bin, split.Type, leftNode, rightNode, UINT32_MAX};
                nodes.push_back({0, 0, 0, 0, 0, parent}); nodes.push_back({0, 0, 0, 0, 0, right});
                leafNodes[parent] = leftNode; leafNodes[right] = rightNode;
                branches.push_back({parent, right, split});
            }
            if (!ForcedGreedyRounds) SelectedGreedyRounds.push_back(std::move(branches));
            Command c(Stats);
            c.Dispatch("RouteGreedySplitRows", {Bins, LeafIds, GreedyWinners, GreedyRightIds}, G, G.Rows);
            c.Dispatch("UpdateGreedyLeafDepths", {GreedyDepths, GreedyRightIds, GreedyNextDepths}, G, G.Leaves);
            c.Dispatch("CountGreedyPartitionBits", {RowIndices, LeafIds, RowPrefix, TilePrefix}, G, (G.Rows + 255) / 256, true);
            c.Dispatch("ScanGreedyPartitionTiles", {TilePrefix}, G, 1, true);
            c.Dispatch("BuildGreedyPartitionOffsets", {Offsets, GreedyRightIds, RowPrefix, TilePrefix, GreedyFrontier, NextOffsets}, G, G.Leaves);
            c.Dispatch("ScatterGreedyPartitionRows", {RowIndices, LeafIds, RowPrefix, TilePrefix, NextRows}, G, G.Rows);
            c.Wait(); std::swap(RowIndices, NextRows); std::swap(Offsets, NextOffsets); std::swap(GreedyDepths, GreedyNextDepths);
            K.Leaves = G.Leaves = frontier.NewLeaves; ++round;
        }
        if (Langevin && !ForcedGreedyRounds && !FixedSplits.Enabled() && Options.candidates && !scoredBatches)
            ReadLangevinSeed(CBM_LANGEVIN_SEARCH);
        if (!DeferLeafEstimation) EstimateAndPublish(values, weights);
        std::copy(nodes.begin(), nodes.end(), output);
        *info = {}; info->completed_iterations = Completed; info->finished = Completed == Options.iterations;
        info->node_count = nodes.size(); info->leaf_count = K.Leaves; info->loss = Loss; info->stats = Stats;
    }
    void Predictions(float* output) const {
        Require(output != nullptr, "Prediction output is required");
        std::memcpy(output, Published.contents, uint64_t(Options.rows) * Options.classes * 4);
    }
    void ConfigurePermutations(uint32_t count, const uint8_t* const* bins, const float* const* initial,
                               const float* lambdas, const uint8_t* valid) {
        Require(Completed == 0 && Permutations.empty(), "Permutations can be configured once before training");
        Require(count >= 1 && count <= 64 && bins, "Permutation count must be in [1,64] with matrices supplied");
        Require((lambdas == nullptr) == (valid == nullptr), "MVS placeholder arrays must be supplied together");
        const uint64_t cells = uint64_t(Options.rows) * Options.features;
        const uint64_t activeBytes = uint64_t(Options.rows) * D * 4;
        const uint64_t fullBytes = uint64_t(Options.rows) * Options.classes * 4;
        const uint64_t extra = uint64_t(count - 1) * (cells + activeBytes + fullBytes) + uint64_t(count) * (activeBytes + fullBytes);
        const uint64_t leafCells = uint64_t(count) * MaxLeaves * Options.classes;
        // Both new banks coexist with the previous banks until configuration
        // succeeds. Include that allocation peak in the ordinary workspace cap.
        const uint64_t retainedLeafBytes = leafCells * sizeof(float) * 2;
        Require(WorkingBytes + BootstrapBytes + NoiseBytes + BacktrackingBytes + extra + LangevinBytes + retainedLeafBytes <= MemoryLimit,
                "Multiclass permutation datasets and rollback cursors exceed 1 GiB working set");
        // Validate every permutation before mutating any existing resident state.
        for (uint32_t permutation = 0; permutation < count; ++permutation) {
            Require(bins[permutation] != nullptr, "Permutation bin matrix is null");
            Require(!valid || (!valid[permutation] && lambdas[permutation] == 0), "Multiclass does not support MVS permutation state");
            for (uint64_t cell = 0; cell < cells; ++cell)
                Require(bins[permutation][cell] < Options.bins_per_feature, "Permutation bin exceeds shared feature grid");
            if (initial) {
                Require(initial[permutation] != nullptr, "Permutation prediction matrix is null");
                for (uint64_t cell = 0; cell < fullBytes / 4; ++cell)
                    Require(std::isfinite(initial[permutation][cell]), "Permutation predictions must be finite");
                if (Options.objective == 0) for (uint32_t row = 0; row < Options.rows; ++row)
                    for (uint32_t k = 0; k < D; ++k)
                        Require(std::isfinite(initial[permutation][uint64_t(row) * Options.classes + k] -
                            initial[permutation][uint64_t(row) * Options.classes + D]), "Permutation gauge difference overflows float32");
            }
        }
        const auto* original = static_cast<const float*>(Published.contents);
        std::vector<float> lastLeaves(leafCells), pendingLeaves(leafCells);
        std::vector<float> commonInitial(original, original + fullBytes / 4);
        std::vector<PermutationData> states;
        states.reserve(count);
        // Allocate every buffer first, keeping configuration failure reversible.
        for (uint32_t permutation = 0; permutation < count; ++permutation) {
            const float* values = initial ? initial[permutation] : commonInitial.data();
            std::vector<float> active(activeBytes / 4);
            for (uint32_t row = 0; row < Options.rows; ++row) {
                const float anchor = Options.objective == 0 ? values[uint64_t(row) * Options.classes + D] : 0;
                for (uint32_t k = 0; k < D; ++k) active[uint64_t(k) * Options.rows + row] = values[uint64_t(row) * Options.classes + k] - anchor;
            }
            PermutationData state;
            if (permutation + 1 == count) {
                state.Bins = Bins; state.Cursor = Cursor; state.Published = Published;
                // Defer writes to original buffers until allocations succeed.
            } else {
                state.Bins = Context().Buffer(cells, bins[permutation]);
                state.Cursor = Context().Buffer(activeBytes, active.data());
                state.Published = Context().Buffer(fullBytes, values);
            }
            state.BackupCursor = Context().Buffer(activeBytes, active.data()); state.BackupPublished = Context().Buffer(fullBytes);
            states.push_back(state);
        }
        // Evaluate staged cursors before changing the original buffers. Finite
        // input logits can still overflow a weighted objective on the GPU.
        const id<MTLBuffer> previousCursor = Cursor;
        const float previousLoss = Loss;
        try {
            for (uint32_t permutation = 0; permutation < count; ++permutation) {
                Cursor = permutation + 1 == count ? states[permutation].BackupCursor : states[permutation].Cursor;
                Command c(Stats); Derivatives(c); c.Wait(); UpdateLoss();
                states[permutation].Loss = Loss;
            }
        } catch (...) { Cursor = previousCursor; Loss = previousLoss; throw; }
        Cursor = previousCursor; Loss = previousLoss;
        const float* last = initial ? initial[count - 1] : commonInitial.data();
        std::memcpy(Bins.contents, bins[count - 1], cells);
        std::memcpy(Published.contents, last, fullBytes);
        auto* active = static_cast<float*>(Cursor.contents);
        for (uint32_t row = 0; row < Options.rows; ++row) {
            const float anchor = Options.objective == 0 ? last[uint64_t(row) * Options.classes + D] : 0;
            for (uint32_t k = 0; k < D; ++k) active[uint64_t(k) * Options.rows + row] = last[uint64_t(row) * Options.classes + k] - anchor;
        }
        WorkingBytes += retainedLeafBytes - LastPermutationLeaves.size() * sizeof(float) * 2;
        LastPermutationLeaves.swap(lastLeaves); PendingPermutationLeaves.swap(pendingLeaves);
        Permutations = std::move(states); PermutationBytes = extra; SearchPermutation = count - 1;
        LoadPermutation(count - 1);
    }
    void SelectPermutation(uint32_t index) {
        Require(!Permutations.empty() && index < Permutations.size(), "Invalid multiclass search permutation");
        SearchPermutation = index;
    }
    void CopyPermutationState(uint32_t capacity, float* output, float* lambdas, uint8_t* valid) const {
        const uint32_t count = std::max<size_t>(1, Permutations.size());
        Require(capacity >= count && output && lambdas && valid, "Permutation state output capacity is insufficient");
        const uint64_t cells = uint64_t(Options.rows) * Options.classes;
        for (uint32_t permutation = 0; permutation < count; ++permutation) {
            const id<MTLBuffer> source = Permutations.empty() ? Published : Permutations[permutation].Published;
            std::memcpy(output + permutation * cells, source.contents, cells * 4);
            lambdas[permutation] = 0; valid[permutation] = 0;
        }
    }
    void CopyOptimizationState(uint32_t capacity, float* output) const {
        const uint32_t count = std::max<size_t>(1, Permutations.size());
        Require(capacity >= count && output, "Optimization state output capacity is insufficient");
        const uint64_t cells = uint64_t(Options.rows) * D;
        for (uint32_t permutation = 0; permutation < count; ++permutation) {
            const id<MTLBuffer> source = Permutations.empty() ? Cursor : Permutations[permutation].Cursor;
            std::memcpy(output + permutation * cells, source.contents, cells * 4);
        }
    }
    void CopyLastPermutationLeaves(uint32_t count, uint32_t maxLeaves, float* output) const {
        Require(LeafCaptureHealthy && !TreeActive && !CapturingLeaves && HasLastPermutationLeaves && Completed,
                "Last permutation leaves require an idle session with a successful tree");
        Require(count == std::max<size_t>(1, Permutations.size()) && maxLeaves == MaxLeaves && output,
                "Last permutation leaf output must match the exact session geometry");
        std::memcpy(output, LastPermutationLeaves.data(), LastPermutationLeaves.size() * sizeof(float));
    }
    void RestoreOptimizationState(uint32_t count, const float* input) {
        Require(Completed == 0 && count == std::max<size_t>(1, Permutations.size()) && input,
                "Optimization state must match the configured permutations and precede training");
        const uint64_t cells = uint64_t(Options.rows) * D;
        for (uint64_t cell = 0; cell < cells * count; ++cell)
            Require(std::isfinite(input[cell]), "Optimization state must be finite float32");
        const float previousLoss = Loss;
        std::vector<float> priorLoss;
        Command backup(Stats);
        if (Permutations.empty()) backup.Copy(Cursor, Base, cells * 4);
        else for (const auto& state : Permutations) {
            backup.Copy(state.Cursor, state.BackupCursor, cells * 4); priorLoss.push_back(state.Loss);
        }
        backup.Wait();
        try {
            for (uint32_t permutation = 0; permutation < count; ++permutation) {
                if (!Permutations.empty()) LoadPermutation(permutation);
                std::memcpy(Cursor.contents, input + permutation * cells, cells * 4);
                Command c(Stats); Derivatives(c); c.Wait(); UpdateLoss();
                if (!Permutations.empty()) Permutations[permutation].Loss = Loss;
            }
        } catch (...) {
            Command restore(Stats);
            if (Permutations.empty()) restore.Copy(Base, Cursor, cells * 4);
            else for (uint32_t permutation = 0; permutation < count; ++permutation) {
                auto& state = Permutations[permutation];
                restore.Copy(state.BackupCursor, state.Cursor, cells * 4); state.Loss = priorLoss[permutation];
            }
            restore.Wait(); Loss = previousLoss;
            if (!Permutations.empty()) LoadPermutation(count - 1);
            throw;
        }
        if (!Permutations.empty()) LoadPermutation(count - 1);
    }
private:
    CBMFixedSplits FixedSplits;
    bool Greedy = false;
    CBMVectorGreedyOptions GreedyOptions = {};
    CBMGreedyParams G = {};
    struct GreedyBranch { uint32_t Parent, Right; CBMGreedySplit Split; };
    using GreedyRounds = std::vector<std::vector<GreedyBranch>>;
    GreedyRounds SelectedGreedyRounds;
    const GreedyRounds* ForcedGreedyRounds = nullptr;
    id<MTLBuffer> GreedyDepths, GreedyNextDepths, GreedySelected, GreedyRightIds, GreedySampledOffsets;
    id<MTLBuffer> GreedyWinners, GreedyPartials, GreedyFrontier, GreedyFeatureOffsets;
    MathParams M;
    KernelParams K;
    uint32_t D, MaxLeaves, HistCells;
    double TotalWeight;
    bool TreeActive = false;
    bool CapturingLeaves = false, HasLastPermutationLeaves = false, LeafCaptureHealthy = true;
    std::vector<float> LastPermutationLeaves, PendingPermutationLeaves;
    void BeginLeafCapture() {
        Require(LeafCaptureHealthy && !TreeActive && !CapturingLeaves,
                "Multiclass session is not healthy and idle");
        std::fill(PendingPermutationLeaves.begin(), PendingPermutationLeaves.end(), 0.f);
        CapturingLeaves = true;
    }
    void StagePermutationLeaves(uint32_t index, const float* values) {
        const uint64_t cells = uint64_t(MaxLeaves) * Options.classes;
        std::memcpy(PendingPermutationLeaves.data() + index * cells, values, cells * sizeof(float));
    }
    void CommitLeafCapture() noexcept {
        LastPermutationLeaves.swap(PendingPermutationLeaves);
        HasLastPermutationLeaves = true;
        CapturingLeaves = false;
    }
    uint64_t WorkingBytes = 0, BootstrapBytes = 0, NoiseBytes = 0, BacktrackingBytes = 0, LangevinBytes = 0;
    bool Langevin = false, DeferLeafEstimation = false;
    CBMLangevinNoiseCallback LangevinNoise = nullptr;
    CBMLangevinSeedCallback LangevinSeed = nullptr;
    void* LangevinContext = nullptr;
    id<MTLBuffer> LangevinGradientNoise, LangevinHessianNoise, LangevinWorkspace;
    id<MTLBuffer> LangevinPoint, LangevinTrial, LangevinDirection;
    CBMBootstrapOptions BootstrapOptions = {0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0};
    float MVSLambda = 0, RandomStrength = 0;
    bool HasMVSLambda = false;
    id<MTLBuffer> StructureWeights, Multipliers;
    id<MTLBuffer> FeatureNoise, NoiseStatistics;
    id<MTLBuffer> FeatureWeights;
    std::vector<uint32_t> CtrUniqueValues;
    std::vector<uint8_t> UsedFeatures;
    std::vector<float> UserFeatureWeights;
    float ModelSizeReg = .5f;
    uint32_t BacktrackingType = 0;
    id<MTLBuffer> TrialValues, DirectionDots, TrialLoss;
    struct PermutationData {
        id<MTLBuffer> Bins, Cursor, Published, BackupCursor, BackupPublished;
        float Loss = 0;
    };
    std::vector<PermutationData> Permutations;
    uint32_t SearchPermutation = 0;
    uint64_t PermutationBytes = 0;
    const std::vector<SplitState>* ForcedTree = nullptr;
    id<MTLBuffer> Bins, Labels, Weights, Cursor, Base, Published, NextPublished, Gradients, Probabilities, Losses;
    id<MTLBuffer> RowIndices, NextRows, Offsets, NextOffsets, LeafIds, RowPrefix, TilePrefix, BlockPrefix;
    id<MTLBuffer> RawValues, Directions, LeafStats, SolveWorkspace, Status;
    id<MTLBuffer> HistSums, HistWeights, LeafSums, LeafWeights, Partials;
    id<MTLBuffer> CandidateFeatures, CandidateBins, CandidateTypes, FeatureTypes, Winner, Winners;
    void Derivatives(Command& command) {
        command.Dispatch(M.Objective < 2 ? "MulticlassDerivatives" : "MultioutputDerivatives", {Labels, Weights, Cursor, Gradients, Probabilities, Losses}, M, M.Rows);
    }
    void LoadPermutation(uint32_t index) {
        Bins = Permutations[index].Bins; Cursor = Permutations[index].Cursor;
        Published = Permutations[index].Published; Loss = Permutations[index].Loss;
    }
    void StepPermutations(CBMStepInfo* info, uint32_t* depth, uint32_t* features, uint32_t* borders,
                          uint8_t* types, float* values, float* weights) {
        Require(info && depth && values && weights && (!Options.depth || (features && borders && types)), "Step output buffers are required");
        Require(Completed < Options.iterations, "Multiclass session is already complete");
        const uint32_t priorCompleted = Completed, count = Permutations.size();
        const auto priorUsed = UsedFeatures;
        std::vector<float> priorLoss;
        Command backup(Stats);
        for (const auto& state : Permutations) {
            priorLoss.push_back(state.Loss);
            backup.Copy(state.Cursor, state.BackupCursor, uint64_t(Options.rows) * D * 4);
            backup.Copy(state.Published, state.BackupPublished, uint64_t(Options.rows) * Options.classes * 4);
        }
        backup.Wait();
        std::vector<uint32_t> temporaryFeatures(Options.depth), temporaryBins(Options.depth);
        std::vector<uint8_t> temporaryTypes(Options.depth);
        std::vector<float> temporaryValues(uint64_t(MaxLeaves) * Options.classes), temporaryWeights(MaxLeaves);
        std::vector<SplitState> selected;
        try {
            // Search the chosen weak history first, then evaluate every leaf
            // oracle in source history order. Structure-only work draws no leaf noise.
            const bool defer = Langevin && M.LeafMethod != 3;
            for (uint32_t order = 0; order < count + uint32_t(defer); ++order) {
                const uint32_t index = defer ? (order ? order - 1 : SearchPermutation)
                    : (order == 0 ? SearchPermutation : (order - 1 < SearchPermutation ? order - 1 : order));
                DeferLeafEstimation = defer && order == 0;
                LoadPermutation(index); Completed = priorCompleted;
                ForcedTree = order ? &selected : nullptr;
                const bool exported = index + 1 == count;
                CBMStepInfo temporaryInfo; uint32_t temporaryDepth;
                uint32_t* treeDepth = exported ? depth : &temporaryDepth;
                uint32_t* treeFeatures = exported ? features : temporaryFeatures.data();
                uint32_t* treeBins = exported ? borders : temporaryBins.data();
                uint8_t* treeTypes = exported ? types : temporaryTypes.data();
                StepOne(exported ? info : &temporaryInfo, treeDepth, treeFeatures, treeBins, treeTypes,
                        exported ? values : temporaryValues.data(), exported ? weights : temporaryWeights.data());
                if (!DeferLeafEstimation)
                    StagePermutationLeaves(index, exported ? values : temporaryValues.data());
                Permutations[index].Cursor = Cursor; Permutations[index].Published = Published;
                Permutations[index].Loss = Loss;
                if (order == 0) {
                    for (uint32_t level = 0; level < *treeDepth; ++level)
                        selected.push_back({level, treeFeatures[level], treeBins[level], treeTypes[level], 0, 1, 0, 0});
                    if (M.LeafMethod == 3 && count > 1) {
                        // CUDA NeedEstimation=false copies the weak values
                        // and sampled masses into every history. Reuse the
                        // solver buffers, which Simple never consumes.
                        Command cache(Stats);
                        cache.Copy(RawValues, Directions, uint64_t(K.Leaves) * D * 4);
                        cache.Copy(LeafStats, SolveWorkspace, uint64_t(K.Leaves) * 4);
                        cache.Wait();
                    }
                }
            }
            ForcedTree = nullptr; DeferLeafEstimation = false; Completed = priorCompleted + 1;
            LoadPermutation(count - 1); Info(*info);
        } catch (...) {
            ForcedTree = nullptr; DeferLeafEstimation = false;
            LeafCaptureHealthy = false;
            Command restore(Stats);
            for (uint32_t index = 0; index < count; ++index) {
                auto& state = Permutations[index];
                restore.Copy(state.BackupCursor, state.Cursor, uint64_t(Options.rows) * D * 4);
                restore.Copy(state.BackupPublished, state.Published, uint64_t(Options.rows) * Options.classes * 4);
                state.Loss = priorLoss[index];
            }
            restore.Wait(); Completed = priorCompleted; LoadPermutation(count - 1);
            UsedFeatures = priorUsed;
            LeafCaptureHealthy = true;
            throw;
        }
    }
    void UpdateFeatureWeights() {
        uint32_t maximum = 1;
        for (uint32_t feature = 0; feature < Options.features; ++feature)
            if (!UsedFeatures[feature]) maximum = std::max(maximum, CtrUniqueValues[feature]);
        auto* output = static_cast<float*>(FeatureWeights.contents);
        for (uint32_t feature = 0; feature < Options.features; ++feature) {
            if (!CtrUniqueValues[feature]) output[feature] = UserFeatureWeights[feature];
            else if (UsedFeatures[feature]) output[feature] = 1;
            else {
                const float base = 1.0f + float(CtrUniqueValues[feature]) / float(maximum);
                output[feature] = float(std::pow(double(base), -double(ModelSizeReg)));
            }
        }
    }
    double ReadExpansion(id<MTLBuffer> buffer, uint32_t count) const {
        const auto* values = static_cast<const float*>(buffer.contents);
        double sum = 0;
        for (uint32_t i = 0; i < count; ++i) sum += double(values[2 * i]) + values[2 * i + 1];
        return sum;
    }
    uint64_t ReadLangevinSeed(uint32_t event) {
        uint64_t seed = 0;
        Require(LangevinSeed && LangevinSeed(LangevinContext, event, &seed) == 0,
            "Vector Langevin seed callback failed");
        return seed;
    }
    void EstimateLangevinLeaves() {
        const uint32_t groups = std::min(256u, (Options.rows + 255) / 256);
        const uint32_t gradientCells = M.Leaves * M.Classes;
        const uint32_t hessianCells = gradientCells * ((M.LeafMethod == 1 || M.Objective == 1) ? 1 : M.Classes);
        std::memset(LangevinPoint.contents, 0, uint64_t(gradientCells) * 4);
        auto evaluate = [&](id<MTLBuffer> point) {
            Command oracle(Stats);
            oracle.Dispatch("MulticlassBacktrackingBuildCursor", {Base, point, LeafIds, Cursor}, M, uint64_t(M.Rows) * D);
            Derivatives(oracle);
            oracle.Dispatch(M.Objective < 2 ? "MulticlassReduceLeafStats" : "MultioutputReduceLeafStats",
                {Gradients, Probabilities, Weights, RowIndices, Offsets, LeafStats},
                M, uint64_t(M.Leaves) * StatsWidth(M), true);
            oracle.Dispatch(M.Objective < 2 ? "MulticlassBacktrackingReduceObjective" : "MultioutputBacktrackingReduceObjective",
                {Labels, Weights, Base, LeafIds, point, TrialLoss}, M, groups, true);
            oracle.Wait();
            return ReadExpansion(TrialLoss, groups);
        };
        auto noise = [&](uint32_t event, bool hessian, bool add) {
            const uint32_t count = hessian ? hessianCells : gradientCells;
            std::vector<double> values(count);
            Require(LangevinNoise(LangevinContext, event, count, values.data()) == 0,
                "Vector Langevin noise callback failed");
            auto* output = static_cast<float*>((hessian ? LangevinHessianNoise : LangevinGradientNoise).contents);
            for (uint32_t i = 0; i < count; ++i) {
                double value = values[i];
                if (add) value += double(output[2 * i]) + output[2 * i + 1];
                Require(std::isfinite(value) && std::isfinite(float(value)),
                    "Vector Langevin callback noise must be finite and fit float expansions");
                output[2 * i] = float(value); output[2 * i + 1] = float(value - double(output[2 * i]));
            }
        };
        auto solve = [&]() {
            Command direction(Stats);
            direction.Dispatch("VectorLangevinDirection", {LeafStats, LangevinGradientNoise, LangevinHessianNoise,
                LangevinWorkspace, LangevinDirection, DirectionDots, Status}, M, M.Leaves);
            direction.Wait();
            const auto* status = static_cast<const uint32_t*>(Status.contents);
            for (uint32_t leaf = 0; leaf < M.Leaves; ++leaf)
                Require(status[leaf] == 0, "Nonfinite vector Langevin leaf direction");
            const auto* partials = static_cast<const float*>(DirectionDots.contents);
            double dot = 0;
            for (uint32_t leaf = 0; leaf < M.Leaves; ++leaf) {
                Require(std::isfinite(partials[4 * leaf + 2]) && partials[4 * leaf + 2] >= -1022 && partials[4 * leaf + 2] <= 1023,
                    "Invalid vector Langevin direction exponent");
                dot += std::ldexp(double(partials[4 * leaf]) + partials[4 * leaf + 1], int(partials[4 * leaf + 2]));
            }
            Require(std::isfinite(dot), "Nonfinite vector Langevin direction dot product");
            return dot;
        };
        double currentValue = evaluate(RawValues);
        Require(std::isfinite(currentValue), "Nonfinite vector Langevin initial objective");
        noise(CBM_LANGEVIN_INITIAL_GRADIENT, false, false);
        noise(CBM_LANGEVIN_INITIAL_HESSIAN, true, false);
        double directionDot = solve();
        BacktrackingParams b = {1, BacktrackingType, 0, 0};
        bool accepted = false;
        for (uint32_t attempt = 0; attempt < Options.leaf_iterations || (!accepted && attempt < 100); ++attempt) {
            Command move(Stats);
            move.Dispatch("VectorLangevinCandidate", {LangevinPoint, LangevinDirection, LeafStats, LangevinTrial},
                M, gradientCells, false, 1, 1, &b, sizeof(b));
            move.Dispatch("VectorLangevinGauge", {LangevinTrial, TrialValues}, M, uint64_t(M.Leaves) * D);
            move.Wait();
            if (Options.leaf_iterations == 1) {
                std::swap(RawValues, TrialValues); std::swap(LangevinPoint, LangevinTrial); break;
            }
            const double trialValue = evaluate(TrialValues);
            noise(CBM_LANGEVIN_TRIAL_GRADIENT, false, false);
            const double threshold = currentValue + (b.Type == 2 ? 1e-5 * double(b.Step) * directionDot : 0);
            if (b.Type == 0 || (std::isfinite(trialValue) && trialValue >= threshold)) {
                std::swap(RawValues, TrialValues); std::swap(LangevinPoint, LangevinTrial);
                noise(CBM_LANGEVIN_ACCEPTED_GRADIENT, false, true);
                std::memset(LangevinHessianNoise.contents, 0, uint64_t(hessianCells) * 8);
                // NextPoint computes the accepted direction even at the
                // terminal iteration; no new target evaluation/noise occurs.
                directionDot = solve();
                currentValue = trialValue; accepted = true; b.Step = 1;
            } else b.Step *= .5f;
        }
        Command finish(Stats);
        finish.Dispatch("MulticlassBuildCursor", {Base, RawValues, LeafIds, Cursor}, M, uint64_t(M.Rows) * D,
            false, 1, 1, &Options.learning_rate, sizeof(float));
        Derivatives(finish); finish.Wait();
    }
    void EstimateBacktracking() {
        const uint32_t groups = std::min(256u, (Options.rows + 255) / 256);
        BacktrackingParams b = {1, BacktrackingType, 0, 0};
        Command initial(Stats);
        initial.Dispatch(M.Objective < 2 ? "MulticlassBacktrackingReduceObjective" : "MultioutputBacktrackingReduceObjective", {Labels, Weights, Base, LeafIds, RawValues, TrialLoss},
                         M, groups, true);
        initial.Wait();
        double currentValue = ReadExpansion(TrialLoss, groups), directionDot = 0;
        Require(std::isfinite(currentValue), "Nonfinite current multiclass leaf objective");
        bool accepted = false, newDirection = true;
        for (uint32_t attempt = 0; attempt < Options.leaf_iterations || (!accepted && attempt < 100); ++attempt) {
            if (newDirection) {
                Command solve(Stats); Derivatives(solve);
                solve.Dispatch(M.Objective < 2 ? "MulticlassReduceLeafStats" : "MultioutputReduceLeafStats", {Gradients, Probabilities, Weights, RowIndices, Offsets, LeafStats},
                               M, uint64_t(M.Leaves) * StatsWidth(M), true);
                MathParams unmasked = M; unmasked.MinLeafWeight = 0;
                solve.Dispatch(M.Objective < 2 ? "MulticlassSolveLeaves" : "MultioutputSolveLeaves", {LeafStats, SolveWorkspace, Directions, Status}, unmasked, M.Leaves);
                solve.Wait();
                const auto* status = static_cast<const uint32_t*>(Status.contents);
                for (uint32_t leaf = 0; leaf < M.Leaves; ++leaf)
                    Require(status[leaf] == 0, "Multiclass backtracking direction solve failed (status " + std::to_string(status[leaf]) + ")");
            }
            Command trial(Stats);
            if (newDirection) trial.Dispatch("MulticlassBacktrackingDirectionDot", {LeafStats, Directions, DirectionDots},
                                              M, M.Leaves, true);
            trial.Dispatch("MulticlassBacktrackingBuildCandidate", {RawValues, Directions, LeafStats, TrialValues},
                           M, M.Leaves * D, false, 1, 1, &b, sizeof(b));
            trial.Dispatch(M.Objective < 2 ? "MulticlassBacktrackingReduceObjective" : "MultioutputBacktrackingReduceObjective", {Labels, Weights, Base, LeafIds, TrialValues, TrialLoss},
                           M, groups, true);
            trial.Wait();
            if (newDirection) {
                const auto* dots = static_cast<const float*>(DirectionDots.contents);
                directionDot = 0;
                for (uint32_t leaf = 0; leaf < M.Leaves; ++leaf) {
                    Require(std::isfinite(dots[4 * leaf + 2]) && dots[4 * leaf + 2] >= -1022 && dots[4 * leaf + 2] <= 1023,
                            "Invalid scaled multiclass direction exponent");
                    directionDot += std::ldexp(double(dots[4 * leaf]) + dots[4 * leaf + 1], int(dots[4 * leaf + 2]));
                }
                Require(std::isfinite(directionDot), "Nonfinite multiclass backtracking direction dot product");
            }
            const double trialValue = ReadExpansion(TrialLoss, groups);
            const double threshold = currentValue + (BacktrackingType == 2 ? 1e-5 * double(b.Step) * directionDot : 0);
            if (std::isfinite(trialValue) && trialValue >= threshold) {
                std::swap(RawValues, TrialValues); currentValue = trialValue;
                accepted = true; newDirection = true; b.Step = 1;
                Command cursor(Stats);
                cursor.Dispatch("MulticlassBacktrackingBuildCursor", {Base, RawValues, LeafIds, Cursor}, M, uint64_t(M.Rows) * D);
                cursor.Wait();
            } else { b.Step *= .5f; newDirection = false; }
        }
        Command finish(Stats);
        finish.Dispatch("MulticlassBuildCursor", {Base, RawValues, LeafIds, Cursor}, M, uint64_t(M.Rows) * D,
                        false, 1, 1, &Options.learning_rate, sizeof(float));
        Derivatives(finish); finish.Wait();
    }
    BootstrapParams BootstrapParameters() const {
        return {Options.rows, BootstrapOptions.bootstrap_type, BootstrapOptions.random_seed_low,
            BootstrapOptions.random_seed_high, BootstrapOptions.iteration_offset + Completed, 0, 0, 0,
            BootstrapOptions.bagging_temperature, BootstrapOptions.subsample, 0, 0};
    }
    float PrepareStructure() {
        if (!BootstrapOptions.bootstrap_type && !NoiseStatistics) return 0;
        const MulticlassBootstrapParams m = {Options.rows, D, uint32_t(Options.objective == 0), 0};
        const uint32_t groups = std::min(4096u, (Options.rows + 255) / 256);
        Command c(Stats);
        if (BootstrapOptions.bootstrap_type) {
            c.Dispatch("GenerateBootstrapWeights", {Multipliers, Gradients}, BootstrapParameters(), Options.rows);
            c.Dispatch("ApplyMulticlassBootstrap", {Gradients, Weights, Multipliers, StructureWeights}, m,
                       uint64_t(Options.rows) * D);
        }
        if (NoiseStatistics) c.Dispatch("ReduceMulticlassScoreStatistics", {Gradients, StructureWeights, NoiseStatistics},
                                        m, groups, true);
        c.Wait();
        if (!NoiseStatistics) return 0;
        const auto* partials = static_cast<const float*>(NoiseStatistics.contents);
        double numerator = 0, denominator = 0;
        for (uint32_t i = 0; i < groups; ++i) {
            Require(std::isfinite(partials[2 * i]) && partials[2 * i] >= 0 &&
                    std::isfinite(partials[2 * i + 1]) && partials[2 * i + 1] >= 0,
                    "Nonfinite multiclass score noise statistic");
            numerator += partials[2 * i]; denominator += partials[2 * i + 1];
        }
        const double logRemaining = std::log(double(Options.rows)) -
            double(BootstrapOptions.iteration_offset + Completed) * Options.learning_rate;
        const double multiplier = logRemaining >= 0 ? 1 / (1 + std::exp(-logRemaining)) :
            std::exp(logRemaining) / (1 + std::exp(logRemaining));
        const float scale = float(RandomStrength * multiplier * (denominator > 0 ? std::sqrt(numerator / denominator) : 0));
        Require(std::isfinite(scale), "Multiclass score noise scale exceeds float32");
        return scale;
    }
    void UpdateLoss() {
        // Only the final O(N) metric reduction is on the host; derivatives,
        // Hessians, histogram scores, leaf solves and prediction updates are GPU.
        const auto* losses = static_cast<const float*>(Losses.contents);
        double sum = 0;
        for (uint32_t row = 0; row < Options.rows; ++row) {
            Require(std::isfinite(losses[row]), "Nonfinite multiclass objective loss"); sum += losses[row];
        }
        Loss = float(sum / TotalWeight);
        if (M.Objective == 2) Loss = std::sqrt(Loss);
        Require(std::isfinite(Loss), "Nonfinite multiclass objective reduction");
    }
};
template <class F> int Guard(char* error, size_t capacity, F action) {
    Text(error, capacity, "");
    struct Invocation { F& Function; char* Error; size_t Capacity; } invocation = {action, error, capacity};
    auto invoke = [](void* context) -> int {
        auto& call = *static_cast<Invocation*>(context);
        @try { call.Function(); return 0; }
        @catch (NSException* exception) { Text(call.Error, call.Capacity, [[exception reason] UTF8String]); }
        return 1;
    };
    @autoreleasepool {
        return CBMInvokeCppGuard(invoke, &invocation, error, capacity);
    }
}
Session& Get(void* session) { Require(session != nullptr, "Multiclass session is closed"); return *static_cast<Session*>(session); }
}

extern "C" int cbm_multiclass_session_set_fixed_splits(void* session, uint32_t count,
    const uint32_t* features, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.ConfigureFixedSplits(count, features); });
}
extern "C" int cbm_multiclass_session_set_langevin(void* session, float temperature,
    CBMLangevinNoiseCallback noise, CBMLangevinSeedCallback seed, void* context, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.ConfigureLangevin(temperature, noise, seed, context); });
}
extern "C" int cbm_multiclass_session_create(const CBMMulticlassParams* params, const uint8_t* bins,
    const uint32_t* labels, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint8_t* types, void** session, char* error, size_t capacity) {
    if (session) *session = nullptr;
    return Guard(error, capacity, [&] {
        Require(session != nullptr, "Session output is required");
        Require(params && params->objective < 2, "Use the float-target constructor for multioutput objectives");
        auto result = std::make_unique<Session>(params, bins, labels, weights, initial, features, borders, types);
        *session = result.release();
    });
}
extern "C" int cbm_multioutput_session_create(const CBMMulticlassParams* params, const uint8_t* bins,
    const float* targets, const float* weights, const float* initial, const uint32_t* features,
    const uint32_t* borders, const uint8_t* types, void** session, char* error, size_t capacity) {
    if (session) *session = nullptr;
    return Guard(error, capacity, [&] {
        Require(session != nullptr, "Session output is required");
        Require(params && params->objective >= 2 && params->objective <= 5, "Invalid multioutput objective");
        auto result = std::make_unique<Session>(params, bins, nullptr, weights, initial, features, borders, types, targets);
        *session = result.release();
    });
}
extern "C" int cbm_multiclass_session_step(void* session, CBMStepInfo* info, uint32_t* depth,
    uint32_t* features, uint32_t* borders, uint8_t* types, float* values, float* weights,
    char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.Step(info, depth, features, borders, types, values, weights); });
}
extern "C" int cbm_multiclass_session_create_greedy(const CBMMulticlassParams* params,
    const CBMVectorGreedyOptions* greedy, const uint8_t* bins, const uint32_t* labels, const float* targets,
    const float* weights, const float* initial, const uint32_t* features, const uint32_t* borders,
    const uint8_t* types, void** session, char* error, size_t capacity) {
    if (session) *session = nullptr;
    return Guard(error, capacity, [&] {
        Require(session && greedy, "Vector greedy options and session output are required");
        auto result = std::make_unique<Session>(params, bins, labels, weights, initial, features, borders, types, targets, greedy);
        *session = result.release();
    });
}
extern "C" int cbm_multiclass_session_step_greedy(void* session, CBMGreedyStepInfo* info,
    CBMGreedyNode* nodes, float* values, float* weights, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.StepGreedy(info, nodes, values, weights); });
}
extern "C" int cbm_multiclass_session_copy_predictions(void* session, float* predictions, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex); s.Predictions(predictions); });
}
extern "C" int cbm_multiclass_session_info(void* session, CBMStepInfo* info, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        Require(info != nullptr, "Info output is required"); s.Info(*info); });
}
extern "C" int cbm_multiclass_session_set_bootstrap(void* session, const CBMBootstrapOptions* options, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex); s.ConfigureBootstrap(options); });
}
extern "C" int cbm_multiclass_session_get_bootstrap_state(void* session, uint32_t* iteration, float* lambda,
    uint32_t* valid, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex); s.BootstrapState(iteration, lambda, valid); });
}
extern "C" int cbm_multiclass_session_set_score_noise(void* session, const CBMScoreNoiseOptions* options, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex); s.ConfigureNoise(options); });
}
extern "C" int cbm_multiclass_session_set_backtracking(void* session, uint32_t type, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex); s.ConfigureBacktracking(type); });
}
extern "C" int cbm_multiclass_session_set_permutations(void* session, uint32_t count, const uint8_t* const* bins,
    const float* const* initial, const float* lambdas, const uint8_t* valid, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.ConfigurePermutations(count, bins, initial, lambdas, valid); });
}
extern "C" int cbm_multiclass_session_select_permutation(void* session, uint32_t index, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex); s.SelectPermutation(index); });
}
extern "C" int cbm_multiclass_session_copy_permutation_state(void* session, uint32_t count, float* output,
    float* lambdas, uint8_t* valid, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.CopyPermutationState(count, output, lambdas, valid); });
}
extern "C" int cbm_multiclass_session_copy_optimization_state(void* session, uint32_t count, float* output,
    char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.CopyOptimizationState(count, output); });
}
extern "C" int cbm_multiclass_session_copy_last_permutation_leaves(void* session, uint32_t count,
    uint32_t maxLeaves, float* output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.CopyLastPermutationLeaves(count, maxLeaves, output); });
}
extern "C" int cbm_multiclass_session_restore_optimization_state(void* session, uint32_t count, const float* input,
    char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.RestoreOptimizationState(count, input); });
}
extern "C" int cbm_multiclass_session_set_feature_penalties(void* session, const CBMFeaturePenaltyOptions* options,
    const uint32_t* counts, const float* weights, const uint8_t* used, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.ConfigureFeaturePenalties(options, counts, weights, used); });
}
extern "C" int cbm_multiclass_session_copy_feature_penalty_state(void* session, uint8_t* output, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.CopyFeaturePenaltyState(output); });
}
extern "C" int cbm_multiclass_session_set_greedy_feature_weights(void* session, uint32_t count,
    const float* weights, char* error, size_t capacity) {
    return Guard(error, capacity, [&] { auto& s = Get(session); std::lock_guard<std::mutex> lock(s.Mutex);
        s.ConfigureGreedyFeatureWeights(count, weights); });
}
extern "C" void cbm_multiclass_session_close(void* session) { @autoreleasepool { delete static_cast<Session*>(session); } }

extern "C" int cbm_multiclass_math(uint32_t rows, uint32_t classes, uint32_t objective, uint32_t leaves,
    uint32_t method, float l2, const uint32_t* labels, const float* weights, const float* logits,
    const uint32_t* leafIds, float* gradients, float* probabilities, float* losses, float* directions,
    CBMTrainStats* stats, char* error, size_t capacity) {
    return Guard(error, capacity, [&] {
        MathParams p = {rows, classes, objective, leaves, l2, 1e-20f, method, 0};
        ValidateMath(p, labels, weights, logits);
        Require(weights && logits && leafIds && gradients && probabilities && losses && directions && stats,
                "Multiclass math input and output buffers are required");
        const uint32_t d = Dimension(p), width = StatsWidth(p);
        const uint64_t bytes = uint64_t(rows) * (16 + 4 * (2 * d + classes)) + uint64_t(leaves) *
            (12 + 4 * (width + classes * classes + classes + d));
        Require(bytes <= MemoryLimit, "Multiclass math working set exceeds 1 GiB");
        std::vector<uint32_t> offsets(leaves + 1, 0), indices(rows), positions;
        for (uint32_t row = 0; row < rows; ++row) { Require(leafIds[row] < leaves, "Invalid leaf id"); ++offsets[leafIds[row] + 1]; }
        for (uint32_t leaf = 0; leaf < leaves; ++leaf) offsets[leaf + 1] += offsets[leaf];
        positions = offsets;
        for (uint32_t row = 0; row < rows; ++row) indices[positions[leafIds[row]]++] = row;
        auto& r = Context(); *stats = {}; Text(stats->device_name, sizeof(stats->device_name), r.Device.name.UTF8String);
        auto label = r.Buffer(uint64_t(rows) * 4, labels), weight = r.Buffer(uint64_t(rows) * 4, weights);
        auto cursor = r.Buffer(uint64_t(rows) * d * 4, logits), gradient = r.Buffer(uint64_t(rows) * d * 4);
        auto probability = r.Buffer(uint64_t(rows) * classes * 4), loss = r.Buffer(uint64_t(rows) * 4);
        auto partition = r.Buffer(uint64_t(rows) * 4, indices.data()), offset = r.Buffer(uint64_t(leaves + 1) * 4, offsets.data());
        auto statistics = r.Buffer(uint64_t(leaves) * width * 4);
        auto workspace = r.Buffer(uint64_t(leaves) * (classes * classes + classes) * 4);
        auto direction = r.Buffer(uint64_t(leaves) * d * 4), status = r.Buffer(uint64_t(leaves) * 4);
        Command c(*stats);
        c.Dispatch("MulticlassDerivatives", {label, weight, cursor, gradient, probability, loss}, p, rows);
        c.Dispatch("MulticlassReduceLeafStats", {gradient, probability, weight, partition, offset, statistics}, p,
                   uint64_t(leaves) * width, true);
        c.Dispatch("MulticlassSolveLeaves", {statistics, workspace, direction, status}, p, leaves); c.Wait();
        const auto* result = static_cast<const uint32_t*>(status.contents);
        for (uint32_t leaf = 0; leaf < leaves; ++leaf) Require(result[leaf] == 0,
            "Multiclass leaf solve failed (status " + std::to_string(result[leaf]) + ")");
        std::memcpy(gradients, gradient.contents, uint64_t(rows) * d * 4);
        std::memcpy(probabilities, probability.contents, uint64_t(rows) * classes * 4);
        std::memcpy(losses, loss.contents, uint64_t(rows) * 4);
        std::memcpy(directions, direction.contents, uint64_t(leaves) * d * 4);
    });
}
