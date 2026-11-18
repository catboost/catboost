#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_greedy_kernels.h"
#ifdef CBM_GREEDY_VECTOR_PROBE
#include "../native/metal_multiclass_scores.h"
#include "../native/metal_greedy_vector_scores.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Runtime {
    id<MTLDevice> Device;
    id<MTLLibrary> Library;
    id<MTLCommandQueue> Queue;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device != nil, "No Metal device");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        options.fastMathEnabled = NO;
        NSError* error = nil;
#ifdef CBM_GREEDY_VECTOR_PROBE
        NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s", CBMMetalMulticlassScoresSource,
            CBMMetalGreedySource, CBMMetalGreedyVectorScoresSource];
#else
        NSString* source = [NSString stringWithUTF8String:CBMMetalGreedySource];
#endif
        Library = [Device newLibraryWithSource:source
            options:options error:&error];
        if (!Library) throw std::runtime_error([[error localizedDescription] UTF8String]);
        Queue = [Device newCommandQueue];
    }
    id<MTLBuffer> Buffer(const void* data, size_t bytes) {
        Require(bytes && bytes <= (1ull << 30), "Probe allocation exceeds 1 GiB");
        id<MTLBuffer> result = data ? [Device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Require(result != nil, "Metal greedy probe allocation failed");
        return result;
    }
    void Dispatch(id<MTLCommandBuffer> command, const char* name,
        std::initializer_list<id<MTLBuffer>> buffers, const CBMGreedyParams* p,
        MTLSize grid, bool groups = false) {
        NSError* error = nil;
        id<MTLFunction> function = [Library newFunctionWithName:[NSString stringWithUTF8String:name]];
        id<MTLComputePipelineState> pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        NSUInteger index = 0;
        for (id<MTLBuffer> buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
        if (p) [encoder setBytes:p length:sizeof(*p) atIndex:index];
        if (groups) [encoder dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    void Complete(id<MTLCommandBuffer> command) {
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error([[command.error localizedDescription] UTF8String]);
    }
};
Runtime& GetRuntime() { static Runtime runtime; return runtime; }
void Validate(const CBMGreedyParams& p) {
    Require(p.Rows && p.Rows <= (1u << 24), "Invalid row count");
    Require(p.Features && p.Features <= 65536, "Invalid feature count");
    Require(p.Leaves && p.Leaves <= 65536 && p.MaxLeaves >= p.Leaves && p.MaxLeaves <= 65536,
        "Invalid leaf count or capacity");
    Require(p.MaxDepth && p.MaxDepth <= 64 && p.Policy <= 2, "Invalid grow policy or depth");
    Require(p.Reserved0 == 0 && p.Reserved1 == 0, "Reserved fields must be zero");
}
int Error(const std::exception& exception, char* text, uint32_t capacity) {
    if (text && capacity) {
        std::strncpy(text, exception.what(), capacity - 1);
        text[capacity - 1] = '\0';
    }
    return 1;
}
}

extern "C" int cbm_greedy_score_probe(const CBMGreedyParams* params,
    const float* sums, const float* weights, const float* leafSums, const float* leafWeights,
    const uint32_t* features, const uint32_t* bins, const uint8_t* types,
    const uint32_t* featureOffsets, const float* featureWeights, const float* featureNoise,
    CBMGreedySplit* output, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            Require(params && sums && weights && leafSums && leafWeights && features && bins && types &&
                featureOffsets && featureWeights && featureNoise && output, "Null score probe buffer");
            const auto& p = *params;
            Validate(p);
            Require(p.Candidates && p.Candidates <= (1u << 24) && p.ScoreGroups && p.ScoreGroups <= 1024,
                "Invalid candidate or score group count");
            Require(p.Dimensions && p.Dimensions <= 64 && p.MulticlassOptimization <= 1 && p.Normalize <= 1 &&
                p.ScoreFunction <= 6 && std::isfinite(p.L2) && p.L2 >= 0, "Invalid scoring parameter");
            const uint64_t cells = uint64_t(p.Leaves) * p.TotalBins;
            Require(p.TotalBins && p.HistogramStride >= cells && p.LeafStride >= p.Leaves,
                "Invalid histogram plane strides");
            Require(featureOffsets[0] == 0 && featureOffsets[p.Features] == p.TotalBins,
                "Feature offsets must cover histogram bins");
            for (uint32_t feature = 0; feature < p.Features; ++feature)
                Require(featureOffsets[feature] < featureOffsets[feature + 1], "Feature offsets must increase");
            uint32_t globalFeatures = p.FeatureBegin + p.Features;
            Require(globalFeatures >= p.Features && globalFeatures <= (1u << 20), "Feature tile is too large");
            for (uint32_t candidate = 0; candidate < p.Candidates; ++candidate) {
                Require(features[candidate] < (1u << 20) && types[candidate] <= 1, "Invalid candidate");
                globalFeatures = std::max(globalFeatures, features[candidate] + 1);
                if (features[candidate] >= p.FeatureBegin && features[candidate] - p.FeatureBegin < p.Features) {
                    const uint32_t local = features[candidate] - p.FeatureBegin;
                    Require(bins[candidate] < featureOffsets[local + 1] - featureOffsets[local],
                        "Candidate bin is outside its feature");
                }
            }
            for (uint32_t feature = 0; feature < globalFeatures; ++feature)
                Require(std::isfinite(featureWeights[feature]) && featureWeights[feature] >= 0 &&
                    std::isfinite(featureNoise[feature]), "Invalid feature weight or noise");
            Runtime& runtime = GetRuntime();
#ifdef CBM_GREEDY_VECTOR_PROBE
            Require(p.ScoreFunction == 0 || p.ScoreFunction == 1 || p.ScoreFunction >= 4,
                "Vector greedy scoring does not support Newton structure scores");
            const uint32_t leafBytes = 8;
            const char* scoreKernel = "FindGreedyVectorSplitWinners";
#else
            const uint32_t leafBytes = 4;
            const char* scoreKernel = "FindGreedySplitWinners";
#endif
            auto s = runtime.Buffer(sums, uint64_t(p.Dimensions) * p.HistogramStride * 4);
            auto w = runtime.Buffer(weights, cells * 4);
            auto ls = runtime.Buffer(leafSums, uint64_t(p.Dimensions) * p.LeafStride * leafBytes);
            auto lw = runtime.Buffer(leafWeights, p.Leaves * leafBytes);
            auto f = runtime.Buffer(features, p.Candidates * 4), b = runtime.Buffer(bins, p.Candidates * 4);
            auto t = runtime.Buffer(types, p.Candidates), offsets = runtime.Buffer(featureOffsets, (p.Features + 1) * 4);
            auto fw = runtime.Buffer(featureWeights, globalFeatures * 4), noise = runtime.Buffer(featureNoise, globalFeatures * 4);
            auto partials = runtime.Buffer(nullptr, uint64_t(p.Leaves) * p.ScoreGroups * sizeof(CBMGreedySplit));
            auto winners = runtime.Buffer(nullptr, p.Leaves * sizeof(CBMGreedySplit));
            id<MTLCommandBuffer> command = [runtime.Queue commandBuffer];
            runtime.Dispatch(command, scoreKernel, {s, w, ls, lw, f, b, t, offsets, fw, noise, partials},
                &p, MTLSizeMake(p.ScoreGroups, p.Leaves, 1), true);
            runtime.Dispatch(command, "ReduceGreedySplitWinners", {partials, winners}, &p,
                MTLSizeMake(p.Leaves, 1, 1), true);
            runtime.Complete(command);
            std::memcpy(output, winners.contents, p.Leaves * sizeof(CBMGreedySplit));
            return 0;
        } catch (const std::exception& error) { return Error(error, errorText, capacity); }
    }
}

extern "C" int cbm_greedy_frontier_probe(const CBMGreedyParams* params,
    const CBMGreedySplit* winners, const uint32_t* offsets, const uint32_t* depths,
    const uint8_t* bins, const uint32_t* indices, const uint32_t* leafIds,
    uint32_t* selected, uint32_t* rightIds, uint32_t* newLeafIds, uint32_t* newIndices,
    uint32_t* newOffsets, uint32_t* newDepths, CBMGreedyFrontier* info,
    char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            Require(params && winners && offsets && depths && bins && indices && leafIds && selected &&
                rightIds && newLeafIds && newIndices && newOffsets && newDepths && info, "Null frontier probe buffer");
            const auto& p = *params;
            Validate(p);
            Require(p.FeatureBegin == 0, "Routing requires the complete feature matrix");
            Require(offsets[0] == 0 && offsets[p.Leaves] == p.Rows, "Partitions must cover every row");
            std::vector<uint8_t> seen(p.Rows, 0);
            for (uint32_t leaf = 0; leaf < p.Leaves; ++leaf) {
                Require(offsets[leaf] <= offsets[leaf + 1] && offsets[leaf + 1] <= p.Rows,
                    "Partition offsets must be monotone");
                Require(depths[leaf] <= p.MaxDepth, "Invalid leaf depth");
                if (winners[leaf].Valid)
                    Require(winners[leaf].Feature < p.Features && winners[leaf].Bin <= 255 &&
                        winners[leaf].Type <= 1 && std::isfinite(winners[leaf].Gain), "Invalid split winner");
                for (uint32_t position = offsets[leaf]; position < offsets[leaf + 1]; ++position) {
                    const uint32_t row = indices[position];
                    Require(row < p.Rows && !seen[row] && leafIds[row] == leaf, "Invalid row partition permutation");
                    seen[row] = 1;
                }
            }
            Runtime& runtime = GetRuntime();
            auto win = runtime.Buffer(winners, p.Leaves * sizeof(CBMGreedySplit));
            auto oldOffsets = runtime.Buffer(offsets, (p.Leaves + 1) * 4), oldDepths = runtime.Buffer(depths, p.Leaves * 4);
            auto matrix = runtime.Buffer(bins, uint64_t(p.Features) * p.Rows);
            auto oldIndices = runtime.Buffer(indices, p.Rows * 4), rowLeaves = runtime.Buffer(leafIds, p.Rows * 4);
            auto selection = runtime.Buffer(nullptr, p.Leaves * 4), right = runtime.Buffer(nullptr, p.Leaves * 4);
            auto result = runtime.Buffer(nullptr, sizeof(CBMGreedyFrontier));
            const uint32_t tiles = (p.Rows - 1) / 256 + 1;
            auto ranks = runtime.Buffer(nullptr, p.Rows * 4), tilePrefix = runtime.Buffer(nullptr, tiles * 4);
            auto outIndices = runtime.Buffer(nullptr, p.Rows * 4), outOffsets = runtime.Buffer(nullptr, (p.MaxLeaves + 1) * 4);
            auto outDepths = runtime.Buffer(nullptr, p.MaxLeaves * 4);
            id<MTLCommandBuffer> command = [runtime.Queue commandBuffer];
            runtime.Dispatch(command, "SelectGreedyLeaves", {win, oldOffsets, oldDepths, selection, right, result}, &p,
                MTLSizeMake(1, 1, 1));
            runtime.Dispatch(command, "RouteGreedySplitRows", {matrix, rowLeaves, win, right}, &p,
                MTLSizeMake(p.Rows, 1, 1));
            runtime.Dispatch(command, "UpdateGreedyLeafDepths", {oldDepths, right, outDepths}, &p,
                MTLSizeMake(p.Leaves, 1, 1));
            runtime.Dispatch(command, "CountGreedyPartitionBits", {oldIndices, rowLeaves, ranks, tilePrefix}, &p,
                MTLSizeMake(tiles, 1, 1), true);
            runtime.Dispatch(command, "ScanGreedyPartitionTiles", {tilePrefix}, &p,
                MTLSizeMake(1, 1, 1), true);
            runtime.Dispatch(command, "BuildGreedyPartitionOffsets", {oldOffsets, right, ranks, tilePrefix, result, outOffsets}, &p,
                MTLSizeMake(p.Leaves, 1, 1));
            runtime.Dispatch(command, "ScatterGreedyPartitionRows", {oldIndices, rowLeaves, ranks, tilePrefix, outIndices}, &p,
                MTLSizeMake(p.Rows, 1, 1));
            runtime.Complete(command);
            std::memcpy(info, result.contents, sizeof(*info));
            std::memcpy(selected, selection.contents, info->Selected * 4);
            std::memcpy(rightIds, right.contents, p.Leaves * 4);
            std::memcpy(newLeafIds, rowLeaves.contents, p.Rows * 4);
            std::memcpy(newIndices, outIndices.contents, p.Rows * 4);
            std::memcpy(newOffsets, outOffsets.contents, (info->NewLeaves + 1) * 4);
            std::memcpy(newDepths, outDepths.contents, info->NewLeaves * 4);
            return 0;
        } catch (const std::exception& error) { return Error(error, errorText, capacity); }
    }
}
