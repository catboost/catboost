#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../native/metal_kernels.h"
#include "../native/metal_additional_objective_kernels.h"
#include "../native/metal_objective_kernels.h"
#include "../native/metal_streaming_score_kernels.h"
#include "../native/metal_dynamic_score_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static_assert(CBMMetalKernelAbiVersion == 2, "Review dynamic probe after shared ABI changes");
struct ProbeScoreTileParams {
    uint32_t FeatureBegin, FeatureEnd, TotalBins, Reserved[5];
};
static_assert(sizeof(ProbeScoreTileParams) == 32);

namespace {
std::string ErrorText(NSError* error) {
    return error ? [[error localizedDescription] UTF8String] : "Unknown Metal probe error";
}
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    id<MTLLibrary> Library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device != nil, "No Metal GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "No Metal command queue");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        options.fastMathEnabled = NO;
        NSError* error = nil;
        Library = [Device newLibraryWithSource:[NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s",
            CBMMetalSource, CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
            CBMMetalStreamingScoreSource, CBMMetalDynamicScoreSource] options:options error:&error];
        if (!Library) throw std::runtime_error(ErrorText(error));
        for (const char* name : {"FindTileSplitWinners", "FindDynamicTileSplitWinners",
                                 "ReduceSplitWinners", "MergeTileSplitWinner"}) {
            id<MTLFunction> function = [Library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, "Missing dynamic score probe kernel");
            id<MTLComputePipelineState> pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error(ErrorText(error));
            Require(pipeline.maxTotalThreadsPerThreadgroup >= 256, "Probe kernel requires 256 threads");
            Pipelines.emplace(name, pipeline);
        }
    }
    id<MTLBuffer> Buffer(const void* source, size_t bytes) {
        Require(bytes <= 128ull * 1024 * 1024 && bytes <= Device.maxBufferLength,
                "Dynamic score probe exceeds buffer memory guard");
        const size_t size = std::max<size_t>(bytes, 1);
        id<MTLBuffer> result = source && bytes
            ? [Device newBufferWithBytes:source length:size options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:size options:MTLResourceStorageModeShared];
        Require(result != nil, "Dynamic score probe allocation failed");
        return result;
    }
    void Dispatch(id<MTLCommandBuffer> command, const char* name,
                  std::initializer_list<id<MTLBuffer>> buffers, const CBMMetalKernelParams& p,
                  const ProbeScoreTileParams* tile, uint32_t count, bool groups) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:Pipelines.at(name)];
        NSUInteger index = 0;
        for (id<MTLBuffer> buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
        [encoder setBytes:&p length:sizeof(p) atIndex:index++];
        if (tile) [encoder setBytes:tile length:sizeof(*tile) atIndex:index];
        if (groups) [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
};
}

// out contains one reduced winner per tile followed by the final merged
// winner. useDynamic=0 exercises the canonical scorer as an independent GPU
// arithmetic reference on a caller-supplied stable subsequence of candidates.
extern "C" int cbm_dynamic_score_probe(
    const CBMMetalKernelParams* params, const float* histogramSums,
    const float* histogramWeights, const float* leafSums, const float* leafWeights,
    const uint32_t* candidateFeatures, const uint32_t* candidateBins,
    const uint8_t* candidateTypes, const float* featureNoise,
    const uint32_t* featureOffsets, const float* featurePenalties,
    const uint8_t* candidateActive, const uint32_t* tileRanges,
    uint32_t tileCount, uint32_t useDynamic, CBMMetalSplitState* out,
    char* errorText, uint32_t errorCapacity) {
    @autoreleasepool {
        try {
            Require(params && histogramSums && histogramWeights && leafSums && leafWeights &&
                    candidateFeatures && candidateBins && candidateTypes && featureNoise &&
                    featureOffsets && featurePenalties && candidateActive && tileRanges && out,
                    "Null dynamic score probe argument");
            const auto& p = *params;
            Require(p.Features > 0 && p.Features <= 4096 && p.Leaves > 0 && p.Leaves <= 65536 &&
                    p.Candidates <= (1u << 20) && p.ScoreGroups > 0 && p.ScoreGroups <= 256 &&
                    p.ScoreFunction <= 5 && tileCount > 0 && tileCount <= 64 && useDynamic <= 1,
                    "Invalid dynamic score probe dimensions");
            Require(featureOffsets[0] == 0, "Feature offsets must start at zero");
            for (uint32_t feature = 0; feature < p.Features; ++feature)
                Require(featureOffsets[feature + 1] >= featureOffsets[feature] &&
                        featureOffsets[feature + 1] - featureOffsets[feature] <= 256,
                        "Invalid compact feature span");
            const uint32_t totalBins = featureOffsets[p.Features];
            const uint64_t cells = uint64_t(p.Leaves) * totalBins;
            Require(16 * cells + 8ull * p.Leaves + 10ull * p.Candidates +
                    32ull * p.Features + 32ull * (p.ScoreGroups + 2) <= 128ull * 1024 * 1024,
                    "Dynamic score probe exceeds aggregate 128 MiB memory guard");
            for (uint32_t candidate = 0; candidate < p.Candidates; ++candidate) {
                if (useDynamic && !candidateActive[candidate]) continue;
                const uint32_t feature = candidateFeatures[candidate];
                Require(feature < p.Features && candidateTypes[candidate] <= 1,
                        "Invalid active candidate metadata");
                Require(candidateBins[candidate] < featureOffsets[feature + 1] - featureOffsets[feature],
                        "Active candidate bin exceeds compact feature span");
            }
            for (uint32_t tile = 0; tile < tileCount; ++tile)
                Require(tileRanges[2 * tile] <= tileRanges[2 * tile + 1] &&
                        tileRanges[2 * tile + 1] <= p.Features, "Invalid feature tile range");
            static Runtime runtime;
            auto& r = runtime;
            id<MTLBuffer> leafS = r.Buffer(leafSums, p.Leaves * sizeof(float));
            id<MTLBuffer> leafW = r.Buffer(leafWeights, p.Leaves * sizeof(float));
            id<MTLBuffer> features = r.Buffer(candidateFeatures, p.Candidates * sizeof(uint32_t));
            id<MTLBuffer> bins = r.Buffer(candidateBins, p.Candidates * sizeof(uint32_t));
            id<MTLBuffer> types = r.Buffer(candidateTypes, p.Candidates);
            id<MTLBuffer> active = r.Buffer(candidateActive, p.Candidates);
            id<MTLBuffer> noise = r.Buffer(featureNoise, p.Features * sizeof(float));
            id<MTLBuffer> penalties = r.Buffer(featurePenalties, p.Features * 2 * sizeof(float));
            id<MTLBuffer> partials = r.Buffer(nullptr, p.ScoreGroups * sizeof(CBMMetalSplitState));
            id<MTLBuffer> global = r.Buffer(nullptr, sizeof(CBMMetalSplitState));
            id<MTLBuffer> next = r.Buffer(nullptr, sizeof(CBMMetalSplitState));
            for (uint32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
                @autoreleasepool {
                    const uint32_t begin = tileRanges[2 * tileIndex], end = tileRanges[2 * tileIndex + 1];
                    const uint32_t tileBins = featureOffsets[end] - featureOffsets[begin];
                    const ProbeScoreTileParams tile{begin, end, tileBins, {0, 0, 0, 0, 0}};
                    std::vector<uint32_t> offsets(end - begin + 1);
                    for (uint32_t feature = begin; feature <= end; ++feature)
                        offsets[feature - begin] = featureOffsets[feature] - featureOffsets[begin];
                    std::vector<float> sums(uint64_t(p.Leaves) * tileBins), weights(sums.size());
                    for (uint32_t leaf = 0; leaf < p.Leaves; ++leaf) {
                        if (!tileBins) continue;
                        const size_t source = size_t(leaf) * totalBins + featureOffsets[begin];
                        std::copy_n(histogramSums + source, tileBins, sums.data() + size_t(leaf) * tileBins);
                        std::copy_n(histogramWeights + source, tileBins, weights.data() + size_t(leaf) * tileBins);
                    }
                    id<MTLBuffer> histS = r.Buffer(sums.data(), sums.size() * sizeof(float));
                    id<MTLBuffer> histW = r.Buffer(weights.data(), weights.size() * sizeof(float));
                    id<MTLBuffer> localOffsets = r.Buffer(offsets.data(), offsets.size() * sizeof(uint32_t));
                    id<MTLCommandBuffer> command = [r.Queue commandBuffer];
                    if (useDynamic)
                        r.Dispatch(command, "FindDynamicTileSplitWinners",
                            {histS, histW, leafS, leafW, features, bins, types, partials, noise, localOffsets, penalties, active},
                            p, &tile, p.ScoreGroups, true);
                    else
                        r.Dispatch(command, "FindTileSplitWinners",
                            {histS, histW, leafS, leafW, features, bins, types, partials, noise, localOffsets, penalties},
                            p, &tile, p.ScoreGroups, true);
                    id<MTLBuffer> reduced = tileIndex == 0 ? global : next;
                    r.Dispatch(command, "ReduceSplitWinners", {partials, reduced}, p, nullptr, 1, true);
                    if (tileIndex)
                        r.Dispatch(command, "MergeTileSplitWinner", {global, next}, p, nullptr, 1, false);
                    [command commit];
                    [command waitUntilCompleted];
                    if (command.status != MTLCommandBufferStatusCompleted)
                        throw std::runtime_error(ErrorText(command.error));
                    std::memcpy(out + tileIndex, reduced.contents, sizeof(CBMMetalSplitState));
                }
            }
            std::memcpy(out + tileCount, global.contents, sizeof(CBMMetalSplitState));
            return 0;
        } catch (const std::exception& error) {
            if (errorText && errorCapacity) {
                std::strncpy(errorText, error.what(), errorCapacity - 1);
                errorText[errorCapacity - 1] = '\0';
            }
            return 1;
        }
    }
}
