#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../native/metal_kernels.h"
#include "../native/metal_additional_objective_kernels.h"
#include "../native/metal_objective_kernels.h"
#include "../native/metal_histogram_kernels.h"
#include "../native/metal_histogram_reuse_kernels.h"
#include "../native/metal_incremental_partition_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(CBMMetalKernelAbiVersion == 2, "Review probe bindings after ABI changes");

// Each histogram output has ten feature-by-bin planes, in this order:
// root raw, root prefix, smaller raw, smaller prefix, reused children (2),
// rebuilt children raw (2), rebuilt children prefix (2). Smaller snapshots are
// taken from the temporary right-child storage before sibling subtraction.
// Partition outputs contain root, left child, right child statistics.
extern "C" int cbm_scalar_histogram_probe(
    uint32_t rows, uint32_t features, uint32_t bins, uint32_t tiles, uint32_t rebuild,
    const uint8_t* binData, const float* gradients, const float* weights,
    const uint32_t* childIndices, const uint32_t* childOffsets, const uint8_t* featureTypes,
    float* gradientStages, float* weightStages, float* partitionSums, float* partitionWeights,
    char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            const size_t cells = size_t(features) * bins;
            if (!rows || rows > (1u << 24) || !features || !bins || bins > 256 ||
                !tiles || tiles > 256 || cells > (1u << 24) || size_t(rows) * features > (1u << 29))
                throw std::runtime_error("Invalid histogram probe dimensions");
            if (!binData || !gradients || !weights || !childIndices || !childOffsets ||
                !featureTypes || !gradientStages || !weightStages || !partitionSums || !partitionWeights)
                throw std::runtime_error("Null histogram probe buffer");
            if (childOffsets[0] != 0 || childOffsets[1] > rows || childOffsets[2] != rows)
                throw std::runtime_error("Invalid child partition offsets");
            std::vector<bool> seen(rows, false);
            for (uint32_t i = 0; i < rows; ++i) {
                if (childIndices[i] >= rows || seen[childIndices[i]])
                    throw std::runtime_error("Child row indices must be a permutation");
                seen[childIndices[i]] = true;
                if (!std::isfinite(gradients[i]) || !std::isfinite(weights[i]) || weights[i] < 0)
                    throw std::runtime_error("Gradients and nonnegative weights must be finite");
            }
            for (size_t i = 0; i < size_t(rows) * features; ++i)
                if (binData[i] >= bins) throw std::runtime_error("Feature bin exceeds histogram width");

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:
                [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s",
                    CBMMetalSource, CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
                    CBMMetalHistogramSource, CBMMetalHistogramReuseSource, CBMMetalIncrementalPartitionSource]
                options:options error:&error];
            if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            auto buffer = [&](const void* data, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = data
                    ? [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Histogram probe allocation failed");
                return result;
            };
            id<MTLBuffer> data = buffer(binData, size_t(rows) * features);
            id<MTLBuffer> g = buffer(gradients, size_t(rows) * sizeof(float));
            id<MTLBuffer> w = buffer(weights, size_t(rows) * sizeof(float));
            id<MTLBuffer> types = buffer(featureTypes, features);
            id<MTLBuffer> rootIndices = buffer(nullptr, size_t(rows) * sizeof(uint32_t));
            id<MTLBuffer> rootOffsets = buffer(nullptr, 2 * sizeof(uint32_t));
            id<MTLBuffer> indices = buffer(childIndices, size_t(rows) * sizeof(uint32_t));
            id<MTLBuffer> offsets = buffer(childOffsets, 3 * sizeof(uint32_t));
            id<MTLBuffer> sums = buffer(nullptr, 2 * cells * sizeof(float));
            id<MTLBuffer> histogramWeights = buffer(nullptr, 2 * cells * sizeof(float));
            id<MTLBuffer> partials = buffer(nullptr, size_t(2) * tiles * 2 * 4 * sizeof(float));
            id<MTLBuffer> leafSums = buffer(nullptr, 2 * sizeof(float));
            id<MTLBuffer> leafWeights = buffer(nullptr, 2 * sizeof(float));
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!queue) throw std::runtime_error("Could not create Metal command queue");
            id<MTLCommandBuffer> command = [queue commandBuffer];
            CBMMetalKernelParams p{};
            p.Rows = rows;
            p.Features = features;
            p.Bins = bins;
            p.Leaves = 1;
            p.HistogramTiles = tiles;
            p.PartitionTiles = tiles;
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                                MTLSize size, bool groups = true) {
                NSError* pipelineError = nil;
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (!function) throw std::runtime_error(std::string("Missing probe kernel: ") + name);
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                    error:&pipelineError];
                if (!pipeline) throw std::runtime_error([[pipelineError localizedDescription] UTF8String]);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> item : buffers) [encoder setBuffer:item offset:0 atIndex:index++];
                [encoder setBytes:&p length:sizeof(p) atIndex:index];
                if (groups) [encoder dispatchThreadgroups:size threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else [encoder dispatchThreads:size threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            auto finish = [&]() {
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error([[command.error localizedDescription] UTF8String]);
                command = [queue commandBuffer];
            };
            auto capture = [&](size_t plane, size_t sourcePlane, size_t count) {
                finish();
                std::memcpy(gradientStages + plane * cells,
                    static_cast<const float*>(sums.contents) + sourcePlane * cells, count * cells * sizeof(float));
                std::memcpy(weightStages + plane * cells,
                    static_cast<const float*>(histogramWeights.contents) + sourcePlane * cells, count * cells * sizeof(float));
            };
            auto statistics = [&](id<MTLBuffer> rowIndex, id<MTLBuffer> partitionOffset, size_t destination) {
                dispatch("ReduceStructurePartials", {g, w, rowIndex, partitionOffset, partials},
                    MTLSizeMake(tiles, p.Leaves, 1));
                dispatch("CollectPartitionStatistics", {partials, leafSums, leafWeights}, MTLSizeMake(p.Leaves, 1, 1));
                finish();
                std::memcpy(partitionSums + destination, leafSums.contents, p.Leaves * sizeof(float));
                std::memcpy(partitionWeights + destination, leafWeights.contents, p.Leaves * sizeof(float));
            };
            dispatch("InitializeRootPartition", {rootIndices, rootOffsets}, MTLSizeMake(rows, 1, 1), false);
            statistics(rootIndices, rootOffsets, 0);
            dispatch("ClearHistograms", {sums, histogramWeights}, MTLSizeMake(cells, 1, 1), false);
            dispatch("ComputeHistograms", {data, g, w, rootIndices, rootOffsets, sums, histogramWeights},
                MTLSizeMake(tiles, features, 1));
            capture(0, 0, 1);
            dispatch("ScanHistograms", {sums, histogramWeights, types}, MTLSizeMake(features, 1, 1));
            capture(1, 0, 1);

            p.Leaves = 2;
            statistics(indices, offsets, 1);
            dispatch("ClearChildHistograms", {sums, histogramWeights}, MTLSizeMake(cells, 1, 1), false);
            dispatch("ComputeSmallerChildHistograms", {data, g, w, indices, offsets, sums, histogramWeights},
                MTLSizeMake(tiles, features, 1));
            capture(2, 1, 1);
            dispatch("ScanChildHistograms", {sums, histogramWeights, types}, MTLSizeMake(features, 1, 1));
            capture(3, 1, 1);
            dispatch("SubtractSiblingHistograms", {sums, histogramWeights, offsets}, MTLSizeMake(cells, 1, 1), false);
            capture(4, 0, 2);

            if (rebuild) {
                dispatch("ClearHistograms", {sums, histogramWeights}, MTLSizeMake(2 * cells, 1, 1), false);
                dispatch("ComputeHistograms", {data, g, w, indices, offsets, sums, histogramWeights},
                    MTLSizeMake(tiles, features, 2));
                capture(6, 0, 2);
                dispatch("ScanHistograms", {sums, histogramWeights, types}, MTLSizeMake(2 * features, 1, 1));
                capture(8, 0, 2);
            } else {
                std::fill(gradientStages + 6 * cells, gradientStages + 10 * cells, 0.0f);
                std::fill(weightStages + 6 * cells, weightStages + 10 * cells, 0.0f);
            }
            return 0;
        } catch (const std::exception& error) {
            if (errorText && capacity) {
                std::strncpy(errorText, error.what(), capacity - 1);
                errorText[capacity - 1] = '\0';
            }
            return 1;
        }
    }
}

// Score the supplied snapshots with the real production kernel. Each dispatch
// has one candidate, preserving its arithmetic while exposing scores that the
// normal winner reduction discards. Feature penalties are float2 [F, 2].
extern "C" int cbm_scalar_histogram_scores(
    uint32_t rows, uint32_t features, uint32_t bins, uint32_t leaves,
    uint32_t candidates, uint32_t scoreFunction, float l2,
    const float* histogramSums, const float* histogramWeights,
    const float* partitionSums, const float* partitionWeights,
    const uint32_t* candidateFeatures, const uint32_t* candidateBins,
    const uint8_t* candidateTypes, const float* featurePenalties,
    float* scores, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!rows || !features || !bins || bins > 256 || !leaves || leaves > 256 ||
                !candidates || candidates > (1u << 20) || scoreFunction > 6 ||
                !std::isfinite(l2) || l2 < 0 || size_t(features) * bins > (1u << 24))
                throw std::runtime_error("Invalid histogram score probe dimensions");
            for (uint32_t i = 0; i < candidates; ++i)
                if (candidateFeatures[i] >= features || candidateBins[i] >= bins || candidateTypes[i] > 1)
                    throw std::runtime_error("Invalid histogram score probe candidate");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:
                [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s",
                    CBMMetalSource, CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
                    CBMMetalHistogramSource, CBMMetalHistogramReuseSource, CBMMetalIncrementalPartitionSource]
                options:options error:&error];
            if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            auto buffer = [&](const void* data, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = data
                    ? [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Histogram score probe allocation failed");
                return result;
            };
            const size_t histogramBytes = size_t(leaves) * features * bins * sizeof(float);
            id<MTLBuffer> g = buffer(histogramSums, histogramBytes);
            id<MTLBuffer> w = buffer(histogramWeights, histogramBytes);
            id<MTLBuffer> pg = buffer(partitionSums, leaves * sizeof(float));
            id<MTLBuffer> pw = buffer(partitionWeights, leaves * sizeof(float));
            id<MTLBuffer> cf = buffer(candidateFeatures, candidates * sizeof(uint32_t));
            id<MTLBuffer> cb = buffer(candidateBins, candidates * sizeof(uint32_t));
            // Metal buffer offsets are four-byte aligned, including uchar data.
            std::vector<uint32_t> paddedTypes(candidates);
            for (uint32_t i = 0; i < candidates; ++i) paddedTypes[i] = candidateTypes[i];
            id<MTLBuffer> ct = buffer(paddedTypes.data(), candidates * sizeof(uint32_t));
            id<MTLBuffer> winners = buffer(nullptr, candidates * sizeof(CBMMetalSplitState));
            std::vector<float> zeroNoise(features, 0);
            std::vector<uint32_t> zeroOffsets(features + 1, 0);
            id<MTLBuffer> noise = buffer(zeroNoise.data(), features * sizeof(float));
            id<MTLBuffer> offsets = buffer(zeroOffsets.data(), (features + 1) * sizeof(uint32_t));
            id<MTLBuffer> penalties = buffer(featurePenalties, size_t(features) * 2 * sizeof(float));
            id<MTLFunction> function = [library newFunctionWithName:@"FindSplitWinners"];
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            CBMMetalKernelParams p{};
            p.Rows = rows; p.Features = features; p.Bins = bins; p.Leaves = leaves;
            p.Candidates = 1; p.ScoreGroups = 1; p.ScoreFunction = scoreFunction; p.L2 = l2;
            for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> item : {g, w, pg, pw, cf, cb, ct, winners, noise, offsets, penalties}) {
                    const size_t offset = index >= 4 && index <= 6 ? candidate * sizeof(uint32_t)
                        : index == 7 ? candidate * sizeof(CBMMetalSplitState) : 0;
                    [encoder setBuffer:item offset:offset atIndex:index++];
                }
                [encoder setBytes:&p length:sizeof(p) atIndex:index];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            }
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            const auto* output = static_cast<const CBMMetalSplitState*>(winners.contents);
            for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
                if (!output[candidate].Valid || output[candidate].InvalidScore)
                    throw std::runtime_error("Nonfinite or invalid split score");
                scores[candidate] = output[candidate].Score;
            }
            return 0;
        } catch (const std::exception& error) {
            if (errorText && capacity) {
                std::strncpy(errorText, error.what(), capacity - 1);
                errorText[capacity - 1] = '\0';
            }
            return 1;
        }
    }
}
