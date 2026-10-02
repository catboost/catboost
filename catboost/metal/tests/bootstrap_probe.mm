#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../native/metal_bootstrap_kernels.h"
#include "../native/metal_score_noise_kernels.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>

struct ProbeParams {
    uint32_t Rows, Type, SeedLow, SeedHigh;
    uint32_t Iteration, Stream, Reserved0, Reserved1;
    float Temperature, Subsample, MvsLambda, NoiseScale;
};
static_assert(sizeof(ProbeParams) == 48);

static const char* ProbeSource = R"METAL(
kernel void BootstrapProbeRandom(device float* uniforms [[buffer(0)]],
                                 device uint* words [[buffer(1)]],
                                 constant BootstrapParams& p [[buffer(2)]],
                                 uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    ulong seed = BootstrapSeedForItem(row, p);
    words[row] = BootstrapNextUint(seed);
    seed = BootstrapSeedForItem(row, p);
    uniforms[row] = BootstrapUniform(seed);
}
)METAL";

extern "C" int cbm_bootstrap_probe(const ProbeParams* params, const float* derivatives,
    const float* weights, float* output, float* scaledDerivatives, float* structureWeights,
    float* thresholds, uint32_t* words, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            const ProbeParams& p = *params;
            if (!p.Rows || p.Rows > (1u << 22)) throw std::runtime_error("Invalid probe row count");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:
                [NSString stringWithFormat:@"%s\n%s\n%s", CBMMetalBootstrapSource, CBMMetalScoreNoiseSource, ProbeSource]
                options:options error:&error];
            if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            auto makeBuffer = [&](const void* source, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = source
                    ? [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Metal probe allocation failed");
                return result;
            };
            const size_t bytes = p.Rows * sizeof(float);
            const uint32_t tiles = (p.Rows + 8191) / 8192;
            const uint32_t statisticsGroups = std::min<uint32_t>((p.Rows + 255) / 256, 4096);
            id<MTLBuffer> source = makeBuffer(derivatives, bytes);
            id<MTLBuffer> weightBuffer = makeBuffer(weights, bytes);
            id<MTLBuffer> result = makeBuffer(nullptr, std::max(bytes, size_t(8) * statisticsGroups));
            id<MTLBuffer> structure = makeBuffer(nullptr, bytes);
            id<MTLBuffer> thresholdBuffer = makeBuffer(nullptr, tiles * sizeof(float));
            id<MTLBuffer> wordBuffer = makeBuffer(nullptr, bytes);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                                uint32_t count, bool groups = false) {
                NSError* pipelineError = nil;
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                    error:&pipelineError];
                if (!pipeline) throw std::runtime_error([[pipelineError localizedDescription] UTF8String]);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
                [encoder setBytes:&p length:sizeof(p) atIndex:index];
                if (groups) [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            if (p.Type == 4) {
                dispatch("ComputeMvsThresholds", {source, thresholdBuffer}, tiles, true);
                dispatch("GenerateMvsBootstrapWeights", {result, source, thresholdBuffer}, p.Rows);
            } else if (p.Type == 5) {
                dispatch("BootstrapProbeRandom", {result, wordBuffer}, p.Rows);
            } else if (p.Type == 6) {
                dispatch("GenerateBootstrapNormals", {result}, p.Rows);
            } else if (p.Type == 7) {
                dispatch("ReduceBootstrapStatistics", {source, result}, statisticsGroups, true);
            } else if (p.Type == 8) {
                dispatch("ReduceScoreNoiseStatistics", {source, weightBuffer, result}, statisticsGroups, true);
            } else if (p.Type == 9) {
                dispatch("GenerateScoreFeatureNoise", {result}, p.Rows);
            } else {
                dispatch("GenerateBootstrapWeights", {result, source}, p.Rows);
            }
            if (p.Type <= 4) dispatch("ApplyBootstrapWeights", {source, weightBuffer, result, structure}, p.Rows);
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(output, result.contents, bytes);
            if (p.Type <= 4) {
                std::memcpy(scaledDerivatives, source.contents, bytes);
                std::memcpy(structureWeights, structure.contents, bytes);
            }
            if (p.Type == 4) std::memcpy(thresholds, thresholdBuffer.contents, tiles * sizeof(float));
            if (p.Type == 5) std::memcpy(words, wordBuffer.contents, bytes);
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
