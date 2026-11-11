#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../native/metal_bootstrap_kernels.h"
#include "../native/metal_langevin_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

struct BootstrapProbeParams {
    uint32_t Rows, Type, SeedLow, SeedHigh;
    uint32_t Iteration, Stream, Reserved0, Reserved1;
    float Temperature, Subsample, MvsLambda, NoiseScale;
};
struct LangevinWeakProbeParams {
    BootstrapProbeParams Random;
    uint32_t Offset, Stride, FilterBootstrap, Reserved;
};
static_assert(sizeof(BootstrapProbeParams) == 48);
static_assert(sizeof(LangevinWeakProbeParams) == 64);
static_assert(offsetof(LangevinWeakProbeParams, Offset) == 48);

// Expose raw words to distinguish integer RNG regressions from tolerances in
// GPU logarithm/cosine. Weak Langevin uses the first two MWC draws directly;
// BootstrapNormalForItem's four warm-up draws are not part of this kernel.
static const char* ProbeSource = R"METAL(
kernel void LangevinWeakProbeWords(device uint* words [[buffer(0)]],
    constant LangevinWeakParams& p [[buffer(1)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.random.rows) return;
    ulong seed = BootstrapSeedForItem(row, p.random);
    words[2 * row] = BootstrapNextUint(seed);
    words[2 * row + 1] = BootstrapNextUint(seed);
}
)METAL";

extern "C" int cbm_langevin_weak_probe(const LangevinWeakProbeParams* params,
    uint32_t valueCount, uint32_t launchCount, uint32_t threadgroupSize,
    const float* values, const float* multipliers, float* output, uint32_t* words,
    char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!params || !values || !multipliers || !output || !words)
                throw std::runtime_error("Missing Langevin weak probe input");
            const auto& p = *params;
            if (!p.Random.Rows || p.Random.Rows > (1u << 22) || !valueCount || valueCount > (1u << 24) ||
                !p.Stride || p.Stride > 16 || p.FilterBootstrap > 1 || p.Reserved ||
                uint64_t(p.Offset) + uint64_t(p.Random.Rows - 1) * p.Stride >= valueCount ||
                launchCount < p.Random.Rows || launchCount > (1u << 23) ||
                !threadgroupSize || threadgroupSize > 256 ||
                !std::isfinite(p.Random.NoiseScale) || p.Random.NoiseScale < 0)
                throw std::runtime_error("Invalid Langevin weak probe dimensions or coefficient");

            static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("Metal device is unavailable");
            static id<MTLLibrary> library = nil;
            NSError* error = nil;
            if (!library) {
                MTLCompileOptions* options = [MTLCompileOptions new];
                options.languageVersion = MTLLanguageVersion3_0;
                options.fastMathEnabled = NO;
                NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s",
                    CBMMetalBootstrapSource, CBMMetalLangevinSource, ProbeSource];
                library = [device newLibraryWithSource:source options:options error:&error];
            }
            if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
            auto pipeline = [&](const char* name) {
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
                if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
                return result;
            };
            id<MTLComputePipelineState> noisePipeline = pipeline("AddLangevinWeakNoise");
            id<MTLComputePipelineState> wordsPipeline = pipeline("LangevinWeakProbeWords");
            auto buffer = [&](const void* data, size_t bytes) {
                id<MTLBuffer> result = data ?
                    [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared] :
                    [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Metal Langevin probe allocation failed");
                return result;
            };
            id<MTLBuffer> valueBuffer = buffer(values, uint64_t(valueCount) * sizeof(float));
            id<MTLBuffer> multiplierBuffer = buffer(multipliers, uint64_t(p.Random.Rows) * sizeof(float));
            id<MTLBuffer> wordsBuffer = buffer(nullptr, uint64_t(p.Random.Rows) * 2 * sizeof(uint32_t));
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:noisePipeline];
            [encoder setBuffer:valueBuffer offset:0 atIndex:0];
            [encoder setBuffer:multiplierBuffer offset:0 atIndex:1];
            [encoder setBytes:&p length:sizeof(p) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(launchCount, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
            [encoder endEncoding];
            encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:wordsPipeline];
            [encoder setBuffer:wordsBuffer offset:0 atIndex:0];
            [encoder setBytes:&p length:sizeof(p) atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(launchCount, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error(command.error.localizedDescription.UTF8String);
            std::memcpy(output, valueBuffer.contents, uint64_t(valueCount) * sizeof(float));
            std::memcpy(words, wordsBuffer.contents, uint64_t(p.Random.Rows) * 2 * sizeof(uint32_t));
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
