#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_greedy_bootstrap_kernels.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    id<MTLComputePipelineState> Count, Prefix, Noise;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device != nil && Device.hasUnifiedMemory, "Apple Silicon Metal required");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        options.fastMathEnabled = NO;
        NSError* error = nil;
        auto library = [Device newLibraryWithSource:[NSString stringWithUTF8String:CBMMetalGreedyBootstrapSource]
            options:options error:&error];
        if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
        Count = [Device newComputePipelineStateWithFunction:[library newFunctionWithName:@"CountGreedyBootstrapRows"] error:&error];
        Prefix = [Device newComputePipelineStateWithFunction:[library newFunctionWithName:@"PrefixGreedyBootstrapOffsets"] error:&error];
        Noise = [Device newComputePipelineStateWithFunction:[library newFunctionWithName:@"ReduceGreedyScoreNoiseStatistics"] error:&error];
        Require(Count && Prefix && Noise && Noise.maxTotalThreadsPerThreadgroup >= 256 && Count.maxTotalThreadsPerThreadgroup >= 256 && Prefix.maxTotalThreadsPerThreadgroup >= 256,
            "Bootstrap probe requires 256-thread compute pipelines");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Could not create Metal command queue");
    }
    id<MTLBuffer> Buffer(const void* data, uint64_t bytes) {
        Require(bytes && bytes <= (1ull << 30), "Invalid probe buffer size");
        id<MTLBuffer> buffer = data ? [Device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Require(buffer != nil, "Could not allocate bootstrap probe buffer");
        return buffer;
    }
};
Runtime& GetRuntime() { static Runtime runtime; return runtime; }
}

extern "C" int cbm_greedy_noise_statistics_probe(const CBMGreedyBootstrapParams* params,
    const float* gradient, const float* weight, uint32_t groups,
    float* result, char* errorText, uint32_t errorCapacity) {
    @autoreleasepool {
        try {
            Require(params && gradient && weight && result, "Null noise probe argument");
            const auto& p = *params;
            Require(p.Rows && p.Rows <= (1u << 24) && p.Leaves && p.Leaves <= 65536, "Invalid row or leaf count");
            Require(p.BootstrapType <= 3 && !p.Reserved, "Invalid bootstrap type or reserved field");
            Require(groups && groups <= 65536, "Invalid reduction group count");
            for (uint32_t row = 0; row < p.Rows; ++row) {
                Require(std::isfinite(gradient[row]), "Gradients must be finite");
                Require(std::isfinite(weight[row]) && weight[row] >= 0, "Weights must be finite and nonnegative");
            }
            Runtime& runtime = GetRuntime();
            auto gradients = runtime.Buffer(gradient, 4ull * p.Rows);
            auto weights = runtime.Buffer(weight, 4ull * p.Rows);
            auto output = runtime.Buffer(nullptr, 16ull * groups);
            auto command = [runtime.Queue commandBuffer];
            Require(command != nil, "Could not create noise command buffer");
            auto encoder = [command computeCommandEncoder];
            Require(encoder != nil, "Could not create noise encoder");
            [encoder setComputePipelineState:runtime.Noise];
            [encoder setBuffer:gradients offset:0 atIndex:0]; [encoder setBuffer:weights offset:0 atIndex:1];
            [encoder setBuffer:output offset:0 atIndex:2]; [encoder setBytes:&p length:sizeof(p) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted, "Noise statistics GPU command failed");
            std::memcpy(result, output.contents, 16ull * groups);
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

extern "C" int cbm_greedy_bootstrap_counts_probe(const CBMGreedyBootstrapParams* params,
    const float* multipliers, const uint32_t* rowIndices, const uint32_t* offsets,
    uint32_t* result, char* errorText, uint32_t errorCapacity) {
    @autoreleasepool {
        try {
            Require(params && multipliers && rowIndices && offsets && result, "Null bootstrap probe argument");
            const auto& p = *params;
            Require(p.Rows && p.Rows <= (1u << 24) && p.Leaves && p.Leaves <= 65536, "Invalid row or leaf count");
            Require(p.BootstrapType <= 3 && !p.Reserved, "Invalid bootstrap type or reserved field");
            Require(offsets[0] == 0 && offsets[p.Leaves] == p.Rows, "Offsets must cover all rows");
            for (uint32_t leaf = 0; leaf < p.Leaves; ++leaf)
                Require(offsets[leaf] <= offsets[leaf + 1] && offsets[leaf + 1] <= p.Rows, "Offsets must be nondecreasing");
            std::vector<uint8_t> seen(p.Rows, 0);
            for (uint32_t row = 0; row < p.Rows; ++row) {
                Require(std::isfinite(multipliers[row]) && multipliers[row] >= 0, "Multipliers must be finite and nonnegative");
                Require(rowIndices[row] < p.Rows && !seen[rowIndices[row]], "Row indices must be a permutation");
                seen[rowIndices[row]] = 1;
            }
            Runtime& runtime = GetRuntime();
            auto m = runtime.Buffer(multipliers, 4ull * p.Rows);
            auto rows = runtime.Buffer(rowIndices, 4ull * p.Rows);
            auto parts = runtime.Buffer(offsets, 4ull * (p.Leaves + 1));
            auto output = runtime.Buffer(nullptr, 4ull * (p.Leaves + 1));
            auto command = [runtime.Queue commandBuffer];
            Require(command != nil, "Could not create Metal command buffer");
            auto count = [command computeCommandEncoder];
            Require(count != nil, "Could not create count encoder");
            [count setComputePipelineState:runtime.Count];
            [count setBuffer:m offset:0 atIndex:0]; [count setBuffer:rows offset:0 atIndex:1];
            [count setBuffer:parts offset:0 atIndex:2]; [count setBuffer:output offset:0 atIndex:3];
            [count setBytes:&p length:sizeof(p) atIndex:4];
            [count dispatchThreadgroups:MTLSizeMake(p.Leaves, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [count endEncoding];
            auto prefix = [command computeCommandEncoder];
            Require(prefix != nil, "Could not create prefix encoder");
            [prefix setComputePipelineState:runtime.Prefix];
            [prefix setBuffer:output offset:0 atIndex:0]; [prefix setBytes:&p length:sizeof(p) atIndex:1];
            [prefix dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [prefix endEncoding];
            [command commit]; [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted, "Bootstrap count GPU command failed");
            std::memcpy(result, output.contents, 4ull * (p.Leaves + 1));
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
