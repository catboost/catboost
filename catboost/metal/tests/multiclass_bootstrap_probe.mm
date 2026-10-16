#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_bootstrap_kernels.h"
#include "../native/metal_multiclass_bootstrap.h"
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

struct MulticlassBootstrapParams { uint32_t Rows, Dimensions, MultiLogit, Reserved; };
static_assert(sizeof(MulticlassBootstrapParams) == 16);

extern "C" int cbm_multiclass_bootstrap_probe(const MulticlassBootstrapParams* params,
    uint32_t groups, const float* gradients, const float* weights, const float* multipliers,
    float* sampledGradients, float* sampledWeights, float* partials, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            const auto& p = *params;
            if (!p.Rows || !p.Dimensions || !groups || p.Dimensions > 1024 ||
                uint64_t(p.Rows) * p.Dimensions > (1u << 26) || groups > 65536)
                throw std::runtime_error("Invalid multiclass bootstrap dimensions");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithFormat:@"%s\n%s",
                CBMMetalBootstrapSource, CBMMetalMulticlassBootstrapSource] options:options error:&error];
            if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            auto buffer = [&](const void* source, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = source
                    ? [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Metal multiclass probe allocation failed");
                return result;
            };
            const size_t rowBytes = p.Rows * 4, gradientBytes = rowBytes * p.Dimensions;
            id<MTLBuffer> gradientBuffer = buffer(gradients, gradientBytes), weightBuffer = buffer(weights, rowBytes);
            id<MTLBuffer> multiplierBuffer = buffer(multipliers, rowBytes), sampledWeightBuffer = buffer(nullptr, rowBytes);
            id<MTLBuffer> partialBuffer = buffer(nullptr, groups * 8);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers, bool grouped) {
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> entry : buffers) [encoder setBuffer:entry offset:0 atIndex:index++];
                [encoder setBytes:&p length:sizeof(p) atIndex:index];
                if (grouped) [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else [encoder dispatchThreads:MTLSizeMake(p.Rows * p.Dimensions, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            dispatch("ApplyMulticlassBootstrap", {gradientBuffer, weightBuffer, multiplierBuffer, sampledWeightBuffer}, false);
            dispatch("ReduceMulticlassScoreStatistics", {gradientBuffer, sampledWeightBuffer, partialBuffer}, true);
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(sampledGradients, gradientBuffer.contents, gradientBytes);
            std::memcpy(sampledWeights, sampledWeightBuffer.contents, rowBytes);
            std::memcpy(partials, partialBuffer.contents, groups * 8);
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
