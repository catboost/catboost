#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_leaf_matrix_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

struct LeafMatrixParams {
    uint32_t Leaves, HasDiagonalPart, Reserved0, Reserved1;
    float L2, NonDiagL2, MinLeafWeight, Step;
};
static_assert(sizeof(LeafMatrixParams) == 32);

extern "C" int cbm_leaf_matrix_probe(const LeafMatrixParams* params,
    const float* hessian, const float* gradient, const float* weights, const float* point,
    float* regularized, float* direction, float* updated, float* dot, uint32_t* status,
    char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!params) throw std::runtime_error("Missing matrix parameters");
            const auto& p = *params;
            if (!p.Leaves || p.Leaves > 256 || p.HasDiagonalPart > 1 || p.Reserved0 || p.Reserved1
                    || !std::isfinite(p.L2) || p.L2 < 0 || !std::isfinite(p.NonDiagL2) || p.NonDiagL2 < 0
                    || !std::isfinite(p.MinLeafWeight) || p.MinLeafWeight < 0 || !std::isfinite(p.Step))
                throw std::runtime_error("Invalid leaf matrix parameters");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0; options.fastMathEnabled = NO;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:CBMMetalLeafMatrixSource]
                options:options error:&error];
            if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            auto buffer = [&](const void* source, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = source
                    ? [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Matrix probe allocation failed");
                if (!source) std::memset(result.contents, 0, bytes);
                return result;
            };
            const size_t bytes = p.Leaves * 4, matrixBytes = p.Leaves * bytes;
            id<MTLBuffer> h = buffer(hessian, matrixBytes), g = buffer(gradient, bytes);
            id<MTLBuffer> w = buffer(weights, bytes), initial = buffer(point, bytes);
            id<MTLBuffer> work = buffer(nullptr, matrixBytes * 2), preserved = buffer(nullptr, matrixBytes * 2);
            id<MTLBuffer> d = buffer(nullptr, bytes), result = buffer(nullptr, bytes);
            id<MTLBuffer> product = buffer(nullptr, 8), flag = buffer(nullptr, 4);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers, uint32_t count, bool grouped) {
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> value : buffers) [encoder setBuffer:value offset:0 atIndex:index++];
                [encoder setBytes:&p length:sizeof(p) atIndex:index];
                if (grouped) [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            dispatch("RegularizeLeafMatrix", {h, work}, p.Leaves * p.Leaves, false);
            id<MTLBlitCommandEncoder> copy = [command blitCommandEncoder];
            [copy copyFromBuffer:work sourceOffset:0 toBuffer:preserved destinationOffset:0 size:matrixBytes * 2];
            [copy endEncoding];
            dispatch("SolveLeafMatrix", {work, g, d, flag}, 1, true);
            dispatch("UpdateLeafMatrixPoint", {initial, d, w, result, flag}, p.Leaves, false);
            dispatch("ReduceLeafMatrixDirectionalDot", {g, d, product}, 1, true);
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(regularized, preserved.contents, matrixBytes * 2);
            std::memcpy(direction, d.contents, bytes); std::memcpy(updated, result.contents, bytes);
            std::memcpy(dot, product.contents, 8); std::memcpy(status, flag.contents, 4);
            return 0;
        } catch (const std::exception& error) {
            if (errorText && capacity) { std::strncpy(errorText, error.what(), capacity - 1); errorText[capacity - 1] = '\0'; }
            return 1;
        }
    }
}
