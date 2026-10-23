#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_query_cross_entropy_kernels.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

struct QCEParams { uint32_t Rows, Groups, Leaves; float Alpha; };
static_assert(sizeof(QCEParams) == 16);

extern "C" int cbm_qce_probe(const QCEParams* config, const float* targets,
    const float* weights, const float* point, const uint32_t* offsets,
    const float* scales, const uint32_t* leafIds, float* rowStats, float* groupStats,
    uint32_t* singleClass, float* gradient, float* hessian, char* message, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!config || !targets || !weights || !point || !offsets || !scales || !leafIds ||
                !rowStats || !groupStats || !singleClass || !gradient || !hessian)
                throw std::runtime_error("Null QueryCrossEntropy probe buffer");
            const auto& p = *config;
            if (!p.Rows || p.Rows > (1u << 24) || !p.Groups || p.Groups > p.Rows ||
                !p.Leaves || p.Leaves > 64 || !std::isfinite(p.Alpha))
                throw std::runtime_error("Invalid QueryCrossEntropy dimensions or alpha");
            if (uint64_t(p.Groups) * p.Leaves * 16 > (1ull << 29))
                throw std::runtime_error("QueryCrossEntropy probe projection workspace exceeds 512 MiB");
            if (offsets[0] || offsets[p.Groups] != p.Rows)
                throw std::runtime_error("Query offsets must cover every row");
            for (uint32_t q = 0; q < p.Groups; ++q) {
                if (offsets[q] >= offsets[q + 1] || offsets[q + 1] - offsets[q] > 256 ||
                    !std::isfinite(scales[q]))
                    throw std::runtime_error("QueryCrossEntropy requires 1 to 256 rows per group and finite scales");
                for (uint32_t i = offsets[q]; i < offsets[q + 1]; ++i) {
                    if (!std::isfinite(targets[i]) || targets[i] < 0 || targets[i] > 1 ||
                        !std::isfinite(weights[i]) || weights[i] < 0 || !std::isfinite(point[i]) ||
                        !std::isfinite(point[i] * scales[q]) || leafIds[i] >= p.Leaves)
                        throw std::runtime_error("Invalid QueryCrossEntropy observation");
                }
            }
            static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            static id<MTLLibrary> library = [&]() {
                NSError* error = nil;
                MTLCompileOptions* options = [MTLCompileOptions new];
                options.languageVersion = MTLLanguageVersion3_0;
                options.fastMathEnabled = NO;
                id<MTLLibrary> result = [device newLibraryWithSource:
                    [NSString stringWithUTF8String:CBMMetalQueryCrossEntropySource]
                    options:options error:&error];
                if (!result) throw std::runtime_error([[error localizedDescription] UTF8String]);
                return result;
            }();
            auto buffer = [&](const void* source, size_t size) -> id<MTLBuffer> {
                id<MTLBuffer> result = source
                    ? [device newBufferWithBytes:source length:size options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:size options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("QueryCrossEntropy buffer allocation failed");
                return result;
            };
            const size_t rows = size_t(p.Rows) * 4, groups = size_t(p.Groups) * 4;
            id<MTLBuffer> y = buffer(targets, rows), w = buffer(weights, rows), x = buffer(point, rows);
            id<MTLBuffer> off = buffer(offsets, groups + 4), scale = buffer(scales, groups);
            id<MTLBuffer> ids = buffer(leafIds, rows), rs = buffer(nullptr, rows * 4);
            id<MTLBuffer> qs = buffer(nullptr, groups * 4), single = buffer(nullptr, groups);
            id<MTLBuffer> g = buffer(nullptr, size_t(p.Leaves) * 4);
            id<MTLBuffer> h = buffer(nullptr, size_t(p.Leaves) * p.Leaves * 4);
            id<MTLBuffer> cache = buffer(nullptr, size_t(p.Groups) * p.Leaves * 16);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (!command) throw std::runtime_error("Could not allocate Metal command buffer");
            auto dispatch = [&](NSString* name, std::initializer_list<id<MTLBuffer>> buffers, uint32_t mode) {
                NSError* error = nil;
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
                    [library newFunctionWithName:name] error:&error];
                if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!encoder) throw std::runtime_error("Could not create Metal encoder");
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> item : buffers) [encoder setBuffer:item offset:0 atIndex:index++];
                [encoder setBytes:&p length:sizeof(p) atIndex:index];
                if (mode == 2) [encoder dispatchThreadgroups:MTLSizeMake(p.Leaves, p.Groups, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else if (mode == 1) [encoder dispatchThreadgroups:MTLSizeMake(p.Groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else [encoder dispatchThreads:MTLSizeMake(p.Leaves * p.Leaves, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            dispatch(@"QueryCrossEntropyStatistics", {y, w, x, off, scale, rs, qs, single}, 1);
            dispatch(@"CacheQueryCrossEntropyLeafSums", {rs, off, ids, cache}, 2);
            dispatch(@"SumQueryCrossEntropyLeafMatrix", {cache, qs, g, h}, 0);
            [command commit]; [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(rowStats, rs.contents, rows * 4);
            std::memcpy(groupStats, qs.contents, groups * 4);
            std::memcpy(singleClass, single.contents, groups);
            std::memcpy(gradient, g.contents, size_t(p.Leaves) * 4);
            std::memcpy(hessian, h.contents, size_t(p.Leaves) * p.Leaves * 4);
            return 0;
        } catch (const std::exception& error) {
            if (message && capacity) {
                std::strncpy(message, error.what(), capacity - 1);
                message[capacity - 1] = '\0';
            }
            return 1;
        }
    }
}
