#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_yeti_rank_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>

struct YetiRankParams {
    uint32_t Rows, Groups, Tasks, Permutations, SeedLow, SeedHigh;
    float Decay;
    uint32_t CenterRows;
};
static_assert(sizeof(YetiRankParams) == 32);

extern "C" int cbm_yeti_rank_probe(const YetiRankParams* config, const float* targets,
    const float* weights, const float* point, const uint32_t* offsets,
    float* prepared, float* derivatives, char* message, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!config || !targets || !weights || !point || !offsets || !prepared || !derivatives)
                throw std::runtime_error("Null YetiRank probe buffer");
            YetiRankParams p = *config;
            if (!p.Rows || p.Rows > (1u << 24) || !p.Groups || p.Groups > p.Rows ||
                !p.Permutations || p.Permutations > 10000 || p.CenterRows > p.Rows ||
                !std::isfinite(p.Decay) || p.Decay < 0 || p.Decay > 1 ||
                offsets[0] || offsets[p.Groups] != p.Rows)
                throw std::runtime_error("Invalid YetiRank dimensions, permutations, decay or offsets");
            std::vector<uint32_t> queryIds(p.Rows), tasks;
            for (uint32_t q = 0; q < p.Groups; ++q) {
                if (offsets[q] >= offsets[q + 1] || offsets[q + 1] - offsets[q] > 1023)
                    throw std::runtime_error("YetiRank requires 1 to 1023 rows per query");
                for (uint32_t row = offsets[q]; row < offsets[q + 1]; ++row) {
                    if (!std::isfinite(targets[row]) || !std::isfinite(weights[row]) || weights[row] < 0 ||
                        !std::isfinite(point[row]) || !std::isfinite(targets[row] * weights[row]))
                        throw std::runtime_error("Invalid YetiRank observation");
                    queryIds[row] = q;
                }
            }
            for (uint32_t q = 0; q < p.Groups;) {
                const uint32_t limit = std::min<uint32_t>(p.Rows, offsets[q] + 1024);
                const uint32_t next = limit == p.Rows ? p.Groups : queryIds[limit];
                if (next <= q) throw std::runtime_error("Invalid YetiRank task packing");
                tasks.push_back(q); tasks.push_back(next); q = next;
            }
            p.Tasks = tasks.size() / 2;
            static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            static id<MTLLibrary> library = [&]() {
                NSError* error = nil;
                MTLCompileOptions* options = [MTLCompileOptions new];
                options.languageVersion = MTLLanguageVersion3_0; options.fastMathEnabled = NO;
                id<MTLLibrary> value = [device newLibraryWithSource:
                    [NSString stringWithUTF8String:CBMMetalYetiRankSource] options:options error:&error];
                if (!value) throw std::runtime_error([[error localizedDescription] UTF8String]);
                return value;
            }();
            auto buffer = [&](const void* source, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> value = source ? [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!value) throw std::runtime_error("YetiRank allocation failed");
                return value;
            };
            const size_t bytes = size_t(p.Rows) * 4;
            auto y = buffer(targets, bytes), w = buffer(weights, bytes), x = buffer(point, bytes);
            auto off = buffer(offsets, size_t(p.Groups + 1) * 4), qid = buffer(queryIds.data(), bytes);
            auto task = buffer(tasks.data(), tasks.size() * 4), prep = buffer(nullptr, bytes * 2);
            auto deriv = buffer(nullptr, bytes * 2);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](NSString* name, std::initializer_list<id<MTLBuffer>> buffers, uint32_t count) {
                NSError* error = nil;
                auto pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
                if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
                auto encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (auto value : buffers) [encoder setBuffer:value offset:0 atIndex:index++];
                [encoder setBytes:&p length:sizeof(p) atIndex:index];
                [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            dispatch(@"PrepareYetiRankApprox", {x, off, prep}, p.Groups);
            dispatch(@"YetiRankPointwise", {prep, y, w, qid, off, task, deriv}, p.Tasks);
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(prepared, prep.contents, bytes * 2);
            std::memcpy(derivatives, deriv.contents, bytes * 2);
            return 0;
        } catch (const std::exception& error) {
            if (message && capacity) { std::strncpy(message, error.what(), capacity - 1); message[capacity - 1] = '\0'; }
            return 1;
        }
    }
}
