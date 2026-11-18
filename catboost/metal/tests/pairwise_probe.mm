#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_pairwise_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>

struct PairwiseParams {
    uint32_t Rows, Pairs, Objective, ApplyLeafValues;
    uint32_t Leaves, Reserved0, Reserved1, Reserved2;
};
static_assert(sizeof(PairwiseParams) == 32);

extern "C" int cbm_pairwise_probe(const PairwiseParams* params, uint32_t groups,
    const float* cursor, const uint32_t* winners, const uint32_t* losers, const float* weights,
    const uint32_t* leafIds, const float* leafValues, float* point, float* edges,
    float* gradients, float* curvature, float* incidentWeights, float* partials,
    char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            const auto& p = *params;
            if (!p.Rows || !p.Leaves || !groups || p.Rows > (1u << 22) || p.Pairs > (1u << 24) ||
                groups > 65536 || p.Objective != 14) throw std::runtime_error("Invalid pairwise dimensions");
            std::vector<uint32_t> offsets(p.Rows + 1), incidence(p.Pairs * 2);
            std::vector<int32_t> signs(p.Pairs * 2);
            for (uint32_t edge = 0; edge < p.Pairs; ++edge) {
                if (winners[edge] >= p.Rows || losers[edge] >= p.Rows || winners[edge] == losers[edge])
                    throw std::runtime_error("Invalid pair endpoints");
                if (!std::isfinite(weights[edge]) || weights[edge] < 0)
                    throw std::runtime_error("Invalid pair weight");
                ++offsets[winners[edge] + 1]; ++offsets[losers[edge] + 1];
            }
            for (uint32_t row = 0; row < p.Rows; ++row) {
                if (!std::isfinite(cursor[row]) || (p.ApplyLeafValues && leafIds[row] >= p.Leaves))
                    throw std::runtime_error("Invalid pairwise point or leaf id");
                offsets[row + 1] += offsets[row];
            }
            auto next = offsets;
            for (uint32_t edge = 0; edge < p.Pairs; ++edge) {
                const uint32_t first = next[winners[edge]]++, second = next[losers[edge]]++;
                incidence[first] = incidence[second] = edge; signs[first] = 1; signs[second] = -1;
            }
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0; options.fastMathEnabled = NO;
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:CBMMetalPairwiseSource]
                options:options error:&error];
            if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            auto buffer = [&](const void* source, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = source && bytes
                    ? [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:std::max<size_t>(bytes, 4) options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Pairwise probe allocation failed");
                return result;
            };
            const size_t rowBytes = p.Rows * 4, pairBytes = p.Pairs * 4;
            id<MTLBuffer> cursorBuffer = buffer(cursor, rowBytes), winnerBuffer = buffer(winners, pairBytes);
            id<MTLBuffer> loserBuffer = buffer(losers, pairBytes), weightBuffer = buffer(weights, pairBytes);
            id<MTLBuffer> leafIdBuffer = buffer(leafIds, rowBytes), leafBuffer = buffer(leafValues, p.Leaves * 4);
            id<MTLBuffer> offsetBuffer = buffer(offsets.data(), (p.Rows + 1) * 4);
            id<MTLBuffer> incidentBuffer = buffer(incidence.data(), pairBytes * 2), signBuffer = buffer(signs.data(), pairBytes * 2);
            id<MTLBuffer> pointBuffer = buffer(nullptr, rowBytes), edgeBuffer = buffer(nullptr, pairBytes * 4);
            id<MTLBuffer> gradientBuffer = buffer(nullptr, rowBytes), curvatureBuffer = buffer(nullptr, rowBytes);
            id<MTLBuffer> incidentWeightBuffer = buffer(nullptr, rowBytes), partialBuffer = buffer(nullptr, groups * 8);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers, uint32_t count, bool grouped) {
                if (!count) return;
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
            dispatch("PreparePairwisePoint", {cursorBuffer, leafBuffer, leafIdBuffer, pointBuffer}, p.Rows, false);
            dispatch("PairLogitEdgeDerivatives", {pointBuffer, winnerBuffer, loserBuffer, weightBuffer, edgeBuffer}, p.Pairs, false);
            dispatch("ReducePairwiseRows", {offsetBuffer, incidentBuffer, signBuffer, edgeBuffer, gradientBuffer,
                curvatureBuffer, incidentWeightBuffer}, p.Rows, true);
            dispatch("ReducePairwiseObjective", {edgeBuffer, partialBuffer}, groups, true);
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(point, pointBuffer.contents, rowBytes); std::memcpy(edges, edgeBuffer.contents, pairBytes * 4);
            std::memcpy(gradients, gradientBuffer.contents, rowBytes); std::memcpy(curvature, curvatureBuffer.contents, rowBytes);
            std::memcpy(incidentWeights, incidentWeightBuffer.contents, rowBytes); std::memcpy(partials, partialBuffer.contents, groups * 8);
            return 0;
        } catch (const std::exception& error) {
            if (errorText && capacity) { std::strncpy(errorText, error.what(), capacity - 1); errorText[capacity - 1] = '\0'; }
            return 1;
        }
    }
}
