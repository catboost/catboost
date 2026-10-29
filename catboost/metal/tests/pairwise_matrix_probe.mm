#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_pairwise_matrix_kernels.h"
#include "../native/metal_leaf_matrix_kernels.h"
#include "../native/metal_sort.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

struct PairwiseMatrixParams {
    uint32_t Rows, Pairs, Leaves, LeafMethod, Reserved0, Reserved1, Reserved2, Reserved3;
};
struct LeafMatrixParams {
    uint32_t Leaves, HasDiagonalPart, Reserved0, Reserved1;
    float L2, NonDiagL2, MinLeafWeight, Step;
};
static_assert(sizeof(PairwiseMatrixParams) == 32 && sizeof(LeafMatrixParams) == 32);

extern "C" int cbm_pairwise_matrix_probe(const PairwiseMatrixParams* params, float l2, float nonDiagL2,
    const float* point, const uint32_t* winners, const uint32_t* losers, const float* weights,
    const uint32_t* leafIds, float* edges, uint32_t* keys, uint32_t* indices, uint32_t* offsets,
    float* gradient, float* hessian, float* directions, uint32_t* status, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!params) throw std::runtime_error("Missing pairwise matrix parameters");
            const auto& p = *params;
            if (!p.Rows || p.Rows > (1u << 22) || p.Pairs > (1u << 24) || !p.Leaves || p.Leaves > 256
                    || p.LeafMethod > 1 || p.Reserved0 || p.Reserved1 || p.Reserved2 || p.Reserved3
                    || !std::isfinite(l2) || l2 < 0 || !std::isfinite(nonDiagL2) || nonDiagL2 < 0)
                throw std::runtime_error("Invalid pairwise matrix parameters");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0; options.fastMathEnabled = NO;
            NSError* error = nil;
            NSString* source = [NSString stringWithFormat:@"%s\n%s", CBMMetalPairwiseMatrixSource, CBMMetalLeafMatrixSource];
            id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
            if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            auto buffer = [&](const void* data, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = data && bytes
                    ? [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:std::max<size_t>(4, bytes) options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Pairwise matrix probe allocation failed");
                if (!data || !bytes) std::memset(result.contents, 0, std::max<size_t>(4, bytes));
                return result;
            };
            const size_t rb = 4ull * p.Rows, eb = 4ull * p.Pairs, lb = 4ull * p.Leaves;
            const uint32_t cells = p.Leaves * p.Leaves;
            id<MTLBuffer> raw = buffer(point, rb), win = buffer(winners, eb), lose = buffer(losers, eb);
            id<MTLBuffer> weight = buffer(weights, eb), leaves = buffer(leafIds, rb), edge = buffer(nullptr, eb * 4);
            id<MTLBuffer> keyA = buffer(nullptr, eb), keyB = buffer(nullptr, eb), indexA = buffer(nullptr, eb), indexB = buffer(nullptr, eb);
            id<MTLBuffer> off = buffer(nullptr, 4ull * (cells + 1)), stats = buffer(nullptr, 32ull * cells);
            id<MTLBuffer> g = buffer(nullptr, lb), h = buffer(nullptr, 4ull * cells), factor = buffer(nullptr, 8ull * cells);
            id<MTLBuffer> d = buffer(nullptr, lb), flag = buffer(nullptr, 4);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                                const auto& parameters, uint32_t count, bool grouped) {
                if (!count) return;
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> value : buffers) [encoder setBuffer:value offset:0 atIndex:index++];
                [encoder setBytes:&parameters length:sizeof(parameters) atIndex:index];
                if (grouped) [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            dispatch("ComputePairwiseMatrixEdges", {raw, win, lose, weight, edge, flag}, p, p.Pairs, false);
            dispatch("BuildPairwiseLeafKeys", {leaves, win, lose, keyA, indexA, flag}, p, p.Pairs, false);
            CBMEncodeSortU32(command, keyA, indexA, p.Pairs, keyB, indexB, nullptr);
            dispatch("BuildPairwiseCellOffsets", {keyB, off}, p, cells + 1, false);
            dispatch("ReducePairwiseLeafCells", {off, indexB, edge, stats}, p, cells, true);
            dispatch("AssemblePairwiseLeafMatrix", {stats, g, h, flag}, p, cells, false);
            const LeafMatrixParams m = {p.Leaves, 0, 0, 0, l2, nonDiagL2, 1e-20f, 1};
            dispatch("RegularizeLeafMatrix", {h, factor}, m, cells, false);
            dispatch("SolveLeafMatrix", {factor, g, d, flag}, m, 1, true);
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(edges, edge.contents, eb * 4); std::memcpy(keys, keyB.contents, eb);
            std::memcpy(indices, indexB.contents, eb); std::memcpy(offsets, off.contents, 4ull * (cells + 1));
            std::memcpy(gradient, g.contents, lb); std::memcpy(hessian, h.contents, 4ull * cells);
            std::memcpy(directions, d.contents, lb); std::memcpy(status, flag.contents, 4);
            return 0;
        } catch (const std::exception& error) {
            if (errorText && capacity) { std::strncpy(errorText, error.what(), capacity - 1); errorText[capacity - 1] = '\0'; }
            return 1;
        }
    }
}
