#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_pairwise_matrix_kernels.h"
#include "../native/metal_pairwise_candidate_kernels.h"
#include "../native/metal_leaf_matrix_kernels.h"
#include "../native/metal_pairwise_score_kernels.h"
#include "../native/metal_sort.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

struct CandidateParams {
    uint32_t Rows, Pairs, ParentLeaves, Candidates, Features, FirstCandidate, LeafMethod, Reserved;
};
struct MatrixParams { uint32_t Rows, Pairs, Leaves, LeafMethod, R0, R1, R2, R3; };
struct LeafParams { uint32_t Leaves, HasDiagonal, R0, R1; float L2, NonDiagL2, MinWeight, Step; };
static_assert(sizeof(CandidateParams) == 32 && sizeof(MatrixParams) == 32 && sizeof(LeafParams) == 32);

extern "C" int cbm_pairwise_candidate_probe(const CandidateParams* options, uint32_t totalCandidates,
    float l2, float nonDiagL2, const uint8_t* bins, const uint32_t* leafIds, const float* point,
    const uint32_t* winners, const uint32_t* losers, const float* weights,
    const uint32_t* features, const uint32_t* borders, const uint8_t* types,
    float* gradient, float* hessian, float* direction, float* scores, uint32_t* status,
    char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!options) throw std::runtime_error("Missing candidate parameters");
            const auto& p = *options;
            const uint64_t entries = uint64_t(p.Candidates) * p.Pairs;
            const uint32_t leaves = p.ParentLeaves * 2;
            const uint64_t cells = uint64_t(p.Candidates) * leaves * leaves;
            const uint64_t ownBytes = uint64_t(p.Rows) * (p.Features + 8) + 28ull * p.Pairs +
                16 * entries + 48 * cells + 8ull * p.Candidates * leaves + 12ull * p.Candidates + 4 + 9ull * totalCandidates;
            if (!p.Rows || p.Rows > (1u << 24) || p.Pairs > (1u << 24) || !p.Features || !p.Candidates ||
                p.Candidates > 64 || !p.ParentLeaves || p.ParentLeaves > 128 ||
                p.LeafMethod > 1 || p.Reserved || uint64_t(p.FirstCandidate) + p.Candidates > totalCandidates ||
                entries > (1u << 22) || ownBytes > (1u << 29) ||
                !std::isfinite(l2) || l2 < 0 || !std::isfinite(nonDiagL2) || nonDiagL2 < 0)
                throw std::runtime_error("Invalid or excessive candidate dimensions");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            MTLCompileOptions* compile = [MTLCompileOptions new];
            compile.languageVersion = MTLLanguageVersion3_0; compile.fastMathEnabled = NO;
            NSError* error = nil;
            NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s\n%s", CBMMetalPairwiseMatrixSource,
                CBMMetalPairwiseCandidateSource, CBMMetalLeafMatrixSource, CBMMetalPairwiseScoreSource];
            id<MTLLibrary> library = [device newLibraryWithSource:source options:compile error:&error];
            if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            auto buffer = [&](const void* input, uint64_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = input && bytes ? [device newBufferWithBytes:input length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:std::max<uint64_t>(4, bytes) options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Candidate allocation failed");
                if (!input || !bytes) std::memset(result.contents, 0, std::max<uint64_t>(4, bytes));
                return result;
            };
            auto data = buffer(bins, uint64_t(p.Features) * p.Rows), ids = buffer(leafIds, 4ull * p.Rows);
            auto raw = buffer(point, 4ull * p.Rows), win = buffer(winners, 4ull * p.Pairs), lose = buffer(losers, 4ull * p.Pairs);
            auto weight = buffer(weights, 4ull * p.Pairs), edges = buffer(nullptr, 16ull * p.Pairs);
            auto cf = buffer(features, 4ull * totalCandidates), cb = buffer(borders, 4ull * totalCandidates), ct = buffer(types, totalCandidates);
            auto ka = buffer(nullptr, 4 * entries), kb = buffer(nullptr, 4 * entries);
            auto ia = buffer(nullptr, 4 * entries), ib = buffer(nullptr, 4 * entries);
            auto offsets = buffer(nullptr, 4 * (cells + 1)), values = buffer(nullptr, 32 * cells);
            auto g = buffer(nullptr, 4ull * p.Candidates * leaves), h = buffer(nullptr, 4 * cells);
            auto d = buffer(nullptr, 4ull * p.Candidates * leaves), work = buffer(nullptr, 8 * cells);
            auto score = buffer(nullptr, 8ull * p.Candidates), flag = buffer(nullptr, 4ull * p.Candidates);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                                const auto& params, uint32_t count, bool grouped) {
                if (!count) return;
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> b : buffers) [encoder setBuffer:b offset:0 atIndex:index++];
                [encoder setBytes:&params length:sizeof(params) atIndex:index];
                if (grouped) [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            const MatrixParams matrix = {p.Rows, p.Pairs, leaves, p.LeafMethod, 0, 0, 0, 0};
            const LeafParams leaf = {leaves, 0, 0, 0, l2, nonDiagL2, 1e-20f, 1};
            dispatch("ComputePairwiseMatrixEdges", {raw, win, lose, weight, edges, flag}, matrix, p.Pairs, false);
            dispatch("BuildPairwiseCandidateKeys", {data, ids, win, lose, cf, cb, ct, ka, ia, flag}, p, entries, false);
            CBMEncodeSortU32(command, ka, ia, entries, kb, ib, nullptr);
            dispatch("BuildPairwiseCandidateOffsets", {kb, offsets}, p, cells + 1, false);
            dispatch("ReducePairwiseCandidateCells", {offsets, ib, edges, values}, p, cells, true);
            dispatch("AssemblePairwiseCandidateMatrices", {values, g, h, flag}, p, cells, false);
            dispatch("RegularizePairwiseSplitMatrix", {h, work}, leaf, p.Candidates, true);
            dispatch("SolveLeafMatrix", {work, g, d, flag}, leaf, p.Candidates, true);
            dispatch("CenterPairwiseSplitSolution", {d}, leaf, p.Candidates, true);
            dispatch("ScorePairwiseSplitSolution", {h, g, d, score}, leaf, p.Candidates, true);
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(gradient, g.contents, 4ull * p.Candidates * leaves);
            std::memcpy(hessian, h.contents, 4 * cells);
            std::memcpy(direction, d.contents, 4ull * p.Candidates * leaves);
            std::memcpy(scores, score.contents, 8ull * p.Candidates);
            std::memcpy(status, flag.contents, 4ull * p.Candidates);
            return 0;
        } catch (const std::exception& e) {
            if (errorText && capacity) { std::strncpy(errorText, e.what(), capacity - 1); errorText[capacity - 1] = '\0'; }
            return 1;
        }
    }
}
