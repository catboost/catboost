#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_leaf_matrix_kernels.h"
#include "../native/metal_pairwise_score_kernels.h"
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
struct SelectionParams {
    uint32_t Candidates, Features, Reserved0, Reserved1;
    float Previous, Reserved2, Reserved3, Reserved4;
};

struct ScoreProbeRuntime {
    id<MTLDevice> Device;
    id<MTLLibrary> Library;
    id<MTLCommandQueue> Queue;
    id<MTLCommandBuffer> Command;
    ScoreProbeRuntime() {
        Device = MTLCreateSystemDefaultDevice();
        if (!Device) throw std::runtime_error("No Metal device");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0; options.fastMathEnabled = NO;
        NSError* error = nil;
        NSString* source = [NSString stringWithFormat:@"%s\n%s", CBMMetalLeafMatrixSource, CBMMetalPairwiseScoreSource];
        Library = [Device newLibraryWithSource:source options:options error:&error];
        if (!Library) throw std::runtime_error([[error localizedDescription] UTF8String]);
        Queue = [Device newCommandQueue]; Command = [Queue commandBuffer];
    }
    id<MTLBuffer> Buffer(const void* data, size_t bytes) {
        id<MTLBuffer> result = data && bytes ? [Device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:std::max<size_t>(4, bytes) options:MTLResourceStorageModeShared];
        if (!result) throw std::runtime_error("Score probe allocation failed");
        if (!data || !bytes) std::memset(result.contents, 0, std::max<size_t>(4, bytes));
        return result;
    }
    template <class TParams>
    void Dispatch(const char* name, std::initializer_list<id<MTLBuffer>> buffers, const TParams& params) {
        NSError* error = nil;
        id<MTLFunction> function = [Library newFunctionWithName:[NSString stringWithUTF8String:name]];
        id<MTLComputePipelineState> pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
        id<MTLComputeCommandEncoder> encoder = [Command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        NSUInteger index = 0;
        for (id<MTLBuffer> value : buffers) [encoder setBuffer:value offset:0 atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    void Wait() {
        [Command commit]; [Command waitUntilCompleted];
        if (Command.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error([[Command.error localizedDescription] UTF8String]);
    }
};

static int Failure(const std::exception& error, char* text, uint32_t capacity) {
    if (text && capacity) { std::strncpy(text, error.what(), capacity - 1); text[capacity - 1] = '\0'; }
    return 1;
}

extern "C" int cbm_pairwise_score_probe(const LeafMatrixParams* params, const float* hessian,
    const float* gradient, float* regularized, float* direction, float* score, uint32_t* status,
    char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!params) throw std::runtime_error("Missing score parameters");
            const auto& p = *params;
            if (!p.Leaves || p.Leaves > 256 || p.HasDiagonalPart > 1 || p.Reserved0 || p.Reserved1
                    || !std::isfinite(p.L2) || p.L2 < 0 || !std::isfinite(p.NonDiagL2) || p.NonDiagL2 < 0)
                throw std::runtime_error("Invalid score parameters");
            ScoreProbeRuntime runtime;
            const size_t bytes = p.Leaves * 4, matrixBytes = p.Leaves * bytes;
            auto h = runtime.Buffer(hessian, matrixBytes), g = runtime.Buffer(gradient, bytes);
            auto work = runtime.Buffer(nullptr, matrixBytes * 2), preserved = runtime.Buffer(nullptr, matrixBytes * 2);
            auto d = runtime.Buffer(nullptr, bytes), result = runtime.Buffer(nullptr, 8), flag = runtime.Buffer(nullptr, 4);
            runtime.Dispatch("RegularizePairwiseSplitMatrix", {h, work}, p);
            id<MTLBlitCommandEncoder> copy = [runtime.Command blitCommandEncoder];
            [copy copyFromBuffer:work sourceOffset:0 toBuffer:preserved destinationOffset:0 size:matrixBytes * 2];
            [copy endEncoding];
            runtime.Dispatch("SolveLeafMatrix", {work, g, d, flag}, p);
            runtime.Dispatch("CenterPairwiseSplitSolution", {d}, p);
            runtime.Dispatch("ScorePairwiseSplitSolution", {h, g, d, result}, p);
            runtime.Wait();
            std::memcpy(regularized, preserved.contents, matrixBytes * 2);
            std::memcpy(direction, d.contents, bytes); std::memcpy(score, result.contents, 8);
            std::memcpy(status, flag.contents, 4);
            return 0;
        } catch (const std::exception& error) { return Failure(error, errorText, capacity); }
    }
}

extern "C" int cbm_pairwise_selection_probe(uint32_t candidates, uint32_t features, float previous,
    const float* scores, const uint32_t* featureIds, const float* featureWeights,
    uint32_t* winner, float* scoreGain, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (candidates > (1u << 20) || !features || features > (1u << 20) || !std::isfinite(previous))
                throw std::runtime_error("Invalid selection parameters");
            ScoreProbeRuntime runtime;
            auto values = runtime.Buffer(scores, 8ull * candidates), ids = runtime.Buffer(featureIds, 4ull * candidates);
            auto weights = runtime.Buffer(featureWeights, 4ull * features), index = runtime.Buffer(nullptr, 4);
            auto result = runtime.Buffer(nullptr, 8);
            SelectionParams p = {candidates, features, 0, 0, previous, 0, 0, 0};
            runtime.Dispatch("SelectPairwiseSplitWinner", {values, ids, weights, index, result, values}, p);
            runtime.Wait();
            std::memcpy(winner, index.contents, 4); std::memcpy(scoreGain, result.contents, 8);
            return 0;
        } catch (const std::exception& error) { return Failure(error, errorText, capacity); }
    }
}
