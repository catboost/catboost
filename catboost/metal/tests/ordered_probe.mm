#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_ordered_kernels.h"
#include <cstring>
#include <initializer_list>

struct OrderedParams {
    uint32_t Rows, Features, Folds, Leaves;
    uint32_t Candidates, PackedRows, TestOnly, ScoreFunction;
    float L2;
    uint32_t Normalize;
    float ScoreBefore, LearningRate;
};
static_assert(sizeof(OrderedParams) == 48);

static int Error(const std::exception& error, char* output, uint32_t capacity) {
    if (output && capacity) {
        std::strncpy(output, error.what(), capacity - 1);
        output[capacity - 1] = '\0';
    }
    return 1;
}

extern "C" int cbm_ordered_folds(uint32_t rows, float growth, uint32_t minSize,
    CBMOrderedFold* output, uint32_t capacity, uint32_t* count, char* error, uint32_t errorCapacity) {
    try {
        const auto folds = CBMCreateNumericOrderedFolds(rows, growth, minSize);
        *count = folds.size();
        if (capacity < folds.size()) throw std::invalid_argument("Ordered fold output is too small");
        std::memcpy(output, folds.data(), folds.size() * sizeof(CBMOrderedFold));
        return 0;
    } catch (const std::exception& e) { return Error(e, error, errorCapacity); }
}

extern "C" int cbm_ordered_probe(const OrderedParams* params,
    const uint8_t* bins, const uint32_t* permutation, const uint32_t* leafIds,
    const uint32_t* candidates, const uint32_t* splitTypes, const CBMOrderedFold* folds,
    const float* targets, const float* weights, const float* cursor, const float* multipliers,
    const float* featureOptions, float* derivatives, float* sampled, float* statistics,
    float* scores, float* qualityStatistics, float* leafValues, float* updatedCursor,
    char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            const OrderedParams& p = *params;
            if (!p.Rows || p.Rows > (1u << 24) || !p.Features || p.Features > 4096 ||
                !p.Folds || p.Folds > 4096 || !p.Leaves || p.Leaves > 65536 ||
                !p.Candidates || p.Candidates > (1u << 20) || p.ScoreFunction > 1 ||
                p.TestOnly > 1 || p.Normalize > 1 || !std::isfinite(p.L2) || p.L2 < 0 ||
                !std::isfinite(p.ScoreBefore) || !std::isfinite(p.LearningRate))
                throw std::invalid_argument("Invalid Ordered probe configuration");
            uint64_t packedRows = 0;
            for (uint32_t i = 0; i < p.Folds; ++i) {
                const auto& f = folds[i];
                if (!f.EstimateEnd || f.EstimateEnd > f.QualityEnd || f.QualityEnd > p.Rows ||
                    f.CursorOffset != packedRows) throw std::invalid_argument("Invalid Ordered fold descriptor");
                packedRows += f.QualityEnd;
            }
            if (packedRows != p.PackedRows) throw std::invalid_argument("Invalid Ordered packed cursor length");
            const uint64_t statBytes = uint64_t(p.Candidates) * p.Leaves * p.Folds * 2 * 16;
            const uint64_t workingBytes = statBytes + packedRows * 28 + uint64_t(p.Rows) * (p.Features + 16);
            if (workingBytes > (1u << 29)) throw std::invalid_argument("Ordered probe workspace is too large");
            std::vector<bool> seen(p.Rows, false);
            for (uint32_t row = 0; row < p.Rows; ++row) {
                if (permutation[row] >= p.Rows || seen[permutation[row]])
                    throw std::invalid_argument("Invalid Ordered permutation");
                seen[permutation[row]] = true;
                if (leafIds[row] >= p.Leaves || !std::isfinite(targets[row]) ||
                    !std::isfinite(weights[row]) || weights[row] < 0)
                    throw std::invalid_argument("Invalid Ordered observation");
            }
            for (uint32_t row = 0; row < p.PackedRows; ++row) {
                if (!std::isfinite(cursor[row]) || !std::isfinite(multipliers[row]) || multipliers[row] < 0)
                    throw std::invalid_argument("Invalid Ordered packed value");
            }
            for (uint32_t c = 0; c < p.Candidates; ++c) {
                if (candidates[c * 2] >= p.Features || candidates[c * 2 + 1] > 255 || splitTypes[c] > 1)
                    throw std::invalid_argument("Invalid Ordered split candidate");
            }
            for (uint32_t f = 0; f < p.Features; ++f) {
                for (uint32_t part = 0; part < 3; ++part)
                    if (!std::isfinite(featureOptions[f * 4 + part]))
                        throw std::invalid_argument("Invalid Ordered feature option");
                if (featureOptions[f * 4] < 0 || featureOptions[f * 4 + 1] < 0)
                    throw std::invalid_argument("Invalid Ordered feature multiplier");
            }
            static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            static id<MTLLibrary> library = nil;
            NSError* error = nil;
            if (!library) {
                MTLCompileOptions* options = [MTLCompileOptions new];
                options.languageVersion = MTLLanguageVersion3_0;
                options.fastMathEnabled = NO;
                library = [device newLibraryWithSource:[NSString stringWithUTF8String:CBMMetalOrderedSource]
                    options:options error:&error];
                if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
            }
            auto buffer = [&](const void* source, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> value = source
                    ? [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!value) throw std::runtime_error("Metal Ordered probe allocation failed");
                return value;
            };
            id<MTLBuffer> binBuffer = buffer(bins, p.Rows * p.Features);
            id<MTLBuffer> permutationBuffer = buffer(permutation, p.Rows * 4);
            id<MTLBuffer> leafIdsBuffer = buffer(leafIds, p.Rows * 4);
            id<MTLBuffer> candidateBuffer = buffer(candidates, p.Candidates * 8);
            id<MTLBuffer> typeBuffer = buffer(splitTypes, p.Candidates * 4);
            id<MTLBuffer> foldsBuffer = buffer(folds, p.Folds * 16);
            id<MTLBuffer> targetBuffer = buffer(targets, p.Rows * 4), weightBuffer = buffer(weights, p.Rows * 4);
            id<MTLBuffer> cursorBuffer = buffer(cursor, p.PackedRows * 4);
            id<MTLBuffer> multiplierBuffer = buffer(multipliers, p.PackedRows * 4);
            id<MTLBuffer> featureBuffer = buffer(featureOptions, p.Features * 16);
            id<MTLBuffer> derivativeBuffer = buffer(nullptr, p.PackedRows * 8);
            id<MTLBuffer> sampledBuffer = buffer(nullptr, p.PackedRows * 8);
            id<MTLBuffer> statisticsBuffer = buffer(nullptr, statBytes);
            id<MTLBuffer> scoresBuffer = buffer(nullptr, p.Candidates * 8);
            id<MTLBuffer> qualityBuffer = buffer(nullptr, p.Folds * 8);
            id<MTLBuffer> leafValuesBuffer = buffer(nullptr, p.Folds * p.Leaves * 4);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                                MTLSize count, bool grouped) {
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> entry : buffers) [encoder setBuffer:entry offset:0 atIndex:index++];
                [encoder setBytes:&p length:sizeof(p) atIndex:index];
                if (grouped) [encoder dispatchThreadgroups:count threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else [encoder dispatchThreads:count threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            const MTLSize packedGrid = MTLSizeMake(p.Rows, p.Folds, 1);
            dispatch("PrepareOrderedRmseDerivatives", {targetBuffer, weightBuffer, cursorBuffer,
                permutationBuffer, foldsBuffer, derivativeBuffer}, packedGrid, false);
            dispatch("ApplyOrderedBootstrap", {derivativeBuffer, multiplierBuffer, foldsBuffer, sampledBuffer}, packedGrid, false);
            dispatch("OrderedCandidateStatistics", {binBuffer, permutationBuffer, leafIdsBuffer, candidateBuffer,
                typeBuffer, sampledBuffer, foldsBuffer, statisticsBuffer}, MTLSizeMake(p.Candidates, p.Leaves, p.Folds), true);
            dispatch("ScoreOrderedCandidates", {statisticsBuffer, candidateBuffer, featureBuffer, scoresBuffer},
                MTLSizeMake(p.Candidates, 1, 1), false);
            dispatch("OrderedQualityStatistics", {derivativeBuffer, foldsBuffer, qualityBuffer}, MTLSizeMake(p.Folds, 1, 1), true);
            dispatch("EstimateOrderedRmseLeaves", {derivativeBuffer, permutationBuffer, leafIdsBuffer,
                foldsBuffer, leafValuesBuffer}, MTLSizeMake(p.Leaves, p.Folds, 1), true);
            dispatch("ApplyOrderedFoldValues", {cursorBuffer, permutationBuffer, leafIdsBuffer, foldsBuffer,
                leafValuesBuffer}, packedGrid, false);
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(derivatives, derivativeBuffer.contents, p.PackedRows * 8);
            std::memcpy(sampled, sampledBuffer.contents, p.PackedRows * 8);
            std::memcpy(statistics, statisticsBuffer.contents, statBytes);
            std::memcpy(scores, scoresBuffer.contents, p.Candidates * 8);
            std::memcpy(qualityStatistics, qualityBuffer.contents, p.Folds * 8);
            std::memcpy(leafValues, leafValuesBuffer.contents, p.Folds * p.Leaves * 4);
            std::memcpy(updatedCursor, cursorBuffer.contents, p.PackedRows * 4);
            return 0;
        } catch (const std::exception& e) { return Error(e, errorText, capacity); }
    }
}
