#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_querywise_kernels.h"
#include "../native/metal_kernels.h"
#include "../native/metal_objective_kernels.h"
#include "../native/metal_additional_objective_kernels.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>

struct QuerywiseParams {
    uint32_t Rows, Groups, Objective, ApplyLeafValues;
    float Beta, Lambda;
    uint32_t Leaves, Reserved;
};
static_assert(sizeof(QuerywiseParams) == 32);

struct QuerywiseProjectionParams {
    uint32_t Rows, Leaves, Tiles, LeafMethod;
};
static_assert(sizeof(QuerywiseProjectionParams) == 16);

static id<MTLLibrary> QuerywiseProbeLibrary(id<MTLDevice> device) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_0;
    options.fastMathEnabled = NO;
    NSError* error = nil;
    const std::string source = std::string(CBMMetalSource) + CBMMetalAdditionalObjectiveSource
        + CBMMetalObjectiveSource + CBMMetalQuerywiseSource;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
        options:options error:&error];
    if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
    return library;
}

static void QuerywiseProbeError(const std::exception& error, char* text, uint32_t capacity) {
    if (text && capacity) {
        std::strncpy(text, error.what(), capacity - 1);
        text[capacity - 1] = '\0';
    }
}

extern "C" int cbm_querywise_probe(const QuerywiseParams* params,
    const float* targets, const float* weights, const float* cursor, const uint32_t* offsets,
    const float* leafValues, const uint32_t* leafIds, float* point, float* gradients,
    float* curvature, float* queryStats, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            const QuerywiseParams& p = *params;
            if (!p.Rows || !p.Groups || p.Groups > p.Rows || p.Rows > (1u << 24) || !p.Leaves)
                throw std::runtime_error("Invalid query dimensions");
            if (p.Objective != 12 && p.Objective != 13) throw std::runtime_error("Invalid query objective");
            if (offsets[0] != 0 || offsets[p.Groups] != p.Rows)
                throw std::runtime_error("Query offsets must cover every row");
            for (uint32_t q = 0; q < p.Groups; ++q)
                if (offsets[q] >= offsets[q + 1]) throw std::runtime_error("Query offsets must increase");
            if (!std::isfinite(p.Beta) || !std::isfinite(p.Lambda))
                throw std::runtime_error("Nonfinite query parameter");
            for (uint32_t row = 0; row < p.Rows; ++row) {
                if (!std::isfinite(targets[row]) || !std::isfinite(cursor[row]) ||
                    !std::isfinite(weights[row]) || weights[row] < 0 ||
                    (p.Objective == 13 && targets[row] < 0))
                    throw std::runtime_error("Invalid query observation");
                if (p.ApplyLeafValues && leafIds[row] >= p.Leaves)
                    throw std::runtime_error("Invalid query leaf id");
            }
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            NSError* error = nil;
            id<MTLLibrary> library = QuerywiseProbeLibrary(device);
            auto buffer = [&](const void* source, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = source
                    ? [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Metal query probe allocation failed");
                return result;
            };
            const size_t bytes = p.Rows * sizeof(float);
            id<MTLBuffer> targetBuffer = buffer(targets, bytes), weightBuffer = buffer(weights, bytes);
            id<MTLBuffer> cursorBuffer = buffer(cursor, bytes), offsetBuffer = buffer(offsets, (p.Groups + 1) * 4);
            id<MTLBuffer> leafBuffer = buffer(leafValues, p.Leaves * 4), leafIdBuffer = buffer(leafIds, bytes);
            id<MTLBuffer> pointBuffer = buffer(nullptr, bytes), gradientBuffer = buffer(nullptr, bytes);
            id<MTLBuffer> curvatureBuffer = buffer(nullptr, bytes), statsBuffer = buffer(nullptr, p.Groups * 8);
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
                if (grouped) [encoder dispatchThreadgroups:MTLSizeMake(p.Groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                else [encoder dispatchThreads:MTLSizeMake(p.Rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            dispatch("PrepareQuerywisePoint", {cursorBuffer, leafBuffer, leafIdBuffer, pointBuffer}, false);
            dispatch(p.Objective == 12 ? "QueryRmseDerivatives" : "QuerySoftMaxDerivatives",
                {targetBuffer, weightBuffer, pointBuffer, offsetBuffer, gradientBuffer, curvatureBuffer, statsBuffer}, true);
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(point, pointBuffer.contents, bytes);
            std::memcpy(gradients, gradientBuffer.contents, bytes);
            std::memcpy(curvature, curvatureBuffer.contents, bytes);
            std::memcpy(queryStats, statsBuffer.contents, p.Groups * 8);
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

// Directly consume querywise partials with the production scalar leaf solver.
// This verifies the byte layout and bindings in addition to the projected sums.
extern "C" int cbm_querywise_project_probe(const QuerywiseProjectionParams* params,
    const float* gradients, const float* curvature, const float* weights,
    const uint32_t* rowIndices, const uint32_t* offsets, float* partials,
    float* leafValues, float* leafWeights, float l2, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            const QuerywiseProjectionParams& p = *params;
            if (!p.Rows || !p.Leaves || !p.Tiles || p.Tiles > 4096 || p.LeafMethod > 1)
                throw std::runtime_error("Invalid projection dimensions");
            if (offsets[0] != 0 || offsets[p.Leaves] != p.Rows)
                throw std::runtime_error("Projection offsets must cover every row");
            for (uint32_t leaf = 0; leaf < p.Leaves; ++leaf)
                if (offsets[leaf] > offsets[leaf + 1]) throw std::runtime_error("Projection offsets must increase");
            for (uint32_t row = 0; row < p.Rows; ++row)
                if (rowIndices[row] >= p.Rows) throw std::runtime_error("Invalid projection row index");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            id<MTLLibrary> library = QuerywiseProbeLibrary(device);
            auto buffer = [&](const void* source, size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = source
                    ? [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Metal projection probe allocation failed");
                return result;
            };
            const size_t rowBytes = p.Rows * 4, leafBytes = p.Leaves * 4;
            const size_t partialBytes = size_t(p.Leaves) * p.Tiles * 2 * 16;
            id<MTLBuffer> gradientBuffer = buffer(gradients, rowBytes), curvatureBuffer = buffer(curvature, rowBytes);
            id<MTLBuffer> weightBuffer = buffer(weights, rowBytes), rowBuffer = buffer(rowIndices, rowBytes);
            id<MTLBuffer> offsetBuffer = buffer(offsets, (p.Leaves + 1) * 4);
            id<MTLBuffer> partialBuffer = buffer(nullptr, partialBytes);
            id<MTLBuffer> valuesBuffer = buffer(leafValues, leafBytes), leafWeightBuffer = buffer(nullptr, leafBytes);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                                const void* settings, size_t settingsSize, MTLSize groups) {
                NSError* error = nil;
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
                id<MTLCommandEncoder> generic = [command computeCommandEncoder];
                id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)generic;
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> entry : buffers) [encoder setBuffer:entry offset:0 atIndex:index++];
                [encoder setBytes:settings length:settingsSize atIndex:index];
                [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            dispatch("ReduceQuerywiseLeafPartials", {gradientBuffer, curvatureBuffer, weightBuffer, rowBuffer,
                offsetBuffer, partialBuffer}, &p, sizeof(p), MTLSizeMake(p.Tiles, p.Leaves, 1));
            CBMMetalKernelParams scalar = {};
            scalar.Rows = p.Rows; scalar.Leaves = p.Leaves; scalar.HistogramTiles = p.Tiles;
            scalar.LeafMethod = p.LeafMethod; scalar.L2 = l2; scalar.Objective = 12;
            dispatch("EstimateNewtonLeafValues", {partialBuffer, valuesBuffer, leafWeightBuffer},
                &scalar, sizeof(scalar), MTLSizeMake(p.Leaves, 1, 1));
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(partials, partialBuffer.contents, partialBytes);
            std::memcpy(leafValues, valuesBuffer.contents, leafBytes);
            std::memcpy(leafWeights, leafWeightBuffer.contents, leafBytes);
            return 0;
        } catch (const std::exception& error) {
            QuerywiseProbeError(error, errorText, capacity); return 1;
        }
    }
}

extern "C" int cbm_querywise_objective_probe(const float* queryStats, uint32_t queries,
    uint32_t groups, float* partials, uint32_t* leafIds, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!queries || !groups || groups > 4096) throw std::runtime_error("Invalid reduction dimensions");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            id<MTLLibrary> library = QuerywiseProbeLibrary(device);
            id<MTLBuffer> stats = [device newBufferWithBytes:queryStats length:size_t(queries) * 8
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> output = [device newBufferWithLength:size_t(groups) * 8 options:MTLResourceStorageModeShared];
            id<MTLBuffer> ids = [device newBufferWithBytes:leafIds length:size_t(queries) * 4
                options:MTLResourceStorageModeShared];
            if (!stats || !output || !ids) throw std::runtime_error("Metal reduction probe allocation failed");
            QuerywiseParams p = {queries, queries, 12, 0, 1, .01f, 1, 0};
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                                uint32_t groupCount) {
                NSError* error = nil;
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (id<MTLBuffer> entry : buffers) [encoder setBuffer:entry offset:0 atIndex:index++];
                [encoder setBytes:&p length:sizeof(p) atIndex:index];
                [encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
            };
            dispatch("ReduceQuerywiseObjective", {stats, output}, groups);
            dispatch("ResetQuerywiseLeafIds", {ids}, (queries + 255) / 256);
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(partials, output.contents, size_t(groups) * 8);
            std::memcpy(leafIds, ids.contents, size_t(queries) * 4);
            return 0;
        } catch (const std::exception& error) {
            QuerywiseProbeError(error, errorText, capacity); return 1;
        }
    }
}

extern "C" int cbm_querywise_curvature_probe(const float* curvature, uint32_t rows,
    uint32_t* status, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            if (!rows) throw std::runtime_error("Empty curvature input");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device");
            id<MTLLibrary> library = QuerywiseProbeLibrary(device);
            id<MTLBuffer> input = [device newBufferWithBytes:curvature length:size_t(rows) * 4
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> flag = [device newBufferWithBytes:status length:4 options:MTLResourceStorageModeShared];
            if (!input || !flag) throw std::runtime_error("Metal curvature probe allocation failed");
            QuerywiseParams p = {rows, 1, 13, 0, 1, .01f, 1, 0};
            NSError* error = nil;
            id<MTLFunction> function = [library newFunctionWithName:@"ValidateQuerywiseStructureCurvature"];
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLBlitCommandEncoder> clear = [command blitCommandEncoder];
            [clear fillBuffer:flag range:NSMakeRange(0, 4) value:0];
            [clear endEncoding];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBuffer:flag offset:0 atIndex:1];
            [encoder setBytes:&p length:sizeof(p) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            std::memcpy(status, flag.contents, 4);
            return 0;
        } catch (const std::exception& error) {
            QuerywiseProbeError(error, errorText, capacity); return 1;
        }
    }
}
