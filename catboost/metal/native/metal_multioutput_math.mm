#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_multioutput_math.h"
#include "metal_multioutput_math_kernels.h"
#include "metal_multiclass_scores.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct Params {
    uint32_t Rows, Dimensions, Objective, Leaves;
    float L2, MinLeafWeight;
    uint32_t LeafMethod, Reserved;
};
static_assert(sizeof(Params) == 32, "Multioutput shader ABI mismatch");
constexpr uint64_t MemoryLimit = uint64_t(1) << 30;
void Require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
// Literal validation messages allocate only when a check fails.
void Require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void Text(char* output, size_t capacity, const char* value) {
    if (!output || !capacity) return;
    const size_t length = std::min(capacity - 1, std::strlen(value));
    std::memcpy(output, value, length); output[length] = 0;
}
std::string Error(NSError* error) { return error ? error.localizedDescription.UTF8String : "unknown Metal error"; }
struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device && Device.hasUnifiedMemory && [Device supportsFamily:MTLGPUFamilyApple7],
                "Multioutput math requires Apple Silicon Metal");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Metal command queue allocation failed");
        MTLCompileOptions* options = [MTLCompileOptions new];
        if (@available(macOS 13.0, *)) options.languageVersion = MTLLanguageVersion3_0;
        else throw std::runtime_error("Multioutput math requires macOS 13 or newer");
        options.fastMathEnabled = NO;
        NSError* error = nil;
        NSString* source = [NSString stringWithFormat:@"%s\n%s", CBMMetalMultioutputMathSource, CBMMetalMulticlassScoresSource];
        id<MTLLibrary> library = [Device newLibraryWithSource:source
                                                    options:options error:&error];
        Require(library != nil, "Multioutput shader compilation failed: " + Error(error));
        for (const char* name : {"MultioutputDerivatives", "MultioutputReduceLeafStats", "MultioutputSolveLeaves", "MulticlassExtraScoreMath"}) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing Metal kernel: ") + name);
            id<MTLComputePipelineState> state = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(state != nil && state.maxTotalThreadsPerThreadgroup >= 256,
                    "Multioutput pipeline compilation failed: " + Error(error));
            Pipelines.emplace(name, state);
        }
    }
    id<MTLBuffer> Buffer(uint64_t bytes, const void* source = nullptr) {
        Require(bytes <= MemoryLimit && bytes <= Device.maxBufferLength, "Multioutput buffer exceeds memory limit");
        id<MTLBuffer> result = source && bytes
            ? [Device newBufferWithBytes:source length:std::max<uint64_t>(bytes, 1) options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:std::max<uint64_t>(bytes, 1) options:MTLResourceStorageModeShared];
        Require(result != nil, "Multioutput buffer allocation failed");
        return result;
    }
};
Runtime& Context() { static Runtime result; return result; }
template <class P> void Dispatch(id<MTLCommandBuffer> command, const char* name, std::initializer_list<id<MTLBuffer>> buffers,
              const P& p, uint64_t count, bool groups = false) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    Require(encoder != nil, "Multioutput encoder allocation failed");
    [encoder setComputePipelineState:Context().Pipelines.at(name)];
    NSUInteger index = 0;
    for (auto buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
    [encoder setBytes:&p length:sizeof(p) atIndex:index];
    if (groups) [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    else [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
}
}

extern "C" int cbm_vector_score_math(uint32_t candidates, uint32_t terms, uint32_t kind,
    const float* gradients, const float* weights, float* scores, char* error, size_t capacity) {
    @autoreleasepool {
        try {
            Require(candidates > 0 && terms > 0 && kind >= 4 && kind <= 6, "Invalid vector score dimensions or kind");
            const uint64_t cells = uint64_t(candidates) * terms;
            Require(cells <= UINT32_MAX && cells * 12 + uint64_t(candidates) * 4 <= MemoryLimit,
                    "Vector score working set exceeds limits");
            Require(gradients && weights && scores, "Vector score input and output pointers are required");
            for (uint64_t i = 0; i < cells; ++i) {
                Require(std::isfinite(gradients[2*i]) && std::isfinite(gradients[2*i+1]), "Gradient pairs must be finite");
                Require(std::isfinite(weights[i]) && weights[i] >= 0, "Score weights must be finite and nonnegative");
            }
            auto& r = Context();
            auto g = r.Buffer(cells * 8, gradients), w = r.Buffer(cells * 4, weights);
            auto output = r.Buffer(uint64_t(candidates) * 4);
            id<MTLCommandBuffer> command = [r.Queue commandBuffer];
            Require(command != nil, "Vector score command allocation failed");
            const uint32_t p[4] = {candidates, terms, kind, 0};
            Dispatch(command, "MulticlassExtraScoreMath", {g, w, output}, p, candidates);
            [command commit]; [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted, "Vector score command failed: " + Error(command.error));
            std::memcpy(scores, output.contents, uint64_t(candidates) * 4);
            Text(error, capacity, ""); return 0;
        } catch (const std::exception& exception) { Text(error, capacity, exception.what()); return 1; }
        catch (...) { Text(error, capacity, "Unknown vector score failure"); return 1; }
    }
}

extern "C" int cbm_multioutput_math(uint32_t rows, uint32_t dimensions, uint32_t objective,
    uint32_t leaves, uint32_t method, float l2, const float* targets, const float* weights,
    const float* predictions, const uint32_t* leafIds, float* gradients, float* hessians,
    float* losses, float* directions, CBMTrainStats* stats, char* error, size_t capacity) {
    @autoreleasepool {
        try {
            Require(rows > 0 && rows <= (1u << 24), "rows must be in [1,16777216]");
            Require(dimensions >= 2 && dimensions <= 64, "dimensions must be in [2,64]");
            Require(objective <= 3 && (objective != 1 || dimensions == 2), "Invalid multioutput objective/dimensions");
            Require(leaves > 0 && leaves <= 65536 && method <= 1, "Invalid leaves or leaf method");
            Require(std::isfinite(l2) && l2 >= 0, "l2 must be finite and nonnegative");
            Require(targets && weights && predictions && leafIds && gradients && hessians && losses && directions && stats,
                    "Multioutput input and output pointers are required");
            const uint32_t targetDimensions = objective == 1 ? 1 : dimensions;
            const uint32_t width = 1 + 2 * dimensions;
            const uint64_t cells = uint64_t(rows) * dimensions;
            const uint64_t bytes = 4 * (uint64_t(rows) * (targetDimensions + 3 * dimensions + 3)
                + uint64_t(leaves) * (width + dimensions + 2) + 2);
            Require(bytes <= MemoryLimit, "Multioutput math working set exceeds 1 GiB");
            double totalWeight = 0;
            std::vector<uint32_t> offsets(leaves + 1, 0), indices(rows), positions;
            for (uint32_t row = 0; row < rows; ++row) {
                Require(std::isfinite(weights[row]) && weights[row] >= 0, "weights must be finite and nonnegative");
                totalWeight += weights[row];
                Require(leafIds[row] < leaves, "Invalid leaf id");
                ++offsets[leafIds[row] + 1];
            }
            Require(totalWeight > 0 && totalWeight < 1e30, "Total weight must be positive and below 1e30");
            for (uint64_t i = 0; i < uint64_t(rows) * targetDimensions; ++i) {
                Require(std::isfinite(targets[i]), "Targets must be finite");
                if (objective >= 2) Require(targets[i] >= 0 && targets[i] <= 1, "Multilabel targets must be in [0,1]");
                if (objective == 2) Require(targets[i] == 0 || targets[i] == 1, "MultiLogloss targets must be binary");
            }
            for (uint64_t i = 0; i < cells; ++i) Require(std::isfinite(predictions[i]), "Predictions must be finite");
            for (uint32_t leaf = 0; leaf < leaves; ++leaf) offsets[leaf + 1] += offsets[leaf];
            positions = offsets;
            for (uint32_t row = 0; row < rows; ++row) indices[positions[leafIds[row]]++] = row;
            auto& r = Context(); *stats = {}; Text(stats->device_name, sizeof(stats->device_name), r.Device.name.UTF8String);
            auto target = r.Buffer(uint64_t(rows) * targetDimensions * 4, targets);
            auto weight = r.Buffer(uint64_t(rows) * 4, weights);
            auto cursor = r.Buffer(cells * 4, predictions), gradient = r.Buffer(cells * 4), hessian = r.Buffer(cells * 4);
            auto loss = r.Buffer(uint64_t(rows) * 4), partition = r.Buffer(uint64_t(rows) * 4, indices.data());
            auto offset = r.Buffer(uint64_t(leaves + 1) * 4, offsets.data());
            auto statistics = r.Buffer(uint64_t(leaves) * width * 4), workspace = r.Buffer(4);
            auto direction = r.Buffer(uint64_t(leaves) * dimensions * 4), status = r.Buffer(uint64_t(leaves) * 4);
            Params p = {rows, dimensions, objective + 2, leaves, l2, 1e-20f, method, 0};
            id<MTLCommandBuffer> command = [r.Queue commandBuffer];
            Require(command != nil, "Multioutput command allocation failed");
            Dispatch(command, "MultioutputDerivatives", {target, weight, cursor, gradient, hessian, loss}, p, rows);
            Dispatch(command, "MultioutputReduceLeafStats", {gradient, hessian, weight, partition, offset, statistics}, p,
                     uint64_t(leaves) * width, true);
            Dispatch(command, "MultioutputSolveLeaves", {statistics, workspace, direction, status}, p, leaves);
            [command commit]; [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted, "Multioutput command failed: " + Error(command.error));
            stats->kernel_dispatches = 3;
            stats->gpu_seconds = std::max(0.0, command.GPUEndTime - command.GPUStartTime);
            const auto* state = static_cast<const uint32_t*>(status.contents);
            for (uint32_t leaf = 0; leaf < leaves; ++leaf)
                Require(state[leaf] == 0, "Multioutput leaf solve failed (status " + std::to_string(state[leaf]) + ")");
            for (uint32_t row = 0; row < rows; ++row)
                Require(std::isfinite(static_cast<const float*>(loss.contents)[row]), "Multioutput objective overflow");
            std::memcpy(gradients, gradient.contents, cells * 4);
            std::memcpy(hessians, hessian.contents, cells * 4);
            std::memcpy(losses, loss.contents, uint64_t(rows) * 4);
            std::memcpy(directions, direction.contents, uint64_t(leaves) * dimensions * 4);
            Text(error, capacity, ""); return 0;
        } catch (const std::exception& exception) { Text(error, capacity, exception.what()); return 1; }
        catch (...) { Text(error, capacity, "Unknown multioutput Metal failure"); return 1; }
    }
}
