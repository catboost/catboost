#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_projection.h"
#include "metal_projection_kernels.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
void Require(bool valid, const std::string& text) {
    if (!valid) throw std::runtime_error(text);
}
// Literal validation messages allocate only when a check fails.
void Require(bool valid, const char* text) {
    if (!valid) throw std::runtime_error(text);
}
void CopyText(char* output, size_t capacity, const char* input) {
    if (!output || !capacity) return;
    const size_t count = std::min(capacity - 1, std::strlen(input));
    std::memcpy(output, input, count);
    output[count] = '\0';
}
std::string ErrorText(NSError* error) {
    return error ? error.localizedDescription.UTF8String : "Unknown Metal error";
}
bool Overlap(const void* a, size_t aSize, const void* b, size_t bSize) {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < aSize : x - y < bSize;
}
struct Params { uint32_t Rows, Cats, Bins, Components, HasPermutation; };
static_assert(sizeof(Params) == 20, "Projection Metal parameter layout mismatch");

struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device != nil && Device.hasUnifiedMemory,
                "Projection kernels require an Apple unified-memory GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Cannot create projection Metal queue");
        NSError* error = nil;
        id<MTLLibrary> library = [Device newLibraryWithSource:
            [NSString stringWithUTF8String:CBMProjectionMetalSource] options:nil error:&error];
        Require(library != nil, "Projection Metal compilation failed: " + ErrorText(error));
        for (const char* name : {"ProjectionHash", "ProjectionHighKeys", "ProjectionGatherHashes"}) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing projection kernel ") + name);
            id<MTLComputePipelineState> pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline != nil, "Projection pipeline creation failed: " + ErrorText(error));
            Pipelines.emplace(name, pipeline);
        }
    }
    id<MTLBuffer> Buffer(size_t bytes, const void* input = nullptr, bool shared = true) {
        const size_t capacity = std::max<size_t>(4, bytes);
        Require(capacity <= Device.maxBufferLength, "Projection buffer exceeds device limit");
        const MTLResourceOptions mode = shared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        id<MTLBuffer> output = input && bytes
            ? [Device newBufferWithBytes:input length:bytes options:mode]
            : [Device newBufferWithLength:capacity options:mode];
        Require(output != nil, "Projection Metal buffer allocation failed");
        return output;
    }
};
Runtime& GetRuntime() { static Runtime runtime; return runtime; }
}

extern "C" int cbm_projection_group(const CBMProjectionParams* p,
    const uint32_t* cats, const uint8_t* bins,
    const uint8_t* types, const uint32_t* features, const uint32_t* thresholds,
    const uint32_t* permutation, uint64_t* rowHashes, uint64_t* sortedHashes,
    uint32_t* sortedRows, CBMSortStats* stats, char* error, size_t errorCapacity) {
    @autoreleasepool {
        try {
            Require(p && stats, "Null projection parameters or statistics pointer");
            *stats = {};
            Require(p->rows <= (1u << 24), "Projection rows exceed the 16777216 limit");
            Require(p->components > 0 && types && features && thresholds,
                    "Projection requires component types, features, and thresholds");
            for (uint32_t component = 0; component < p->components; ++component) {
                Require(types[component] <= 2 && (!component || types[component - 1] <= types[component]),
                        "Projection components must be ordered categorical, numeric, then one-hot");
                Require(features[component] < (types[component] ? p->bin_features : p->cat_features),
                        "Projection component feature index is out of range");
                Require(!types[component] || thresholds[component] <= 255,
                        "Projection bin threshold exceeds uint8 range");
            }
            if (!p->rows) return 0;
            Require(rowHashes && sortedHashes && sortedRows && (!p->cat_features || cats) && (!p->bin_features || bins),
                    "Null projection feature or output pointer");
            const size_t rowBytes = size_t(p->rows) * 4;
            Require(!Overlap(rowHashes, rowBytes * 2, sortedHashes, rowBytes * 2) &&
                    !Overlap(rowHashes, rowBytes * 2, sortedRows, rowBytes) &&
                    !Overlap(sortedHashes, rowBytes * 2, sortedRows, rowBytes),
                    "Projection output arrays must not overlap");
            const uint64_t catBytes = uint64_t(p->cat_features) * rowBytes;
            const uint64_t binBytes = uint64_t(p->bin_features) * p->rows;
            Require(catBytes + binBytes + uint64_t(p->rows) * 64 + uint64_t(p->components) * 9 <= (uint64_t(1) << 30),
                    "Projection work buffers exceed the 1 GiB limit");
            if (permutation) {
                std::vector<uint8_t> seen(p->rows, 0);
                for (uint32_t i = 0; i < p->rows; ++i) {
                    Require(permutation[i] < p->rows && !seen[permutation[i]],
                            "Projection permutation must contain each row exactly once");
                    seen[permutation[i]] = 1;
                }
            }
            auto& runtime = GetRuntime();
            CopyText(stats->device_name, sizeof(stats->device_name), runtime.Device.name.UTF8String);
            auto catBuffer = runtime.Buffer(catBytes, cats), binBuffer = runtime.Buffer(binBytes, bins);
            auto typeBuffer = runtime.Buffer(p->components, types);
            auto featureBuffer = runtime.Buffer(size_t(p->components) * 4, features);
            auto thresholdBuffer = runtime.Buffer(size_t(p->components) * 4, thresholds);
            auto permutationBuffer = runtime.Buffer(permutation ? rowBytes : 0, permutation);
            auto hashBuffer = runtime.Buffer(rowBytes * 2), sortedHashBuffer = runtime.Buffer(rowBytes * 2);
            auto lowBuffer = runtime.Buffer(rowBytes, nullptr, false);
            auto highBuffer = runtime.Buffer(rowBytes, nullptr, false);
            auto rowBuffer = runtime.Buffer(rowBytes);
            id<MTLCommandBuffer> command = [runtime.Queue commandBuffer];
            Require(command != nil, "Cannot create projection command buffer");
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers,
                                const void* params, size_t paramsSize) {
                auto pipeline = runtime.Pipelines.at(name);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                Require(encoder != nil, "Cannot create projection encoder");
                [encoder setComputePipelineState:pipeline];
                NSUInteger slot = 0;
                for (auto buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:slot++];
                [encoder setBytes:params length:paramsSize atIndex:slot];
                [encoder dispatchThreads:MTLSizeMake(p->rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
                [encoder endEncoding];
                ++stats->kernel_dispatches;
            };
            Params params{p->rows, p->cat_features, p->bin_features, p->components, permutation != nullptr};
            dispatch("ProjectionHash", {catBuffer, binBuffer, typeBuffer, featureBuffer, thresholdBuffer,
                permutationBuffer, hashBuffer, lowBuffer, rowBuffer}, &params, sizeof(params));
            CBMEncodeSortU32(command, lowBuffer, rowBuffer, p->rows, lowBuffer, rowBuffer, &stats->kernel_dispatches);
            dispatch("ProjectionHighKeys", {hashBuffer, rowBuffer, highBuffer}, &p->rows, sizeof(p->rows));
            CBMEncodeSortU32(command, highBuffer, rowBuffer, p->rows, highBuffer, rowBuffer, &stats->kernel_dispatches);
            dispatch("ProjectionGatherHashes", {hashBuffer, rowBuffer, sortedHashBuffer}, &p->rows, sizeof(p->rows));
            [command commit];
            [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted,
                    "Projection GPU command failed: " + ErrorText(command.error));
            if (command.GPUEndTime >= command.GPUStartTime)
                stats->gpu_seconds = command.GPUEndTime - command.GPUStartTime;
            std::memcpy(rowHashes, hashBuffer.contents, rowBytes * 2);
            std::memcpy(sortedHashes, sortedHashBuffer.contents, rowBytes * 2);
            std::memcpy(sortedRows, rowBuffer.contents, rowBytes);
            return 0;
        } catch (const std::exception& exception) {
            CopyText(error, errorCapacity, exception.what());
            return 1;
        }
    }
}
