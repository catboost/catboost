#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_ctrs.h"
#include "metal_ctr_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
void Require(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error(message);
}
// Literal validation messages allocate only when a check fails.
void Require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
void CopyText(char* dst, size_t capacity, const char* source) {
    if (!dst || !capacity) return;
    const size_t n = std::min(capacity - 1, std::strlen(source));
    std::memcpy(dst, source, n); dst[n] = '\0';
}
std::string ErrorText(NSError* error) {
    return error ? error.localizedDescription.UTF8String : "Unknown Metal error";
}
struct KernelParams {
    uint32_t Rows, Categories, Type, Border;
    float PriorNumerator, PriorDenominator;
    uint32_t Offset;
};
static_assert(sizeof(KernelParams) == 28, "CTR Metal parameter layout mismatch");

struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device != nil && Device.hasUnifiedMemory, "CTR kernels require an Apple unified-memory GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Cannot create CTR Metal command queue");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [Device newLibraryWithSource:[NSString stringWithUTF8String:CBMCtrMetalSource]
                                                     options:options error:&error];
        Require(library != nil, "CTR Metal compilation failed: " + ErrorText(error));
        for (const char* name : {"CtrInitialize", "CtrSegmentedScan", "CtrWriteHistory", "CtrWriteFrequency",
                                 "CtrInitializeGroupHeads", "CtrScanWithGroupHeads", "CtrWriteGroupedHistory"}) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing CTR kernel ") + name);
            id<MTLComputePipelineState> pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline != nil, "CTR pipeline creation failed: " + ErrorText(error));
            Pipelines.emplace(name, pipeline);
        }
    }
    id<MTLBuffer> Buffer(size_t bytes, const void* input = nullptr) {
        Require(bytes > 0 && bytes <= Device.maxBufferLength, "CTR buffer exceeds device limit");
        id<MTLBuffer> result = input ? [Device newBufferWithBytes:input length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Require(result != nil, "CTR Metal buffer allocation failed");
        return result;
    }
};
Runtime& GetRuntime() { static Runtime runtime; return runtime; }
} // namespace

static int ComputeCtrs(const CBMCtrParams* p, const uint32_t* categories,
                       const uint32_t* indices, const float* targets, const uint32_t* groupIds,
                       float* values, float* sums, uint32_t* counts,
                       CBMCtrStats* stats, char* error, size_t errorCapacity) {
    @autoreleasepool {
        try {
            Require(p && categories && indices && targets && values && sums && counts && stats,
                    "Null CTR input or output pointer");
            Require(p->rows > 0 && p->rows <= (1u << 24) && p->categories > 0 && p->categories <= p->rows,
                    "Invalid CTR row/category count");
            Require(p->ctr_type <= 3 && p->target_border <= 255, "Invalid CTR type/target border");
            Require(std::isfinite(p->prior_numerator) && std::isfinite(p->prior_denominator)
                    && p->prior_denominator > 0, "CTR priors must be finite with positive denominator");
            const bool grouped = groupIds && p->ctr_type != 3;
            Require(uint64_t(p->rows) * (grouped ? 44 : 32) + uint64_t(p->categories) * 8 <= (uint64_t(1) << 30),
                    "CTR work buffers exceed the 1 GiB limit");
            std::vector<uint8_t> seen(p->rows, 0);
            std::unordered_set<uint32_t> categoryGroups;
            uint32_t expectedCategory = 0;
            for (uint32_t i = 0; i < p->rows; ++i) {
                Require(indices[i] < p->rows && !seen[indices[i]], "CTR indices must be a permutation");
                seen[indices[i]] = 1;
                Require(categories[i] < p->categories, "CTR category out of range");
                if (i && categories[i] != categories[i - 1]) ++expectedCategory;
                Require(categories[i] == expectedCategory, "CTR categories must be sorted contiguous bins starting at zero");
                if (grouped) {
                    if (!i || categories[i] != categories[i - 1]) categoryGroups.clear();
                    if (!i || categories[i] != categories[i - 1] || groupIds[indices[i]] != groupIds[indices[i - 1]])
                        Require(categoryGroups.insert(groupIds[indices[i]]).second,
                                "CTR rows of each group must be contiguous within each category");
                }
                Require(std::isfinite(targets[i]), "CTR targets must be finite");
                if (p->ctr_type <= 1)
                    Require(targets[i] >= 0 && targets[i] <= 255 && std::floor(targets[i]) == targets[i],
                            "Borders/Buckets CTR targets must be bins in [0, 255]");
            }
            Require(expectedCategory + 1 == p->categories, "Every CTR category must have a row");
            Runtime& runtime = GetRuntime();
            *stats = {};
            CopyText(stats->device_name, sizeof(stats->device_name), runtime.Device.name.UTF8String);
            const size_t rowBytes = size_t(p->rows) * 4, categoryBytes = size_t(p->categories) * 4;
            auto catBuffer = runtime.Buffer(rowBytes, categories), indexBuffer = runtime.Buffer(rowBytes, indices);
            auto targetBuffer = runtime.Buffer(rowBytes, targets), outputBuffer = runtime.Buffer(rowBytes);
            auto source = runtime.Buffer(rowBytes * 2), destination = runtime.Buffer(rowBytes * 2);
            auto sumBuffer = runtime.Buffer(categoryBytes), countBuffer = runtime.Buffer(categoryBytes);
            id<MTLBuffer> groupBuffer = grouped ? runtime.Buffer(rowBytes, groupIds) : nil;
            id<MTLBuffer> sourceHeads = grouped ? runtime.Buffer(rowBytes) : nil;
            id<MTLBuffer> destinationHeads = grouped ? runtime.Buffer(rowBytes) : nil;
            id<MTLCommandBuffer> command = [runtime.Queue commandBuffer];
            Require(command != nil, "Cannot create CTR command buffer");
            KernelParams params{p->rows, p->categories, p->ctr_type, p->target_border,
                                p->prior_numerator, p->prior_denominator, 0};
            auto dispatch = [&](const char* name, std::initializer_list<id<MTLBuffer>> buffers) {
                id<MTLComputePipelineState> pipeline = runtime.Pipelines.at(name);
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                Require(encoder != nil, "Cannot create CTR encoder");
                [encoder setComputePipelineState:pipeline];
                NSUInteger index = 0;
                for (auto buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
                [encoder setBytes:&params length:sizeof(params) atIndex:index];
                [encoder dispatchThreads:MTLSizeMake(p->rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
                [encoder endEncoding];
                ++stats->kernel_dispatches;
            };
            dispatch("CtrInitialize", {indexBuffer, targetBuffer, source});
            if (grouped) dispatch("CtrInitializeGroupHeads", {catBuffer, indexBuffer, groupBuffer, sourceHeads});
            for (params.Offset = 1; params.Offset < p->rows; params.Offset <<= 1) {
                if (grouped) {
                    dispatch("CtrScanWithGroupHeads", {catBuffer, source, destination, sourceHeads, destinationHeads});
                    std::swap(sourceHeads, destinationHeads);
                } else dispatch("CtrSegmentedScan", {catBuffer, source, destination});
                std::swap(source, destination);
            }
            if (grouped)
                dispatch("CtrWriteGroupedHistory", {catBuffer, indexBuffer, source, sourceHeads, outputBuffer, sumBuffer, countBuffer});
            else dispatch("CtrWriteHistory", {catBuffer, indexBuffer, source, outputBuffer, sumBuffer, countBuffer});
            if (p->ctr_type == 3) dispatch("CtrWriteFrequency", {catBuffer, indexBuffer, countBuffer, outputBuffer});
            [command commit]; [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted, "CTR GPU command failed: " + ErrorText(command.error));
            stats->gpu_seconds = std::max(0.0, command.GPUEndTime - command.GPUStartTime);
            const float* result = static_cast<const float*>(outputBuffer.contents);
            const float* finalSums = static_cast<const float*>(sumBuffer.contents);
            for (uint32_t i = 0; i < p->rows; ++i) Require(std::isfinite(result[i]), "CTR output exceeds float32 range");
            for (uint32_t i = 0; i < p->categories; ++i) Require(std::isfinite(finalSums[i]), "CTR category sum exceeds float32 range");
            std::memcpy(values, result, rowBytes);
            std::memcpy(sums, finalSums, categoryBytes);
            std::memcpy(counts, countBuffer.contents, categoryBytes);
            return 0;
        } catch (const std::exception& ex) {
            CopyText(error, errorCapacity, ex.what());
            return 1;
        }
    }
}

extern "C" int cbm_compute_ctrs(const CBMCtrParams* p, const uint32_t* categories,
                                const uint32_t* indices, const float* targets,
                                float* values, float* sums, uint32_t* counts,
                                CBMCtrStats* stats, char* error, size_t errorCapacity) {
    return ComputeCtrs(p, categories, indices, targets, nullptr, values, sums, counts, stats, error, errorCapacity);
}

extern "C" int cbm_compute_ctrs_grouped(const CBMCtrParams* p, const uint32_t* categories,
                                        const uint32_t* indices, const float* targets, const uint32_t* groupIds,
                                        float* values, float* sums, uint32_t* counts,
                                        CBMCtrStats* stats, char* error, size_t errorCapacity) {
    return ComputeCtrs(p, categories, indices, targets, groupIds, values, sums, counts, stats, error, errorCapacity);
}
