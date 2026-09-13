#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_sort.h"
#include "metal_sort_kernels.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr uint32_t MaxSortRows = 1u << 24;
constexpr uint32_t Threads = 256;
struct SortParams { uint32_t Rows, Tiles, Shift, Elements; };
static_assert(sizeof(SortParams) == 16, "Metal radix-sort parameter layout mismatch");

void Require(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error(message);
}
// Literal validation messages allocate only when a check fails.
void Require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
void CopyText(char* target, size_t capacity, const char* source) {
    if (!target || !capacity) return;
    const size_t bytes = std::min(capacity - 1, std::strlen(source));
    std::memcpy(target, source, bytes);
    target[bytes] = '\0';
}
std::string ErrorText(NSError* error) {
    return error ? error.localizedDescription.UTF8String : "Unknown Metal error";
}
uint32_t Groups(uint32_t elements) { return (elements - 1) / Threads + 1; }

struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device != nil && Device.hasUnifiedMemory, "Metal radix sort requires an Apple unified-memory GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Cannot create Metal radix-sort command queue");
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        id<MTLLibrary> library = [Device newLibraryWithSource:[NSString stringWithUTF8String:CBMSortMetalSource]
                                                     options:options error:&error];
        Require(library != nil, "Metal radix-sort compilation failed: " + ErrorText(error));
        for (const char* name : {"SortInitializePayload", "SortCountTiles", "SortScanPrefixes",
                                 "SortAddBlockPrefixes", "SortScatter"}) {
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing Metal radix-sort function ") + name);
            id<MTLComputePipelineState> pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline != nil, "Cannot create Metal radix-sort pipeline: " + ErrorText(error));
            Require(pipeline.maxTotalThreadsPerThreadgroup >= Threads && pipeline.threadExecutionWidth >= 8 &&
                        Threads % pipeline.threadExecutionWidth == 0,
                    "Metal radix-sort pipeline does not support the required cooperative group size");
            Pipelines.emplace(name, pipeline);
        }
    }
    id<MTLBuffer> Buffer(size_t bytes, const void* data = nullptr) {
        Require(bytes > 0 && bytes <= Device.maxBufferLength, "Metal radix-sort buffer exceeds the device limit");
        id<MTLBuffer> buffer = data
            ? [Device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        Require(buffer != nil, "Cannot allocate Metal radix-sort buffer");
        return buffer;
    }
    void Dispatch(id<MTLCommandBuffer> command, const char* name,
                  std::initializer_list<id<MTLBuffer>> buffers, const SortParams& params,
                  uint32_t groups, uint64_t* dispatches) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        Require(encoder != nil, "Cannot create Metal radix-sort compute encoder");
        [encoder setComputePipelineState:Pipelines.at(name)];
        NSUInteger index = 0;
        for (id<MTLBuffer> buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(Threads, 1, 1)];
        [encoder endEncoding];
        if (dispatches) ++*dispatches;
    }
};

Runtime& GetRuntime() { static Runtime runtime; return runtime; }

struct ScanLevel {
    id<MTLBuffer> Prefix;
    uint32_t Elements;
};
} // namespace

struct CBMSortU32Workspace::TState {
    id<MTLDevice> Device;
    id<MTLCommandBuffer> LastCommand;
    id<MTLBuffer> ScratchKeys, ScratchPayload, Ranks;
    std::vector<id<MTLBuffer>> Prefixes;
    uint32_t Capacity = 0;
    uint64_t Bytes = 0;
};

CBMSortU32Workspace::CBMSortU32Workspace(id<MTLDevice> device, uint32_t capacity)
    : State(std::make_unique<TState>()) {
    Require(capacity <= MaxSortRows, "Metal radix-sort workspace supports at most 16777216 rows");
    Runtime& runtime = GetRuntime();
    Require(device && device == runtime.Device, "Metal radix-sort workspace requires the active Apple GPU");
    State->Device = device;
    State->Capacity = capacity;
    if (!capacity) return;
    const uint64_t bytes = uint64_t(capacity) * 4;
    State->ScratchKeys = runtime.Buffer(bytes);
    State->ScratchPayload = runtime.Buffer(bytes);
    State->Ranks = runtime.Buffer(bytes);
    State->Bytes = 3 * bytes;
    uint32_t elements = Groups(capacity) * 16;
    while (true) {
        State->Prefixes.push_back(runtime.Buffer(uint64_t(elements) * 4));
        State->Bytes += uint64_t(elements) * 4;
        if (elements == 1) break;
        elements = Groups(elements);
    }
}
CBMSortU32Workspace::~CBMSortU32Workspace() = default;
uint64_t CBMSortU32Workspace::AllocatedBytes() const { return State->Bytes; }

void CBMSortU32Workspace::Encode(id<MTLCommandBuffer> command, id<MTLBuffer> keys,
                      id<MTLBuffer> payload, uint32_t count,
                      id<MTLBuffer> out_keys, id<MTLBuffer> out_payload,
                      uint64_t* kernel_dispatches, uint32_t low_key_bits) {
    Require(command != nil && command.status == MTLCommandBufferStatusNotEnqueued,
            "Metal radix sort needs an uncommitted command buffer");
    Require(command.retainedReferences, "Metal radix-sort scratch requires retained command-buffer references");
    Require(count <= MaxSortRows, "Metal radix sort supports at most 16777216 rows");
    Require(count <= State->Capacity, "Metal radix-sort count exceeds workspace capacity");
    Require(low_key_bits>=4 && low_key_bits<=32 && low_key_bits%4==0,
            "Metal radix-sort low key bits must be a multiple of four in [4,32]");
    Require(command.device == State->Device, "Metal radix-sort workspace uses a different device");
    Require(!State->LastCommand || State->LastCommand == command ||
            State->LastCommand.status == MTLCommandBufferStatusCompleted ||
            State->LastCommand.status == MTLCommandBufferStatusError,
            "Metal radix-sort workspace is still in use by another command");
    if (!count) return;
    Runtime& runtime = GetRuntime();
    Require(command.commandQueue.device == runtime.Device, "Metal radix-sort command uses a different device");
    const size_t bytes = size_t(count) * sizeof(uint32_t);
    for (id<MTLBuffer> buffer : {keys, payload, out_keys, out_payload}) {
        Require(buffer != nil && buffer.device == runtime.Device && buffer.length >= bytes,
                "Metal radix-sort buffer is missing, too small, or on a different device");
    }
    Require(out_keys != out_payload, "Metal radix-sort output key and payload buffers must be distinct");
    State->LastCommand = command;
    // Even passes write scratch, odd passes write the caller's output. Eight
    // passes leave the final result in output, including for in-place callers.
    auto scratchKeys = State->ScratchKeys, scratchPayload = State->ScratchPayload;
    auto ranks = State->Ranks;
    const uint32_t tiles = Groups(count);
    std::vector<ScanLevel> levels;
    uint32_t elements = tiles * 16;
    while (true) {
        levels.push_back({State->Prefixes.at(levels.size()), elements});
        if (elements == 1) break;
        elements = Groups(elements);
    }
    id<MTLBuffer> sourceKeys = keys, sourcePayload = payload;
    for (uint32_t shift = 0; shift < low_key_bits; shift += 4) {
        SortParams params{count, tiles, shift, levels.front().Elements};
        runtime.Dispatch(command, "SortCountTiles", {sourceKeys, ranks, levels.front().Prefix},
                         params, tiles, kernel_dispatches);
        for (size_t level = 0; level + 1 < levels.size(); ++level) {
            params.Elements = levels[level].Elements;
            runtime.Dispatch(command, "SortScanPrefixes", {levels[level].Prefix, levels[level + 1].Prefix},
                             params, Groups(params.Elements), kernel_dispatches);
        }
        // The top nontrivial level fits one group. Propagate its global offsets
        // down to every earlier level before stable scatter reads the histogram.
        for (size_t level = levels.size() - 2; level > 0; --level) {
            params.Elements = levels[level - 1].Elements;
            runtime.Dispatch(command, "SortAddBlockPrefixes", {levels[level - 1].Prefix, levels[level].Prefix},
                             params, Groups(params.Elements), kernel_dispatches);
        }
        id<MTLBuffer> destinationKeys = (shift & 4) ? out_keys : scratchKeys;
        id<MTLBuffer> destinationPayload = (shift & 4) ? out_payload : scratchPayload;
        runtime.Dispatch(command, "SortScatter",
                         {sourceKeys, sourcePayload, ranks, levels.front().Prefix, destinationKeys, destinationPayload},
                         params, tiles, kernel_dispatches);
        sourceKeys = destinationKeys;
        sourcePayload = destinationPayload;
    }
    if (sourceKeys != out_keys) {
        // Odd pass counts leave scratch as the source. Copy both arrays after
        // the final scatter, including when caller inputs and outputs alias.
        auto copy=[command blitCommandEncoder];Require(copy!=nil,"Radix output copy allocation failed");
        [copy copyFromBuffer:sourceKeys sourceOffset:0 toBuffer:out_keys destinationOffset:0 size:bytes];
        [copy copyFromBuffer:sourcePayload sourceOffset:0 toBuffer:out_payload destinationOffset:0 size:bytes];
        [copy endEncoding];
    }
}

void CBMEncodeSortU32(id<MTLCommandBuffer> command, id<MTLBuffer> keys,
                      id<MTLBuffer> payload, uint32_t count,
                      id<MTLBuffer> out_keys, id<MTLBuffer> out_payload,
                      uint64_t* kernel_dispatches) {
    Require(command && command.status == MTLCommandBufferStatusNotEnqueued && command.retainedReferences,
        "Metal radix sort needs an uncommitted command with retained references");
    Require(count <= MaxSortRows, "Metal radix sort supports at most 16777216 rows");
    if (!count) return;
    CBMSortU32Workspace workspace(command.device, count);
    workspace.Encode(command, keys, payload, count, out_keys, out_payload, kernel_dispatches);
}

extern "C" int cbm_sort_u32(const uint32_t* keys, const uint32_t* payload, uint32_t count,
                             uint32_t* out_keys, uint32_t* out_payload, CBMSortStats* stats,
                             char* error, size_t error_capacity) {
    @autoreleasepool {
        try {
            Require(stats != nullptr, "Metal radix-sort stats pointer is required");
            *stats = {};
            CopyText(error, error_capacity, "");
            Require(count <= MaxSortRows, "Metal radix sort supports at most 16777216 rows");
            Require(count == 0 || (keys && out_keys && out_payload), "Metal radix-sort array pointer is null");
            const size_t bytes = size_t(count) * sizeof(uint32_t);
            if (count) {
                const uintptr_t keyAddress = reinterpret_cast<uintptr_t>(out_keys);
                const uintptr_t payloadAddress = reinterpret_cast<uintptr_t>(out_payload);
                Require((keyAddress > payloadAddress ? keyAddress - payloadAddress : payloadAddress - keyAddress) >= bytes,
                        "Metal radix-sort output arrays must not overlap");
            }
            Runtime& runtime = GetRuntime();
            CopyText(stats->device_name, sizeof(stats->device_name), runtime.Device.name.UTF8String);
            if (!count) return 0;
            auto keyBuffer = runtime.Buffer(bytes, keys), payloadBuffer = runtime.Buffer(bytes, payload);
            id<MTLBuffer> outputKeys = [runtime.Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputPayload = [runtime.Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            Require(outputKeys != nil && outputPayload != nil, "Cannot allocate Metal radix-sort output buffers");
            id<MTLCommandBuffer> command = [runtime.Queue commandBuffer];
            Require(command != nil, "Cannot create Metal radix-sort command buffer");
            if (!payload) {
                SortParams params{count, Groups(count), 0, 0};
                runtime.Dispatch(command, "SortInitializePayload", {payloadBuffer}, params, Groups(count),
                                 &stats->kernel_dispatches);
            }
            CBMEncodeSortU32(command, keyBuffer, payloadBuffer, count, outputKeys, outputPayload,
                             &stats->kernel_dispatches);
            [command commit];
            [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted,
                    "Metal radix-sort GPU command failed: " + ErrorText(command.error));
            stats->gpu_seconds = std::max(0.0, command.GPUEndTime - command.GPUStartTime);
            std::memcpy(out_keys, outputKeys.contents, bytes);
            std::memcpy(out_payload, outputPayload.contents, bytes);
            return 0;
        } catch (const std::exception& ex) {
            CopyText(error, error_capacity, ex.what());
            return 1;
        }
    }
}
