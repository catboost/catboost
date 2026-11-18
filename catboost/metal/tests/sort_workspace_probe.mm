#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_sort.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" int cbm_sort_workspace_probe(uint32_t capacity, uint32_t batches, const uint32_t* counts,
    const uint32_t* keys, const uint32_t* payloads, uint32_t mode, uint32_t inPlace, uint32_t lowBits,
    uint32_t* sortedKeys, uint32_t* sortedPayloads, uint64_t* bytes, uint64_t* dispatches,
    char* text, uint32_t textCapacity) {
    @autoreleasepool {
        try {
            if (batches > 128 || mode > 2 || inPlace > 1) throw std::runtime_error("Invalid workspace probe options");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            CBMSortU32Workspace workspace(device, capacity);
            *bytes = workspace.AllocatedBytes(); *dispatches = 0;
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            std::vector<id<MTLBuffer>> results;
            uint64_t offset = 0;
            auto wait = [&]() {
                [command commit]; [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("Sort workspace GPU failure");
            };
            for (uint32_t batch = 0; batch < batches; ++batch) {
                if (counts[batch] > (1u << 20)) throw std::runtime_error("Probe batch too large");
                const uint64_t size = 4ull * counts[batch];
                auto buffer = [&](const uint32_t* input) -> id<MTLBuffer> {
                    auto value = input && size ? [device newBufferWithBytes:input length:size options:MTLResourceStorageModeShared]
                        : [device newBufferWithLength:std::max<uint64_t>(4,size) options:MTLResourceStorageModeShared];
                    if (!value) throw std::runtime_error("Probe buffer allocation failed");
                    return value;
                };
                auto inputKeys = buffer(keys+offset), inputPayloads = buffer(payloads+offset);
                auto outKeys = inPlace ? inputKeys : buffer(nullptr), outPayloads = inPlace ? inputPayloads : buffer(nullptr);
                if (batch && mode == 2) command = [queue commandBuffer]; // Must reject overlapping ownership.
                workspace.Encode(command,inputKeys,inputPayloads,counts[batch],outKeys,outPayloads,dispatches,lowBits);
                if (*bytes != workspace.AllocatedBytes()) throw std::runtime_error("Sort scratch grew during reuse");
                results.push_back(outKeys); results.push_back(outPayloads); offset += counts[batch];
                if (mode == 1) { wait(); command = [queue commandBuffer]; }
            }
            if (mode != 1) wait();
            offset = 0;
            for (uint32_t batch = 0; batch < batches; ++batch) {
                std::memcpy(sortedKeys+offset,results[2*batch].contents,4ull*counts[batch]);
                std::memcpy(sortedPayloads+offset,results[2*batch+1].contents,4ull*counts[batch]);
                offset += counts[batch];
            }
            return 0;
        } catch (const std::exception& error) {
            if (text && textCapacity) { std::strncpy(text,error.what(),textCapacity-1);text[textCapacity-1]='\0'; }
            return 1;
        }
    }
}
