#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_pairwise_runtime.h"

extern "C" int cbm_pairwise_difference_probe(uint32_t rows, uint32_t pairs, uint32_t leaves,
    uint32_t groups, const float* cursor, const uint32_t* winners, const uint32_t* losers,
    const float* weights, const uint32_t* ids, const float* current, const float* trial,
    double* result, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            CBMPairwiseRuntime runtime(device, rows, pairs, winners, losers, weights, leaves, groups);
            for (uint32_t row = 0; row < rows; ++row)
                if (ids[row] >= leaves) throw std::runtime_error("Invalid probe leaf ID");
            auto buffer = [&](const void* values, uint64_t count) {
                return [device newBufferWithBytes:values length:count * 4 options:MTLResourceStorageModeShared];
            };
            id<MTLBuffer> cursorBuffer = buffer(cursor, rows), idsBuffer = buffer(ids, rows);
            id<MTLBuffer> currentBuffer = buffer(current, leaves), trialBuffer = buffer(trial, leaves);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            runtime.EncodeObjectiveDifference(command, cursorBuffer, currentBuffer, trialBuffer, idsBuffer, leaves, nullptr);
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error localizedDescription] UTF8String]);
            *result = runtime.ReadObjectiveDifference();
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
