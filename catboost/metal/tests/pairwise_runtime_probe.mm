#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_pairwise_runtime.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
struct RuntimeProbe {
    std::unique_ptr<CBMPairwiseRuntime> Runtime;
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    id<MTLBuffer> Cursor, LeafValues, LeafIds, Gradients, Hessian, Incident;
    uint32_t Rows = 0, MaxLeaves = 0;
};

void WriteError(char* text, uint32_t capacity, const std::string& message) {
    if (text && capacity) {
        std::strncpy(text, message.c_str(), capacity - 1);
        text[capacity - 1] = '\0';
    }
}

void Wait(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        const char* message = [[command.error localizedDescription] UTF8String];
        throw std::runtime_error(message ? message : "Metal command failed");
    }
}
}

// The handle deliberately survives multiple transactions, so pipeline/CSR
// lifetime and sticky GPU status are exercised independently of the trainer.
extern "C" void* cbm_pairwise_runtime_create(uint32_t rows, uint32_t pairs,
    const uint32_t* winners, const uint32_t* losers, const float* weights,
    uint32_t maxLeaves, uint32_t lossGroups, uint32_t groupCount,
    const uint32_t* groupOffsets, float* incidentWeights, double* totalIncidentWeight,
    uint64_t* allocatedBytes, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            auto probe = std::make_unique<RuntimeProbe>();
            probe->Device = MTLCreateSystemDefaultDevice();
            if (!probe->Device) throw std::runtime_error("No Metal device");
            probe->Runtime = std::make_unique<CBMPairwiseRuntime>(probe->Device,
                rows, pairs, winners, losers, weights, maxLeaves, lossGroups,
                groupCount, groupOffsets);
            // Queue creation is intentionally after helper construction.
            probe->Queue = [probe->Device newCommandQueue];
            if (!probe->Queue) throw std::runtime_error("No Metal command queue");
            probe->Rows = rows;
            probe->MaxLeaves = maxLeaves;
            auto buffer = [&](size_t bytes) -> id<MTLBuffer> {
                id<MTLBuffer> result = [probe->Device newBufferWithLength:std::max<size_t>(bytes, 4)
                    options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Runtime probe allocation failed");
                return result;
            };
            probe->Cursor = buffer(size_t(rows) * 4);
            probe->LeafValues = buffer(size_t(maxLeaves) * 4);
            probe->LeafIds = buffer(size_t(rows) * 4);
            probe->Gradients = buffer(size_t(rows) * 4);
            probe->Hessian = buffer(size_t(rows) * 4);
            probe->Incident = buffer(size_t(rows) * 4);
            const auto& incident = probe->Runtime->IncidentWeights();
            if (incident.size() != rows) throw std::runtime_error("Incorrect incident-weight shape");
            std::memcpy(incidentWeights, incident.data(), size_t(rows) * 4);
            *totalIncidentWeight = probe->Runtime->TotalIncidentWeight();
            *allocatedBytes = probe->Runtime->AllocatedBytes();
            return probe.release();
        } catch (const std::exception& error) {
            WriteError(errorText, capacity, error.what());
            return nullptr;
        }
    }
}

extern "C" int cbm_pairwise_runtime_evaluate(void* handle,
    const float* cursor, const float* leafValues, const uint32_t* leafIds,
    uint32_t leaves, uint32_t applyShift, uint32_t clearStatus, uint32_t allowNonfiniteTrial,
    float* gradients, float* hessian, float* incidentWeights, double* lossAndWeight,
    uint64_t* dispatches, uint64_t* allocatedBytes, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        const char* stage = "Host";
        try {
            auto& probe = *static_cast<RuntimeProbe*>(handle);
            if (!leaves || leaves > probe.MaxLeaves) throw std::runtime_error("Invalid probe leaf count");
            std::memcpy(probe.Cursor.contents, cursor, size_t(probe.Rows) * 4);
            std::memcpy(probe.LeafValues.contents, leafValues, size_t(leaves) * 4);
            std::memcpy(probe.LeafIds.contents, leafIds, size_t(probe.Rows) * 4);
            if (clearStatus) probe.Runtime->ClearStatus();
            id<MTLCommandBuffer> command = [probe.Queue commandBuffer];
            stage = "Encode";
            probe.Runtime->EncodePointDerivatives(command, probe.Cursor, probe.LeafValues,
                probe.LeafIds, leaves, bool(applyShift), probe.Gradients,
                probe.Hessian, probe.Incident, dispatches, bool(allowNonfiniteTrial));
            probe.Runtime->EncodeLossReduction(command, dispatches);
            stage = "Metal command";
            Wait(command);
            stage = "GPU status";
            probe.Runtime->CheckStatus();
            const auto loss = probe.Runtime->ReadLossPartials(bool(allowNonfiniteTrial));
            std::memcpy(gradients, probe.Gradients.contents, size_t(probe.Rows) * 4);
            std::memcpy(hessian, probe.Hessian.contents, size_t(probe.Rows) * 4);
            std::memcpy(incidentWeights, probe.Incident.contents, size_t(probe.Rows) * 4);
            lossAndWeight[0] = loss[0];
            lossAndWeight[1] = loss[1];
            *allocatedBytes = probe.Runtime->AllocatedBytes();
            return 0;
        } catch (const std::exception& error) {
            WriteError(errorText, capacity, std::string(stage) + ": " + error.what());
            return 1;
        }
    }
}

extern "C" int cbm_pairwise_runtime_center(void* handle, const float* leafValues,
    uint32_t leaves, uint32_t clearStatus, float* output, uint64_t* dispatches,
    uint64_t* allocatedBytes, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        const char* stage = "Host";
        try {
            auto& probe = *static_cast<RuntimeProbe*>(handle);
            if (!leaves || leaves > probe.MaxLeaves) throw std::runtime_error("Invalid probe leaf count");
            std::memcpy(probe.LeafValues.contents, leafValues, size_t(leaves) * 4);
            if (clearStatus) probe.Runtime->ClearStatus();
            id<MTLCommandBuffer> command = [probe.Queue commandBuffer];
            stage = "Encode";
            probe.Runtime->EncodeCenterLeafValues(command, probe.LeafValues, leaves, dispatches);
            stage = "Metal command";
            Wait(command);
            stage = "GPU status";
            probe.Runtime->CheckStatus();
            std::memcpy(output, probe.LeafValues.contents, size_t(leaves) * 4);
            *allocatedBytes = probe.Runtime->AllocatedBytes();
            return 0;
        } catch (const std::exception& error) {
            WriteError(errorText, capacity, std::string(stage) + ": " + error.what());
            return 1;
        }
    }
}

extern "C" void cbm_pairwise_runtime_destroy(void* handle) {
    @autoreleasepool {
        delete static_cast<RuntimeProbe*>(handle);
    }
}
