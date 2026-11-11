#include "../native/metal_combination_runtime.h"
#include <cstdio>

namespace {
id<MTLBuffer> TestBuffer(id<MTLDevice> device, uint64_t bytes, const void* input = nullptr) {
    auto result = input ? [device newBufferWithBytes:input length:bytes options:MTLResourceStorageModeShared]
        : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Combination probe buffer allocation failed");
    return result;
}
}

// Exercise the resident target's trial contract directly: session parameters
// never admit an infinite starting cursor, while a line-search trial may
// overflow after adding its finite direction to the accepted point.
extern "C" int cbm_combination_trial_probe(float shift, uint32_t trial,
    double* output, char* error, size_t errorCapacity) {
    @autoreleasepool {
        try {
            auto device = MTLCreateSystemDefaultDevice();
            auto queue = [device newCommandQueue];
            if (!device || !queue || !output || trial > 1) throw std::runtime_error("Invalid Combination probe options");
            constexpr uint32_t rows = 8;
            const uint32_t offsets[] = {0, 4, 8}, ids[rows] = {};
            const float targets[] = {.1f, .3f, .6f, .9f, .2f, .4f, .7f, 1.f};
            const float weights[] = {1, 2, .5f, 1, .5f, 1, 2, 1};
            const float cursor[rows] = {}, zero = 0;
            CBMCombinationComponent components[2] = {};
            components[0].objective = 17; components[0].weight = .03f;
            components[0].permutations = 3; components[0].decay = .85f;
            components[1].objective = 0; components[1].weight = 1.5f;
            CBMCombinationRuntime target(device, rows, 2, components, 2, offsets, 0,
                nullptr, nullptr, nullptr, 1, 1);
            target.ValidateTargets(targets, weights);
            uint64_t seedCount = 0, dispatches = 0;
            target.SetYetiSeedCallback([](void* context, uint64_t* seed) -> int {
                auto& count = *static_cast<uint64_t*>(context);
                *seed = 7919 + (++count) * 104729;
                return 0;
            }, &seedCount);
            auto t = TestBuffer(device, sizeof(targets), targets), w = TestBuffer(device, sizeof(weights), weights);
            auto p = TestBuffer(device, sizeof(cursor), cursor), leaves = TestBuffer(device, sizeof(zero), &zero);
            auto bins = TestBuffer(device, sizeof(ids), ids);
            auto gradient = TestBuffer(device, sizeof(cursor)), hessian = TestBuffer(device, sizeof(cursor));
            auto gradientWeights = TestBuffer(device, sizeof(cursor));
            auto evaluate = [&](float value, bool isTrial) {
                *static_cast<float*>(leaves.contents) = value;
                auto command = [queue commandBuffer];
                target.EncodeOracle(command, p, leaves, bins, 1, true, t, w,
                    gradient, hessian, gradientWeights, &dispatches, isTrial);
                [command commit]; [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error("Combination probe command failed");
                return target.ReadObjective(isTrial);
            };
            output[0] = evaluate(0, false);
            output[1] = evaluate(shift, trial != 0);
            output[2] = evaluate(0, false);
            output[3] = seedCount;
            output[4] = dispatches;
            return 0;
        } catch (const std::exception& exception) {
            if (error && errorCapacity) std::snprintf(error, errorCapacity, "%s", exception.what());
            return 1;
        }
    }
}
