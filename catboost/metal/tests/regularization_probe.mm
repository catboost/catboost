#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../native/metal_kernel_abi.h"
#include "../native/metal_kernels.h"
#include "../native/metal_additional_objective_kernels.h"
#include "../native/metal_objective_kernels.h"
#include "../native/metal_backtracking_kernels.h"
#include "../native/metal_streaming_score_kernels.h"
#include "../native/metal_dynamic_score_kernels.h"
#include "../native/metal_regularization_kernels.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

// Fixed one-border features let the independent Python oracle assess every
// scoring variant with identical uploaded float histogram statistics.
extern "C" int cbm_regularization_score_probe(uint32_t leaves, uint32_t features,
    uint32_t score, uint32_t variant, uint32_t normalize, float exponent, float l2,
    const float* sums, const float* weights, const float* leafSums, const float* leafWeights,
    uint32_t* selected, float* selectedScore, char* errorText, uint32_t capacity) {
    @autoreleasepool {
        try {
            static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("Metal device is unavailable");
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0; options.fastMathEnabled = NO;
            NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s\n%s",
                CBMMetalSource, CBMMetalAdditionalObjectiveSource, CBMMetalObjectiveSource,
                CBMMetalBacktrackingSource, CBMMetalStreamingScoreSource,
                CBMMetalDynamicScoreSource, CBMMetalRegularizationSource];
            NSError* error = nil;
            static id<MTLLibrary> library = nil;
            if (!library) library = [device newLibraryWithSource:source options:options error:&error];
            if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
            const char* names[] = {"FindSplitWinners", "FindSplitWinnersRegularized",
                "FindTileSplitWinnersRegularized", "FindDynamicTileSplitWinnersRegularized"};
            if (variant > 3) throw std::runtime_error("Invalid score variant");
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:names[variant]]];
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
            auto buffer = [&](uint64_t size, const void* data = nullptr) {
                id<MTLBuffer> result = [device newBufferWithLength:std::max<uint64_t>(size, 4) options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Metal allocation failed");
                if (data && size) std::memcpy(result.contents, data, size);
                else std::memset(result.contents, 0, result.length);
                return result;
            };
            std::vector<uint32_t> ids(features), zeros(features), offsets(features + 1);
            for (uint32_t i = 0; i <= features; ++i) { offsets[i] = i; if (i < features) ids[i] = i; }
            std::vector<float> penalties(2 * features, 1);
            std::vector<uint8_t> active(features, 1);
            std::vector<id<MTLBuffer>> buffers = {
                buffer(4ull * leaves * features, sums), buffer(4ull * leaves * features, weights),
                buffer(4ull * leaves, leafSums), buffer(4ull * leaves, leafWeights),
                buffer(4ull * features, ids.data()), buffer(4ull * features, zeros.data()),
                buffer(features), buffer(sizeof(CBMMetalSplitState)), buffer(4ull * features),
                buffer(4ull * (features + 1), offsets.data()), buffer(8ull * features, penalties.data())};
            if (variant == 3) buffers.push_back(buffer(features, active.data()));
            CBMMetalKernelParams p{};
            p.Rows = 1; p.Features = features; p.Bins = 1; p.Leaves = leaves;
            p.Candidates = features; p.ScoreFunction = score; p.L2 = l2; p.ScoreGroups = 1;
            p.Reserved3 = 1;
            struct ScoreOptions { uint32_t Normalize; float Exponent; uint32_t Reserved[2]; } regularization{normalize, exponent, {0, 0}};
            uint32_t bits; std::memcpy(&bits, &exponent, 4);
            struct Tile { uint32_t Begin, End, Bins, Reserved[5]; } tile{0, features, features, {normalize, bits, 0, 0, 0}};
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            NSUInteger index = 0;
            for (id<MTLBuffer> item : buffers) [encoder setBuffer:item offset:0 atIndex:index++];
            [encoder setBytes:&p length:sizeof(p) atIndex:index++];
            if (variant == 1) [encoder setBytes:&regularization length:sizeof(regularization) atIndex:index];
            if (variant >= 2) [encoder setBytes:&tile length:sizeof(tile) atIndex:index];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(command.error.localizedDescription.UTF8String);
            const auto& winner = *static_cast<const CBMMetalSplitState*>(buffers[7].contents);
            if (!winner.Valid || winner.InvalidScore) throw std::runtime_error("Invalid score result");
            *selected = winner.Index; *selectedScore = winner.Score;
            return 0;
        } catch (const std::exception& error) {
            if (errorText && capacity) { std::strncpy(errorText, error.what(), capacity - 1); errorText[capacity - 1] = 0; }
            return 1;
        }
    }
}
