#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_inference.h"
#include "metal_inference_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint64_t MaxBytes = uint64_t(1) << 30;
constexpr uint32_t TreesPerTile = 32;
struct Pair { float High, Low; };
struct KernelParams {
    uint32_t Rows, SplitStride, LeafStride, TreeStart, TreeEnd, Tiles, TreesPerTile;
    uint32_t Dimensions;
};
static_assert(sizeof(CBMInferenceParams) == 48, "Inference ABI mismatch");
static_assert(sizeof(KernelParams) == 32 && sizeof(Pair) == 8, "MSL ABI mismatch");

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
// Literal validation messages allocate only when a check fails.
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void CopyText(char* destination, size_t capacity, const char* source) {
    if (!destination || !capacity) return;
    size_t length = std::min(capacity - 1, std::strlen(source));
    std::memcpy(destination, source, length);
    destination[length] = 0;
}
std::string ErrorText(NSError* error) {
    const char* description = error ? error.localizedDescription.UTF8String : nullptr;
    return description ? description : "Unknown Metal error";
}
void CheckBuffer(const void* data, uint64_t count, uint64_t required, const char* name) {
    Require(count == required, std::string(name) + " element count does not match its shape");
    Require(!required || data, std::string(name) + " is null");
}

struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    id<MTLComputePipelineState> Evaluate, Reduce, EvaluateNonSymmetric;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device && Device.hasUnifiedMemory, "Metal inference requires an Apple unified-memory GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Could not create Metal inference command queue");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [Device newLibraryWithSource:
            [NSString stringWithUTF8String:CBMInferenceSource] options:options error:&error];
        Require(library != nil, "Metal inference shader compilation failed: " + ErrorText(error));
        Evaluate = [Device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"EvaluateTreeTiles"] error:&error];
        Require(Evaluate != nil, "Could not create inference pipeline: " + ErrorText(error));
        EvaluateNonSymmetric = [Device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"EvaluateNonSymmetricTreeTiles"] error:&error];
        Require(EvaluateNonSymmetric != nil, "Could not create variable-tree inference pipeline: " + ErrorText(error));
        Reduce = [Device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"ReduceTreeTiles"] error:&error];
        Require(Reduce != nil, "Could not create inference reduction: " + ErrorText(error));
    }
    id<MTLBuffer> Buffer(uint64_t bytes, const void* source = nullptr) {
        Require(bytes <= MaxBytes && bytes <= Device.maxBufferLength, "Inference buffer exceeds memory limit");
        id<MTLBuffer> buffer = source && bytes
            ? [Device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:std::max<uint64_t>(bytes, 1) options:MTLResourceStorageModeShared];
        Require(buffer != nil, "Could not allocate Metal inference buffer");
        return buffer;
    }
};
Runtime& GetRuntime() { static Runtime runtime; return runtime; }

void Predict(const CBMInferenceParams& p, uint32_t dimensions,
             const double* biases, uint64_t biasesCount,
             const uint8_t* bins, uint64_t binsCount,
             const uint32_t* depths, uint64_t depthsCount,
             const uint32_t* features, uint64_t featuresCount,
             const uint32_t* borders, uint64_t bordersCount,
             const uint8_t* types, uint64_t typesCount,
             const double* leaves, uint64_t leavesCount,
             double* predictions, uint64_t predictionsCount, CBMInferenceStats& stats) {
    Require(p.trees <= 1000000 && p.split_stride <= 16 && p.leaf_stride > 0,
            "Invalid inference tree count, split stride, or leaf stride");
    Require(dimensions >= 1 && dimensions <= 64, "Inference dimensions must be in [1,64]");
    Require(p.tree_start <= p.tree_end && p.tree_end <= p.trees, "Invalid inference tree range");
    Require(std::isfinite(p.scale), "Inference scale must be finite");
    Require(p.batch_rows > 0, "Inference batch size must be positive");
    CheckBuffer(biases, biasesCount, dimensions, "biases");
    for (uint32_t dimension = 0; dimension < dimensions; ++dimension)
        Require(std::isfinite(biases[dimension]), "Inference biases must be finite");
    CheckBuffer(bins, binsCount, uint64_t(p.rows) * p.features, "bins");
    CheckBuffer(depths, depthsCount, p.trees, "depths");
    CheckBuffer(features, featuresCount, uint64_t(p.trees) * p.split_stride, "split_features");
    CheckBuffer(borders, bordersCount, uint64_t(p.trees) * p.split_stride, "split_bins");
    CheckBuffer(types, typesCount, uint64_t(p.trees) * p.split_stride, "split_types");
    CheckBuffer(leaves, leavesCount, uint64_t(p.trees) * p.leaf_stride * dimensions, "leaf_values");
    CheckBuffer(predictions, predictionsCount, uint64_t(p.rows) * dimensions, "predictions");
    const uint64_t modelBytes = depthsCount * 4 + featuresCount * 9 + leavesCount * 8;
    Require(modelBytes <= MaxBytes && uint64_t(p.rows) * dimensions * 8 <= MaxBytes,
            "Inference model or output exceeds the 1 GiB memory limit");
    Require(featuresCount <= UINT32_MAX && leavesCount <= UINT32_MAX,
            "Inference model exceeds the GPU index limit");
    double maximumLeafMagnitude = 0;
    for (uint32_t tree = 0; tree < p.trees; ++tree) {
        Require(depths[tree] <= p.split_stride && depths[tree] <= 16
                && (uint32_t(1) << depths[tree]) <= p.leaf_stride, "Invalid actual tree depth");
        for (uint32_t depth = 0; depth < depths[tree]; ++depth) {
            uint64_t index = uint64_t(tree) * p.split_stride + depth;
            Require(features[index] < p.features && types[index] <= 1
                    && borders[index] < (types[index] == 1 ? 256 : 255), "Invalid active split index");
        }
        double treeMagnitude = 0;
        for (uint32_t leaf = 0; leaf < p.leaf_stride; ++leaf) {
            for (uint32_t dimension = 0; dimension < dimensions; ++dimension) {
                double value = leaves[(uint64_t(tree) * p.leaf_stride + leaf) * dimensions + dimension];
                Require(std::isfinite(value), "Inference leaf values must be finite");
                if (leaf < (uint32_t(1) << depths[tree])) treeMagnitude = std::max(treeMagnitude, std::abs(value));
            }
        }
        if (tree >= p.tree_start && tree < p.tree_end) maximumLeafMagnitude += treeMagnitude;
    }
    // Bound every possible partial sum so compensated float operations cannot
    // overflow, including a model whose final prediction happens to cancel.
    Require(maximumLeafMagnitude <= std::numeric_limits<float>::max() / 4,
            "Inference leaf magnitudes exceed the compensated float range");
    if (!p.rows) return;
    if (p.tree_start == p.tree_end) {
        for (uint32_t row = 0; row < p.rows; ++row)
            for (uint32_t dimension = 0; dimension < dimensions; ++dimension)
                predictions[uint64_t(row) * dimensions + dimension] = p.tree_start == 0 ? biases[dimension] : 0.0;
        return;
    }
    Runtime& runtime = GetRuntime();
    CopyText(stats.device_name, sizeof(stats.device_name), runtime.Device.name.UTF8String);
    std::vector<Pair> pairs(leavesCount);
    for (uint64_t index = 0; index < leavesCount; ++index) {
        double value = leaves[index];
        Require(std::abs(value) <= std::numeric_limits<float>::max() / 4,
                "Inference leaf value exceeds the compensated float range");
        pairs[index].High = static_cast<float>(value);
        pairs[index].Low = static_cast<float>(value - double(pairs[index].High));
    }
    id<MTLBuffer> depthBuffer = runtime.Buffer(depthsCount * 4, depths);
    id<MTLBuffer> featureBuffer = runtime.Buffer(featuresCount * 4, features);
    id<MTLBuffer> borderBuffer = runtime.Buffer(bordersCount * 4, borders);
    id<MTLBuffer> typeBuffer = runtime.Buffer(typesCount, types);
    id<MTLBuffer> leafBuffer = runtime.Buffer(leavesCount * sizeof(Pair), pairs.data());
    const uint32_t tiles = (p.tree_end - p.tree_start + TreesPerTile - 1) / TreesPerTile;
    const uint64_t bytesPerRow = uint64_t(p.features) + uint64_t(tiles + 1) * dimensions * sizeof(Pair);
    Require(modelBytes + bytesPerRow <= MaxBytes, "Inference model leaves no room for row buffers");
    const uint32_t batchRows = std::min<uint64_t>({p.rows, p.batch_rows,
        (MaxBytes - modelBytes) / bytesPerRow, UINT32_MAX / std::max<uint64_t>(p.features, uint64_t(tiles) * dimensions)});
    Require(batchRows > 0, "Inference batch cannot fit in GPU memory");
    for (uint64_t offset = 0; offset < p.rows; offset += batchRows) {
        @autoreleasepool {
            uint32_t count = std::min<uint64_t>(batchRows, uint64_t(p.rows) - offset);
            id<MTLBuffer> binBuffer = runtime.Buffer(uint64_t(count) * p.features);
            auto* destination = static_cast<uint8_t*>(binBuffer.contents);
            for (uint32_t feature = 0; feature < p.features; ++feature) {
                std::memcpy(destination + uint64_t(feature) * count,
                            bins + uint64_t(feature) * p.rows + offset, count);
            }
            id<MTLBuffer> partials = runtime.Buffer(uint64_t(count) * tiles * dimensions * sizeof(Pair));
            id<MTLBuffer> output = runtime.Buffer(uint64_t(count) * dimensions * sizeof(Pair));
            KernelParams parameters{count, p.split_stride, p.leaf_stride,
                p.tree_start, p.tree_end, tiles, TreesPerTile, dimensions};
            id<MTLCommandBuffer> command = [runtime.Queue commandBuffer];
            Require(command != nil, "Could not create inference command buffer");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            Require(encoder != nil, "Could not create inference encoder");
            [encoder setComputePipelineState:runtime.Evaluate];
            [encoder setBuffer:binBuffer offset:0 atIndex:0];
            [encoder setBuffer:depthBuffer offset:0 atIndex:1];
            [encoder setBuffer:featureBuffer offset:0 atIndex:2];
            [encoder setBuffer:borderBuffer offset:0 atIndex:3];
            [encoder setBuffer:leafBuffer offset:0 atIndex:4];
            [encoder setBuffer:partials offset:0 atIndex:5];
            [encoder setBuffer:typeBuffer offset:0 atIndex:6];
            [encoder setBytes:&parameters length:sizeof(parameters) atIndex:7];
            NSUInteger width = std::min<NSUInteger>(256, runtime.Evaluate.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, tiles, dimensions)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            encoder = [command computeCommandEncoder];
            Require(encoder != nil, "Could not create inference reduction encoder");
            [encoder setComputePipelineState:runtime.Reduce];
            [encoder setBuffer:partials offset:0 atIndex:0];
            [encoder setBuffer:output offset:0 atIndex:1];
            [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
            width = std::min<NSUInteger>(256, runtime.Reduce.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, dimensions, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted,
                    "Metal inference failed: " + ErrorText(command.error));
            stats.kernel_dispatches += 2;
            if (std::isfinite(command.GPUStartTime) && std::isfinite(command.GPUEndTime)
                    && command.GPUStartTime > 0 && command.GPUEndTime >= command.GPUStartTime) {
                stats.gpu_seconds += command.GPUEndTime - command.GPUStartTime;
            }
            const auto* values = static_cast<const Pair*>(output.contents);
            for (uint32_t row = 0; row < count; ++row) {
                for (uint32_t dimension = 0; dimension < dimensions; ++dimension) {
                    const auto value = values[uint64_t(row) * dimensions + dimension];
                    const double sum = double(value.High) + double(value.Low);
                    // Reconstruct on the host and apply float64 model scale/bias,
                    // preserving CatBoost's public double prediction interface.
                    auto& result = predictions[(offset + row) * dimensions + dimension];
                    result = p.scale * sum + (p.tree_start == 0 ? biases[dimension] : 0.0);
                    Require(std::isfinite(result), "Inference result overflowed");
                }
            }
        }
    }
}
void PredictNonSymmetric(const CBMInferenceParams& p, uint32_t dimensions,
    const double* biases, uint64_t biasesCount, const uint8_t* bins, uint64_t binsCount,
    const uint32_t* roots, uint64_t rootsCount, const CBMInferenceNode* nodes, uint64_t nodesCount,
    const double* leaves, uint64_t leavesCount, double* predictions, uint64_t predictionsCount,
    CBMInferenceStats& stats) {
    static_assert(sizeof(CBMInferenceNode) == 24, "Variable-tree node ABI mismatch");
    Require(p.trees <= 1000000 && !p.split_stride && !p.leaf_stride,
        "Non-symmetric inference requires compact tree storage with zero strides");
    Require(dimensions >= 1 && dimensions <= 64 && p.rows <= (1u << 27), "Invalid variable-tree dimensions");
    Require(p.tree_start <= p.tree_end && p.tree_end <= p.trees && p.batch_rows > 0,
        "Invalid variable-tree range or batch size");
    Require(std::isfinite(p.scale) && p.bias == 0, "Variable-tree inference uses a finite scale and explicit biases");
    CheckBuffer(biases, biasesCount, dimensions, "biases");
    CheckBuffer(bins, binsCount, uint64_t(p.rows) * p.features, "bins");
    CheckBuffer(roots, rootsCount, p.trees, "roots");
    Require(nodesCount <= UINT32_MAX && leavesCount <= UINT32_MAX && leavesCount % dimensions == 0,
        "Variable-tree storage exceeds the GPU index limit or has an incomplete leaf");
    CheckBuffer(nodes, nodesCount, nodesCount, "nodes");
    CheckBuffer(leaves, leavesCount, leavesCount, "leaf_values");
    CheckBuffer(predictions, predictionsCount, uint64_t(p.rows) * dimensions, "predictions");
    const uint64_t modelBytes = rootsCount * 4 + nodesCount * sizeof(CBMInferenceNode) + leavesCount * sizeof(Pair);
    Require(modelBytes <= MaxBytes && predictionsCount * 8 <= MaxBytes && binsCount <= MaxBytes,
        "Variable-tree model, input or output exceeds the 1 GiB memory limit");
    for (uint32_t d = 0; d < dimensions; ++d) Require(std::isfinite(biases[d]), "Nonfinite inference bias");
    for (uint64_t value = 0; value < leavesCount; ++value)
        Require(std::isfinite(leaves[value]) && std::abs(leaves[value]) <= std::numeric_limits<float>::max() / 4,
            "Variable-tree leaf exceeds the compensated float range");
    std::vector<uint8_t> visited(nodesCount, 0);
    std::vector<std::pair<uint32_t, uint32_t>> pending;
    double maximumMagnitude = 0;
    for (uint32_t tree = 0; tree < p.trees; ++tree) {
        pending.emplace_back(roots[tree], 0);
        double treeMagnitude = 0;
        while (!pending.empty()) {
            const auto [index, depth] = pending.back(); pending.pop_back();
            Require(index < nodesCount && !visited[index] && depth <= 65535,
                "Variable-tree graph has a cycle, shared node, invalid index or excessive depth");
            visited[index] = 1;
            const auto& node = nodes[index];
            if (node.leaf == UINT32_MAX) {
                Require(node.feature < p.features && node.type <= 1 && node.bin < (node.type ? 256u : 255u),
                    "Invalid active variable-tree split");
                Require(depth < 65535, "Variable-tree depth exceeds 65535");
                pending.emplace_back(node.right, depth + 1); pending.emplace_back(node.left, depth + 1);
            } else {
                Require(uint64_t(node.leaf) * dimensions + dimensions <= leavesCount, "Invalid variable-tree leaf index");
                for (uint32_t d = 0; d < dimensions; ++d)
                    treeMagnitude = std::max(treeMagnitude, std::abs(leaves[uint64_t(node.leaf) * dimensions + d]));
            }
        }
        if (tree >= p.tree_start && tree < p.tree_end) maximumMagnitude += treeMagnitude;
    }
    Require(std::all_of(visited.begin(), visited.end(), [](uint8_t value) { return value != 0; }),
        "Variable-tree storage contains unreachable nodes");
    Require(maximumMagnitude <= std::numeric_limits<float>::max() / 4,
        "Variable-tree partial sums exceed the compensated float range");
    if (!p.rows) return;
    if (p.tree_start == p.tree_end) {
        for (uint32_t row = 0; row < p.rows; ++row)
            for (uint32_t d = 0; d < dimensions; ++d)
                predictions[uint64_t(row) * dimensions + d] = p.tree_start == 0 ? biases[d] : 0;
        return;
    }
    Runtime& runtime = GetRuntime();
    CopyText(stats.device_name, sizeof(stats.device_name), runtime.Device.name.UTF8String);
    std::vector<Pair> pairs(leavesCount);
    for (uint64_t value = 0; value < leavesCount; ++value) {
        pairs[value].High = static_cast<float>(leaves[value]);
        pairs[value].Low = static_cast<float>(leaves[value] - double(pairs[value].High));
    }
    auto rootBuffer = runtime.Buffer(rootsCount * 4, roots);
    auto nodeBuffer = runtime.Buffer(nodesCount * sizeof(CBMInferenceNode), nodes);
    auto leafBuffer = runtime.Buffer(leavesCount * sizeof(Pair), pairs.data());
    const uint32_t tiles = (p.tree_end - p.tree_start + TreesPerTile - 1) / TreesPerTile;
    const uint64_t rowBytes = uint64_t(p.features) + uint64_t(tiles + 1) * dimensions * sizeof(Pair);
    Require(modelBytes + rowBytes <= MaxBytes, "Variable-tree model leaves no row workspace");
    const uint32_t batchRows = std::min<uint64_t>({p.rows, p.batch_rows, (MaxBytes - modelBytes) / rowBytes,
        UINT32_MAX / std::max<uint64_t>(p.features, uint64_t(tiles) * dimensions)});
    Require(batchRows > 0, "Variable-tree batch cannot fit in GPU memory");
    for (uint64_t offset = 0; offset < p.rows; offset += batchRows) {
        @autoreleasepool {
            const uint32_t count = std::min<uint64_t>(batchRows, uint64_t(p.rows) - offset);
            auto binBuffer = runtime.Buffer(uint64_t(count) * p.features);
            auto* destination = static_cast<uint8_t*>(binBuffer.contents);
            for (uint32_t feature = 0; feature < p.features; ++feature)
                std::memcpy(destination + uint64_t(feature) * count, bins + uint64_t(feature) * p.rows + offset, count);
            auto partials = runtime.Buffer(uint64_t(count) * tiles * dimensions * sizeof(Pair));
            auto output = runtime.Buffer(uint64_t(count) * dimensions * sizeof(Pair));
            KernelParams params{count, 0, 0, p.tree_start, p.tree_end, tiles, TreesPerTile, dimensions};
            auto command = [runtime.Queue commandBuffer];
            Require(command != nil, "Could not create variable-tree command buffer");
            auto encoder = [command computeCommandEncoder];
            Require(encoder != nil, "Could not create variable-tree encoder");
            [encoder setComputePipelineState:runtime.EvaluateNonSymmetric];
            [encoder setBuffer:binBuffer offset:0 atIndex:0];
            [encoder setBuffer:rootBuffer offset:0 atIndex:1];
            [encoder setBuffer:nodeBuffer offset:0 atIndex:2];
            [encoder setBuffer:leafBuffer offset:0 atIndex:3];
            [encoder setBuffer:partials offset:0 atIndex:4];
            [encoder setBytes:&params length:sizeof(params) atIndex:5];
            const NSUInteger width = std::min<NSUInteger>(256, runtime.EvaluateNonSymmetric.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, tiles, dimensions) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            encoder = [command computeCommandEncoder];
            Require(encoder != nil, "Could not create variable-tree reduction encoder");
            [encoder setComputePipelineState:runtime.Reduce];
            [encoder setBuffer:partials offset:0 atIndex:0];
            [encoder setBuffer:output offset:0 atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(count, dimensions, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            [command commit]; [command waitUntilCompleted];
            Require(command.status == MTLCommandBufferStatusCompleted, "Variable-tree inference failed: " + ErrorText(command.error));
            stats.kernel_dispatches += 2;
            if (std::isfinite(command.GPUStartTime) && std::isfinite(command.GPUEndTime) &&
                command.GPUStartTime > 0 && command.GPUEndTime >= command.GPUStartTime)
                stats.gpu_seconds += command.GPUEndTime - command.GPUStartTime;
            const auto* values = static_cast<const Pair*>(output.contents);
            for (uint32_t row = 0; row < count; ++row) {
                for (uint32_t d = 0; d < dimensions; ++d) {
                    const Pair value = values[uint64_t(row) * dimensions + d];
                    double& result = predictions[(offset + row) * dimensions + d];
                    result = p.scale * (double(value.High) + double(value.Low)) + (p.tree_start == 0 ? biases[d] : 0);
                    Require(std::isfinite(result), "Variable-tree inference result overflowed");
                }
            }
        }
    }
}
} // namespace

extern "C" int cbm_predict_non_symmetric_bins_multidim(
    const CBMInferenceParams* params, uint32_t dimensions, const double* biases, uint64_t biasesCount,
    const uint8_t* bins, uint64_t binsCount, const uint32_t* roots, uint64_t rootsCount,
    const CBMInferenceNode* nodes, uint64_t nodesCount, const double* leaves, uint64_t leavesCount,
    double* predictions, uint64_t predictionsCount, CBMInferenceStats* stats, char* error, size_t errorCapacity) {
    @autoreleasepool {
        try {
            Require(params && stats, "Inference parameters and statistics must not be null");
            std::memset(stats, 0, sizeof(*stats));
            PredictNonSymmetric(*params, dimensions, biases, biasesCount, bins, binsCount, roots, rootsCount,
                nodes, nodesCount, leaves, leavesCount, predictions, predictionsCount, *stats);
            return 0;
        } catch (const std::exception& exception) {
            CopyText(error, errorCapacity, exception.what());
            return 1;
        } catch (...) {
            CopyText(error, errorCapacity, "Unknown variable-tree inference failure");
            return 1;
        }
    }
}

extern "C" int cbm_predict_bins(
    const CBMInferenceParams* params, const uint8_t* bins, uint64_t binsCount,
    const uint32_t* depths, uint64_t depthsCount,
    const uint32_t* features, uint64_t featuresCount,
    const uint32_t* borders, uint64_t bordersCount,
    const uint8_t* types, uint64_t typesCount,
    const double* leaves, uint64_t leavesCount,
    double* predictions, uint64_t predictionsCount,
    CBMInferenceStats* stats, char* error, size_t errorCapacity) {
    @autoreleasepool {
        try {
            Require(params && stats, "Inference parameters and statistics must not be null");
            std::memset(stats, 0, sizeof(*stats));
            Predict(*params, 1, &params->bias, 1, bins, binsCount, depths, depthsCount, features, featuresCount,
                    borders, bordersCount, types, typesCount, leaves, leavesCount,
                    predictions, predictionsCount, *stats);
            return 0;
        } catch (const std::exception& exception) {
            CopyText(error, errorCapacity, exception.what());
            return 1;
        } catch (...) {
            CopyText(error, errorCapacity, "Unknown Metal inference failure");
            return 1;
        }
    }
}

extern "C" int cbm_predict_bins_multidim(
    const CBMInferenceParams* params, uint32_t dimensions, const double* biases, uint64_t biasesCount,
    const uint8_t* bins, uint64_t binsCount, const uint32_t* depths, uint64_t depthsCount,
    const uint32_t* features, uint64_t featuresCount, const uint32_t* borders, uint64_t bordersCount,
    const uint8_t* types, uint64_t typesCount, const double* leaves, uint64_t leavesCount,
    double* predictions, uint64_t predictionsCount, CBMInferenceStats* stats, char* error, size_t errorCapacity) {
    @autoreleasepool {
        try {
            Require(params && stats, "Inference parameters and statistics must not be null");
            std::memset(stats, 0, sizeof(*stats));
            Require(params->bias == 0.0, "Multidimensional inference uses the explicit bias vector");
            Predict(*params, dimensions, biases, biasesCount, bins, binsCount, depths, depthsCount,
                    features, featuresCount, borders, bordersCount, types, typesCount, leaves, leavesCount,
                    predictions, predictionsCount, *stats);
            return 0;
        } catch (const std::exception& exception) {
            CopyText(error, errorCapacity, exception.what());
            return 1;
        } catch (...) {
            CopyText(error, errorCapacity, "Unknown multidimensional Metal inference failure");
            return 1;
        }
    }
}
