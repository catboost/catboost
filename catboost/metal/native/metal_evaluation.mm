#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_evaluation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
constexpr uint64_t MaxBytes = uint64_t(1) << 30;
constexpr uint32_t MaxRows = uint32_t(1) << 27;
struct KernelParams { uint32_t Rows, Depth, Classes; };
static_assert(sizeof(CBMEvaluationParams) == 16, "Evaluation parameter ABI mismatch");
static_assert(sizeof(CBMEvaluationMatrixParams) == 16, "Matrix evaluation parameter ABI mismatch");
static_assert(sizeof(CBMEvaluationStats) == 304, "Evaluation statistics ABI mismatch");
static_assert(sizeof(KernelParams) == 12, "Evaluation shader ABI mismatch");

const char* Source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint Rows, Depth, Classes; };
kernel void AddEvaluationTree(
    device const uchar* bins [[buffer(0)]],
    device const uint* features [[buffer(1)]],
    device const uint* borders [[buffer(2)]],
    device const uchar* types [[buffer(3)]],
    device const float* leaves [[buffer(4)]],
    device float* predictions [[buffer(5)]],
    constant Params& p [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.Rows * p.Classes) return;
    uint row = index / p.Classes;
    uint column = index % p.Classes;
    uint leaf = 0;
    for (uint level = 0; level < p.Depth; ++level) {
        uint bin = bins[features[level] * p.Rows + row];
        bool right = types[level] == 1 ? bin == borders[level] : bin > borders[level];
        leaf |= uint(right) << level;
    }
    predictions[index] += leaves[leaf * p.Classes + column];
}
)MSL";

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
// Literal validation messages allocate only when a check fails.
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void CopyText(char* destination, size_t capacity, const char* source) {
    if (!destination || !capacity) return;
    const size_t length = std::min(capacity - 1, std::strlen(source));
    std::memcpy(destination, source, length);
    destination[length] = 0;
}
std::string ErrorText(NSError* error) {
    const char* text = error ? error.localizedDescription.UTF8String : nullptr;
    return text ? text : "Unknown Metal error";
}
void CheckBuffer(const void* pointer, uint64_t count, uint64_t required, const char* name) {
    Require(count == required, std::string(name) + " element count does not match its shape");
    Require(!required || pointer, std::string(name) + " is null");
}
uint64_t AllocationBytes(uint64_t bytes) { return std::max<uint64_t>(bytes, 1); }

struct Runtime {
    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    id<MTLComputePipelineState> AddTree;
    Runtime() {
        Device = MTLCreateSystemDefaultDevice();
        Require(Device && Device.hasUnifiedMemory,
                "Metal evaluation requires an Apple unified-memory GPU");
        Queue = [Device newCommandQueue];
        Require(Queue != nil, "Could not create Metal evaluation queue");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [Device newLibraryWithSource:
            [NSString stringWithUTF8String:Source] options:options error:&error];
        Require(library != nil, "Metal evaluation shader compilation failed: " + ErrorText(error));
        AddTree = [Device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"AddEvaluationTree"] error:&error];
        Require(AddTree != nil, "Could not create Metal evaluation pipeline: " + ErrorText(error));
    }
    id<MTLBuffer> Buffer(uint64_t bytes, const void* source = nullptr) {
        const uint64_t allocation = AllocationBytes(bytes);
        Require(allocation <= MaxBytes && allocation <= Device.maxBufferLength,
                "Evaluation buffer exceeds the device memory limit");
        id<MTLBuffer> buffer = source && bytes
            ? [Device newBufferWithBytes:source length:allocation options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:allocation options:MTLResourceStorageModeShared];
        Require(buffer != nil, "Could not allocate Metal evaluation buffer");
        return buffer;
    }
};
Runtime& GetRuntime() { static Runtime runtime; return runtime; }
} // namespace

struct CBMEvaluationSession {
    CBMEvaluationParams Params;
    uint32_t Classes = 1;
    CBMEvaluationStats Stats{};
    id<MTLBuffer> Bins, Predictions, Features, Borders, Types, Leaves;
    std::array<double, 64> MagnitudeBounds{};
    bool Failed = false;
    std::mutex Mutex;
};

namespace {
// The registry validates opaque handles without dereferencing an unknown or
// already-destroyed pointer. A shared owner also makes concurrent close safe.
std::mutex RegistryMutex;
std::unordered_map<CBMEvaluationSession*, std::shared_ptr<CBMEvaluationSession>> Sessions;
std::shared_ptr<CBMEvaluationSession> GetSession(CBMEvaluationSession* handle) {
    std::lock_guard<std::mutex> lock(RegistryMutex);
    auto found = Sessions.find(handle);
    Require(found != Sessions.end(), "Evaluation cursor is closed or invalid");
    return found->second;
}

template <class Operation>
int Checked(char* error, size_t capacity, Operation operation) {
    @autoreleasepool {
        try {
            CopyText(error, capacity, "");
            operation();
            return 0;
        } catch (const std::exception& exception) {
            CopyText(error, capacity, exception.what());
            return 1;
        } catch (...) {
            CopyText(error, capacity, "Unknown Metal evaluation failure");
            return 1;
        }
    }
}

void CopyPredictions(CBMEvaluationSession& session, float* output) {
    Require(!session.Failed, "Evaluation cursor has failed; close it and restore a new cursor");
    if (session.Params.rows) {
        std::memcpy(output, session.Predictions.contents,
                    uint64_t(session.Params.rows) * session.Classes * sizeof(float));
    }
}

void CreateSession(
    const CBMEvaluationParams& p, uint32_t classes,
    const uint8_t* bins, uint64_t binsCount,
    const float* bias, uint64_t biasCount, const float* initial, uint64_t initialCount,
    CBMEvaluationSession** handle) {
        Require(p.max_depth <= 16 && p.rows <= MaxRows, "Invalid evaluation depth or row count");
        Require(classes >= 1 && classes <= 64, "Evaluation classes must be in [1,64]");
        const uint64_t binsBytes = uint64_t(p.rows) * p.features;
        const uint64_t predictionCount = uint64_t(p.rows) * classes;
        Require(binsBytes <= UINT32_MAX && predictionCount <= UINT32_MAX,
                "Evaluation data exceed the GPU index limit");
        CheckBuffer(bins, binsCount, binsBytes, "bins");
        CheckBuffer(bias, biasCount, classes, "bias");
        CheckBuffer(initial, initialCount, initial ? predictionCount : 0, "initial_predictions");
        const uint64_t predictionBytes = predictionCount * sizeof(float);
        const uint64_t splitBytes = uint64_t(p.max_depth) * sizeof(uint32_t);
        const uint64_t leafBytes = (uint64_t(1) << p.max_depth) * classes * sizeof(float);
        const uint64_t totalBytes = AllocationBytes(binsBytes) + AllocationBytes(predictionBytes)
            + AllocationBytes(splitBytes) * 2 + AllocationBytes(p.max_depth) + leafBytes;
        Require(totalBytes <= MaxBytes, "Evaluation cursor exceeds the 1 GiB resident memory limit");
        std::array<double, 64> magnitudes{};
        for (uint32_t column = 0; column < classes; ++column) {
            Require(std::isfinite(bias[column]), "Evaluation bias must be finite");
            if (!initial) magnitudes[column] = std::abs(double(bias[column]));
        }
        for (uint64_t index = 0; index < initialCount; ++index) {
            Require(std::isfinite(initial[index]), "Initial predictions must be finite");
            const uint32_t column = index % classes;
            magnitudes[column] = std::max(magnitudes[column], std::abs(double(initial[index])));
        }
        Runtime& runtime = GetRuntime();
        auto session = std::make_shared<CBMEvaluationSession>();
        session->Params = p;
        session->Classes = classes;
        session->MagnitudeBounds = magnitudes;
        session->Bins = runtime.Buffer(binsBytes, bins);
        session->Predictions = runtime.Buffer(predictionBytes, initial);
        if (!initial && p.rows) {
            auto* cursor = static_cast<float*>(session->Predictions.contents);
            for (uint32_t row = 0; row < p.rows; ++row) {
                std::copy(bias, bias + classes, cursor + uint64_t(row) * classes);
            }
        }
        session->Features = runtime.Buffer(splitBytes);
        session->Borders = runtime.Buffer(splitBytes);
        session->Types = runtime.Buffer(p.max_depth);
        session->Leaves = runtime.Buffer(leafBytes);
        session->Stats.dataset_uploads = 1;
        session->Stats.bins_upload_bytes = binsBytes;
        session->Stats.resident_bytes = totalBytes;
        CopyText(session->Stats.device_name, sizeof(session->Stats.device_name), runtime.Device.name.UTF8String);
        std::lock_guard<std::mutex> lock(RegistryMutex);
        Sessions.emplace(session.get(), session);
        *handle = session.get();
}
} // namespace

extern "C" int cbm_evaluation_create(
    const CBMEvaluationParams* params, const uint8_t* bins, uint64_t binsCount,
    const float* initial, uint64_t initialCount, CBMEvaluationSession** handle,
    char* error, size_t capacity) {
    return Checked(error, capacity, [&] {
        Require(handle != nullptr, "Evaluation session output is null");
        *handle = nullptr;
        Require(params != nullptr, "Evaluation parameters are null");
        CreateSession(*params, 1, bins, binsCount, &params->bias, 1, initial, initialCount, handle);
    });
}

extern "C" int cbm_evaluation_create_matrix(
    const CBMEvaluationMatrixParams* params, const uint8_t* bins, uint64_t binsCount,
    const float* bias, uint64_t biasCount, const float* initial, uint64_t initialCount,
    CBMEvaluationSession** handle, char* error, size_t capacity) {
    return Checked(error, capacity, [&] {
        Require(handle != nullptr, "Evaluation session output is null");
        *handle = nullptr;
        Require(params != nullptr, "Matrix evaluation parameters are null");
        const CBMEvaluationParams scalar{params->rows, params->features, params->max_depth, 0.0f};
        CreateSession(scalar, params->classes, bins, binsCount, bias, biasCount, initial, initialCount, handle);
    });
}

static int AddTree(
    CBMEvaluationSession* handle, uint32_t depth,
    const uint32_t* features, uint64_t featuresCount,
    const uint32_t* borders, uint64_t bordersCount,
    const uint8_t* types, uint64_t typesCount,
    const float* leaves, uint64_t leavesCount,
    float* predictions, uint64_t predictionsCount, bool scalarOnly, char* error, size_t capacity) {
    return Checked(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> lock(session->Mutex);
        Require(!scalarOnly || session->Classes == 1, "Scalar evaluation entry point requires one output dimension");
        Require(!session->Failed, "Evaluation cursor has failed; close it and restore a new cursor");
        Require(depth <= session->Params.max_depth, "Tree depth exceeds evaluation cursor max_depth");
        CheckBuffer(features, featuresCount, depth, "split_features");
        CheckBuffer(borders, bordersCount, depth, "split_bins");
        CheckBuffer(types, typesCount, depth, "split_types");
        CheckBuffer(leaves, leavesCount, (uint64_t(1) << depth) * session->Classes, "leaf_values");
        CheckBuffer(predictions, predictionsCount, uint64_t(session->Params.rows) * session->Classes, "predictions");
        for (uint32_t level = 0; level < depth; ++level) {
            Require(features[level] < session->Params.features && types[level] <= 1
                    && borders[level] < (types[level] == 1 ? 256 : 255), "Invalid active split index");
        }
        std::array<double, 64> bounds{};
        for (uint64_t index = 0; index < leavesCount; ++index) {
            Require(std::isfinite(leaves[index]), "Evaluation leaf values must be finite");
            const uint32_t column = index % session->Classes;
            bounds[column] = std::max(bounds[column], std::abs(double(leaves[index])));
        }
        // Bound all possible paths before mutating GPU state, including a tree
        // whose actual predictions could cancel a large existing cursor value.
        for (uint32_t column = 0; column < session->Classes; ++column) {
            bounds[column] += session->MagnitudeBounds[column];
            Require(bounds[column] <= std::numeric_limits<float>::max(), "Evaluation predictions can overflow float32");
        }
        if (depth) {
            std::memcpy(session->Features.contents, features, featuresCount * sizeof(uint32_t));
            std::memcpy(session->Borders.contents, borders, bordersCount * sizeof(uint32_t));
            std::memcpy(session->Types.contents, types, typesCount);
        }
        std::memcpy(session->Leaves.contents, leaves, leavesCount * sizeof(float));
        if (session->Params.rows) {
            Runtime& runtime = GetRuntime();
            id<MTLCommandBuffer> command = [runtime.Queue commandBuffer];
            Require(command != nil, "Could not create Metal evaluation command");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            Require(encoder != nil, "Could not create Metal evaluation encoder");
            [encoder setComputePipelineState:runtime.AddTree];
            [encoder setBuffer:session->Bins offset:0 atIndex:0];
            [encoder setBuffer:session->Features offset:0 atIndex:1];
            [encoder setBuffer:session->Borders offset:0 atIndex:2];
            [encoder setBuffer:session->Types offset:0 atIndex:3];
            [encoder setBuffer:session->Leaves offset:0 atIndex:4];
            [encoder setBuffer:session->Predictions offset:0 atIndex:5];
            const KernelParams p{session->Params.rows, depth, session->Classes};
            [encoder setBytes:&p length:sizeof(p) atIndex:6];
            const NSUInteger width = std::min<NSUInteger>(256, runtime.AddTree.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(uint64_t(p.Rows) * p.Classes, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                session->Failed = true;
                throw std::runtime_error("Metal evaluation failed: " + ErrorText(command.error));
            }
            ++session->Stats.kernel_dispatches;
            if (std::isfinite(command.GPUStartTime) && std::isfinite(command.GPUEndTime)
                    && command.GPUStartTime > 0 && command.GPUEndTime >= command.GPUStartTime) {
                session->Stats.gpu_seconds += command.GPUEndTime - command.GPUStartTime;
            }
        }
        session->Stats.tree_upload_bytes += uint64_t(depth) * 9 + leavesCount * sizeof(float);
        // Propagate the same round-to-float step as the cursor after each tree;
        // a sum of unrounded magnitudes can underestimate accumulated rounding.
        for (uint32_t column = 0; column < session->Classes; ++column) {
            session->MagnitudeBounds[column] = double(static_cast<float>(bounds[column]));
        }
        CopyPredictions(*session, predictions);
    });
}

extern "C" int cbm_evaluation_add_tree(
    CBMEvaluationSession* handle, uint32_t depth,
    const uint32_t* features, uint64_t featuresCount,
    const uint32_t* borders, uint64_t bordersCount,
    const uint8_t* types, uint64_t typesCount,
    const float* leaves, uint64_t leavesCount,
    float* predictions, uint64_t predictionsCount, char* error, size_t capacity) {
    return AddTree(handle, depth, features, featuresCount, borders, bordersCount, types, typesCount,
                   leaves, leavesCount, predictions, predictionsCount, true, error, capacity);
}

extern "C" int cbm_evaluation_add_tree_matrix(
    CBMEvaluationSession* handle, uint32_t depth,
    const uint32_t* features, uint64_t featuresCount,
    const uint32_t* borders, uint64_t bordersCount,
    const uint8_t* types, uint64_t typesCount,
    const float* leaves, uint64_t leavesCount,
    float* predictions, uint64_t predictionsCount, char* error, size_t capacity) {
    return AddTree(handle, depth, features, featuresCount, borders, bordersCount, types, typesCount,
                   leaves, leavesCount, predictions, predictionsCount, false, error, capacity);
}

static int ReadPredictions(
    CBMEvaluationSession* handle, float* predictions, uint64_t count,
    bool scalarOnly, char* error, size_t capacity) {
    return Checked(error, capacity, [&] {
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> lock(session->Mutex);
        Require(!scalarOnly || session->Classes == 1, "Scalar evaluation entry point requires one output dimension");
        CheckBuffer(predictions, count, uint64_t(session->Params.rows) * session->Classes, "predictions");
        CopyPredictions(*session, predictions);
    });
}

extern "C" int cbm_evaluation_predictions(
    CBMEvaluationSession* handle, float* predictions, uint64_t count, char* error, size_t capacity) {
    return ReadPredictions(handle, predictions, count, true, error, capacity);
}

extern "C" int cbm_evaluation_predictions_matrix(
    CBMEvaluationSession* handle, float* predictions, uint64_t count, char* error, size_t capacity) {
    return ReadPredictions(handle, predictions, count, false, error, capacity);
}

extern "C" int cbm_evaluation_stats(
    CBMEvaluationSession* handle, CBMEvaluationStats* stats, char* error, size_t capacity) {
    return Checked(error, capacity, [&] {
        Require(stats != nullptr, "Evaluation statistics output is null");
        auto session = GetSession(handle);
        std::lock_guard<std::mutex> lock(session->Mutex);
        *stats = session->Stats;
    });
}

extern "C" void cbm_evaluation_destroy(CBMEvaluationSession* handle) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(RegistryMutex);
        Sessions.erase(handle);
    }
}
