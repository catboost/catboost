#pragma once

// Resident YetiRank target. The trainer supplies every oracle seed explicitly,
// so derivative calls consume no hidden RNG and can be replayed from a snapshot.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_yeti_rank_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static const char* CBMMetalYetiRankBridge = R"METAL(
struct YetiPointParams { uint rows, leaves, apply_shift, reserved; };

kernel void PrepareValidatedYetiPoint(const device float* cursor [[buffer(0)]],
    const device float* leaves [[buffer(1)]], const device uint* leaf_ids [[buffer(2)]],
    const device float* targets [[buffer(3)]], const device float* weights [[buffer(4)]],
    device float* point [[buffer(5)]], device atomic_uint* status [[buffer(6)]],
    constant YetiPointParams& p [[buffer(7)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const bool valid_leaf = !p.apply_shift || leaf_ids[row] < p.leaves;
    float value = cursor[row];
    if (p.apply_shift && valid_leaf) value += leaves[leaf_ids[row]];
    const bool valid = valid_leaf && isfinite(value) && isfinite(targets[row]) &&
        isfinite(weights[row]) && weights[row] >= 0.0f && isfinite(targets[row] * weights[row]);
    if (!valid) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    point[row] = valid ? value : 0.0f;
}

kernel void PublishYetiPointDerivatives(const device float2* derivatives [[buffer(0)]],
    device float* gradient [[buffer(1)]], device float* incident_mass [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant YetiPointParams& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const float2 value = derivatives[row];
    const bool valid = all(isfinite(value)) && value.y >= 0.0f;
    if (!valid) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    gradient[row] = valid ? value.x : 0.0f;
    incident_mass[row] = valid ? value.y : 0.0f;
}

// CUDA NeedZeroAverage/MakeEstimationResult centers ALL leaf coordinates,
// including empty leaves, after the leaf walk and before learning-rate scaling.
kernel void CenterYetiLeafValues(device float* values [[buffer(0)]],
    device atomic_uint* status [[buffer(1)]], constant YetiPointParams& p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float partials[256];
    float sum = 0.0f, correction = 0.0f;
    for (uint leaf = tid; leaf < p.leaves; leaf += 256) {
        const float value = values[leaf];
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
        const float adjusted = value / float(p.leaves) - correction;
        const float next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128; width; width >>= 1) {
        if (tid < width) partials[tid] += partials[tid + width];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = partials[0];
    for (uint leaf = tid; leaf < p.leaves; leaf += 256) {
        const float value = values[leaf] - mean;
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
        values[leaf] = value;
    }
}
)METAL";

class CBMYetiRankRuntime {
public:
    CBMYetiRankRuntime(id<MTLDevice> device, uint32_t rows, uint32_t groups,
        const uint32_t* offsets, uint32_t permutations, float decay,
        bool legacyPrefixCentering = false)
        : Device(device), Rows(rows), Groups(groups), Permutations(permutations), Decay(decay),
          CenterRows(legacyPrefixCentering ? groups : rows), LegacyPrefixCentering(legacyPrefixCentering) {
        Require(Device != nil, "YetiRank requires a Metal device");
        Require(rows && rows <= (1u << 24) && groups && groups <= rows && offsets,
            "Invalid YetiRank rows or query count");
        Require(permutations && permutations <= 10000 && std::isfinite(decay) && decay >= 0 && decay <= 1,
            "Invalid YetiRank permutations or decay");
        Require(offsets[0] == 0 && offsets[groups] == rows, "YetiRank query offsets must span all rows");
        // Check the conservative target-only workspace before staging metadata.
        Require(24ull * rows + 12ull * groups + 8 <= (1ull << 30),
            "YetiRank target exceeds the 1 GiB workspace guard");
        std::vector<uint32_t> ids(rows), tasks;
        for (uint32_t q = 0; q < groups; ++q) {
            Require(offsets[q] < offsets[q + 1] && offsets[q + 1] <= rows &&
                offsets[q + 1] - offsets[q] <= 1023, "YetiRank requires 1 to 1023 rows per query");
            std::fill(ids.begin() + offsets[q], ids.begin() + offsets[q + 1], q);
        }
        for (uint32_t q = 0; q < groups;) {
            const uint32_t limit = std::min(rows, offsets[q] + 1024);
            const uint32_t end = limit == rows ? groups : ids[limit];
            Require(end > q, "Invalid YetiRank task packing");
            tasks.push_back(q); tasks.push_back(end); q = end;
        }
        Tasks = tasks.size() / 2;
        Offsets = Buffer(offsets, 4ull * (groups + 1));
        QueryIds = Buffer(ids.data(), 4ull * rows);
        TaskRanges = Buffer(tasks.data(), 8ull * Tasks);
        Point = Buffer(nullptr, 4ull * rows);
        Exponents = Buffer(nullptr, 8ull * rows);
        Derivatives = Buffer(nullptr, 8ull * rows);
        Status = Buffer(nullptr, 4);
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        options.fastMathEnabled = NO;
        NSError* error = nil;
        Library = [Device newLibraryWithSource:[NSString stringWithFormat:@"%s\n%s",
            CBMMetalYetiRankSource, CBMMetalYetiRankBridge] options:options error:&error];
        Require(Library != nil, ErrorText(error, "YetiRank Metal source compilation failed"));
        for (const char* name : {"PrepareValidatedYetiPoint", "PrepareYetiRankApprox",
                "YetiRankPointwise", "PublishYetiPointDerivatives", "CenterYetiLeafValues"}) {
            auto function = [Library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing YetiRank function: ") + name);
            auto pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline != nil, ErrorText(error, "YetiRank pipeline compilation failed"));
            Require(pipeline.maxTotalThreadsPerThreadgroup >= 256,
                "YetiRank pipeline requires 256 threads per group");
            Pipelines.emplace(name, pipeline);
        }
    }

    uint64_t AllocatedBytes() const { return Bytes; }
    uint32_t TaskCount() const { return Tasks; }
    bool UsesLegacyPrefixCentering() const { return LegacyPrefixCentering; }

    // Only call these host methods while no command using this target is in
    // flight. Status remains sticky across calls until explicitly cleared.
    void ClearStatus() { *static_cast<uint32_t*>(Status.contents) = 0; }
    void CheckStatus() const {
        const auto value = *static_cast<const uint32_t*>(Status.contents);
        Require(value == 0, "Invalid YetiRank GPU point or statistics (status " + std::to_string(value) + ")");
    }

    void EncodePointDerivatives(id<MTLCommandBuffer> command, id<MTLBuffer> cursor,
        id<MTLBuffer> rawLeaves, id<MTLBuffer> leafIds, uint32_t leaves, bool applyShift,
        id<MTLBuffer> targets, id<MTLBuffer> originalWeights, id<MTLBuffer> gradients,
        id<MTLBuffer> incidentMass, uint64_t seed, uint64_t* dispatches = nullptr) {
        Require(command && command.device == Device && command.status == MTLCommandBufferStatusNotEnqueued,
            "YetiRank requires an uncommitted command buffer on its Metal device");
        Require(leaves && leaves <= 65536, "Invalid YetiRank active leaf count");
        CheckBuffer(rawLeaves, 4ull * leaves);
        for (auto buffer : {cursor, leafIds, targets, originalWeights, gradients, incidentMass})
            CheckBuffer(buffer, 4ull * Rows);
        Require(gradients != incidentMass, "YetiRank derivative outputs cannot alias");
        for (auto input : {cursor, rawLeaves, leafIds, targets, originalWeights})
            Require(input != gradients && input != incidentMass, "YetiRank outputs cannot alias source buffers");
        const PointParams point = {Rows, leaves, uint32_t(applyShift), 0};
        const RankParams rank = {Rows, Groups, Tasks, Permutations, uint32_t(seed), uint32_t(seed >> 32), Decay, CenterRows};
        Dispatch(command, "PrepareValidatedYetiPoint", {cursor, rawLeaves, leafIds, targets, originalWeights, Point, Status},
            point, Rows, false, dispatches);
        Dispatch(command, "PrepareYetiRankApprox", {Point, Offsets, Exponents}, rank, Groups, true, dispatches);
        Dispatch(command, "YetiRankPointwise", {Exponents, targets, originalWeights, QueryIds, Offsets, TaskRanges, Derivatives},
            rank, Tasks, true, dispatches);
        Dispatch(command, "PublishYetiPointDerivatives", {Derivatives, gradients, incidentMass, Status},
            point, Rows, false, dispatches);
    }

    void EncodeCenterLeafValues(id<MTLCommandBuffer> command, id<MTLBuffer> rawLeaves,
        uint32_t leaves, uint64_t* dispatches = nullptr) {
        Require(command && command.device == Device && command.status == MTLCommandBufferStatusNotEnqueued,
            "YetiRank requires an uncommitted command buffer on its Metal device");
        Require(leaves && leaves <= 65536, "Invalid YetiRank active leaf count");
        CheckBuffer(rawLeaves, 4ull * leaves);
        Dispatch(command, "CenterYetiLeafValues", {rawLeaves, Status}, PointParams{Rows, leaves, 0, 0},
            1, true, dispatches);
    }

private:
    struct PointParams { uint32_t Rows, Leaves, ApplyShift, Reserved; };
    struct RankParams {
        uint32_t Rows, Groups, Tasks, Permutations, SeedLow, SeedHigh;
        float Decay; uint32_t CenterRows;
    };
    static_assert(sizeof(PointParams) == 16 && sizeof(RankParams) == 32);
    id<MTLDevice> Device;
    id<MTLLibrary> Library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    uint32_t Rows, Groups, Permutations, Tasks = 0;
    float Decay;
    uint32_t CenterRows;
    bool LegacyPrefixCentering;
    uint64_t Bytes = 0;
    id<MTLBuffer> Offsets, QueryIds, TaskRanges, Point, Exponents, Derivatives, Status;
    static void Require(bool condition, const std::string& message) {
        if (!condition) throw std::runtime_error(message);
    }
    // Literal validation messages allocate only when a check fails.
    static void Require(bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    }
    static std::string ErrorText(NSError* error, const char* fallback) {
        return error ? std::string([[error localizedDescription] UTF8String]) : std::string(fallback);
    }
    void CheckBuffer(id<MTLBuffer> buffer, uint64_t bytes) const {
        Require(buffer && buffer.device == Device && buffer.length >= bytes, "Invalid YetiRank Metal buffer");
    }
    id<MTLBuffer> Buffer(const void* source, uint64_t bytes) {
        Require(bytes <= Device.maxBufferLength, "YetiRank buffer exceeds the Metal device limit");
        auto result = source ? [Device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Require(result != nil, "YetiRank GPU allocation failed");
        if (!source) std::memset(result.contents, 0, bytes);
        Bytes += bytes;
        return result;
    }
    template<class TParams>
    void Dispatch(id<MTLCommandBuffer> command, const char* name,
        std::initializer_list<id<MTLBuffer>> buffers, const TParams& params,
        uint32_t count, bool grouped, uint64_t* dispatches) {
        auto encoder = [command computeCommandEncoder];
        Require(encoder != nil, "YetiRank compute encoder allocation failed");
        [encoder setComputePipelineState:Pipelines.at(name)];
        NSUInteger index = 0;
        for (auto buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index];
        if (grouped) [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        if (dispatches) ++*dispatches;
    }
};
