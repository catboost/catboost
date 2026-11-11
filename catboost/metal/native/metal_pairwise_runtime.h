#pragma once

// Caller-owned command buffers allow this target to share the persistent
// trainer's transaction, timing, rollback and resource lifetime boundaries.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_pairwise_kernels.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class CBMPairwiseRuntime {
public:
    CBMPairwiseRuntime(id<MTLDevice> device, uint32_t rows, uint32_t pairs,
        const uint32_t* winners, const uint32_t* losers, const float* pairWeights,
        uint32_t maxLeaves, uint32_t lossGroups, uint32_t groupCount = 0,
        const uint32_t* groupOffsets = nullptr)
        : Device(device), Rows(rows), Pairs(pairs), MaxLeaves(maxLeaves), LossGroups(lossGroups) {
        Require(Device != nil, "PairLogit requires a Metal device");
        Require(rows > 0 && rows <= (1u << 24), "Invalid PairLogit row count");
        Require(pairs > 0 && pairs <= std::numeric_limits<uint32_t>::max() / 2,
            "PairLogit requires a nonempty representable supplied pair list");
        Require(winners && losers && pairWeights, "PairLogit supplied pair arrays are required");
        Require(maxLeaves > 0 && maxLeaves <= 65536 && lossGroups > 0 && lossGroups <= 4096,
            "Invalid PairLogit leaf or loss reduction dimensions");
        Require(groupCount <= rows && ((groupCount > 0) == (groupOffsets != nullptr)),
            "PairLogit group count and offsets must be supplied together");
        MeanGroups = (maxLeaves + 255) / 256;
        const uint64_t bytes = 44ull * pairs + 8ull * rows + 4 + 8ull * lossGroups + 8ull * MeanGroups + 8;
        Require(bytes <= (1ull << 30), "PairLogit target exceeds the 1 GiB workspace guard");
        if (groupCount) {
            Require(groupOffsets[0] == 0 && groupOffsets[groupCount] == rows,
                "PairLogit group offsets must span all rows");
            for (uint32_t group = 0; group < groupCount; ++group)
                Require(groupOffsets[group] < groupOffsets[group + 1] && groupOffsets[group + 1] <= rows,
                    "PairLogit group offsets must be strictly increasing");
        }
        std::vector<uint32_t> offsets(rows + 1, 0), incidence(2ull * pairs);
        std::vector<int32_t> signs(2ull * pairs);
        std::vector<double> weightSums(rows, 0.0);
        double pairMass = 0;
        for (uint32_t edge = 0; edge < pairs; ++edge) {
            const uint32_t winner = winners[edge], loser = losers[edge];
            Require(winner < rows && loser < rows && winner != loser, "Invalid PairLogit pair endpoints");
            Require(std::isfinite(pairWeights[edge]) && pairWeights[edge] >= 0,
                "PairLogit pair weights must be finite and nonnegative");
            if (groupCount) {
                const auto winnerGroup = std::upper_bound(groupOffsets, groupOffsets + groupCount + 1, winner);
                const auto loserGroup = std::upper_bound(groupOffsets, groupOffsets + groupCount + 1, loser);
                Require(winnerGroup == loserGroup, "PairLogit pair endpoints must belong to the same group");
            }
            ++offsets[winner + 1]; ++offsets[loser + 1];
            weightSums[winner] += pairWeights[edge]; weightSums[loser] += pairWeights[edge];
            pairMass += pairWeights[edge];
        }
        Require(pairMass > 0 && std::isfinite(pairMass) && 2 * pairMass <= std::numeric_limits<float>::max(),
            "PairLogit requires positive finite total incident weight representable in float32");
        HostIncidentWeights.resize(rows);
        for (uint32_t row = 0; row < rows; ++row) {
            offsets[row + 1] += offsets[row];
            HostIncidentWeights[row] = static_cast<float>(weightSums[row]);
            TotalWeight += HostIncidentWeights[row];
        }
        Require(TotalWeight > 0 && std::isfinite(TotalWeight) && TotalWeight <= std::numeric_limits<float>::max(),
            "Invalid rounded PairLogit incident weight total");
        auto next = offsets;
        for (uint32_t edge = 0; edge < pairs; ++edge) {
            const uint32_t first = next[winners[edge]]++, second = next[losers[edge]]++;
            incidence[first] = incidence[second] = edge;
            signs[first] = 1; signs[second] = -1;
        }
        WinnerBuffer = Buffer(winners, 4ull * pairs);
        LoserBuffer = Buffer(losers, 4ull * pairs);
        WeightBuffer = Buffer(pairWeights, 4ull * pairs);
        OffsetBuffer = Buffer(offsets.data(), 4ull * (rows + 1));
        IncidenceBuffer = Buffer(incidence.data(), 8ull * pairs);
        SignBuffer = Buffer(signs.data(), 8ull * pairs);
        PointBuffer = Buffer(nullptr, 4ull * rows);
        EdgeBuffer = Buffer(nullptr, 16ull * pairs);
        LossBuffer = Buffer(nullptr, 8ull * lossGroups);
        MeanPartials = Buffer(nullptr, 8ull * MeanGroups);
        MeanBuffer = Buffer(nullptr, 4);
        StatusBuffer = Buffer(nullptr, 4);
        ClearStatus();

        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        options.fastMathEnabled = NO;
        NSError* error = nil;
        Library = [Device newLibraryWithSource:[NSString stringWithUTF8String:CBMMetalPairwiseSource]
            options:options error:&error];
        Require(Library != nil, ErrorText(error, "PairLogit Metal source compilation failed"));
        for (const char* name : {"PrepareValidatedPairwisePoint", "PairLogitEdgeDerivatives",
                "ReducePairwiseRows", "ValidatePairwiseStatistics", "ReducePairwiseObjective", "ReducePairwiseObjectiveDifference",
                "ReducePairwiseLeafMean", "FinalizePairwiseLeafMean", "CenterPairwiseLeafValues"}) {
            id<MTLFunction> function = [Library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing PairLogit Metal function: ") + name);
            id<MTLComputePipelineState> pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline != nil, ErrorText(error, "PairLogit Metal pipeline compilation failed"));
            Require(pipeline.maxTotalThreadsPerThreadgroup >= 256,
                "PairLogit Metal pipeline requires 256 threads per group");
            Pipelines.emplace(name, pipeline);
        }
    }

    const std::vector<float>& IncidentWeights() const { return HostIncidentWeights; }
    double TotalIncidentWeight() const { return TotalWeight; }
    uint64_t AllocatedBytes() const { return Bytes; }
    uint32_t PairCount() const { return Pairs; }
    uint32_t LossGroupCount() const { return LossGroups; }

    // Host calls are legal only when no command using this target is running.
    void ClearStatus() { *static_cast<uint32_t*>(StatusBuffer.contents) = 0; }
    void CheckStatus() const {
        const uint32_t status = *static_cast<const uint32_t*>(StatusBuffer.contents);
        Require(status == 0, "Invalid PairLogit GPU point or statistics (status " + std::to_string(status) + ")");
    }

    void EncodePointDerivatives(id<MTLCommandBuffer> command, id<MTLBuffer> cursor,
        id<MTLBuffer> rawLeaves, id<MTLBuffer> leafIds, uint32_t leaves, bool applyShift,
        id<MTLBuffer> gradients, id<MTLBuffer> hessian, id<MTLBuffer> incidentWeights,
        uint64_t* dispatches, bool allowNonfiniteTrial = false) {
        Require(leaves > 0 && leaves <= MaxLeaves, "Invalid PairLogit active leaf count");
        CheckBuffer(cursor, 4ull * Rows); CheckBuffer(rawLeaves, 4ull * leaves);
        CheckBuffer(leafIds, 4ull * Rows); CheckBuffer(gradients, 4ull * Rows);
        CheckBuffer(hessian, 4ull * Rows); CheckBuffer(incidentWeights, 4ull * Rows);
        Params p = Parameters(leaves, applyShift);
        p.Reserved1 = static_cast<uint32_t>(allowNonfiniteTrial);
        Dispatch(command, "PrepareValidatedPairwisePoint", {cursor, rawLeaves, leafIds, PointBuffer, StatusBuffer}, p, Rows, false, dispatches);
        Dispatch(command, "PairLogitEdgeDerivatives", {PointBuffer, WinnerBuffer, LoserBuffer, WeightBuffer, EdgeBuffer}, p, Pairs, false, dispatches);
        Dispatch(command, "ReducePairwiseRows", {OffsetBuffer, IncidenceBuffer, SignBuffer, EdgeBuffer, gradients, hessian, incidentWeights}, p, Rows, true, dispatches);
        Dispatch(command, "ValidatePairwiseStatistics", {EdgeBuffer, gradients, hessian, incidentWeights, StatusBuffer}, p,
            std::max(Rows, Pairs), false, dispatches);
    }

    void EncodeLossReduction(id<MTLCommandBuffer> command, uint64_t* dispatches) {
        Dispatch(command, "ReducePairwiseObjective", {EdgeBuffer, LossBuffer}, Parameters(1, false), LossGroups, true, dispatches);
    }

    void EncodeObjectiveDifference(id<MTLCommandBuffer> command, id<MTLBuffer> cursor,
        id<MTLBuffer> currentLeaves, id<MTLBuffer> trialLeaves, id<MTLBuffer> leafIds,
        uint32_t leaves, uint64_t* dispatches) {
        Require(leaves && leaves <= MaxLeaves, "Invalid PairLogit backtracking leaf count");
        CheckBuffer(cursor, 4ull * Rows); CheckBuffer(currentLeaves, 4ull * leaves);
        CheckBuffer(trialLeaves, 4ull * leaves); CheckBuffer(leafIds, 4ull * Rows);
        Dispatch(command, "ReducePairwiseObjectiveDifference", {cursor, currentLeaves, trialLeaves,
            leafIds, WinnerBuffer, LoserBuffer, WeightBuffer, LossBuffer}, Parameters(leaves, true),
            LossGroups, true, dispatches);
    }

    double ReadObjectiveDifference() const {
        CheckStatus();
        const float* parts = static_cast<const float*>(LossBuffer.contents);
        double value = 0;
        for (uint32_t group = 0; group < LossGroups; ++group)
            value += double(parts[2 * group]) + parts[2 * group + 1];
        return value * Pairs;
    }

    // Values are unnormalized positive loss and supplied edge mass. The metric
    // is first/second; the maximized backtracking oracle is negative first.
    std::array<double, 2> ReadLossPartials(bool allowNonfiniteLoss = false) const {
        CheckStatus();
        std::array<double, 2> result = {0, 0};
        const float* data = static_cast<const float*>(LossBuffer.contents);
        for (uint32_t group = 0; group < LossGroups; ++group) {
            Require((std::isfinite(data[2 * group]) ? data[2 * group] >= 0 : allowNonfiniteLoss)
                    && std::isfinite(data[2 * group + 1]) && data[2 * group + 1] >= 0,
                "Invalid PairLogit GPU loss partial");
            result[0] += data[2 * group]; result[1] += data[2 * group + 1];
        }
        result[0] *= Pairs; result[1] *= Pairs;
        Require((std::isfinite(result[0]) || allowNonfiniteLoss) && std::isfinite(result[1]) && result[1] > 0,
            "Invalid PairLogit GPU loss or zero edge mass");
        return result;
    }

    void EncodeCenterLeafValues(id<MTLCommandBuffer> command, id<MTLBuffer> rawLeaves,
        uint32_t leaves, uint64_t* dispatches) {
        Require(leaves > 0 && leaves <= MaxLeaves, "Invalid PairLogit centering leaf count");
        CheckBuffer(rawLeaves, 4ull * leaves);
        Params p = Parameters(leaves, false);
        p.Reserved0 = (leaves + 255) / 256;
        Dispatch(command, "ReducePairwiseLeafMean", {rawLeaves, MeanPartials, StatusBuffer}, p, p.Reserved0, true, dispatches);
        Dispatch(command, "FinalizePairwiseLeafMean", {MeanPartials, MeanBuffer}, p, 1, true, dispatches);
        Dispatch(command, "CenterPairwiseLeafValues", {rawLeaves, MeanBuffer, StatusBuffer}, p, leaves, false, dispatches);
    }

private:
    struct Params {
        uint32_t Rows, Pairs, Objective, ApplyLeafValues, Leaves, Reserved0, Reserved1, Reserved2;
    };
    static_assert(sizeof(Params) == 32);
    id<MTLDevice> Device;
    id<MTLLibrary> Library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines;
    uint32_t Rows, Pairs, MaxLeaves, LossGroups, MeanGroups;
    uint64_t Bytes = 0;
    double TotalWeight = 0;
    std::vector<float> HostIncidentWeights;
    id<MTLBuffer> WinnerBuffer, LoserBuffer, WeightBuffer, OffsetBuffer, IncidenceBuffer, SignBuffer;
    id<MTLBuffer> PointBuffer, EdgeBuffer, LossBuffer, MeanPartials, MeanBuffer, StatusBuffer;

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
    Params Parameters(uint32_t leaves, bool applyShift) const {
        return {Rows, Pairs, 14, static_cast<uint32_t>(applyShift), leaves, 0, 0, 0};
    }
    void CheckBuffer(id<MTLBuffer> buffer, uint64_t bytes) const {
        Require(buffer != nil && buffer.device == Device && buffer.length >= bytes,
            "Invalid PairLogit input/output Metal buffer");
    }
    id<MTLBuffer> Buffer(const void* source, uint64_t bytes) {
        Require(bytes <= Device.maxBufferLength, "PairLogit buffer exceeds the Metal device limit");
        id<MTLBuffer> result = source ? [Device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Require(result != nil, "PairLogit GPU allocation failed");
        if (!source) std::memset(result.contents, 0, bytes);
        Bytes += bytes;
        return result;
    }
    void Dispatch(id<MTLCommandBuffer> command, const char* name,
        std::initializer_list<id<MTLBuffer>> buffers, const Params& params,
        uint32_t count, bool grouped, uint64_t* dispatches) {
        Require(command != nil && command.device == Device && command.status == MTLCommandBufferStatusNotEnqueued,
            "PairLogit requires an uncommitted command buffer on its Metal device");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        Require(encoder != nil, "PairLogit compute encoder allocation failed");
        [encoder setComputePipelineState:Pipelines.at(name)];
        NSUInteger index = 0;
        for (id<MTLBuffer> buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index];
        if (grouped) [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        if (dispatches) ++*dispatches;
    }
};
