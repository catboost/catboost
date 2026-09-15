#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_combination.h"
#include "metal_bootstrap_kernels.h"
#include "metal_kernels.h"
#include "metal_additional_objective_kernels.h"
#include "metal_objective_kernels.h"
#include "metal_querywise_kernels.h"
#include "metal_combination_kernels.h"
#include "metal_pairwise_runtime.h"
#include "metal_yeti_rank_runtime.h"
#include <memory>
#include <mutex>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Every operation is encoded into the caller's command buffer. Input object
// weights remain immutable: Combination leaf estimation uses the OUTER target
// weights, while weak GradientAt uses the sum of component-specific weights.
class CBMCombinationRuntime {
public:
    CBMCombinationRuntime(id<MTLDevice> device, uint32_t rows,
        uint32_t count, const CBMCombinationComponent* components,
        uint32_t groups, const uint32_t* offsets, uint32_t pairs,
        const uint32_t* winners, const uint32_t* losers, const float* pairWeights,
        uint32_t maxLeaves, uint32_t lossGroups)
        : Device(device), Rows(rows), Groups(groups), MaxLeaves(maxLeaves), LossGroups(lossGroups) {
        Require(device && rows && rows <= (1u << 24) && count && count <= 128 && components,
                "Combination requires valid Metal device, rows, and 1 to 128 active components");
        Require(maxLeaves && maxLeaves <= 65536 && lossGroups && lossGroups <= 4096,
                "Invalid Combination leaf or reduction dimensions");
        Require(groups <= rows && ((groups != 0) == (offsets != nullptr)),
                "Combination group count and offsets must be supplied together");
        bool needsGroups = false, needsPairs = false;
        for (uint32_t index = 0; index < count; ++index) {
            ValidateComponent(components[index]);
            needsGroups |= components[index].objective >= 12;
            needsPairs |= components[index].objective == 14;
        }
        Require(!needsGroups || (groups && groups <= rows && offsets),
                "Combination query components require complete group offsets");
        Require(!needsPairs || (pairs && winners && losers && pairWeights),
                "Combination PairLogit requires supplied prepared pairs");
        if (groups) {
            Require(offsets && offsets[0] == 0 && offsets[groups] == rows,
                    "Combination group offsets must span every row");
            for (uint32_t group = 0; group < groups; ++group)
                Require(offsets[group] < offsets[group + 1] && offsets[group + 1] <= rows,
                        "Combination groups must be contiguous and nonempty");
        }
        // Bound all shared scratch and every retained metric reduction before
        // allocating. Pair/Yeti runtimes enforce their own target geometry.
        Require(24ull * rows + 8ull * std::max(rows, groups) + 16ull * lossGroups * count
                + 4ull * (groups + 1) + 4 <= (1ull << 30), "Combination scratch exceeds 1 GiB");
        Point = Buffer(nullptr, 4ull * rows);
        ScratchGradient = Buffer(nullptr, 4ull * rows);
        ScratchHessian = Buffer(nullptr, 4ull * rows);
        ScratchWeights = Buffer(nullptr, 4ull * rows);
        RowStats = Buffer(nullptr, 8ull * std::max(rows, groups));
        Status = Buffer(nullptr, 4);
        if (groups) Offsets = Buffer(offsets, 4ull * (groups + 1));
        // Stable partition is CUDA's query-first accumulation order.
        for (uint32_t phase = 0; phase < 2; ++phase) {
            for (uint32_t index = 0; index < count; ++index) {
                const auto& option = components[index];
                if ((option.objective >= 12) != (phase == 0)) continue;
                Component component;
                component.Option = option;
                component.Loss = Buffer(nullptr, 16ull * lossGroups);
                if (option.objective == 14) {
                    component.Pair = std::make_shared<CBMPairwiseRuntime>(device, rows, pairs,
                        winners, losers, pairWeights, maxLeaves, lossGroups, groups, offsets);
                    Bytes += component.Pair->AllocatedBytes();
                }
                if (option.objective == 17) {
                    component.Yeti = std::make_shared<CBMYetiRankRuntime>(device, rows, groups,
                        offsets, option.permutations, option.decay);
                    Bytes += component.Yeti->AllocatedBytes();
                    ++YetiCount;
                }
                Require(Bytes <= (1ull << 30), "Combination target exceeds 1 GiB");
                Components.emplace_back(std::move(component));
            }
        }
        if (YetiCount) {
            SafeYetiPoint = Buffer(nullptr, 4ull * rows);
            TrialStatus = Buffer(nullptr, 4);
        }
        Shaders = GetShaders(Device);
        ClearStatus();
    }

    uint64_t AllocatedBytes() const { return Bytes; }
    bool HasYeti() const { return YetiCount != 0; }
    uint32_t YetiComponentCount() const { return YetiCount; }
    uint32_t YetiSeedsRemaining() const { return Seeds.size() - SeedPosition; }
    uint32_t YetiSeedsUsed() const { return SeedPosition; }
    // Native preparation and private constructors call this on copied input
    // observations. A valid global QuerySoftMax target may have zero-mass local
    // Ordered slices, so their helpers pass requirePositiveMetricMass=false.
    void ValidateTargets(const float* targets, const float* weights,
                         bool requirePositiveMetricMass = true) const {
        Require(targets, "Combination targets are required");
        bool crossEntropy = false, nonnegative = false, querySoftmax = false;
        for (const auto& component : Components) {
            crossEntropy |= component.Option.objective == 2;
            nonnegative |= component.Option.objective == 3 || component.Option.objective == 7 || component.Option.objective == 13;
            querySoftmax |= component.Option.objective == 13;
        }
        double weightSum = 0, weightedTarget = 0;
        for (uint32_t row = 0; row < Rows; ++row) {
            const float target = targets[row], weight = weights ? weights[row] : 1;
            Require(std::isfinite(target) && std::isfinite(weight) && weight >= 0,
                    "Combination targets and nonnegative object weights must be finite");
            Require(!crossEntropy || (target >= 0 && target <= 1), "Combination CrossEntropy labels must be in [0, 1]");
            Require(!nonnegative || target >= 0, "Combination Poisson/Tweedie/QuerySoftMax labels must be nonnegative");
            weightSum += weight;
            if (querySoftmax) {
                Require(std::isfinite(target * weight), "Combination QuerySoftMax weighted labels must be representable in float32");
                weightedTarget += double(target) * weight;
            }
        }
        Require(std::isfinite(weightSum) && weightSum <= std::numeric_limits<float>::max()
                && (!requirePositiveMetricMass || weightSum > 0), "Invalid Combination total object weight");
        Require(!querySoftmax || (std::isfinite(weightedTarget) && weightedTarget <= std::numeric_limits<float>::max()
                && (!requirePositiveMetricMass || weightedTarget > 0)),
                "Combination QuerySoftMax requires positive finite representable weighted target mass");
    }
    void SetYetiSeeds(uint32_t count, const uint64_t* seeds) {
        Require(!count || seeds, "Combination Yeti seeds are required");
        Require(!SeedCallback, "Disable Combination seed callback before supplying a packet");
        Require(SeedPosition == Seeds.size(), "Combination previous Yeti seeds remain unconsumed");
        Seeds.clear(); if (count) Seeds.assign(seeds, seeds + count); SeedPosition = 0;
    }
    void DiscardRemainingYetiSeeds() { Seeds.clear(); SeedPosition = 0; }
    void SetYetiSeedCallback(CBMCombinationYetiSeedCallback callback, void* context) {
        Require(SeedPosition == Seeds.size(), "Consume Combination seed packet before configuring callback");
        SeedCallback = callback; SeedContext = context;
    }
    bool HasYetiSeedCallback() const { return SeedCallback != nullptr; }
    void ClearStatus() {
        *static_cast<uint32_t*>(Status.contents) = 0;
        for (auto& component : Components) {
            if (component.Pair) component.Pair->ClearStatus();
            if (component.Yeti) component.Yeti->ClearStatus();
        }
    }
    void CheckStatus() const {
        Require(!*static_cast<const uint32_t*>(Status.contents), "Nonfinite Combination GPU derivatives or weights");
        for (const auto& component : Components) {
            if (component.Pair) component.Pair->CheckStatus();
            if (component.Yeti) component.Yeti->CheckStatus();
        }
    }

    void EncodePointDerivatives(id<MTLCommandBuffer> command, id<MTLBuffer> cursor,
        id<MTLBuffer> rawLeaves, id<MTLBuffer> leafIds, uint32_t leaves, bool applyShift,
        id<MTLBuffer> target, id<MTLBuffer> sampleWeights,
        id<MTLBuffer> gradients, id<MTLBuffer> hessian, id<MTLBuffer> gradientWeights,
        uint64_t* dispatches, bool allowNonfiniteTrial = false) {
        ValidatePoint(command, cursor, rawLeaves, leafIds, leaves, target, sampleWeights);
        for (auto output : {gradients, hessian, gradientWeights}) CheckBuffer(output, 4ull * Rows);
        Require(gradients != hessian && gradients != gradientWeights && hessian != gradientWeights,
                "Combination derivative outputs cannot alias");
        for (auto input : {cursor, rawLeaves, leafIds, target, sampleWeights})
            Require(input != gradients && input != hessian && input != gradientWeights,
                    "Combination outputs cannot alias input buffers");
        auto blit = [command blitCommandEncoder];
        Require(blit != nil, "Could not create Combination reset encoder");
        for (auto output : {gradients, hessian, gradientWeights})
            [blit fillBuffer:output range:NSMakeRange(0, 4ull * Rows) value:0];
        [blit endEncoding];
        EncodePreparedPoint(command, cursor, rawLeaves, leafIds, leaves, applyShift, dispatches, allowNonfiniteTrial);
        for (auto& component : Components) {
            const auto& option = component.Option;
            id<MTLBuffer> weights = sampleWeights;
            if (component.Pair) {
                component.Pair->EncodePointDerivatives(command, cursor, rawLeaves, leafIds, leaves,
                    applyShift, ScratchGradient, ScratchHessian, ScratchWeights, dispatches, allowNonfiniteTrial);
                weights = ScratchWeights;
            } else if (component.Yeti) {
                component.Yeti->EncodePointDerivatives(command, SafeYetiPoint, rawLeaves, leafIds, leaves,
                    false, target, sampleWeights, ScratchGradient, ScratchHessian,
                    NextYetiSeed(), dispatches);
                weights = ScratchHessian;
            } else EncodeAnalyticComponent(command, component, target, sampleWeights, dispatches);
            auto p = Parameters(option);
            Dispatch(command, "AccumulateCombination", {ScratchGradient, ScratchHessian, weights,
                gradients, hessian, gradientWeights}, p, Rows, false, dispatches);
        }
        if (!allowNonfiniteTrial) Dispatch(command, "ValidateCombinationStatistics",
            {gradients, hessian, gradientWeights, Status}, Parameters(Components.front().Option), Rows, false, dispatches);
    }

    // CUDA's leaf walker evaluates value AND first derivatives at every trial,
    // then reuses the accepted derivatives for the next direction. Call this
    // once at the initial point and once per attempted trial for combinations
    // containing Yeti; projecting a cached accepted point must consume no seed.
    void EncodeOracle(id<MTLCommandBuffer> command, id<MTLBuffer> cursor,
        id<MTLBuffer> rawLeaves, id<MTLBuffer> leafIds, uint32_t leaves, bool applyShift,
        id<MTLBuffer> target, id<MTLBuffer> sampleWeights,
        id<MTLBuffer> gradients, id<MTLBuffer> hessian, id<MTLBuffer> gradientWeights,
        uint64_t* dispatches, bool allowNonfiniteTrial = false) {
        EncodePointDerivatives(command, cursor, rawLeaves, leafIds, leaves, applyShift,
            target, sampleWeights, gradients, hessian, gradientWeights, dispatches, allowNonfiniteTrial);
        EncodeLoss(command, cursor, rawLeaves, leafIds, leaves, applyShift,
            target, sampleWeights, dispatches, allowNonfiniteTrial);
    }

    // Loss-only evaluations consume no Yeti randomness: the pinned CUDA
    // YetiRank optimizer value is identically zero. Native metric tracking
    // evaluates PFound/Combination through CatBoost's shared metric evaluator.
    void EncodeLoss(id<MTLCommandBuffer> command, id<MTLBuffer> cursor,
        id<MTLBuffer> rawLeaves, id<MTLBuffer> leafIds, uint32_t leaves, bool applyShift,
        id<MTLBuffer> target, id<MTLBuffer> sampleWeights, uint64_t* dispatches,
        bool allowNonfiniteTrial = false) {
        ValidatePoint(command, cursor, rawLeaves, leafIds, leaves, target, sampleWeights);
        EncodePreparedPoint(command, cursor, rawLeaves, leafIds, leaves, applyShift, dispatches, allowNonfiniteTrial);
        for (auto& component : Components) {
            if (component.Yeti) continue;
            if (component.Pair) {
                component.Pair->EncodePointDerivatives(command, cursor, rawLeaves, leafIds, leaves,
                    applyShift, ScratchGradient, ScratchHessian, ScratchWeights, dispatches, allowNonfiniteTrial);
                component.Pair->EncodeLossReduction(command, dispatches);
            } else {
                EncodeAnalyticComponent(command, component, target, sampleWeights, dispatches);
                auto p = Parameters(component.Option);
                p.Rows = component.Option.objective >= 12 ? Groups : Rows;
                Dispatch(command, "ReduceCombinationStats", {RowStats, component.Loss}, p, LossGroups, true, dispatches);
            }
        }
    }

    // Unnormalized maximized optimizer value, not the sum of final metrics.
    double ReadObjective(bool allowNonfinite = false) const {
        CheckStatus(); double result = 0;
        if (TrialStatus && *static_cast<const uint32_t*>(TrialStatus.contents)) {
            Require(allowNonfinite, "Nonfinite Combination Yeti trial point");
            return -std::numeric_limits<double>::infinity();
        }
        for (const auto& component : Components) {
            if (component.Yeti) continue;
            result -= component.Option.weight * ReadComponentLoss(component, allowNonfinite)[0];
        }
        Require(allowNonfinite || std::isfinite(result), "Nonfinite Combination objective value");
        return result;
    }

    // YetiRank has a zero private optimizer-reporting slot, as in its standalone
    // runtime. Public Combination metrics, including PFound, use shared metrics.
    float ReadMetric() const {
        CheckStatus(); double result = 0;
        for (const auto& component : Components) {
            if (component.Yeti) continue;
            const auto parts = ReadComponentLoss(component, false);
            Require(parts[1] > 0, "Combination component requires positive metric denominator");
            double value = parts[0] / parts[1];
            if (component.Option.objective == 0 || component.Option.objective == 12) value = std::sqrt(value);
            if (component.Option.objective == 10) value *= 2;
            result += component.Option.weight * value;
        }
        Require(std::isfinite(result) && std::isfinite(static_cast<float>(result)), "Nonfinite Combination metric");
        return result;
    }

private:
    struct Params { uint32_t Rows, Objective, ApplyShift, Leaves; float Coefficient, Param, Border; uint32_t LossGroups; };
    struct QueryParams { uint32_t Rows, Groups, Objective, ApplyShift; float Beta, Lambda; uint32_t Leaves, Reserved; };
    struct Component {
        CBMCombinationComponent Option;
        id<MTLBuffer> Loss;
        std::shared_ptr<CBMPairwiseRuntime> Pair;
        std::shared_ptr<CBMYetiRankRuntime> Yeti;
    };
    id<MTLDevice> Device;
    struct ShaderSet { std::unordered_map<std::string, id<MTLComputePipelineState>> Pipelines; };
    std::shared_ptr<ShaderSet> Shaders;
    uint32_t Rows, Groups, MaxLeaves, LossGroups, YetiCount = 0;
    uint64_t Bytes = 0;
    size_t SeedPosition = 0;
    std::vector<uint64_t> Seeds;
    CBMCombinationYetiSeedCallback SeedCallback = nullptr;
    void* SeedContext = nullptr;
    std::vector<Component> Components;
    id<MTLBuffer> Point, Offsets, ScratchGradient, ScratchHessian, ScratchWeights, RowStats, Status;
    id<MTLBuffer> SafeYetiPoint = nil, TrialStatus = nil;

    static void Require(bool condition, const std::string& message) {
        if (!condition) throw std::runtime_error(message);
    }
    uint64_t NextYetiSeed() {
        if (SeedCallback) {
            uint64_t seed = 0;
            Require(SeedCallback(SeedContext, &seed) == 0, "Combination Yeti seed callback failed");
            return seed;
        }
        Require(SeedPosition < Seeds.size(), "Combination Yeti oracle seed schedule is exhausted");
        return Seeds[SeedPosition++];
    }
    static std::string ErrorText(NSError* error, const char* fallback) {
        return error ? std::string([[error localizedDescription] UTF8String]) : std::string(fallback);
    }
    static std::shared_ptr<ShaderSet> GetShaders(id<MTLDevice> device) {
        // Ordered folds share immutable programs, never target data or seeds.
        static std::mutex mutex;
        static std::unordered_map<uint64_t, std::shared_ptr<ShaderSet>> cache;
        std::lock_guard<std::mutex> guard(mutex);
        auto& cached = cache[device.registryID];
        if (cached) return cached;
        auto result = std::make_shared<ShaderSet>();
        MTLCompileOptions* compile = [MTLCompileOptions new];
        compile.languageVersion = MTLLanguageVersion3_0;
        compile.fastMathEnabled = NO;
        NSError* error = nil;
        NSString* source = [NSString stringWithFormat:@"%s\n%s\n%s\n%s\n%s\n%s",
            CBMMetalBootstrapSource, CBMMetalSource, CBMMetalAdditionalObjectiveSource,
            CBMMetalObjectiveSource, CBMMetalQuerywiseSource, CBMMetalCombinationSource];
        auto library = [device newLibraryWithSource:source options:compile error:&error];
        Require(library != nil, ErrorText(error, "Combination Metal compilation failed"));
        for (const char* name : {"PrepareQuerywisePoint", "QueryRmseDerivatives", "QuerySoftMaxDerivatives",
                "CombinationPointwise", "AccumulateCombination", "ReduceCombinationStats", "ValidateCombinationStatistics",
                "PrepareCombinationYetiPoint"}) {
            auto function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            Require(function != nil, std::string("Missing Combination kernel: ") + name);
            auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            Require(pipeline && pipeline.maxTotalThreadsPerThreadgroup >= 256,
                    ErrorText(error, "Combination pipeline requires 256 threads"));
            result->Pipelines.emplace(name, pipeline);
        }
        cached = result;
        return result;
    }
    static void ValidateComponent(const CBMCombinationComponent& c) {
        Require(c.objective <= 14 || c.objective == 17, "Unsupported Combination component objective");
        Require(std::isfinite(c.weight) && c.weight > 0 && std::isfinite(c.param) && std::isfinite(c.border) &&
                std::isfinite(c.beta) && std::isfinite(c.lambda) && std::isfinite(c.decay) &&
                !c.reserved && !c.reserved1[0] && !c.reserved1[1] && !c.reserved1[2],
                "Combination component weights/parameters must be finite and reserved fields zero");
        Require(c.objective != 4 || c.param >= 0, "Combination Huber delta must be nonnegative");
        Require((c.objective != 5 && c.objective != 8 && c.objective != 9) || (c.param >= 0 && c.param <= 1),
                "Combination alpha must be in [0, 1]");
        Require(c.objective != 6 || c.param >= 1, "Combination Lq q must be at least 1");
        Require(c.objective != 7 || (c.param > 1 && c.param < 2), "Combination Tweedie power must be in (1, 2)");
        Require(c.objective != 17 || (c.permutations && c.permutations <= 10000 && c.decay >= 0 && c.decay <= 1),
                "Invalid Combination YetiRank permutations or decay");
    }
    Params Parameters(const CBMCombinationComponent& option) const {
        return {Rows, option.objective, 0, 1, option.objective == 17 ? -option.weight : option.weight,
                option.param, option.border, LossGroups};
    }
    void CheckBuffer(id<MTLBuffer> buffer, uint64_t bytes) const {
        Require(buffer && buffer.device == Device && buffer.length >= bytes, "Invalid Combination Metal buffer");
    }
    id<MTLBuffer> Buffer(const void* data, uint64_t bytes) {
        Require(bytes && bytes <= Device.maxBufferLength && Bytes + bytes <= (1ull << 30), "Combination buffer exceeds memory limit");
        auto result = data ? [Device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
            : [Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        Require(result != nil, "Could not allocate Combination Metal buffer"); Bytes += bytes; return result;
    }
    void ValidatePoint(id<MTLCommandBuffer> command, id<MTLBuffer> cursor, id<MTLBuffer> rawLeaves,
        id<MTLBuffer> leafIds, uint32_t leaves, id<MTLBuffer> target, id<MTLBuffer> weights) const {
        Require(command && command.device == Device && command.status == MTLCommandBufferStatusNotEnqueued,
                "Combination requires an uncommitted command on its Metal device");
        Require(leaves && leaves <= MaxLeaves, "Invalid Combination active leaf count");
        for (auto buffer : {cursor, leafIds, target, weights}) CheckBuffer(buffer, 4ull * Rows);
        CheckBuffer(rawLeaves, 4ull * leaves);
    }
    void EncodePreparedPoint(id<MTLCommandBuffer> command, id<MTLBuffer> cursor,
        id<MTLBuffer> leaves, id<MTLBuffer> leafIds, uint32_t count, bool applyShift, uint64_t* dispatches,
        bool allowNonfiniteTrial) {
        const QueryParams p = {Rows, Groups, 19, uint32_t(applyShift), 1, 0, count, 0};
        Dispatch(command, "PrepareQuerywisePoint", {cursor, leaves, leafIds, Point}, p, Rows, false, dispatches);
        if (HasYeti()) {
            auto blit = [command blitCommandEncoder];
            Require(blit != nil, "Could not create Combination trial reset encoder");
            [blit fillBuffer:TrialStatus range:NSMakeRange(0, 4) value:0];
            [blit endEncoding];
            auto params = Parameters(Components.front().Option);
            params.ApplyShift = allowNonfiniteTrial;
            Dispatch(command, "PrepareCombinationYetiPoint", {Point, SafeYetiPoint, TrialStatus, Status},
                params, Rows, false, dispatches);
        }
    }
    void EncodeAnalyticComponent(id<MTLCommandBuffer> command, Component& component,
        id<MTLBuffer> target, id<MTLBuffer> weights, uint64_t* dispatches) {
        const auto& option = component.Option;
        if (option.objective == 12 || option.objective == 13) {
            const QueryParams p = {Rows, Groups, option.objective, 0, option.beta, option.lambda, 1, 0};
            Dispatch(command, option.objective == 12 ? "QueryRmseDerivatives" : "QuerySoftMaxDerivatives",
                {target, weights, Point, Offsets, ScratchGradient, ScratchHessian, RowStats}, p, Groups, true, dispatches);
        } else Dispatch(command, "CombinationPointwise", {target, weights, Point, ScratchGradient, ScratchHessian, RowStats},
            Parameters(option), Rows, false, dispatches);
    }
    std::array<double, 2> ReadComponentLoss(const Component& component, bool allowNonfinite) const {
        if (component.Pair) return component.Pair->ReadLossPartials(allowNonfinite);
        std::array<double, 2> result = {0, 0};
        const auto* data = static_cast<const float*>(component.Loss.contents);
        for (uint32_t group = 0; group < LossGroups; ++group) {
            result[0] += double(data[4 * group]) + data[4 * group + 2];
            result[1] += double(data[4 * group + 1]) + data[4 * group + 3];
        }
        Require((allowNonfinite || std::isfinite(result[0])) && std::isfinite(result[1]) && result[1] >= 0,
                "Invalid Combination component objective or metric denominator");
        return result;
    }
    template <class T> void Dispatch(id<MTLCommandBuffer> command, const char* name,
        std::initializer_list<id<MTLBuffer>> buffers, const T& params, uint32_t count,
        bool groups, uint64_t* dispatches) {
        auto encoder = [command computeCommandEncoder]; Require(encoder != nil, "Combination compute encoder failed");
        [encoder setComputePipelineState:Shaders->Pipelines.at(name)];
        NSUInteger index = 0;
        for (auto buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
        [encoder setBytes:&params length:sizeof(params) atIndex:index];
        if (groups) [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        else [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding]; if (dispatches) ++*dispatches;
    }
};
