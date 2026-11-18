#pragma once

// Ordered folds retain independent target slices. Points and published
// derivatives stay on the GPU; only immutable target and query metadata are
// gathered from the original row order when this workspace is constructed.
#import <Metal/Metal.h>
#include "metal_combination_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

struct CBMOrderedCombinationBlock {
    uint32_t CursorOffset = 0;
    std::vector<uint32_t> OriginalRows;
    // Local query boundaries, including zero and OriginalRows.size(). An
    // empty list is valid only for combinations of pointwise components.
    std::vector<uint32_t> GroupOffsets;
};

class CBMOrderedCombinationWorkspace {
public:
    // Includes all block buffers and every component runtime. Plan every
    // block before creating any Metal allocation, so many individually small
    // histories cannot exceed the aggregate workspace bound unnoticed.
    static uint64_t PlannedBytes(const std::vector<CBMOrderedCombinationBlock>& blocks,
        const CBMCombinationOptions& options, const CBMCombinationComponent* components,
        uint32_t originalRows, const uint32_t* winners, const uint32_t* losers,
        const float* pairWeights) {
        return MakePlan(blocks, options, components, originalRows, winners, losers, pairWeights).Bytes;
    }

    // Context exposes Device and Buffer(bytes, optionalSource).
    template<class Context>
    CBMOrderedCombinationWorkspace(Context& context, const std::vector<CBMOrderedCombinationBlock>& blocks,
        const CBMCombinationOptions& options, const CBMCombinationComponent* components,
        const float* originalTargets, const float* originalWeights, uint32_t originalRows,
        const uint32_t* winners, const uint32_t* losers, const float* pairWeights)
        : Device(context.Device) {
        Require(Device && originalTargets, "Ordered Combination requires a Metal device and targets");
        const auto plan = MakePlan(blocks, options, components, originalRows, winners, losers, pairWeights);
        YetiCount = plan.YetiCount;
        for (uint32_t row = 0; row < originalRows; ++row)
            Require(std::isfinite(originalTargets[row]) && (!originalWeights ||
                (std::isfinite(originalWeights[row]) && originalWeights[row] >= 0)),
                "Ordered Combination targets and weights must be finite with nonnegative weights");
        Blocks.reserve(blocks.size());
        for (size_t index = 0; index < blocks.size(); ++index) {
            const auto& input = blocks[index];
            const auto& shape = plan.Blocks[index];
            Block block;
            block.CursorOffset = input.CursorOffset;
            block.Rows = static_cast<uint32_t>(input.OriginalRows.size());
            if (!shape.Components.empty()) {
                std::vector<float> targets(block.Rows), weights(block.Rows);
                std::vector<uint32_t> zeroIds(block.Rows, 0);
                for (uint32_t row = 0; row < block.Rows; ++row) {
                    targets[row] = originalTargets[input.OriginalRows[row]];
                    weights[row] = originalWeights ? originalWeights[input.OriginalRows[row]] : 1.0f;
                }
                const float zero = 0;
                block.Targets = Allocate(context, 4ull * block.Rows, targets.data());
                block.Weights = Allocate(context, 4ull * block.Rows, weights.data());
                block.Point = Allocate(context, 4ull * block.Rows);
                block.LeafIds = Allocate(context, 4ull * block.Rows, zeroIds.data());
                block.ZeroLeaf = Allocate(context, 4, &zero);
                block.Gradient = Allocate(context, 4ull * block.Rows);
                block.Hessian = Allocate(context, 4ull * block.Rows);
                block.GradientWeights = Allocate(context, 4ull * block.Rows);
                const auto groups = static_cast<uint32_t>(input.GroupOffsets.empty() ? 0 : input.GroupOffsets.size() - 1);
                block.Target = std::make_unique<CBMCombinationRuntime>(Device, block.Rows,
                    static_cast<uint32_t>(shape.Components.size()), shape.Components.data(), groups,
                    groups ? input.GroupOffsets.data() : nullptr,
                    static_cast<uint32_t>(shape.Winners.size()), shape.Winners.data(), shape.Losers.data(),
                    shape.PairWeights.data(), 1, shape.LossGroups);
                block.Target->ValidateTargets(targets.data(), weights.data(), block.Rows == originalRows);
                Bytes += block.Target->AllocatedBytes();
            }
            Blocks.push_back(std::move(block));
        }
        Require(Bytes == plan.Bytes, "Ordered Combination workspace allocation does not match its plan");
    }

    uint64_t AllocatedBytes() const { return Bytes; }
    uint32_t BlockCount() const { return static_cast<uint32_t>(Blocks.size()); }
    uint32_t YetiComponentCount() const { return YetiCount; }
    bool HasYeti() const { return YetiCount != 0; }
    bool HasYetiSeedCallback() const { return SeedCallback != nullptr; }
    void SetYetiSeedCallback(CBMCombinationYetiSeedCallback callback, void* context) {
        for (auto& block : Blocks) if (block.Target) block.Target->SetYetiSeedCallback(callback, context);
        SeedCallback = callback;
    }
    void ClearStatus() {
        for (auto& block : Blocks) if (block.Target) block.Target->ClearStatus();
    }
    void CheckStatus() const {
        for (const auto& block : Blocks) if (block.Target) block.Target->CheckStatus();
    }

    // Command exposes Buffer and Stats.kernel_dispatches. The caller prepares
    // globalPoint (including any trial leaf shift), then waits for this command
    // before reading an objective or publishing another point for the block.
    // A normal evaluation calculates value AND derivatives once; the caller
    // retains accepted derivatives across rejected backtracking trials.
    template<class Command>
    void EncodeBlock(Command& command, uint32_t blockId, id<MTLBuffer> globalPoint,
        id<MTLBuffer> globalGradient, id<MTLBuffer> globalHessian, id<MTLBuffer> globalGradientWeights,
        const uint64_t* seeds = nullptr, uint32_t seedCount = 0,
        bool trial = false, bool objectiveOnly = false) {
        auto& block = GetBlock(blockId);
        Require(command.Buffer && command.Buffer.device == Device &&
            command.Buffer.status == MTLCommandBufferStatusNotEnqueued,
            "Ordered Combination requires an uncommitted command on its Metal device");
        const uint64_t offset = 4ull * block.CursorOffset, size = 4ull * block.Rows;
        CheckBuffer(globalPoint, offset + size);
        if (!objectiveOnly) {
            for (auto output : {globalGradient, globalHessian, globalGradientWeights}) CheckBuffer(output, offset + size);
            Require(globalGradient != globalHessian && globalGradient != globalGradientWeights &&
                globalHessian != globalGradientWeights,
                "Ordered Combination derivative outputs cannot alias");
            for (auto output : {globalGradient, globalHessian, globalGradientWeights})
                Require(output != globalPoint, "Ordered Combination outputs cannot alias the point");
        }
        Require((objectiveOnly || SeedCallback) ? seedCount == 0 : seedCount == YetiCount,
            "Ordered Combination block Yeti seed packet has the wrong size");
        Require(!seedCount || seeds, "Ordered Combination block Yeti seeds are missing");
        if (!block.Target) {
            // Pair-only slices without incident pair mass are legitimate CUDA
            // target slices. Their value and all derivatives are identically 0.
            if (!objectiveOnly) {
                auto blit = [command.Buffer blitCommandEncoder];
                Require(blit != nil, "Ordered Combination zero encoder allocation failed");
                for (auto output : {globalGradient, globalHessian, globalGradientWeights})
                    [blit fillBuffer:output range:NSMakeRange(offset, size) value:0];
                [blit endEncoding];
            }
        } else {
            auto blit = [command.Buffer blitCommandEncoder];
            Require(blit != nil, "Ordered Combination point copy encoder allocation failed");
            [blit copyFromBuffer:globalPoint sourceOffset:offset toBuffer:block.Point destinationOffset:0 size:size];
            [blit endEncoding];
            if (objectiveOnly) {
                block.Target->EncodeLoss(command.Buffer, block.Point, block.ZeroLeaf, block.LeafIds, 1, false,
                    block.Targets, block.Weights, &command.Stats.kernel_dispatches, trial);
            } else {
                if (!SeedCallback) block.Target->SetYetiSeeds(seedCount, seeds);
                block.Target->EncodeOracle(command.Buffer, block.Point, block.ZeroLeaf, block.LeafIds, 1, false,
                    block.Targets, block.Weights, block.Gradient, block.Hessian, block.GradientWeights,
                    &command.Stats.kernel_dispatches, trial);
                blit = [command.Buffer blitCommandEncoder];
                Require(blit != nil, "Ordered Combination derivative copy encoder allocation failed");
                [blit copyFromBuffer:block.Gradient sourceOffset:0 toBuffer:globalGradient destinationOffset:offset size:size];
                [blit copyFromBuffer:block.Hessian sourceOffset:0 toBuffer:globalHessian destinationOffset:offset size:size];
                [blit copyFromBuffer:block.GradientWeights sourceOffset:0 toBuffer:globalGradientWeights destinationOffset:offset size:size];
                [blit endEncoding];
            }
        }
        block.LastCommand = command.Buffer;
    }

    double ReadObjective(uint32_t blockId, bool allowNonfinite = false) const {
        const auto& block = ReadableBlock(blockId);
        return block.Target ? block.Target->ReadObjective(allowNonfinite) : 0.0;
    }
    float ReadMetric(uint32_t blockId) const {
        const auto& block = ReadableBlock(blockId);
        return block.Target ? block.Target->ReadMetric() : 0.0f;
    }

private:
    static constexpr uint64_t MemoryLimit = uint64_t(1) << 30;
    struct BlockPlan {
        uint32_t LossGroups = 0;
        std::vector<CBMCombinationComponent> Components;
        std::vector<uint32_t> Winners, Losers;
        std::vector<float> PairWeights;
    };
    struct Plan {
        uint64_t Bytes = 0;
        uint32_t YetiCount = 0;
        std::vector<BlockPlan> Blocks;
    };
    struct Block {
        uint32_t CursorOffset = 0, Rows = 0;
        id<MTLBuffer> Targets = nil, Weights = nil, Point = nil, LeafIds = nil, ZeroLeaf = nil;
        id<MTLBuffer> Gradient = nil, Hessian = nil, GradientWeights = nil;
        id<MTLCommandBuffer> LastCommand = nil;
        std::unique_ptr<CBMCombinationRuntime> Target;
    };
    id<MTLDevice> Device;
    uint64_t Bytes = 0;
    uint32_t YetiCount = 0;
    CBMCombinationYetiSeedCallback SeedCallback = nullptr;
    std::vector<Block> Blocks;

    static void Require(bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    }
    static void ValidateComponent(const CBMCombinationComponent& c) {
        Require(c.objective <= 14 || c.objective == 17, "Unsupported Ordered Combination component objective");
        Require(std::isfinite(c.weight) && c.weight > 0 && std::isfinite(c.param) && std::isfinite(c.border) &&
            std::isfinite(c.beta) && std::isfinite(c.lambda) && std::isfinite(c.decay) &&
            !c.reserved && !c.reserved1[0] && !c.reserved1[1] && !c.reserved1[2],
            "Ordered Combination weights/parameters must be finite and reserved fields zero");
        Require(c.objective != 4 || c.param >= 0, "Ordered Combination Huber delta must be nonnegative");
        Require((c.objective != 5 && c.objective != 8 && c.objective != 9) || (c.param >= 0 && c.param <= 1),
            "Ordered Combination alpha must be in [0, 1]");
        Require(c.objective != 6 || c.param >= 1, "Ordered Combination Lq q must be at least 1");
        Require(c.objective != 7 || (c.param > 1 && c.param < 2), "Ordered Combination Tweedie power must be in (1, 2)");
        Require(c.objective != 17 || (c.permutations && c.permutations <= 10000 && c.decay >= 0 && c.decay <= 1),
            "Invalid Ordered Combination YetiRank permutations or decay");
    }
    static Plan MakePlan(const std::vector<CBMOrderedCombinationBlock>& blocks,
        const CBMCombinationOptions& options, const CBMCombinationComponent* components,
        uint32_t originalRows, const uint32_t* winners, const uint32_t* losers, const float* pairWeights) {
        Require(originalRows && originalRows <= (1u << 24) && !blocks.empty() && blocks.size() <= UINT32_MAX,
            "Ordered Combination requires bounded nonempty original rows and blocks");
        Require(!options.reserved && options.component_count && options.component_count <= 128 && components &&
            options.group_count <= originalRows && options.pair_count <= UINT32_MAX / 2,
            "Invalid Ordered Combination component or target dimensions");
        Plan plan;
        bool needsGroups = false, needsPairs = false;
        for (uint32_t c = 0; c < options.component_count; ++c) {
            ValidateComponent(components[c]);
            needsGroups |= components[c].objective >= 12;
            needsPairs |= components[c].objective == 14;
            plan.YetiCount += components[c].objective == 17;
        }
        Require(!needsGroups || options.group_count, "Ordered Combination query components require original groups");
        Require(!needsPairs || (options.pair_count && winners && losers && pairWeights),
            "Ordered Combination PairLogit requires original prepared pairs");
        double originalPairMass = 0;
        if (needsPairs) for (uint32_t edge = 0; edge < options.pair_count; ++edge) {
            Require(winners[edge] < originalRows && losers[edge] < originalRows && winners[edge] != losers[edge] &&
                std::isfinite(pairWeights[edge]) && pairWeights[edge] >= 0,
                "Invalid Ordered Combination original pair endpoint or weight");
            originalPairMass += pairWeights[edge];
        }
        Require(!needsPairs || (originalPairMass > 0 && std::isfinite(originalPairMass) &&
            2 * originalPairMass <= std::numeric_limits<float>::max()),
            "Ordered Combination requires positive finite representable original pair mass");
        plan.Blocks.reserve(blocks.size());
        for (const auto& block : blocks) {
            Require(!block.OriginalRows.empty() && block.OriginalRows.size() <= originalRows &&
                uint64_t(block.CursorOffset) + block.OriginalRows.size() <= UINT32_MAX,
                "Invalid Ordered Combination block rows or cursor range");
            const auto rows = static_cast<uint32_t>(block.OriginalRows.size());
            const auto& offsets = block.GroupOffsets;
            Require(offsets.empty() ? !needsGroups : (offsets.size() >= 2 && offsets.size() <= uint64_t(rows) + 1 &&
                offsets.front() == 0 && offsets.back() == rows),
                "Ordered Combination blocks require complete local group offsets");
            const uint32_t groups = offsets.empty() ? 0 : static_cast<uint32_t>(offsets.size() - 1);
            for (uint32_t group = 0; group < groups; ++group)
                Require(offsets[group] < offsets[group + 1] && offsets[group + 1] <= rows &&
                    (!plan.YetiCount || offsets[group + 1] - offsets[group] <= 1023),
                    "Invalid Ordered Combination group size (YetiRank permits at most 1023 rows)");
            std::unordered_map<uint32_t, uint32_t> localRows;
            localRows.reserve(rows);
            for (uint32_t row = 0; row < rows; ++row)
                Require(block.OriginalRows[row] < originalRows && localRows.emplace(block.OriginalRows[row], row).second,
                    "Ordered Combination block row mapping must contain distinct valid original rows");
            BlockPlan item;
            double pairMass = 0;
            if (needsPairs) for (uint32_t edge = 0; edge < options.pair_count; ++edge) {
                const auto win = localRows.find(winners[edge]), lose = localRows.find(losers[edge]);
                Require((win == localRows.end()) == (lose == localRows.end()),
                    "Ordered Combination block cuts an original pair");
                if (win == localRows.end()) continue;
                Require(std::upper_bound(offsets.begin(), offsets.end(), win->second) ==
                    std::upper_bound(offsets.begin(), offsets.end(), lose->second),
                    "Ordered Combination pair endpoints must share a local group");
                item.Winners.push_back(win->second); item.Losers.push_back(lose->second);
                item.PairWeights.push_back(pairWeights[edge]); pairMass += pairWeights[edge];
            }
            Require(std::isfinite(pairMass) && 2 * pairMass <= std::numeric_limits<float>::max(),
                "Ordered Combination local pair mass must be representable in float32");
            for (uint32_t c = 0; c < options.component_count; ++c)
                if (components[c].objective != 14 || pairMass > 0) item.Components.push_back(components[c]);
            item.LossGroups = std::min(4096u, (rows + 255) / 256);
            if (!item.Components.empty()) {
                // Adapter: seven row arrays plus a zero leaf. Core target:
                // four row scratch arrays, row/query stats, status and offsets.
                plan.Bytes += 28ull * rows + 4 + 16ull * rows + 8ull * std::max(rows, groups) + 4 +
                    (groups ? 4ull * (uint64_t(groups) + 1) : 0) + 16ull * item.LossGroups * item.Components.size();
                if (plan.YetiCount) plan.Bytes += 4ull * rows + 4; // rejectable nonfinite trial point
                uint32_t yetiTasks = 0;
                if (plan.YetiCount) for (uint32_t query = 0; query < groups; ++yetiTasks) {
                    const uint32_t limit = std::min(rows, offsets[query] + 1024);
                    const auto end = limit == rows ? groups :
                        static_cast<uint32_t>(std::upper_bound(offsets.begin(), offsets.end(), limit) - offsets.begin() - 1);
                    Require(end > query, "Invalid Ordered Combination YetiRank task packing");
                    query = end;
                }
                for (const auto& component : item.Components) {
                    if (component.objective == 14)
                        plan.Bytes += 44ull * item.Winners.size() + 8ull * rows + 20 + 8ull * item.LossGroups;
                    if (component.objective == 17)
                        plan.Bytes += 24ull * rows + 4ull * groups + 8ull * yetiTasks + 8;
                }
                Require(plan.Bytes <= MemoryLimit, "Ordered Combination histories exceed the 1 GiB workspace guard");
            }
            plan.Blocks.push_back(std::move(item));
        }
        return plan;
    }
    template<class Context>
    id<MTLBuffer> Allocate(Context& context, uint64_t bytes, const void* source = nullptr) {
        Require(bytes && Bytes + bytes <= MemoryLimit && bytes <= Device.maxBufferLength,
            "Ordered Combination buffer exceeds its memory limit");
        auto result = context.Buffer(bytes, source);
        CheckBuffer(result, bytes); Bytes += bytes; return result;
    }
    void CheckBuffer(id<MTLBuffer> buffer, uint64_t bytes) const {
        Require(buffer && buffer.device == Device && buffer.length >= bytes, "Invalid Ordered Combination Metal buffer");
    }
    Block& GetBlock(uint32_t blockId) {
        Require(blockId < Blocks.size(), "Invalid Ordered Combination block index");
        return Blocks[blockId];
    }
    const Block& ReadableBlock(uint32_t blockId) const {
        Require(blockId < Blocks.size(), "Invalid Ordered Combination block index");
        const auto& block = Blocks[blockId];
        Require(block.LastCommand && block.LastCommand.status == MTLCommandBufferStatusCompleted,
            "Wait for the Ordered Combination block command before reading its objective");
        return block;
    }
};
