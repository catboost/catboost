#pragma once

// Ordered tasks use flattened cursor slices, each with its own local queries.
// Reuse the original YetiRank kernels and their local task/query seed mapping;
// no oracle seed or derivative state is hidden in this host workspace.
#import <Metal/Metal.h>
#include "metal_yeti_rank_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

static const char* CBMMetalOrderedYetiSource = R"METAL(
kernel void OrderedYetiPublishDerivatives(const device float2* derivatives [[buffer(0)]],
    device float* gradient [[buffer(1)]], device float* incident_mass [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant uint& rows [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    const float2 value = derivatives[row];
    const bool valid = all(isfinite(value)) && value.y >= 0.0f;
    if (!valid) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    gradient[row] = valid ? value.x : 0.0f;
    incident_mass[row] = valid ? value.y : 0.0f;
}
)METAL";

struct CBMOrderedYetiBlock {
    uint32_t CursorOffset = 0;
    // Zero-based local offsets, including both zero and the block row count.
    // Empty slices are omitted by the caller, rather than stored as blocks.
    std::vector<uint32_t> GroupOffsets;
};

class CBMOrderedYetiWorkspace {
public:
    // Includes all resident block metadata plus the two reusable float2 arrays.
    // Validates geometry without allocating Metal buffers or row-sized host maps.
    static uint64_t PlannedBytes(const std::vector<CBMOrderedYetiBlock>& blocks) {
        Require(!blocks.empty() && blocks.size() <= UINT32_MAX,
            "Ordered YetiRank requires a nonempty bounded block list");
        uint32_t maxRows = 0;
        uint64_t bytes = 0;
        for (const auto& block : blocks) {
            const auto shape = ValidateBlock(block);
            maxRows = std::max(maxRows, shape.Rows);
            bytes += 4ull * (uint64_t(shape.Groups) + 1) + 4ull * shape.Rows + 8ull * shape.Tasks;
            Require(bytes + 16ull * maxRows <= MemoryLimit,
                "Ordered YetiRank target exceeds the 1 GiB workspace guard");
        }
        return bytes + 16ull * maxRows;
    }

    // Context exposes Buffer(bytes, optionalSource). Command exposes BindingType
    // and Dispatch(name, bindings, params, width, grouped) with 256-thread groups.
    template<class Context>
    CBMOrderedYetiWorkspace(Context& context, const std::vector<CBMOrderedYetiBlock>& blocks,
        uint32_t permutations, float decay, bool legacyPrefixCentering = false)
        : Permutations(permutations), Decay(decay), LegacyPrefixCentering(legacyPrefixCentering) {
        Require(permutations && permutations <= 10000 && std::isfinite(decay) && decay >= 0 && decay <= 1,
            "Invalid Ordered YetiRank permutations or decay");
        const uint64_t planned = PlannedBytes(blocks);
        Blocks.reserve(blocks.size());
        for (const auto& input : blocks) {
            const auto shape = ValidateBlock(input);
            std::vector<uint32_t> ids(shape.Rows), tasks;
            tasks.reserve(uint64_t(shape.Tasks) * 2);
            for (uint32_t query = 0; query < shape.Groups; ++query)
                std::fill(ids.begin() + input.GroupOffsets[query], ids.begin() + input.GroupOffsets[query + 1], query);
            // Preserve CBMYetiRankRuntime's exact packing, including excluding
            // a query that crosses a task's 1024-row boundary.
            for (uint32_t query = 0; query < shape.Groups;) {
                const uint32_t limit = std::min(shape.Rows, input.GroupOffsets[query] + 1024);
                const uint32_t end = limit == shape.Rows ? shape.Groups : ids[limit];
                Require(end > query, "Invalid Ordered YetiRank task packing");
                tasks.push_back(query); tasks.push_back(end); query = end;
            }
            Require(tasks.size() == uint64_t(shape.Tasks) * 2, "Ordered YetiRank task plan mismatch");
            Block block;
            block.CursorOffset = input.CursorOffset; block.Rows = shape.Rows;
            block.Groups = shape.Groups; block.Tasks = shape.Tasks;
            block.Offsets = Allocate(context, 4ull * (uint64_t(shape.Groups) + 1), input.GroupOffsets.data());
            block.QueryIds = Allocate(context, 4ull * shape.Rows, ids.data());
            block.TaskRanges = Allocate(context, 8ull * shape.Tasks, tasks.data());
            Blocks.push_back(block); MaxRows = std::max(MaxRows, shape.Rows);
        }
        Exponents = Allocate(context, 8ull * MaxRows);
        Derivatives = Allocate(context, 8ull * MaxRows);
        Require(Bytes == planned, "Ordered YetiRank workspace plan mismatch");
    }

    uint64_t AllocatedBytes() const { return Bytes; }
    uint32_t BlockCount() const { return static_cast<uint32_t>(Blocks.size()); }
    bool UsesLegacyPrefixCentering() const { return LegacyPrefixCentering; }

    // Encode blocks serially on the same command queue: Exponents/Derivatives
    // are scratch arrays reused from local row zero for each block. The caller
    // validates/constructs point and expanded targets/weights before this call.
    // All five supplied float arrays use the complete flattened cursor layout.
    // Status is sticky and owned/cleared by the surrounding Ordered session.
    template<class Command>
    void EncodeBlock(Command& command, uint32_t blockId, id<MTLBuffer> point,
        id<MTLBuffer> targets, id<MTLBuffer> originalWeights, id<MTLBuffer> gradient,
        id<MTLBuffer> hessian, id<MTLBuffer> status, uint64_t seed) {
        Require(blockId < Blocks.size(), "Invalid Ordered YetiRank block index");
        const auto& block = Blocks[blockId];
        const uint64_t offset = 4ull * block.CursorOffset;
        const uint64_t end = offset + 4ull * block.Rows;
        for (auto buffer : {point, targets, originalWeights, gradient, hessian})
            CheckBuffer(buffer, end);
        CheckBuffer(status, 4);
        Require(gradient != hessian, "Ordered YetiRank derivative outputs cannot alias");
        for (auto input : {point, targets, originalWeights})
            Require(input != gradient && input != hessian, "Ordered YetiRank outputs cannot alias source buffers");
        const RankParams params = {block.Rows, block.Groups, block.Tasks, Permutations,
            uint32_t(seed), uint32_t(seed >> 32), Decay, LegacyPrefixCentering ? block.Groups : block.Rows};
        using Binding = typename Command::BindingType;
        command.Dispatch("PrepareYetiRankApprox", {Binding(point, offset), block.Offsets, Exponents},
            params, block.Groups, true);
        command.Dispatch("YetiRankPointwise", {Exponents, Binding(targets, offset), Binding(originalWeights, offset),
            block.QueryIds, block.Offsets, block.TaskRanges, Derivatives}, params, block.Tasks, true);
        command.Dispatch("OrderedYetiPublishDerivatives", {Derivatives, Binding(gradient, offset), Binding(hessian, offset), status},
            block.Rows, block.Rows, false);
    }

private:
    static constexpr uint64_t MemoryLimit = uint64_t(1) << 30;
    struct RankParams {
        uint32_t Rows, Groups, Tasks, Permutations, SeedLow, SeedHigh;
        float Decay;
        uint32_t CenterRows;
    };
    static_assert(sizeof(RankParams) == 32, "Ordered YetiRank kernel parameter ABI mismatch");
    struct Shape { uint32_t Rows, Groups, Tasks; };
    struct Block {
        uint32_t CursorOffset = 0, Rows = 0, Groups = 0, Tasks = 0;
        id<MTLBuffer> Offsets = nil, QueryIds = nil, TaskRanges = nil;
    };
    uint32_t Permutations, MaxRows = 0;
    float Decay;
    bool LegacyPrefixCentering;
    uint64_t Bytes = 0;
    std::vector<Block> Blocks;
    id<MTLBuffer> Exponents = nil, Derivatives = nil;

    static void Require(bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    }
    static Shape ValidateBlock(const CBMOrderedYetiBlock& block) {
        const auto& offsets = block.GroupOffsets;
        Require(offsets.size() >= 2 && offsets.size() <= uint64_t(1u << 24) + 1,
            "Ordered YetiRank blocks require local query offsets");
        const uint32_t rows = offsets.back(), groups = static_cast<uint32_t>(offsets.size() - 1);
        Require(offsets.front() == 0 && rows && rows <= (1u << 24) && groups <= rows &&
            uint64_t(block.CursorOffset) + rows <= UINT32_MAX,
            "Invalid Ordered YetiRank block rows or cursor range");
        for (uint32_t query = 0; query < groups; ++query)
            Require(offsets[query] < offsets[query + 1] && offsets[query + 1] <= rows &&
                offsets[query + 1] - offsets[query] <= 1023,
                "Ordered YetiRank requires 1 to 1023 rows per query");
        uint32_t tasks = 0;
        for (uint32_t query = 0; query < groups; ++tasks) {
            const uint32_t limit = std::min(rows, offsets[query] + 1024);
            const uint32_t end = limit == rows ? groups :
                static_cast<uint32_t>(std::upper_bound(offsets.begin(), offsets.end(), limit) - offsets.begin() - 1);
            Require(end > query, "Invalid Ordered YetiRank task packing");
            query = end;
        }
        return {rows, groups, tasks};
    }
    template<class Context>
    id<MTLBuffer> Allocate(Context& context, uint64_t bytes, const void* source = nullptr) {
        auto buffer = context.Buffer(bytes, source);
        Require(buffer && buffer.length >= bytes, "Ordered YetiRank GPU allocation failed");
        Bytes += bytes;
        return buffer;
    }
    void CheckBuffer(id<MTLBuffer> buffer, uint64_t bytes) const {
        Require(buffer && buffer.length >= bytes && buffer.device == Exponents.device,
            "Invalid Ordered YetiRank Metal buffer");
    }
};
