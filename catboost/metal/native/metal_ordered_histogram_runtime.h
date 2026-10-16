#pragma once
#import <Metal/Metal.h>
#include "metal_kernel_abi.h"
#include "metal_ordered_histogram_kernels.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

// Shared host pipeline for the production Ordered session and numerical probe.
// Context exposes Buffer(bytes, optionalSource). Command exposes BindingType,
// Dispatch and DispatchIndirect with exactly256 threads/group. No CPU training
// or histogram statistics are computed here; host work only lays out buffers.
struct CBMOrderedHistogramTile {
    uint32_t Begin = 0, End = 0, Bins = 0, FirstCandidate = 0, Candidates = 0;
    std::vector<uint32_t> Offsets;
    id<MTLBuffer> OffsetBuffer = nil;
};
struct CBMOrderedHistogramPlan {
    uint32_t Rows, PackedRows, FoldCount, FoldSlots = 1, MaxPartitions, HistogramPartitions, JobCapacity, MaxBins = 0;
    std::vector<CBMOrderedHistogramTile> Tiles;
    std::vector<uint32_t> CandidatePairs, CandidateIndices;
    uint64_t Bytes = 0;
    bool AllowReuse = true;

    CBMOrderedHistogramPlan(uint32_t rows, uint32_t packedRows, uint32_t foldCount,
        uint32_t maxLeaves, uint32_t features, uint32_t candidates,
        const uint32_t* candidateFeatures, const uint32_t* candidateBorders,
        uint64_t histogramBudget = uint64_t(128) << 20, bool allowReuse = true, uint32_t histogramMaxLeaves = 0,
        const uint8_t* candidateTypes = nullptr)
        : Rows(rows), PackedRows(packedRows), FoldCount(foldCount), AllowReuse(allowReuse) {
        if (!rows || !packedRows || !foldCount || !maxLeaves || (maxLeaves & (maxLeaves - 1)))
            throw std::runtime_error("Invalid Ordered histogram dimensions");
        while (FoldSlots < foldCount) FoldSlots <<= 1;
        if (uint64_t(FoldSlots) * maxLeaves > UINT32_MAX)
            throw std::runtime_error("Ordered histogram partition count exceeds uint32");
        MaxPartitions = FoldSlots * maxLeaves;
        if (histogramMaxLeaves && (histogramMaxLeaves & (histogramMaxLeaves - 1)))
            throw std::runtime_error("Ordered histogram cache leaf capacity must be a power of two");
        HistogramPartitions = FoldSlots * (histogramMaxLeaves ? std::min(maxLeaves, histogramMaxLeaves) : maxLeaves);
        JobCapacity = (packedRows - 1) / 8192 + 1 + std::min(packedRows, MaxPartitions);
        const uint64_t perBin = uint64_t(HistogramPartitions) * 32;
        const uint64_t capacity = std::max<uint64_t>(1, histogramBudget / perBin);
        std::vector<uint32_t> spans(features, 0);
        for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
            const uint32_t type = candidateTypes ? candidateTypes[candidate] : 0;
            if (candidateFeatures[candidate] >= features || type > 1 || candidateBorders[candidate] > 255 - (type == 0))
                throw std::runtime_error("Invalid Ordered histogram candidate");
            spans[candidateFeatures[candidate]] = std::max(spans[candidateFeatures[candidate]],
                std::min(256u, candidateBorders[candidate] + 2));
        }
        uint32_t begin = 0, count = 0;
        auto add = [&](uint32_t end) {
            if (count) {
                CBMOrderedHistogramTile tile;
                tile.Begin = begin; tile.End = end; tile.Bins = count;
                tile.Offsets.push_back(0);
                for (uint32_t feature = begin; feature < end; ++feature)
                    tile.Offsets.push_back(tile.Offsets.back() + spans[feature]);
                tile.FirstCandidate = CandidateIndices.size();
                for (uint32_t candidate = 0; candidate < candidates; ++candidate) {
                    const uint32_t feature = candidateFeatures[candidate];
                    if (feature >= begin && feature < end) {
                        // The private pair buffer packs comparison type in the high bit;
                        // numeric pairs and all existing kernel binding sizes are unchanged.
                        CandidatePairs.push_back(feature);
                        CandidatePairs.push_back(candidateBorders[candidate] | ((candidateTypes && candidateTypes[candidate]) ? 0x80000000u : 0));
                        CandidateIndices.push_back(candidate); ++tile.Candidates;
                    }
                }
                MaxBins = std::max(MaxBins, count); Tiles.push_back(std::move(tile));
            }
            begin = end; count = 0;
        };
        for (uint32_t feature = 0; feature < features; ++feature) {
            if (count && uint64_t(count) + spans[feature] > capacity) add(feature);
            count += spans[feature];
        }
        add(features);
        Bytes = uint64_t(packedRows) * 20 + uint64_t((packedRows - 1) / 256 + 1) * 4
            + uint64_t((packedRows - 1) / 65536 + 1) * 4 + uint64_t(MaxPartitions + 1ull) * 8
            + uint64_t(JobCapacity) * 16 + uint64_t(MaxPartitions) * 4 + 52
            + perBin * MaxBins + uint64_t(candidates) * 12;
        for (const auto& tile : Tiles) Bytes += tile.Offsets.size() * 4ull;
    }
};

class CBMOrderedHistogramWorkspace {
public:
    CBMOrderedHistogramPlan Plan;
    CBMOrderedHistogramParams P = {};
    id<MTLBuffer> Metadata, RowIndices, NextRows, LeafIds, Offsets, NextOffsets;
    id<MTLBuffer> RowPrefix, TilePrefix, BlockPrefix, Jobs, Active, State, Arguments;
    id<MTLBuffer> High, Low, CandidatePairs, CandidateIndices;

    template<class Context> CBMOrderedHistogramWorkspace(Context& context, CBMOrderedHistogramPlan plan)
        : Plan(std::move(plan)) {
        P.Rows = Plan.Rows; P.PackedRows = Plan.PackedRows; P.FoldCount = Plan.FoldCount; P.FoldSlots = Plan.FoldSlots;
        P.JobCapacity = Plan.JobCapacity; P.TileRows = 8192; P.Partitions = Plan.FoldSlots;
        Metadata = context.Buffer(uint64_t(P.PackedRows) * 4);
        RowIndices = context.Buffer(uint64_t(P.PackedRows) * 4); NextRows = context.Buffer(uint64_t(P.PackedRows) * 4);
        LeafIds = context.Buffer(uint64_t(P.PackedRows) * 4); RowPrefix = context.Buffer(uint64_t(P.PackedRows) * 4);
        TilePrefix = context.Buffer(uint64_t((P.PackedRows - 1) / 256 + 1) * 4);
        BlockPrefix = context.Buffer(uint64_t((P.PackedRows - 1) / 65536 + 1) * 4);
        Offsets = context.Buffer(uint64_t(Plan.MaxPartitions + 1ull) * 4);
        NextOffsets = context.Buffer(uint64_t(Plan.MaxPartitions + 1ull) * 4);
        Jobs = context.Buffer(uint64_t(P.JobCapacity) * 16); Active = context.Buffer(uint64_t(Plan.MaxPartitions) * 4);
        State = context.Buffer(16); Arguments = context.Buffer(36);
        High = context.Buffer(uint64_t(Plan.HistogramPartitions) * Plan.MaxBins * 16);
        Low = context.Buffer(uint64_t(Plan.HistogramPartitions) * Plan.MaxBins * 16);
        CandidatePairs = context.Buffer(Plan.CandidatePairs.size() * 4ull, Plan.CandidatePairs.data());
        CandidateIndices = context.Buffer(Plan.CandidateIndices.size() * 4ull, Plan.CandidateIndices.data());
        for (auto& tile : Plan.Tiles) tile.OffsetBuffer = context.Buffer(tile.Offsets.size() * 4ull, tile.Offsets.data());
    }

    template<class Command, class Binding> void Initialize(Command& command, Binding permutation, Binding folds, uint32_t base,
        uint32_t foldCount = 0, uint32_t packedRows = 0) {
        P.FoldCount = foldCount ? foldCount : Plan.FoldCount;
        P.PackedRows = packedRows ? packedRows : Plan.PackedRows;
        if (P.FoldCount > Plan.FoldCount || P.PackedRows > Plan.PackedRows)
            throw std::runtime_error("Ordered active folds exceed histogram capacity");
        P.DerivativeOffset = base; P.Partitions = P.FoldSlots; P.LeafMask = 0;
        command.Dispatch("InitializeOrderedHistogramOccurrences", {permutation, folds, Metadata, RowIndices, LeafIds, Offsets},
            P, P.Rows, false, P.FoldCount);
    }
    template<class Command, class Binding> void Partition(Command& command, Binding originalLeafIds, uint32_t leaves) {
        P.Partitions = leaves * P.FoldSlots; P.LeafMask = leaves - 1;
        CBMMetalKernelParams k = {}; k.Rows = P.PackedRows; k.Leaves = P.Partitions;
        command.Dispatch("UpdateOrderedHistogramLeafIds", {Metadata, originalLeafIds, LeafIds}, P, P.PackedRows);
        command.Dispatch("CountDeepPartitionBits", {RowIndices, LeafIds, RowPrefix, TilePrefix}, k, (P.PackedRows - 1) / 256 + 1, true);
        command.Dispatch("ScanDeepPartitionTiles", {TilePrefix, BlockPrefix}, k, (P.PackedRows - 1) / 65536 + 1, true);
        command.Dispatch("ScanDeepPartitionBlocks", {BlockPrefix}, k, 1, true);
        command.Dispatch("BuildDeepPartitionOffsets", {Offsets, RowPrefix, TilePrefix, BlockPrefix, NextOffsets}, k, P.Partitions / 2);
        command.Dispatch("ScatterDeepPartitionRows", {RowIndices, LeafIds, RowPrefix, TilePrefix, BlockPrefix, NextRows}, k, P.PackedRows);
        std::swap(RowIndices, NextRows); std::swap(Offsets, NextOffsets);
    }
    template<class Command> void Prepare(Command& command, uint32_t leaves) {
        P.Partitions = leaves * P.FoldSlots;
        P.Reuse = CanHistogram() && Plan.AllowReuse && Plan.Tiles.size() == 1 && leaves > 1;
        command.Dispatch("ResetOrderedHistogramJobs", {State}, P, 1);
        command.Dispatch("BuildOrderedHistogramJobs", {Offsets, Jobs, Active, State}, P, P.Reuse ? P.Partitions / 2 : P.Partitions);
    }
    bool CanHistogram() const { return P.Partitions <= Plan.HistogramPartitions; }
    template<class Command, class Binding> void DirectCandidates(Command& command, id<MTLBuffer> bins,
        id<MTLBuffer> derivatives, Binding candidates, uint32_t count, id<MTLBuffer> statistics) {
        P.Features = count; P.Candidates = count;
        command.Dispatch("OrderedHistogramArguments", {State, Arguments}, P, 1);
        command.Dispatch("ClearOrderedPartitionCandidateStatistics", {statistics}, P,
            uint64_t(count) * (P.Partitions / P.FoldSlots) * P.FoldCount * 2);
        command.DispatchIndirect("ComputeOrderedPartitionCandidates", {bins, derivatives, Metadata, RowIndices,
            Offsets, Active, candidates, statistics}, P, Arguments, 12);
    }
    template<class Command> void Compute(Command& command, const CBMOrderedHistogramTile& tile,
        id<MTLBuffer> bins, id<MTLBuffer> derivatives) {
        P.Features = tile.End - tile.Begin; P.FeatureBegin = tile.Begin; P.TotalBins = tile.Bins;
        command.Dispatch("OrderedHistogramArguments", {State, Arguments}, P, 1);
        command.Dispatch("ClearOrderedHistogram", {High, Low}, P, uint64_t(P.Reuse ? P.Partitions / 2 : P.Partitions) * P.TotalBins);
        command.DispatchIndirect("ComputeOrderedHistogram", {bins, derivatives, Metadata, RowIndices, tile.OffsetBuffer, Jobs, State, High, Low}, P, Arguments, 0);
        command.DispatchIndirect("ScanOrderedHistogram", {High, Low, tile.OffsetBuffer, Active}, P, Arguments, 12);
        if (P.Reuse) command.DispatchIndirect("SubtractOrderedHistogramSibling", {High, Low, Offsets, Active, State}, P, Arguments, 24);
    }
    template<class Command> void Extract(Command& command, const CBMOrderedHistogramTile& tile,
        uint32_t begin, uint32_t count, id<MTLBuffer> statistics) {
        using Binding = typename Command::BindingType;
        P.Candidates = count;
        command.Dispatch("ExtractOrderedHistogramCandidates", {High, Low, tile.OffsetBuffer,
            Binding(CandidatePairs, uint64_t(tile.FirstCandidate + begin) * 8), statistics}, P,
            uint64_t(count) * (P.Partitions / P.FoldSlots) * P.FoldCount);
    }
    template<class Command> void ScatterScores(Command& command, const CBMOrderedHistogramTile& tile,
        uint32_t begin, uint32_t count, id<MTLBuffer> tileScores, id<MTLBuffer> scores) {
        using Binding = typename Command::BindingType;
        P.Candidates = count;
        command.Dispatch("ScatterOrderedHistogramScores", {tileScores,
            Binding(CandidateIndices, uint64_t(tile.FirstCandidate + begin) * 4), scores}, P, count);
    }
    void Check() const {
        if (static_cast<const uint32_t*>(State.contents)[2]) throw std::runtime_error("Ordered histogram job capacity exceeded");
    }
};
