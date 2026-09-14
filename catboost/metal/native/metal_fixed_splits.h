#pragma once

#include "metal_greedy_kernels.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

// Host metadata for CUDA's greedy fixed-prefix search. Numeric routing and
// unconstrained scoring remain on the GPU. Configuration is immutable during
// a tree; per-tree caches contain no state needed across snapshot boundaries.
class CBMFixedSplits {
public:
    void Configure(uint32_t count, const uint32_t* features, uint32_t featureCount,
                   uint32_t candidateCount, const uint32_t* candidateFeatures,
                   const uint32_t* candidateBins, const uint8_t* candidateTypes) {
        if (count > (1u << 24) || (count && !features) ||
            (candidateCount && (!candidateFeatures || !candidateBins)))
            throw std::invalid_argument("Invalid counted fixed binary split buffers");
        std::vector<uint32_t> indices(featureCount, UINT32_MAX), counts(featureCount, 0);
        for (uint32_t candidate = 0; candidate < candidateCount; ++candidate) {
            const auto feature = candidateFeatures[candidate];
            if (feature >= featureCount) throw std::invalid_argument("Fixed split candidate feature is out of range");
            ++counts[feature];
            if (!candidateBins[candidate] && (!candidateTypes || !candidateTypes[candidate])) indices[feature] = candidate;
        }
        std::vector<CBMGreedySplit> splits;
        splits.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            const auto feature = features[i];
            if (feature >= featureCount || counts[feature] != 1 || indices[feature] == UINT32_MAX)
                throw std::invalid_argument("Fixed splits require a numeric feature with exactly one bin-zero candidate");
            splits.push_back({indices[feature], feature, 0, 0, -std::numeric_limits<float>::infinity(), 1, 0, 0});
        }
        Splits.swap(splits);
    }

    bool Enabled() const { return !Splits.empty(); }
    bool AtDepth(uint32_t depth) const { return depth < Splits.size(); }
    bool ForceAll(uint32_t depth) const { return uint64_t(depth) + 1 < Splits.size(); }
    CBMGreedySplit At(uint32_t depth, uint32_t leaf) const {
        auto split = Splits.at(depth);
        split.Leaf = leaf;
        return split;
    }

private:
    std::vector<CBMGreedySplit> Splits;
};

class CBMFixedSplitSearch {
public:
    explicit CBMFixedSplitSearch(const CBMFixedSplits& config) : Config(config) {}

    // CUDA's initial root has no terminal mark, even below min_data_in_leaf.
    // Later children use sampled parent row counts and depth termination.
    bool Begin(uint32_t leaves, const uint32_t* depths, const uint32_t* offsets,
               uint32_t maxDepth, uint32_t minDataInLeaf) {
        Cache.resize(leaves);
        New.resize(leaves, 1);
        ScoreMask.assign(leaves, 0);
        Depth = 0;
        for (uint32_t leaf = 0; leaf < leaves; ++leaf) Depth = std::max(Depth, depths[leaf]);
        bool any = false;
        for (uint32_t leaf = 0; leaf < leaves; ++leaf) {
            const bool root = leaves == 1 && depths[leaf] == 0;
            ScoreMask[leaf] = New[leaf] && (root ||
                (depths[leaf] < maxDepth && offsets[leaf + 1] - offsets[leaf] > minDataInLeaf));
            any |= ScoreMask[leaf] != 0;
            New[leaf] = 0;
        }
        return any && !Config.AtDepth(Depth);
    }

    bool IsForced() const { return Config.AtDepth(Depth); }
    bool ForceAll() const { return Config.ForceAll(Depth); }

    void Merge(CBMGreedySplit* winners) {
        for (uint32_t leaf = 0; leaf < Cache.size(); ++leaf) {
            if (ScoreMask[leaf]) Cache[leaf] = IsForced() ? Config.At(Depth, leaf) : winners[leaf];
            winners[leaf] = Cache[leaf];
        }
    }

    void Split(uint32_t parent, uint32_t right) {
        Cache.resize(std::max<size_t>(Cache.size(), size_t(right) + 1));
        New.resize(Cache.size(), 1);
        Cache[parent] = Cache[right] = {};
        New[parent] = New[right] = 1;
    }

private:
    const CBMFixedSplits& Config;
    uint32_t Depth = 0;
    std::vector<CBMGreedySplit> Cache;
    std::vector<uint8_t> New, ScoreMask;
};
