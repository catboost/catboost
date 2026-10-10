#pragma once

#include "greedy_session.h"

#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>
#include <util/ysaveload.h>

#include <cmath>

namespace NCB {
    struct TMetalGreedyTopology {
        ui32 Depth = 0;
        // Parents precede children; reversing this order permits iterative
        // construction without depending on the session's physical node order.
        TVector<ui32> Preorder;
    };

    inline ui32 MetalGreedyLeafCapacity(ui32 policy, ui32 depth, ui32 maxLeaves) {
        CB_ENSURE(policy <= 2 && maxLeaves && maxLeaves <= 65536,
            "Metal greedy policy or leaf capacity is invalid");
        if (policy == 0) {
            CB_ENSURE(depth <= 16, "Metal Depthwise depth must be at most 16");
            return Min(maxLeaves, 1u << depth);
        }
        if (policy == 2) {
            CB_ENSURE(depth <= 65535, "Metal Region depth must be at most 65535");
            // CUDA Region uses MaxDepth+1, not the generic MaxLeaves option.
            return Min(maxLeaves, depth + 1);
        }
        return maxLeaves;
    }

    inline TMetalGreedyTopology ValidateMetalGreedyTree(
        TConstArrayRef<CBMGreedyNode> nodes,
        TConstArrayRef<float> values,
        TConstArrayRef<float> weights,
        ui32 featureCount,
        ui32 maxDepth = 65535,
        ui32 approxDimension = 1,
        bool allowSignedLeafWeights = false)
    {
        CB_ENSURE(approxDimension >= 1 && approxDimension <= 64 && !weights.empty() && weights.size() <= 65536 &&
            values.size() == weights.size() * approxDimension && nodes.size() == weights.size() * 2 - 1,
            "Metal greedy tree has inconsistent node or leaf counts");
        for (float value : values) CB_ENSURE(std::isfinite(value), "Metal greedy tree contains invalid leaf values");
        for (float weight : weights) CB_ENSURE(std::isfinite(weight) && (allowSignedLeafWeights || weight >= 0),
            "Metal greedy tree contains invalid leaf weights");
        TMetalGreedyTopology result;
        result.Preorder.reserve(nodes.size());
        TVector<ui8> visited(nodes.size(), 0), leaves(weights.size(), 0);
        TVector<std::pair<ui32, ui32>> pending{{0, 0}};
        while (!pending.empty()) {
            const auto [index, depth] = pending.back();
            pending.pop_back();
            CB_ENSURE(index < nodes.size() && !visited[index],
                "Metal greedy tree contains an invalid, shared, or cyclic node");
            CB_ENSURE(depth <= maxDepth, "Metal greedy tree exceeds its configured depth");
            visited[index] = 1;
            result.Preorder.push_back(index);
            result.Depth = Max(result.Depth, depth);
            const auto& node = nodes[index];
            if (node.leaf != Max<ui32>()) {
                CB_ENSURE(node.leaf < weights.size() && !leaves[node.leaf],
                    "Metal greedy tree contains an invalid or repeated leaf reference");
                leaves[node.leaf] = 1;
            } else {
                CB_ENSURE(node.feature < featureCount && node.bin < 256 && node.type <= 1,
                    "Metal greedy tree contains an invalid split");
                CB_ENSURE(depth < maxDepth, "Metal greedy tree exceeds its configured depth");
                pending.emplace_back(node.right, depth + 1);
                pending.emplace_back(node.left, depth + 1);
            }
        }
        CB_ENSURE(result.Preorder.size() == nodes.size(),
            "Metal greedy tree contains unreachable nodes");
        for (ui8 seen : leaves) {
            CB_ENSURE(seen, "Metal greedy tree contains an unreachable leaf value");
        }
        return result;
    }

    // Numeric storage for variable-size trees. Offset units are nodes and leaf
    // IDs respectively; vector values are leaf-major. Nodes stores six uint32 fields per node, with no ABI
    // padding, pointers, or fixed depth/leaf stride. The outer snapshot owns the
    // full raw training cursor, absolute iteration/RNG offset, history, and data
    // checksum; it must save this untrimmed state before best-model selection.
    struct TMetalGreedySnapshotTrees {
        TVector<ui64> NodeOffsets{0};
        TVector<ui64> LeafOffsets{0};
        TVector<ui32> Nodes;
        TVector<float> Values;
        TVector<float> Weights;
        TVector<ui32> Depths;

        Y_SAVELOAD_DEFINE(NodeOffsets, LeafOffsets, Nodes, Values, Weights, Depths);

        size_t GetTreeCount() const noexcept {
            return Depths.size();
        }

        bool Empty() const noexcept {
            return Depths.empty();
        }

        TMetalGreedyTree GetTree(size_t tree, ui32 approxDimension = 1) const {
            CB_ENSURE(approxDimension >= 1 && approxDimension <= 64 && NodeOffsets.size() == Depths.size() + 1 &&
                LeafOffsets.size() == Depths.size() + 1 && tree < Depths.size() && Nodes.size() % 6 == 0,
                "Metal greedy snapshot has inconsistent offset counts");
            const ui64 nodeBegin = NodeOffsets[tree], nodeEnd = NodeOffsets[tree + 1];
            const ui64 leafBegin = LeafOffsets[tree], leafEnd = LeafOffsets[tree + 1];
            CB_ENSURE(nodeBegin < nodeEnd && nodeEnd <= Nodes.size() / 6 &&
                leafBegin < leafEnd && leafEnd <= Weights.size() && Values.size() == Weights.size() * approxDimension &&
                leafEnd - leafBegin <= 65536 && nodeEnd - nodeBegin == (leafEnd - leafBegin) * 2 - 1,
                "Metal greedy snapshot contains invalid tree offsets");
            TMetalGreedyTree result;
            result.Nodes.reserve(nodeEnd - nodeBegin);
            for (ui64 n = nodeBegin; n < nodeEnd; ++n) {
                const ui64 p = n * 6;
                result.Nodes.push_back({Nodes[p], Nodes[p + 1], Nodes[p + 2],
                    Nodes[p + 3], Nodes[p + 4], Nodes[p + 5]});
            }
            result.Values.assign(Values.begin() + leafBegin * approxDimension, Values.begin() + leafEnd * approxDimension);
            result.Weights.assign(Weights.begin() + leafBegin, Weights.begin() + leafEnd);
            result.Info.node_count = result.Nodes.size();
            result.Info.leaf_count = result.Weights.size();
            return result;
        }

        ui32 Append(const TMetalGreedyTree& tree, ui32 featureCount, ui32 maxDepth = 65535, ui32 approxDimension = 1,
                    bool allowSignedLeafWeights = false) {
            const auto topology = ValidateMetalGreedyTree(tree.Nodes, tree.Values, tree.Weights, featureCount, maxDepth,
                approxDimension, allowSignedLeafWeights);
            CB_ENSURE(NodeOffsets.size() == Depths.size() + 1 && LeafOffsets.size() == Depths.size() + 1 &&
                NodeOffsets.front() == 0 && LeafOffsets.front() == 0 && Nodes.size() % 6 == 0 &&
                NodeOffsets.back() == Nodes.size() / 6 && LeafOffsets.back() == Weights.size() &&
                Values.size() == Weights.size() * approxDimension, "Metal greedy snapshot has inconsistent storage");
            for (const auto& node : tree.Nodes) {
                Nodes.insert(Nodes.end(), {node.feature, node.bin, node.type, node.left, node.right, node.leaf});
            }
            Values.insert(Values.end(), tree.Values.begin(), tree.Values.end());
            Weights.insert(Weights.end(), tree.Weights.begin(), tree.Weights.end());
            NodeOffsets.push_back(Nodes.size() / 6);
            LeafOffsets.push_back(Weights.size());
            Depths.push_back(topology.Depth);
            return topology.Depth;
        }

        void Validate(ui32 featureCount, ui32 policy, ui32 maxDepth, ui32 maxLeaves, ui32 maxIterations,
                      ui32 approxDimension = 1, bool allowSignedLeafWeights = false) const {
            const ui32 capacity = MetalGreedyLeafCapacity(policy, maxDepth, maxLeaves);
            const ui32 depthBound = Min(maxDepth, capacity - 1);
            CB_ENSURE(Depths.size() <= maxIterations && NodeOffsets.size() == Depths.size() + 1 &&
                LeafOffsets.size() == Depths.size() + 1 && NodeOffsets.front() == 0 &&
                LeafOffsets.front() == 0 && Nodes.size() % 6 == 0 &&
                NodeOffsets.back() == Nodes.size() / 6 && LeafOffsets.back() == Weights.size() &&
                Values.size() == Weights.size() * approxDimension, "Metal greedy snapshot has inconsistent storage");
            for (size_t t = 0; t < Depths.size(); ++t) {
                const auto tree = GetTree(t, approxDimension);
                CB_ENSURE(tree.Weights.size() <= capacity, "Metal greedy snapshot exceeds its leaf capacity");
                const auto topology = ValidateMetalGreedyTree(tree.Nodes, tree.Values, tree.Weights, featureCount, depthBound,
                    approxDimension, allowSignedLeafWeights);
                CB_ENSURE(topology.Depth == Depths[t], "Metal greedy snapshot depth does not match its tree");
                if (policy == 2) {
                    for (const auto& node : tree.Nodes) {
                        if (node.leaf == Max<ui32>()) {
                            CB_ENSURE(tree.Nodes[node.left].leaf != Max<ui32>() ||
                                tree.Nodes[node.right].leaf != Max<ui32>(),
                                "Metal Region snapshot must contain a single growing path");
                        }
                    }
                }
            }
        }
    };
}
