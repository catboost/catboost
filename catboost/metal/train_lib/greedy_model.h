#pragma once

#include "greedy_snapshot.h"

#include <catboost/libs/model/model_build_helper.h>

namespace NCB {
    // The split lookup maps prepared feature/bin/type IDs to original model
    // features and borders (including one-hot/CTR metadata owned by the caller).
    // Building and validating this host topology never evaluates training rows.
    template <class TSplitLookup>
    THolder<TNonSymmetricTreeNode> MakeMetalGreedyTreeRoot(
        const TMetalGreedyTree& tree,
        ui32 featureCount,
        TSplitLookup&& splitLookup,
        ui32 maxDepth = 16,
        ui32 approxDimension = 1,
        bool allowSignedLeafWeights = false)
    {
        const auto topology = ValidateMetalGreedyTree(tree.Nodes, tree.Values, tree.Weights, featureCount, maxDepth,
            approxDimension, allowSignedLeafWeights);
        TVector<ui32> compactSize(tree.Nodes.size(), 1);
        // TNonSymmetricTreeModelBuilder packs one terminal child into its
        // parent. Calculate its real preorder offsets before its ui16 casts.
        for (auto it = topology.Preorder.rbegin(); it != topology.Preorder.rend(); ++it) {
            const auto& node = tree.Nodes[*it];
            if (node.leaf == Max<ui32>()) {
                const bool leftSplit = tree.Nodes[node.left].leaf == Max<ui32>();
                const bool rightSplit = tree.Nodes[node.right].leaf == Max<ui32>();
                if (leftSplit == rightSplit) {
                    CB_ENSURE(1u + compactSize[node.left] <= Max<ui16>(),
                        "Metal greedy tree cannot fit the native model's uint16 child offsets");
                    compactSize[*it] += compactSize[node.left] + compactSize[node.right];
                } else {
                    compactSize[*it] += compactSize[leftSplit ? node.left : node.right];
                }
            }
        }
        TVector<THolder<TNonSymmetricTreeNode>> built(tree.Nodes.size());
        for (auto it = topology.Preorder.rbegin(); it != topology.Preorder.rend(); ++it) {
            const auto& node = tree.Nodes[*it];
            auto value = MakeHolder<TNonSymmetricTreeNode>();
            if (node.leaf == Max<ui32>()) {
                value->SplitCondition = splitLookup(node.feature, node.bin, node.type);
                value->Left = std::move(built[node.left]);
                value->Right = std::move(built[node.right]);
            } else {
                if (approxDimension == 1) value->Value = double(tree.Values[node.leaf]);
                else {
                    TVector<double> dimensions(approxDimension);
                    for (ui32 k = 0; k < approxDimension; ++k)
                        dimensions[k] = tree.Values[ui64(node.leaf) * approxDimension + k];
                    value->Value = std::move(dimensions);
                }
                value->NodeWeight = double(tree.Weights[node.leaf]);
            }
            built[*it] = std::move(value);
        }
        return std::move(built.front());
    }
}
