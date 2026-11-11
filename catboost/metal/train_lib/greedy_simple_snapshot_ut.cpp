#include "greedy_model.h"
#include "snapshot.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>

#include <limits>
#include <variant>

using namespace NCB;

namespace {
    TMetalGreedyTree SignedTree() {
        TMetalGreedyTree tree;
        // Physical left/right node order deliberately differs from leaf IDs.
        tree.Nodes = {{1, 2, 0, 1, 2, Max<ui32>()},
                      {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0}};
        tree.Values = {.375f, -.125f};
        tree.Weights = {-2.75f, 4.25f};
        tree.Info.completed_iterations = 1;
        tree.Info.finished = 1;
        tree.Info.node_count = 3;
        tree.Info.leaf_count = 2;
        return tree;
    }

    TModelSplit LookupSplit(ui32 feature, ui32 bin, ui32) {
        return TModelSplit(TFloatSplit(int(feature), float(bin) + .5f));
    }
}

Y_UNIT_TEST_SUITE(TMetalGreedySimpleSignedSnapshotWeights) {
    Y_UNIT_TEST(DefaultValidationAndAppendRejectNegativeWeightsWithoutMutation) {
        const auto tree = SignedTree();
        UNIT_ASSERT_EXCEPTION(ValidateMetalGreedyTree(tree.Nodes, tree.Values, tree.Weights, 2, 1),
            TCatBoostException);
        UNIT_ASSERT_EXCEPTION(MakeMetalGreedyTreeRoot(tree, 2, LookupSplit, 1), TCatBoostException);
        TMetalGreedySnapshotTrees trees;
        UNIT_ASSERT_EXCEPTION(trees.Append(tree, 2, 1), TCatBoostException);
        UNIT_ASSERT(trees.Empty() && trees.Nodes.empty() && trees.Values.empty() && trees.Weights.empty());
        UNIT_ASSERT(trees.NodeOffsets == TVector<ui64>({0}));
        UNIT_ASSERT(trees.LeafOffsets == TVector<ui64>({0}));
    }

    Y_UNIT_TEST(ExplicitSignedWeightsKeepTopologyAndValueValidation) {
        auto tree = SignedTree();
        const auto topology = ValidateMetalGreedyTree(tree.Nodes, tree.Values, tree.Weights, 2, 1, 1, true);
        UNIT_ASSERT_VALUES_EQUAL(topology.Depth, 1);
        UNIT_ASSERT(topology.Preorder == TVector<ui32>({0, 1, 2}));
        tree.Nodes[0].feature = 2;
        UNIT_ASSERT_EXCEPTION(ValidateMetalGreedyTree(tree.Nodes, tree.Values, tree.Weights, 2, 1, 1, true),
            TCatBoostException);
        tree.Nodes[0].feature = 1;
        tree.Values[0] = std::numeric_limits<float>::quiet_NaN();
        UNIT_ASSERT_EXCEPTION(ValidateMetalGreedyTree(tree.Nodes, tree.Values, tree.Weights, 2, 1, 1, true),
            TCatBoostException);
    }

    Y_UNIT_TEST(SignedOptInStillRejectsEveryNonfiniteWeight) {
        for (float weight : {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()}) {
            auto tree = SignedTree();
            tree.Weights[0] = weight;
            UNIT_ASSERT_EXCEPTION(ValidateMetalGreedyTree(tree.Nodes, tree.Values, tree.Weights, 2, 1, 1, true),
                TCatBoostException);
            UNIT_ASSERT_EXCEPTION(MakeMetalGreedyTreeRoot(tree, 2, LookupSplit, 1, 1, true), TCatBoostException);
            TMetalGreedySnapshotTrees trees;
            UNIT_ASSERT_EXCEPTION(trees.Append(tree, 2, 1, 1, true), TCatBoostException);
            UNIT_ASSERT(trees.Empty() && trees.Weights.empty());
        }
    }

    Y_UNIT_TEST(SerializedTreeStorePreservesSignedWeightsButDoesNotPersistValidationPermission) {
        const auto tree = SignedTree();
        TMetalGreedySnapshotTrees trees;
        UNIT_ASSERT_VALUES_EQUAL(trees.Append(tree, 2, 1, 1, true), 1);
        TStringStream bytes;
        ::Save(&bytes, trees);
        TMetalGreedySnapshotTrees restored;
        ::Load(&bytes, restored);
        const auto readback = restored.GetTree(0);
        UNIT_ASSERT(readback.Values == tree.Values);
        UNIT_ASSERT(readback.Weights == tree.Weights);
        UNIT_ASSERT_VALUES_EQUAL(readback.Nodes[1].leaf, 1);
        UNIT_ASSERT_VALUES_EQUAL(readback.Nodes[2].leaf, 0);
        UNIT_ASSERT_EXCEPTION(restored.Validate(2, 0, 1, 2, 1), TCatBoostException);
        restored.Validate(2, 0, 1, 2, 1, 1, true);
        UNIT_ASSERT_EXCEPTION(restored.Validate(2, 0, 1, 2, 1), TCatBoostException);
    }

    Y_UNIT_TEST(CompletedSnapshotNeedsTheCurrentConfigurationsExplicitOptIn) {
        TMetalSnapshot snapshot;
        snapshot.Greedy = true;
        snapshot.GreedyTrees.Append(SignedTree(), 2, 1, 1, true);
        snapshot.Depths = snapshot.GreedyTrees.Depths;
        snapshot.Predictions = {.375f, -.125f, .375f, -.125f};
        snapshot.History.TimeHistory.resize(1);
        // All requested iterations are already present, so validation must
        // apply even when resume will not call a native training session.
        UNIT_ASSERT_VALUES_EQUAL(snapshot.GreedyTrees.GetTreeCount(), 1);
        UNIT_ASSERT_EXCEPTION(snapshot.ValidateGreedy(4, 2, 0, 1, 2, 1), TCatBoostException);
        snapshot.ValidateGreedy(4, 2, 0, 1, 2, 1, 1, 1, 0, true);
        UNIT_ASSERT_EXCEPTION(snapshot.ValidateGreedy(4, 2, 0, 1, 2, 1, 1, 1, 0, false), TCatBoostException);
    }

    Y_UNIT_TEST(ModelRootPreservesSignedNodeWeightsAndTheLeafIdMapping) {
        const auto tree = SignedTree();
        auto root = MakeMetalGreedyTreeRoot(tree, 2, LookupSplit, 1, 1, true);
        UNIT_ASSERT(root->IsSplitNode() && root->Left && root->Right);
        root->Validate();
        root->Left->Validate();
        root->Right->Validate();
        UNIT_ASSERT(root->Left->NodeWeight && root->Right->NodeWeight);
        UNIT_ASSERT_VALUES_EQUAL(*root->Left->NodeWeight, 4.25);
        UNIT_ASSERT_VALUES_EQUAL(*root->Right->NodeWeight, -2.75);
        UNIT_ASSERT_VALUES_EQUAL(std::get<double>(root->Left->Value), -.125);
        UNIT_ASSERT_VALUES_EQUAL(std::get<double>(root->Right->Value), .375);
        UNIT_ASSERT_VALUES_EQUAL(root->SplitCondition->FloatFeature.FloatFeature, 1);
        UNIT_ASSERT_VALUES_EQUAL(root->SplitCondition->FloatFeature.Split, 2.5f);
    }
}
