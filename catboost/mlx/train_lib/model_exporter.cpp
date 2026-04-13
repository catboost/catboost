#include "model_exporter.h"

#include <catboost/private/libs/algo/helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/model/model_build_helper.h>

#include <mlx/mlx.h>

#include <unordered_map>

namespace mx = mlx::core;

namespace NCatboostMlx {

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /// Convert a GPU split descriptor (feature column index + bin threshold) to
    /// CatBoost's TModelSplit, using the quantization borders to recover the
    /// original float boundary value.
    ///
    /// \param featureColumnIdx  GPU-local feature index (indexes into externalFeatureIndices).
    /// \param binId             Quantized bin threshold from the split.
    /// \param externalFeatureIndices  Maps GPU feature index → CatBoost external feature index.
    /// \param featuresLayout    Feature layout for internal/external index conversion.
    /// \param quantizedFeaturesInfo   Quantization borders per feature.
    /// \param context           Human-readable context string for error messages.
    static TModelSplit ConvertSplit(
        ui32 featureColumnIdx,
        ui32 binId,
        const TVector<ui32>& externalFeatureIndices,
        const NCB::TFeaturesLayout& featuresLayout,
        const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TString& context
    ) {
        CB_ENSURE(featureColumnIdx < externalFeatureIndices.size(),
            "CatBoost-MLX: " << context << ": split feature column " << featureColumnIdx
            << " out of range (max " << externalFeatureIndices.size() - 1 << ")");

        const ui32 externalIdx = externalFeatureIndices[featureColumnIdx];
        const auto internalIdx = featuresLayout.GetInternalFeatureIdx<NCB::EFeatureType::Float>(externalIdx);
        CB_ENSURE(internalIdx,
            "CatBoost-MLX: " << context << ": external feature " << externalIdx
            << " not found in features layout");

        const auto& borders = quantizedFeaturesInfo.GetBorders(
            NCB::TFloatFeatureIdx(static_cast<ui32>(*internalIdx)));

        CB_ENSURE(binId < borders.size(),
            "CatBoost-MLX: " << context << ": split bin " << binId
            << " out of range for feature " << externalIdx
            << " (has " << borders.size() << " borders)");

        return TModelSplit(TFloatSplit{
            static_cast<int>(*internalIdx),
            borders[binId]
        });
    }

    /// Set a TNonSymmetricTreeNode's leaf value from a flat leaf-values buffer.
    ///
    /// For approxDimension == 1, the node value is set to a scalar double.
    /// For approxDimension > 1, the node value is set to a TVector<double> of
    /// length approxDimension, taken from leafPtr at position
    /// leafIdx * approxDimension .. leafIdx * approxDimension + approxDimension - 1.
    static void SetLeafValue(
        TNonSymmetricTreeNode& node,
        const float* leafPtr,
        ui32 leafIdx,
        ui32 approxDimension
    ) {
        if (approxDimension == 1) {
            node.Value = static_cast<double>(leafPtr[leafIdx]);
        } else {
            TVector<double> dimValues(approxDimension);
            for (ui32 dim = 0; dim < approxDimension; ++dim) {
                dimValues[dim] = static_cast<double>(
                    leafPtr[leafIdx * approxDimension + dim]);
            }
            node.Value = std::move(dimValues);
        }
    }

    /// Recursively build a TNonSymmetricTreeNode subtree for a Depthwise tree.
    ///
    /// The Depthwise tree is a full binary tree of depth `maxDepth`.  Internal
    /// nodes are stored in BFS order in `nodeSplits` (index 0 = root), where
    /// node i's left child is 2*i+1 and right child is 2*i+2.  Leaves occupy
    /// BFS positions [2^maxDepth - 1 .. 2^(maxDepth+1) - 2] and map to dense
    /// leaf indices [0 .. 2^maxDepth - 1].
    ///
    /// \param bfsIdx       BFS index of the current node (0 = root).
    /// \param currentDepth Depth of the current node (root = 0).
    /// \param maxDepth     Total tree depth (leaf depth).
    /// \param nodeSplits   All internal-node split descriptors in BFS order.
    /// \param leafPtr      Flat leaf-values buffer (leaf-major, dim-minor).
    /// \param approxDimension  Output dimension count.
    /// \param externalFeatureIndices, featuresLayout, quantizedFeaturesInfo
    ///                     Feature metadata for split conversion.
    /// \param treeIdx      Tree index (for error messages only).
    static THolder<TNonSymmetricTreeNode> BuildDepthwiseNode(
        ui32 bfsIdx,
        ui32 currentDepth,
        ui32 maxDepth,
        const TVector<TObliviousSplitLevel>& nodeSplits,
        const float* leafPtr,
        ui32 approxDimension,
        const TVector<ui32>& externalFeatureIndices,
        const NCB::TFeaturesLayout& featuresLayout,
        const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        ui32 treeIdx
    ) {
        auto node = MakeHolder<TNonSymmetricTreeNode>();

        if (currentDepth == maxDepth) {
            // Leaf node: BFS index bfsIdx is in [2^maxDepth-1 .. 2^(maxDepth+1)-2].
            // Dense leaf index = bfsIdx - (2^maxDepth - 1).
            const ui32 firstLeafBfsIdx = (1u << maxDepth) - 1u;
            CB_ENSURE(bfsIdx >= firstLeafBfsIdx,
                "CatBoost-MLX: Depthwise tree " << treeIdx
                << ": unexpected leaf bfsIdx " << bfsIdx
                << " at maxDepth " << maxDepth);
            const ui32 leafIdx = bfsIdx - firstLeafBfsIdx;
            SetLeafValue(*node, leafPtr, leafIdx, approxDimension);
            return node;
        }

        // Internal node: use the split stored at BFS position bfsIdx.
        CB_ENSURE(bfsIdx < nodeSplits.size(),
            "CatBoost-MLX: Depthwise tree " << treeIdx
            << ": bfsIdx " << bfsIdx << " out of range (nodeSplits.size()="
            << nodeSplits.size() << ")");

        const auto& splitLevel = nodeSplits[bfsIdx];
        const TString ctx = TStringBuilder()
            << "Depthwise tree " << treeIdx << " node " << bfsIdx;

        node->SplitCondition = ConvertSplit(
            splitLevel.FeatureColumnIdx,
            splitLevel.BinThreshold,
            externalFeatureIndices,
            featuresLayout,
            quantizedFeaturesInfo,
            ctx
        );

        node->Left = BuildDepthwiseNode(
            2 * bfsIdx + 1, currentDepth + 1, maxDepth,
            nodeSplits, leafPtr, approxDimension,
            externalFeatureIndices, featuresLayout, quantizedFeaturesInfo,
            treeIdx
        );
        node->Right = BuildDepthwiseNode(
            2 * bfsIdx + 2, currentDepth + 1, maxDepth,
            nodeSplits, leafPtr, approxDimension,
            externalFeatureIndices, featuresLayout, quantizedFeaturesInfo,
            treeIdx
        );

        return node;
    }

    /// Recursively build a TNonSymmetricTreeNode subtree for a Lossguide tree.
    ///
    /// Lossguide trees are unbalanced: only split nodes are present in
    /// `nodeSplitMap` (keyed by BFS index).  A BFS index absent from the map
    /// is a leaf.  Dense leaf indices are recovered via `reverseLeafMap`
    /// (BFS index → dense leaf index k), which must be pre-built from
    /// `TLossguideTreeStructure::LeafBfsIds`.
    ///
    /// \param bfsIdx          BFS index of the current node (0 = root).
    /// \param nodeSplitMap    Sparse map: BFS index → split descriptor (only internal nodes).
    /// \param reverseLeafMap  Reverse of LeafBfsIds: BFS index → dense leaf index.
    /// \param leafPtr         Flat leaf-values buffer (leaf-major, dim-minor).
    /// \param approxDimension Output dimension count.
    /// \param externalFeatureIndices, featuresLayout, quantizedFeaturesInfo
    ///                        Feature metadata for split conversion.
    /// \param treeIdx         Tree index (for error messages only).
    static THolder<TNonSymmetricTreeNode> BuildLossguideNode(
        ui32 bfsIdx,
        const std::unordered_map<ui32, TObliviousSplitLevel>& nodeSplitMap,
        const std::unordered_map<ui32, ui32>& reverseLeafMap,
        const float* leafPtr,
        ui32 approxDimension,
        const TVector<ui32>& externalFeatureIndices,
        const NCB::TFeaturesLayout& featuresLayout,
        const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        ui32 treeIdx
    ) {
        auto node = MakeHolder<TNonSymmetricTreeNode>();

        auto splitIt = nodeSplitMap.find(bfsIdx);
        if (splitIt == nodeSplitMap.end()) {
            // Leaf node: look up dense leaf index in the reverse map.
            auto leafIt = reverseLeafMap.find(bfsIdx);
            CB_ENSURE(leafIt != reverseLeafMap.end(),
                "CatBoost-MLX: Lossguide tree " << treeIdx
                << ": BFS node " << bfsIdx
                << " is neither a split node nor a known leaf");
            SetLeafValue(*node, leafPtr, leafIt->second, approxDimension);
            return node;
        }

        // Internal node.
        const auto& splitLevel = splitIt->second;
        const TString ctx = TStringBuilder()
            << "Lossguide tree " << treeIdx << " node " << bfsIdx;

        node->SplitCondition = ConvertSplit(
            splitLevel.FeatureColumnIdx,
            splitLevel.BinThreshold,
            externalFeatureIndices,
            featuresLayout,
            quantizedFeaturesInfo,
            ctx
        );

        node->Left = BuildLossguideNode(
            2 * bfsIdx + 1,
            nodeSplitMap, reverseLeafMap, leafPtr, approxDimension,
            externalFeatureIndices, featuresLayout, quantizedFeaturesInfo,
            treeIdx
        );
        node->Right = BuildLossguideNode(
            2 * bfsIdx + 2,
            nodeSplitMap, reverseLeafMap, leafPtr, approxDimension,
            externalFeatureIndices, featuresLayout, quantizedFeaturesInfo,
            treeIdx
        );

        return node;
    }

    // -------------------------------------------------------------------------
    // Grow-policy–specific export implementations
    // -------------------------------------------------------------------------

    /// Export all SymmetricTree (oblivious) trees via TObliviousTreeBuilder.
    ///
    /// Each tree has one split per depth level shared across all leaves at that
    /// level, so the structure is stored as TVector<TBestSplitProperties> in
    /// TObliviousTreeStructure::SplitProperties.
    static void ExportSymmetricTrees(
        const TBoostingResult& boostingResult,
        const TVector<ui32>& externalFeatureIndices,
        const NCB::TFeaturesLayout& featuresLayout,
        const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        ui32 approxDimension,
        const TVector<TFloatFeature>& floatFeatures,
        const TVector<TCatFeature>& catFeatures,
        const TVector<TTextFeature>& textFeatures,
        const TVector<TEmbeddingFeature>& embeddingFeatures,
        TFullModel& fullModel
    ) {
        TObliviousTreeBuilder builder(
            floatFeatures, catFeatures, textFeatures, embeddingFeatures,
            static_cast<int>(approxDimension)
        );

        for (ui32 treeIdx = 0; treeIdx < boostingResult.NumIterations; ++treeIdx) {
            const auto& treeStructure = boostingResult.TreeStructures[treeIdx];
            const auto& leafValuesArr = boostingResult.TreeLeafValues[treeIdx];
            const ui32 depth = treeStructure.Splits.size();

            CB_ENSURE(depth > 0,
                "CatBoost-MLX: SymmetricTree " << treeIdx << " has no splits");
            CB_ENSURE(treeStructure.SplitProperties.size() == depth,
                "CatBoost-MLX: SymmetricTree " << treeIdx
                << " split/properties size mismatch");

            // Convert splits: TBestSplitProperties → TModelSplit
            TVector<TModelSplit> modelSplits;
            modelSplits.reserve(depth);
            for (ui32 level = 0; level < depth; ++level) {
                const auto& sp = treeStructure.SplitProperties[level];
                const TString ctx = TStringBuilder()
                    << "SymmetricTree " << treeIdx << " level " << level;
                modelSplits.push_back(ConvertSplit(
                    sp.FeatureId, sp.BinId,
                    externalFeatureIndices, featuresLayout, quantizedFeaturesInfo,
                    ctx
                ));
            }

            // Materialize leaf values
            mx::eval(leafValuesArr);
            const ui32 numLeaves = 1u << depth;
            const float* leafPtr = leafValuesArr.data<float>();

            if (approxDimension == 1) {
                CB_ENSURE(static_cast<ui32>(leafValuesArr.size()) == numLeaves,
                    "CatBoost-MLX: SymmetricTree " << treeIdx
                    << " leaf count mismatch: " << leafValuesArr.size()
                    << " vs expected " << numLeaves);

                TVector<double> leafValues(numLeaves);
                for (ui32 i = 0; i < numLeaves; ++i) {
                    leafValues[i] = static_cast<double>(leafPtr[i]);
                }
                builder.AddTree(modelSplits, leafValues, TConstArrayRef<double>{});
            } else {
                CB_ENSURE(static_cast<ui32>(leafValuesArr.size()) == numLeaves * approxDimension,
                    "CatBoost-MLX: SymmetricTree " << treeIdx
                    << " leaf count mismatch: " << leafValuesArr.size()
                    << " vs expected " << numLeaves * approxDimension);

                // Deinterleave [numLeaves * approxDim] (leaf-major) → [approxDim][numLeaves]
                TVector<TVector<double>> multiDimLeafValues(approxDimension);
                for (ui32 dim = 0; dim < approxDimension; ++dim) {
                    multiDimLeafValues[dim].resize(numLeaves);
                    for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
                        multiDimLeafValues[dim][leaf] = static_cast<double>(
                            leafPtr[leaf * approxDimension + dim]);
                    }
                }
                builder.AddTree(modelSplits, multiDimLeafValues, TConstArrayRef<double>{});
            }
        }

        builder.Build(fullModel.ModelTrees.GetMutable());
    }

    /// Export all Depthwise (non-symmetric, full binary) trees via
    /// TNonSymmetricTreeModelBuilder.
    ///
    /// Each tree is a full binary tree of depth `tree.Depth`.  The
    /// `NodeSplits` array stores internal-node splits in BFS order;
    /// leaves are at depth `tree.Depth` and are identified by their
    /// BFS position.
    static void ExportDepthwiseTrees(
        const TBoostingResult& boostingResult,
        const TVector<ui32>& externalFeatureIndices,
        const NCB::TFeaturesLayout& featuresLayout,
        const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        ui32 approxDimension,
        const TVector<TFloatFeature>& floatFeatures,
        const TVector<TCatFeature>& catFeatures,
        const TVector<TTextFeature>& textFeatures,
        const TVector<TEmbeddingFeature>& embeddingFeatures,
        TFullModel& fullModel
    ) {
        TNonSymmetricTreeModelBuilder builder(
            floatFeatures, catFeatures, textFeatures, embeddingFeatures,
            static_cast<int>(approxDimension)
        );

        for (ui32 treeIdx = 0; treeIdx < boostingResult.NumIterations; ++treeIdx) {
            const auto& tree = boostingResult.DepthwiseTreeStructures[treeIdx];
            const auto& leafValuesArr = boostingResult.TreeLeafValues[treeIdx];
            const ui32 depth = tree.Depth;

            // A full binary tree of depth D has 2^D - 1 internal nodes.
            const ui32 expectedNodes = (depth == 0) ? 0u : (1u << depth) - 1u;
            CB_ENSURE(tree.NodeSplits.size() == expectedNodes,
                "CatBoost-MLX: Depthwise tree " << treeIdx
                << " NodeSplits size mismatch: got " << tree.NodeSplits.size()
                << " expected " << expectedNodes << " for depth " << depth);

            const ui32 numLeaves = 1u << depth;

            // Materialize leaf values from the GPU array.
            mx::eval(leafValuesArr);
            const ui32 expectedLeafElems = numLeaves * approxDimension;
            CB_ENSURE(static_cast<ui32>(leafValuesArr.size()) == expectedLeafElems,
                "CatBoost-MLX: Depthwise tree " << treeIdx
                << " leaf value count mismatch: " << leafValuesArr.size()
                << " vs expected " << expectedLeafElems);

            const float* leafPtr = leafValuesArr.data<float>();

            if (depth == 0) {
                // Degenerate case: tree is a single leaf (root was never split).
                auto root = MakeHolder<TNonSymmetricTreeNode>();
                SetLeafValue(*root, leafPtr, /*leafIdx=*/0, approxDimension);
                builder.AddTree(std::move(root));
            } else {
                auto root = BuildDepthwiseNode(
                    /*bfsIdx=*/0, /*currentDepth=*/0, depth,
                    tree.NodeSplits, leafPtr, approxDimension,
                    externalFeatureIndices, featuresLayout, quantizedFeaturesInfo,
                    treeIdx
                );
                builder.AddTree(std::move(root));
            }
        }

        builder.Build(fullModel.ModelTrees.GetMutable());
    }

    /// Export all Lossguide (best-first leaf-wise) trees via
    /// TNonSymmetricTreeModelBuilder.
    ///
    /// Each tree is an unbalanced binary tree stored as a sparse map
    /// (NodeSplitMap) of BFS index → split for internal nodes.
    /// LeafBfsIds maps dense leaf indices to BFS node indices; we invert
    /// this to recover the dense leaf index during recursion.
    static void ExportLossguideTrees(
        const TBoostingResult& boostingResult,
        const TVector<ui32>& externalFeatureIndices,
        const NCB::TFeaturesLayout& featuresLayout,
        const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        ui32 approxDimension,
        const TVector<TFloatFeature>& floatFeatures,
        const TVector<TCatFeature>& catFeatures,
        const TVector<TTextFeature>& textFeatures,
        const TVector<TEmbeddingFeature>& embeddingFeatures,
        TFullModel& fullModel
    ) {
        TNonSymmetricTreeModelBuilder builder(
            floatFeatures, catFeatures, textFeatures, embeddingFeatures,
            static_cast<int>(approxDimension)
        );

        for (ui32 treeIdx = 0; treeIdx < boostingResult.NumIterations; ++treeIdx) {
            const auto& tree = boostingResult.LossguideTreeStructures[treeIdx];
            const auto& leafValuesArr = boostingResult.TreeLeafValues[treeIdx];
            const ui32 numLeaves = tree.NumLeaves;

            CB_ENSURE(numLeaves >= 1,
                "CatBoost-MLX: Lossguide tree " << treeIdx
                << " has NumLeaves < 1");
            CB_ENSURE(tree.LeafBfsIds.size() == numLeaves,
                "CatBoost-MLX: Lossguide tree " << treeIdx
                << " LeafBfsIds size " << tree.LeafBfsIds.size()
                << " != NumLeaves " << numLeaves);

            // Materialize leaf values from the GPU array.
            mx::eval(leafValuesArr);
            const ui32 expectedLeafElems = numLeaves * approxDimension;
            CB_ENSURE(static_cast<ui32>(leafValuesArr.size()) == expectedLeafElems,
                "CatBoost-MLX: Lossguide tree " << treeIdx
                << " leaf value count mismatch: " << leafValuesArr.size()
                << " vs expected " << expectedLeafElems);

            const float* leafPtr = leafValuesArr.data<float>();

            // Build reverse map: BFS node index → dense leaf index.
            // This is the inverse of tree.LeafBfsIds[k] = bfsIdx.
            std::unordered_map<ui32, ui32> reverseLeafMap;
            reverseLeafMap.reserve(numLeaves);
            for (ui32 k = 0; k < numLeaves; ++k) {
                const ui32 bfsId = tree.LeafBfsIds[k];
                CB_ENSURE(reverseLeafMap.find(bfsId) == reverseLeafMap.end(),
                    "CatBoost-MLX: Lossguide tree " << treeIdx
                    << " duplicate BFS leaf id " << bfsId
                    << " at dense leaf indices "
                    << reverseLeafMap.at(bfsId) << " and " << k);
                reverseLeafMap[bfsId] = k;
            }

            if (numLeaves == 1) {
                // Degenerate case: the root was never split — single-leaf tree.
                // The only valid BFS id is 0 (root), and it must be in LeafBfsIds.
                CB_ENSURE(reverseLeafMap.count(0u),
                    "CatBoost-MLX: Lossguide tree " << treeIdx
                    << " has 1 leaf but LeafBfsIds[0] != 0 (got "
                    << tree.LeafBfsIds[0] << ")");
                auto root = MakeHolder<TNonSymmetricTreeNode>();
                SetLeafValue(*root, leafPtr, /*leafIdx=*/0, approxDimension);
                builder.AddTree(std::move(root));
            } else {
                auto root = BuildLossguideNode(
                    /*bfsIdx=*/0,
                    tree.NodeSplitMap, reverseLeafMap,
                    leafPtr, approxDimension,
                    externalFeatureIndices, featuresLayout, quantizedFeaturesInfo,
                    treeIdx
                );
                builder.AddTree(std::move(root));
            }
        }

        builder.Build(fullModel.ModelTrees.GetMutable());
    }

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    TFullModel ConvertToFullModel(
        const TBoostingResult& boostingResult,
        const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const NCB::TFeaturesLayout& featuresLayout,
        const TVector<TCFeature>& gpuFeatures,
        const TVector<ui32>& externalFeatureIndices,
        ui32 approxDimension,
        const NCatboostOptions::TCatBoostOptions& catboostOptions
    ) {
        CB_ENSURE(approxDimension >= 1,
            "CatBoost-MLX: approxDimension must be >= 1. Got: " << approxDimension);
        CB_ENSURE(boostingResult.NumIterations > 0,
            "CatBoost-MLX: No trees to export");
        CB_ENSURE(externalFeatureIndices.size() == gpuFeatures.size(),
            "CatBoost-MLX: Feature index mapping size mismatch");

        const TString growPolicyStr = [&]() -> TString {
            switch (boostingResult.GrowPolicy) {
                case EGrowPolicy::SymmetricTree: return "SymmetricTree";
                case EGrowPolicy::Depthwise:     return "Depthwise";
                case EGrowPolicy::Lossguide:     return "Lossguide";
            }
            return "Unknown";
        }();

        CATBOOST_INFO_LOG << "CatBoost-MLX: Exporting model with "
            << boostingResult.NumIterations << " trees"
            << " (GrowPolicy=" << growPolicyStr << ")" << Endl;

        // Step 1: Build feature metadata using existing CatBoost helper
        auto floatFeatures = CreateFloatFeatures(featuresLayout, quantizedFeaturesInfo);
        TVector<TCatFeature>       catFeatures;
        TVector<TTextFeature>      textFeatures;
        TVector<TEmbeddingFeature> embeddingFeatures;

        CATBOOST_INFO_LOG << "CatBoost-MLX: Model has " << floatFeatures.size()
            << " float features" << Endl;

        // Step 2: Validate that the tree structure vectors match the grow policy
        //         and have the expected count.
        switch (boostingResult.GrowPolicy) {
            case EGrowPolicy::SymmetricTree:
                CB_ENSURE(
                    boostingResult.TreeStructures.size() == boostingResult.NumIterations,
                    "CatBoost-MLX: SymmetricTree: TreeStructures.size() "
                    << boostingResult.TreeStructures.size()
                    << " != NumIterations " << boostingResult.NumIterations);
                break;
            case EGrowPolicy::Depthwise:
                CB_ENSURE(
                    boostingResult.DepthwiseTreeStructures.size() == boostingResult.NumIterations,
                    "CatBoost-MLX: Depthwise: DepthwiseTreeStructures.size() "
                    << boostingResult.DepthwiseTreeStructures.size()
                    << " != NumIterations " << boostingResult.NumIterations);
                break;
            case EGrowPolicy::Lossguide:
                CB_ENSURE(
                    boostingResult.LossguideTreeStructures.size() == boostingResult.NumIterations,
                    "CatBoost-MLX: Lossguide: LossguideTreeStructures.size() "
                    << boostingResult.LossguideTreeStructures.size()
                    << " != NumIterations " << boostingResult.NumIterations);
                break;
        }
        CB_ENSURE(
            boostingResult.TreeLeafValues.size() == boostingResult.NumIterations,
            "CatBoost-MLX: TreeLeafValues.size() " << boostingResult.TreeLeafValues.size()
            << " != NumIterations " << boostingResult.NumIterations);

        // Step 3: Export trees using the appropriate builder for the grow policy.
        TFullModel fullModel;

        switch (boostingResult.GrowPolicy) {
            case EGrowPolicy::SymmetricTree:
                ExportSymmetricTrees(
                    boostingResult, externalFeatureIndices,
                    featuresLayout, quantizedFeaturesInfo, approxDimension,
                    floatFeatures, catFeatures, textFeatures, embeddingFeatures,
                    fullModel
                );
                break;

            case EGrowPolicy::Depthwise:
                ExportDepthwiseTrees(
                    boostingResult, externalFeatureIndices,
                    featuresLayout, quantizedFeaturesInfo, approxDimension,
                    floatFeatures, catFeatures, textFeatures, embeddingFeatures,
                    fullModel
                );
                break;

            case EGrowPolicy::Lossguide:
                ExportLossguideTrees(
                    boostingResult, externalFeatureIndices,
                    featuresLayout, quantizedFeaturesInfo, approxDimension,
                    floatFeatures, catFeatures, textFeatures, embeddingFeatures,
                    fullModel
                );
                break;
        }

        // Step 4: Finalize the model
        fullModel.SetScaleAndBias({1.0, {}});
        fullModel.UpdateDynamicData();

        // Store training parameters in model info
        fullModel.ModelInfo["params"] = ToString(catboostOptions);

        CATBOOST_INFO_LOG << "CatBoost-MLX: Model export complete. "
            << boostingResult.NumIterations << " " << growPolicyStr << " trees, "
            << floatFeatures.size() << " float features" << Endl;

        return fullModel;
    }

}  // namespace NCatboostMlx
