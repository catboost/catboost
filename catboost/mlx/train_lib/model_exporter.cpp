#include "model_exporter.h"

#include <catboost/private/libs/algo/helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/model/model_build_helper.h>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace NCatboostMlx {

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

        CATBOOST_INFO_LOG << "CatBoost-MLX: Exporting model with "
            << boostingResult.NumIterations << " trees" << Endl;

        // Step 1: Build feature metadata using existing CatBoost helper
        auto floatFeatures = CreateFloatFeatures(featuresLayout, quantizedFeaturesInfo);
        TVector<TCatFeature> catFeatures;
        TVector<TTextFeature> textFeatures;
        TVector<TEmbeddingFeature> embeddingFeatures;

        CATBOOST_INFO_LOG << "CatBoost-MLX: Model has " << floatFeatures.size()
            << " float features" << Endl;

        // Step 2: Create the oblivious tree builder
        TObliviousTreeBuilder builder(
            floatFeatures,
            catFeatures,
            textFeatures,
            embeddingFeatures,
            static_cast<int>(approxDimension)
        );

        // Step 3: Add each tree
        for (ui32 treeIdx = 0; treeIdx < boostingResult.NumIterations; ++treeIdx) {
            const auto& treeStructure = boostingResult.TreeStructures[treeIdx];
            const auto& leafValuesArr = boostingResult.TreeLeafValues[treeIdx];
            const ui32 depth = treeStructure.Splits.size();

            CB_ENSURE(depth > 0, "CatBoost-MLX: Tree " << treeIdx << " has no splits");
            CB_ENSURE(treeStructure.SplitProperties.size() == depth,
                "CatBoost-MLX: Tree " << treeIdx << " split/properties size mismatch");

            // Convert splits: TBestSplitProperties → TModelSplit
            TVector<TModelSplit> modelSplits;
            modelSplits.reserve(depth);

            for (ui32 level = 0; level < depth; ++level) {
                const auto& splitProps = treeStructure.SplitProperties[level];

                CB_ENSURE(splitProps.FeatureId < externalFeatureIndices.size(),
                    "CatBoost-MLX: Split feature ID " << splitProps.FeatureId
                    << " out of range (max " << externalFeatureIndices.size() - 1 << ")");

                const ui32 externalIdx = externalFeatureIndices[splitProps.FeatureId];
                const auto internalIdx = featuresLayout.GetInternalFeatureIdx<NCB::EFeatureType::Float>(externalIdx);
                CB_ENSURE(internalIdx,
                    "CatBoost-MLX: External feature " << externalIdx
                    << " not found in features layout");

                const auto& borders = quantizedFeaturesInfo.GetBorders(
                    NCB::TFloatFeatureIdx(static_cast<ui32>(*internalIdx)));

                CB_ENSURE(splitProps.BinId < borders.size(),
                    "CatBoost-MLX: Split bin " << splitProps.BinId
                    << " out of range for feature " << externalIdx
                    << " (has " << borders.size() << " borders)");

                const float border = borders[splitProps.BinId];

                modelSplits.push_back(TModelSplit(TFloatSplit{
                    static_cast<int>(*internalIdx),
                    border
                }));
            }

            // Convert leaf values: mx::array → TVector<double> or TVector<TVector<double>>
            mx::eval(leafValuesArr);
            const ui32 numLeaves = 1u << depth;

            if (approxDimension == 1) {
                // Single-dim: leafValuesArr is [numLeaves]
                CB_ENSURE(static_cast<ui32>(leafValuesArr.size()) == numLeaves,
                    "CatBoost-MLX: Tree " << treeIdx << " leaf count mismatch: "
                    << leafValuesArr.size() << " vs expected " << numLeaves);

                const float* leafPtr = leafValuesArr.data<float>();
                TVector<double> leafValues(numLeaves);
                for (ui32 i = 0; i < numLeaves; ++i) {
                    leafValues[i] = static_cast<double>(leafPtr[i]);
                }

                builder.AddTree(modelSplits, leafValues, TConstArrayRef<double>{});
            } else {
                // Multi-dim: leafValuesArr is [numLeaves, approxDimension]
                CB_ENSURE(static_cast<ui32>(leafValuesArr.size()) == numLeaves * approxDimension,
                    "CatBoost-MLX: Tree " << treeIdx << " leaf count mismatch: "
                    << leafValuesArr.size() << " vs expected " << numLeaves * approxDimension);

                const float* leafPtr = leafValuesArr.data<float>();

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

        // Step 4: Build the model
        TFullModel fullModel;
        builder.Build(fullModel.ModelTrees.GetMutable());
        fullModel.SetScaleAndBias({1.0, {}});
        fullModel.UpdateDynamicData();

        // Store training parameters in model info
        fullModel.ModelInfo["params"] = ToString(catboostOptions);

        CATBOOST_INFO_LOG << "CatBoost-MLX: Model export complete. "
            << boostingResult.NumIterations << " oblivious trees, "
            << floatFeatures.size() << " float features" << Endl;

        return fullModel;
    }

}  // namespace NCatboostMlx
