#include "data_set_builder.h"

#include <catboost/libs/data/columns.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/vector.h>

namespace NCatboostMlx {

    TMLXCompressedIndex BuildCompressedIndex(
        const NCB::TQuantizedObjectsDataProvider& objectsData,
        NPar::ILocalExecutor* localExecutor
    ) {
        const auto& featuresLayout = *objectsData.GetFeaturesLayout();
        const ui32 numDocs = objectsData.GetObjectCount();

        // Collect one-byte quantized float feature indices
        TVector<ui32> featureIndices;
        for (auto featureIdx : featuresLayout.GetFloatFeatureInternalIdxToExternalIdx()) {
            featureIndices.push_back(featureIdx);
        }

        const ui32 numFeatures = featureIndices.size();
        // Pack 4 one-byte features per ui32
        const ui32 numUi32PerDoc = (numFeatures + 3) / 4;

        CATBOOST_INFO_LOG << "CatBoost-MLX: Building compressed index for "
            << numDocs << " docs, " << numFeatures << " features ("
            << numUi32PerDoc << " ui32 per doc)" << Endl;

        // Allocate CPU-side packed buffer
        TVector<ui32> packedData(static_cast<size_t>(numDocs) * numUi32PerDoc, 0);

        // Build feature metadata and pack data
        TVector<TCFeature> features;
        features.reserve(numFeatures);

        ui32 totalBinFeatures = 0;

        for (ui32 localIdx = 0; localIdx < numFeatures; ++localIdx) {
            const ui32 wordIdx = localIdx / 4;
            const ui32 posInWord = localIdx % 4;
            const ui32 shift = (3 - posInWord) * 8;  // pack left-to-right: bits 24, 16, 8, 0
            const ui32 mask = 0xFF << shift;

            const auto featureIdx = featureIndices[localIdx];

            // Get quantized feature column.
            // GetNonPackedFloatFeature returns TMaybeData — check it is defined.
            auto maybeColumn = objectsData.GetNonPackedFloatFeature(featureIdx);
            CB_ENSURE(maybeColumn.Defined(),
                "CatBoost-MLX: Feature " << featureIdx << " is not available as non-packed");
            const auto* column = *maybeColumn;
            CB_ENSURE(column != nullptr,
                "CatBoost-MLX: Feature " << featureIdx << " column is null");

            const ui32 numBins = objectsData.GetQuantizedFeaturesInfo()
                ->GetBorders(NCB::TFloatFeatureIdx(featureIdx)).size() + 1;

            features.emplace_back(
                /*offset=*/static_cast<ui64>(wordIdx),
                /*mask=*/mask,
                /*shift=*/shift,
                /*firstFoldIndex=*/totalBinFeatures,
                /*folds=*/numBins - 1,  // number of split candidates = bins - 1
                /*oneHotFeature=*/false,
                /*skipFirstBinInScoreCount=*/false
            );
            totalBinFeatures += (numBins - 1);

            // Extract quantized bin values via CatBoost's ExtractValues API
            auto binsHolder = column->template ExtractValues<ui8>(localExecutor);
            const ui8* bins = binsHolder.data();

            for (ui32 docIdx = 0; docIdx < numDocs; ++docIdx) {
                const size_t flatIdx = static_cast<size_t>(docIdx) * numUi32PerDoc + wordIdx;
                packedData[flatIdx] |= (static_cast<ui32>(bins[docIdx]) << shift);
            }
        }

        CATBOOST_INFO_LOG << "CatBoost-MLX: Compressed index has "
            << totalBinFeatures << " total bin-features" << Endl;

        // Transfer to GPU
        TMLXCompressedIndex index;
        index.Build(packedData.data(), numDocs, numUi32PerDoc, features, featureIndices);
        return index;
    }

    TMLXDataSet BuildMLXDataSet(
        const NCB::TTrainingDataProvider& dataProvider,
        ui32 approxDimension,
        NPar::ILocalExecutor* localExecutor
    ) {
        const ui32 numDocs = dataProvider.GetObjectCount();

        CATBOOST_INFO_LOG << "CatBoost-MLX: Building MLX dataset with "
            << numDocs << " objects" << Endl;

        TMLXDataSet dataset;

        // Build compressed feature index
        auto compressedIndex = BuildCompressedIndex(*dataProvider.ObjectsData, localExecutor);
        dataset.SetCompressedIndex(std::move(compressedIndex));

        // Extract targets — TTargetDataProvider::GetTarget() returns processed float targets:
        // TMaybeData<TConstArrayRef<TConstArrayRef<float>>>  [targetIdx][objectIdx]
        const auto& targetData = *dataProvider.TargetData;
        auto maybeTargets = targetData.GetTarget();
        CB_ENSURE(maybeTargets.Defined(), "CatBoost-MLX: Target data is not available");
        const auto& allTargets = *maybeTargets;
        CB_ENSURE(!allTargets.empty(), "CatBoost-MLX: Target data is empty");
        const auto& targets = allTargets[0];  // first target dimension

        TVector<float> targetValues(targets.begin(), targets.end());
        dataset.SetTargets(targetValues.data(), numDocs);

        // Extract weights (if present)
        const auto& weights = targetData.GetWeights();
        if (!weights.IsTrivial()) {
            TVector<float> weightValues(numDocs);
            for (ui32 i = 0; i < numDocs; ++i) {
                weightValues[i] = weights[i];
            }
            dataset.SetWeights(weightValues.data(), numDocs);
        } else {
            dataset.SetUniformWeights(numDocs);
        }

        // Initialize training state buffers
        dataset.InitCursor(numDocs, approxDimension);
        dataset.InitGradients(numDocs, approxDimension);
        dataset.InitPartitions(numDocs);

        return dataset;
    }

}  // namespace NCatboostMlx
