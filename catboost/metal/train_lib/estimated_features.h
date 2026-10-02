#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/catboost_options.h>

namespace NCB {
    // The shared estimators own their prepared text/embedding data. Training
    // columns remain in original object order; each online bank uses the same
    // exclusive history as the corresponding Metal CTR/prediction cursor.
    struct TMetalEstimatedFeatures {
        TVector<TTextFeature> AllTextFeatures;
        TVector<TEmbeddingFeature> AllEmbeddingFeatures;
        TVector<TModelEstimatedFeature> Features;
        TVector<TVector<float>> Borders;
        TVector<ui8> IsOnline;
        TVector<ui32> BinCountUpperBounds;
        TVector<TVector<ui8>> BinsByPermutation;
        bool HasOnlineFeatures = false;
        ui32 BinsPerFeature = 1;
        ui32 Checksum = 0;
        TFeatureEstimatorsPtr Estimators;
        TQuantizedObjectsDataProviderPtr LearnObjects;

        ui32 GetFeatureCount() const { return Features.size(); }
        ui32 GetPermutationCount() const { return BinsByPermutation.size(); }
        TModelSplit GetSplit(ui32 feature, ui32 bin) const;
        // Build trimmed final calcers for a freshly built tree/model before
        // shared model application. Do not call twice on a remapped model.
        void FinalizeModel(TFullModel* model, NPar::ILocalExecutor* executor) const;
    };

    TMetalEstimatedFeatures PrepareMetalEstimatedFeatures(
        const TTrainingDataProviders& trainingData,
        const NCatboostOptions::TCatBoostOptions& options,
        NPar::ILocalExecutor* executor);
}
