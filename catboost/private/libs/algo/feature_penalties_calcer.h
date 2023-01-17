#pragma once

#include "tensor_search_helpers.h"

#include <catboost/private/libs/options/feature_penalties_options.h>

#include <catboost/libs/data/data_provider.h>


class TFold;


namespace NCB {
    void PenalizeBestSplits(
        const TVector<TIndexType>& leaves,
        const TLearnContext& ctx,
        const TTrainingDataProviders& trainingData,
        const TFold& fold,
        ui32 oneHotMaxSize,
        TVector<TCandidateInfo>* candidates
    );
    float GetSplitFeatureWeight(
        const TSplit& split,
        const TCombinedEstimatedFeaturesContext& estimatedFeaturesContext,
        const TFeaturesLayout& layout,
        const NCatboostOptions::TPerFeaturePenalty& featureWeights
    );
}
