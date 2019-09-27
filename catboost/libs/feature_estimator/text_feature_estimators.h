#pragma once

#include "feature_estimator.h"

#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/text_processing_options.h>
#include <catboost/libs/text_processing/embedding.h>
#include <catboost/libs/text_processing/text_dataset.h>

namespace NCB {
    TVector<TOnlineFeatureEstimatorPtr> CreateEstimators(
        TConstArrayRef<NCatboostOptions::TFeatureCalcerDescription> featureCalcerDescription,
        TEmbeddingPtr embedding,
        TTextClassificationTargetPtr target,
        TTextDataSetPtr learnTexts,
        TArrayRef<TTextDataSetPtr> testText
    );

    //offline
    TVector<TFeatureEstimatorPtr> CreateEstimators(
        TConstArrayRef<NCatboostOptions::TFeatureCalcerDescription> featureCalcerDescription,
        TEmbeddingPtr embedding,
        TTextDataSetPtr learnTexts,
        TArrayRef<TTextDataSetPtr> testText
    );
}
