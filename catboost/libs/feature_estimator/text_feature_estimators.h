#pragma once

#include "feature_estimator.h"

#include <catboost/libs/options/enums.h>
#include <catboost/libs/text_processing/embedding.h>
#include <catboost/libs/text_processing/text_dataset.h>


namespace NCB {

    TVector<TOnlineFeatureEstimatorPtr> CreateEstimators(TConstArrayRef<EFeatureCalcerType> types,
                                                         TEmbeddingPtr embedding,
                                                         TTextClassificationTargetPtr target,
                                                         TTextDataSetPtr learnTexts,
                                                         TArrayRef<TTextDataSetPtr> testText);

    //offline
    TVector<TFeatureEstimatorPtr> CreateEstimators(TConstArrayRef<EFeatureCalcerType> types,
                                                   TEmbeddingPtr embedding,
                                                   TTextDataSetPtr learnTexts,
                                                   TArrayRef<TTextDataSetPtr> testText);
}
