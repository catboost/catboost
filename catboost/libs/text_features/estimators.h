#pragma once

#include "embedding.h"
#include "text_dataset.h"

#include <catboost/libs/feature_estimator/feature_estimator.h>
#include <catboost/libs/options/enums.h>


namespace NCB {

    TVector<TOnlineFeatureEstimatorPtr> CreateEstimators(TConstArrayRef<EFeatureEstimatorType> types,
                                                         TEmbeddingPtr embedding,
                                                         TTextClassificationTargetPtr target,
                                                         TTextDataSetPtr learnTexts,
                                                         TVector<TTextDataSetPtr> testText);

    //offline
    TVector<TFeatureEstimatorPtr> CreateEstimators(TConstArrayRef<EFeatureEstimatorType> types,
                                                   TEmbeddingPtr embedding,
                                                   TTextDataSetPtr learnTexts,
                                                   TVector<TTextDataSetPtr> testText);
}
