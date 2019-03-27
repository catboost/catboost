#pragma once

#include "text_dataset.h"
#include "embedding.h"
#include <catboost/libs/feature_estimator/feature_estimator.h>

namespace NCB {

    TVector<TOnlineFeatureEstimatorPtr> CreateEstimators(TConstArrayRef<EFeatureCalculatorType> types,
                                                         TEmbeddingPtr embedding,
                                                         TTextClassificationTargetPtr target,
                                                         TTextDataSetPtr learnTexts,
                                                         TVector<TTextDataSetPtr> testText);


}
