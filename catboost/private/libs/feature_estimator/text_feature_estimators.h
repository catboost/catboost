#pragma once

#include "classification_target.h"
#include "feature_estimator.h"

#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/private/libs/text_processing/text_dataset.h>

namespace NCB {
    TVector<TOnlineFeatureEstimatorPtr> CreateTextEstimators(
        TConstArrayRef<NCatboostOptions::TFeatureCalcerDescription> featureCalcerDescription,
        TClassificationTargetPtr target,
        TTextDataSetPtr learnTexts,
        TArrayRef<TTextDataSetPtr> testText
    );

    //offline
    TVector<TFeatureEstimatorPtr> CreateTextEstimators(
        TConstArrayRef<NCatboostOptions::TFeatureCalcerDescription> featureCalcerDescription,
        TTextDataSetPtr learnTexts,
        TArrayRef<TTextDataSetPtr> testText
    );
}
