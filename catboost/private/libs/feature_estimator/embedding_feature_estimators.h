#pragma once

#include "classification_target.h"
#include "feature_estimator.h"

#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/private/libs/embeddings/embedding_dataset.h>

namespace NCB {
    TVector<TOnlineFeatureEstimatorPtr> CreateEmbeddingEstimators(
        TConstArrayRef<NCatboostOptions::TFeatureCalcerDescription> featureCalcerDescription,
        TConstArrayRef<float> target,
        TClassificationTargetPtr classificationTarget,  // can be nullptr if regression
        TEmbeddingDataSetPtr learnEmbeddings,
        TArrayRef<TEmbeddingDataSetPtr> testEmbedding
    );
}
