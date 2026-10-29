#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/feature_estimators.h>
#include <catboost/libs/model/model.h>

namespace NCB {

// Apply a finalized model to the quantized sources owned by its training
// estimators. Unlike the generic quantized-Pool applier, this path evaluates
// text/embedding calcers and places their bins before the one-hot/CTR buckets.
// Model evaluation is shared CatBoost code; no objective or tree fitting occurs.
TVector<TVector<double>> ApplyMetalModelWithEstimatedFeatures(
    const TFullModel& model,
    const TTrainingDataProvider& data,
    const TFeatureEstimatorsPtr& estimators,
    NPar::ILocalExecutor* executor);

} // namespace NCB
