#pragma once

#include "learn_context.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/quantized_features_info.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/features.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>


TVector<TFloatFeature> CreateFloatFeatures(const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo);
TVector<TCatFeature> CreateCatFeatures(const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo);


void ConfigureMalloc();

void CalcErrors(
    const NCB::TTrainingForCPUDataProviders& trainingDataProviders,
    const TVector<THolder<IMetric>>& errors,
    bool calcAllMetrics, // bool value for each error
    bool calcErrorTrackerMetric,
    TLearnContext* ctx
);
