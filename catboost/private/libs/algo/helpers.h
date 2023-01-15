#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/features.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>


class TLearnContext;

namespace NCB {
    class TFeaturesLayout;
    class TQuantizedFeaturesInfo;
}


TVector<TFloatFeature> CreateFloatFeatures(
    const NCB::TFeaturesLayout& featuresLayout,
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo);

TVector<TTextFeature> CreateTextFeatures(const NCB::TFeaturesLayout& featuresLayout);

TVector<TCatFeature> CreateCatFeatures(const NCB::TFeaturesLayout& featuresLayout);


void ConfigureMalloc();

double CalcMetric(
    const IMetric& metric,
    const NCB::TTargetDataProviderPtr& targetData,
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* localExecutor
);

void CalcErrors(
    const NCB::TTrainingDataProviders& trainingDataProviders,
    const TVector<THolder<IMetric>>& errors,
    bool calcAllMetrics, // bool value for each error
    bool calcErrorTrackerMetric,
    TLearnContext* ctx
);
