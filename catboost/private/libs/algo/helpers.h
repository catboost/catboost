#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/features.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>

#include <functional>


class TLearnContext;

namespace NCB {
    class TFeaturesLayout;
    class TQuantizedFeaturesInfo;
}


TVector<TFloatFeature> CreateFloatFeatures(
    const NCB::TFeaturesLayout& featuresLayout,
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo
);

TVector<TTextFeature> CreateTextFeatures(const NCB::TFeaturesLayout& featuresLayout);

TVector<TCatFeature> CreateCatFeatures(const NCB::TFeaturesLayout& featuresLayout);

TVector<TEmbeddingFeature> CreateEmbeddingFeatures(const NCB::TFeaturesLayout& featuresLayout);

void ConfigureMalloc();


double CalcMetric(
    const IMetric& metric,
    const NCB::TTargetDataProviderPtr& targetData,
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* localExecutor
);

void CalcErrorsLocally(
    const NCB::TTrainingDataProviders& trainingDataProviders,
    const TVector<THolder<IMetric>>& errors,
    bool calcAllMetrics, // bool value for each error
    bool calcErrorTrackerMetric,
    bool calcNonAdditiveMetricsOnly,
    TLearnContext* ctx
);

void IterateOverMetrics(
    const NCB::TTrainingDataProviders& trainingDataProviders,
    const TVector<THolder<IMetric>>& errors,
    bool calcAllMetrics, // bool value for each error
    bool calcErrorTrackerMetric,
    bool calcAdditiveMetrics,
    bool calcNonAdditiveMetrics,
    std::function<void(TConstArrayRef<const IMetric*> /*metrics*/)> onLearnCallback,
    std::function<
        void(size_t /*testIdx*/, TConstArrayRef<const IMetric*> /*metrics*/, TMaybe<int> /*filteredTrackerIdx*/)
    > onTestCallback
);

