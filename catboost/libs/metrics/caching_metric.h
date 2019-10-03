#pragma once

#include "metric_holder.h"

#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/libs/helpers/maybe_data.h>
#include <catboost/private/libs/data_types/query.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>

struct IMetric;

TVector<THolder<IMetric>> CreateCachingMetrics(
    ELossFunction metric, const TMap<TString, TString>& params, int approxDimension, TSet<TString>* validParams);

TVector<TMetricHolder> EvalErrorsWithCaching(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    NCB::TMaybeData<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    TConstArrayRef<const IMetric *> metrics,
    NPar::TLocalExecutor *localExecutor
);

inline static TVector<TMetricHolder> EvalErrorsWithCaching(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    NCB::TMaybeData<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    TConstArrayRef<THolder<IMetric>> metrics,
    NPar::TLocalExecutor *localExecutor
) {
    TVector<const IMetric *> metricPtrs;
    metricPtrs.reserve(metrics.size());
    for (const auto& metric : metrics) {
        metricPtrs.push_back(metric.Get());
    }
    return EvalErrorsWithCaching(
        approx,
        approxDelta,
        isExpApprox,
        target,
        weight,
        queriesInfo,
        metricPtrs,
        localExecutor
    );
}
