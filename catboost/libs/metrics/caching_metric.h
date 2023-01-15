#pragma once

#include "metric_holder.h"

#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/libs/helpers/maybe.h>
#include <catboost/private/libs/data_types/query.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>

struct IMetric;
struct TMetricConfig;
struct TParamSet;

TVector<THolder<IMetric>> CreateCachingMetrics(const TMetricConfig& config);
TVector<TParamSet> CachingMetricValidParamSets(ELossFunction metric);

TVector<TMetricHolder> EvalErrorsWithCaching(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    TConstArrayRef<const IMetric *> metrics,
    NPar::ILocalExecutor *localExecutor
);

inline static TVector<TMetricHolder> EvalErrorsWithCaching(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    TConstArrayRef<const IMetric*> metrics,
    NPar::ILocalExecutor *localExecutor
) {
    return EvalErrorsWithCaching(
        approx,
        approxDelta,
        isExpApprox,
        TConstArrayRef<TConstArrayRef<float>>(&target, 1),
        weight,
        queriesInfo,
        metrics,
        localExecutor
    );
}
