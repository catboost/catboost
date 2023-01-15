#pragma once

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/private/libs/data_types/query.h>
#include <catboost/private/libs/options/restrictions.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>

TMetricHolder EvalErrorsWithLeaves(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> leafDeltas,
    TConstArrayRef<TIndexType> indices,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const IMetric& error,
    NPar::TLocalExecutor* localExecutor
);
