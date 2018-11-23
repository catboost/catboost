#pragma once

#include "approx_calcer_helpers.h"
#include "approx_updater_helpers.h"

#include <catboost/libs/data_types/query.h>

#include <library/threading/local_executor/local_executor.h>

template <typename TError>
void CalculateDersForQueries(
    const TVector<double>& approxes,
    const TVector<double>& approxesDelta,
    const TVector<float>& targets,
    const TVector<float>& weights,
    const TVector<TQueryInfo>& queriesInfo,
    const TError& error,
    int queryStartIndex,
    int queryEndIndex,
    TVector<TDers>* weightedDers,
    NPar::TLocalExecutor* localExecutor
) {
    if (!approxesDelta.empty()) {
        TVector<double> fullApproxes;
        fullApproxes.yresize(approxes.ysize());
        NPar::ParallelFor(*localExecutor, queriesInfo[queryStartIndex].Begin, queriesInfo[queryEndIndex - 1].End, [&](ui32 docId) {
            fullApproxes[docId] = UpdateApprox<TError::StoreExpApprox>(approxes[docId], approxesDelta[docId]);
        });
        error.CalcDersForQueries(queryStartIndex, queryEndIndex, fullApproxes, targets, weights, queriesInfo, weightedDers, localExecutor);
    } else {
        error.CalcDersForQueries(queryStartIndex, queryEndIndex, approxes, targets, weights, queriesInfo, weightedDers, localExecutor);
    }
}

void UpdateBucketsForQueries(
    const TVector<TDers>& weightedDers,
    const TVector<TIndexType>& indices,
    const TVector<float>& weights,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex,
    ELeavesEstimation estimationMethod,
    int iteration,
    TVector<TSum>* buckets,
    NPar::TLocalExecutor* localExecutor
);
