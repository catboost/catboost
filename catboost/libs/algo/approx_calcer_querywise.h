#pragma once

#include "approx_calcer_helpers.h"
#include "approx_util.h"

#include <catboost/libs/data_types/query.h>

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
    TVector<TDers>* weightedDers
) {
    TVector<double> fullApproxes(approxes);
    if (!approxesDelta.empty()) {
        for (int docId = queriesInfo[queryStartIndex].Begin; docId < queriesInfo[queryEndIndex - 1].End; ++docId) {
            fullApproxes[docId] = UpdateApprox<TError::StoreExpApprox>(approxes[docId], approxesDelta[docId]);
        }
    }

    error.CalcDersForQueries(queryStartIndex, queryEndIndex, fullApproxes, targets, weights, queriesInfo, weightedDers);
}

void UpdateBucketsForQueries(
    TVector<TDers> weightedDers,
    const TVector<TIndexType>& indices,
    const TVector<float>& weights,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex,
    ELeavesEstimation estimationMethod,
    int iteration,
    TVector<TSum>* buckets
);
