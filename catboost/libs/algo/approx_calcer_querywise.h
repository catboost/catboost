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
    TVector<TDer1Der2>* weightedDers
) {
    TVector<double> fullApproxes(approxes);
    if (!approxesDelta.empty()) {
        for (int docId = queriesInfo[queryStartIndex].Begin; docId < queriesInfo[queryEndIndex - 1].End; ++docId) {
            fullApproxes[docId] = UpdateApprox<TError::StoreExpApprox>(approxes[docId], approxesDelta[docId]);
        }
    }

    error.CalcDersForQueries(queryStartIndex, queryEndIndex, fullApproxes, targets, weights, queriesInfo, weightedDers);
}

inline void UpdateBucketsForQueries(
    TVector<TDer1Der2> weightedDers,
    const TVector<TIndexType>& indices,
    const TVector<float>& weights,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex,
    int iteration,
    TVector<TSum>* buckets
) {
    const int leafCount = buckets->ysize();
    TVector<TDer1Der2> bucketDers(leafCount, TDer1Der2{/*Der1*/0.0, /*Der2*/0.0 });
    TVector<double> bucketWeights(leafCount, 0);

    for (int docId = queriesInfo[queryStartIndex].Begin; docId < queriesInfo[queryEndIndex - 1].End; ++docId) {
        TDer1Der2& currentDers = bucketDers[indices[docId]];
        currentDers.Der1 += weightedDers[docId].Der1;
        bucketWeights[indices[docId]] += weights.empty() ? 1.0f : weights[docId];
    }

    for (int leafId = 0; leafId < leafCount; ++leafId) {
        if (bucketWeights[leafId] > FLT_EPSILON) {
            UpdateBucket<ELeavesEstimation::Gradient>(bucketDers[leafId], bucketWeights[leafId], iteration, &(*buckets)[leafId]);
        }
    }
}
