#pragma once

#include "approx_calcer_helpers.h"
#include "approx_util.h"

#include <catboost/libs/data/query.h>

template <typename TError>
void CalcShiftedApproxDersQueries(const TVector<double>& approx,
                                  const TVector<double>& approxDelta,
                                  const TVector<float>& target,
                                  const TVector<float>& weight,
                                  const TVector<TQueryInfo>& queriesInfo,
                                  const TError& error,
                                  int queryStartIndex,
                                  int queryEndIndex,
                                  TVector<TDer1Der2>* scratchDers) {
    TVector<double> fullApproxes(approx);
    if (!approxDelta.empty()) {
        for (int docId = queriesInfo[queryStartIndex].Begin; docId < queriesInfo[queryEndIndex - 1].End; ++docId) {
            fullApproxes[docId] = UpdateApprox<TError::StoreExpApprox>(approx[docId], approxDelta[docId]);
        }
    }

    error.CalcDersForQueries(queryStartIndex, queryEndIndex, fullApproxes, target, weight, queriesInfo, scratchDers);
}

template <typename TError>
void CalcApproxDersQueriesRange(const TVector<TIndexType>& indices,
                                const TVector<double>& approx,
                                const TVector<double>& approxDelta,
                                const TVector<float>& target,
                                const TVector<float>& weight,
                                const TVector<TQueryInfo>& queriesInfo,
                                const TError& error,
                                int queryCount,
                                int iteration,
                                TVector<TSum>* buckets,
                                TVector<TDer1Der2>* scratchDers) {
    const int leafCount = buckets->ysize();

    TVector<double> fullApproxes(approx);
    if (!approxDelta.empty()) {
        for (int docId = 0; docId < queriesInfo[queryCount - 1].End; ++docId) {
            fullApproxes[docId] = UpdateApprox<TError::StoreExpApprox>(approx[docId], approxDelta[docId]);
        }
    }

    error.CalcDersForQueries(/*queryStartIndex=*/0, queryCount, fullApproxes, target, weight, queriesInfo, scratchDers);

    TVector<TDer1Der2> bucketDers(leafCount, TDer1Der2{/*Der1*/0.0, /*Der2*/0.0 });
    TVector<double> bucketWeights(leafCount, 0);
    for (int docId = 0; docId < queriesInfo[queryCount - 1].End; ++docId) {
        TDer1Der2& currentDers = bucketDers[indices[docId]];
        currentDers.Der1 += (*scratchDers)[docId].Der1;
        bucketWeights[indices[docId]] += weight.empty() ? 1.0f : weight[docId];
    }

    for (int leafId = 0; leafId < leafCount; ++leafId) {
        if (bucketWeights[leafId] > FLT_EPSILON) {
            UpdateBucket<ELeavesEstimation::Gradient>(bucketDers[leafId], bucketWeights[leafId], iteration, &(*buckets)[leafId]);
        }
    }
}
