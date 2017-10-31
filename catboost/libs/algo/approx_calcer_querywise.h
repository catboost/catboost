#pragma once

#include "approx_calcer_helpers.h"
#include "approx_util.h"

template <typename TError>
void CalcShiftedApproxDersQueries(const yvector<double>& approx,
                                  const yvector<double>& approxDelta,
                                  const yvector<float>& target,
                                  const yvector<float>& weight,
                                  const yvector<ui32>& queriesId,
                                  const yhash<ui32, ui32>& queriesSize,
                                  const TError& error,
                                  int sampleStart,
                                  int sampleFinish,
                                  yvector<TDer1Der2>* scratchDers) {
    yvector<double> fullApproxes(approx);
    if (!approxDelta.empty()) {
        for (int docId = 0; docId < sampleFinish; ++docId) {
            fullApproxes[docId] = UpdateApprox<TError::StoreExpApprox>(approx[docId], approxDelta[docId]);
        }
    }

    const int dersSize = sampleFinish - sampleStart;
    error.CalcDersForQueries(sampleStart, dersSize, fullApproxes, target, weight, queriesId, queriesSize, scratchDers);
}

template <typename TError>
void CalcApproxDersQueriesRange(const yvector<TIndexType>& indices,
                                const yvector<double>& approx,
                                const yvector<double>& approxDelta,
                                const yvector<float>& target,
                                const yvector<float>& weight,
                                const yvector<ui32>& queriesId,
                                const yhash<ui32, ui32>& queriesSize,
                                const TError& error,
                                int sampleCount,
                                int sampleTotal,
                                int iteration,
                                yvector<TSum>* buckets,
                                yvector<TDer1Der2>* scratchDers) {
    const int leafCount = buckets->ysize();

    yvector<double> fullApproxes(approx);
    if (!approxDelta.empty()) {
        for (int docId = 0; docId < sampleTotal; ++docId) {
            fullApproxes[docId] = UpdateApprox<TError::StoreExpApprox>(approx[docId], approxDelta[docId]);
        }
    }

    error.CalcDersForQueries(0, sampleCount, fullApproxes, target, weight, queriesId, queriesSize, scratchDers);

    yvector<TDer1Der2> bucketDers(leafCount, TDer1Der2{/*Der1*/0.0, /*Der2*/0.0 });
    yvector<double> bucketWeights(leafCount, 0);
    for (int docId = 0; docId < sampleCount; ++docId) {
        TDer1Der2& currentDers = bucketDers[indices[docId]];
        currentDers.Der1 += (*scratchDers)[docId].Der1;
        bucketWeights[indices[docId]] += 1;
    }

    for (int leafId = 0; leafId < leafCount; ++leafId) {
        if (bucketWeights[leafId] > FLT_EPSILON) {
            UpdateBucket<ELeafEstimation::Gradient>(bucketDers[leafId], bucketWeights[leafId], iteration, &(*buckets)[leafId]);
        }
    }
}
