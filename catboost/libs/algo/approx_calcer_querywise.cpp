#include "approx_calcer_querywise.h"

template <ELeavesEstimation estimationMethod>
static void UpdateBucketsForLeaves(
    const TVector<TDers>& bucketDers,
    const TVector<double>& bucketWeights,
    int iteration,
    int leafCount,
    TVector<TSum>* buckets
) {
    for (int leafId = 0; leafId < leafCount; ++leafId) {
        if (bucketWeights[leafId] > FLT_EPSILON) {
            UpdateBucket<estimationMethod>(
                bucketDers[leafId],
                bucketWeights[leafId],
                iteration,
                &(*buckets)[leafId]
            );
        }
    }
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
) {
    const int leafCount = buckets->ysize();
    TVector<TDers> bucketDers(leafCount, TDers{/*Der1*/0.0, /*Der2*/0.0, /*Der3*/0.0});
    TVector<double> bucketWeights(leafCount, 0);

    const int begin = queriesInfo[queryStartIndex].Begin;
    const int end = queriesInfo[queryEndIndex - 1].End;
    for (int docId = begin; docId < end; ++docId) {
        TDers& currentDers = bucketDers[indices[docId]];
        currentDers.Der1 += weightedDers[docId].Der1;
        currentDers.Der2 += weightedDers[docId].Der2;
        bucketWeights[indices[docId]] += weights.empty() ? 1.0f : weights[docId];
    }

    if (estimationMethod == ELeavesEstimation::Newton) {
        UpdateBucketsForLeaves<ELeavesEstimation::Newton>(
            bucketDers,
            bucketWeights,
            iteration,
            leafCount,
            buckets
        );
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        UpdateBucketsForLeaves<ELeavesEstimation::Gradient>(
            bucketDers,
            bucketWeights,
            iteration,
            leafCount,
            buckets
        );
    }
}

