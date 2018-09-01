#include "approx_calcer_querywise.h"

#include <catboost/libs/helpers/index_range.h>
#include <catboost/libs/helpers/map_merge.h>

#include <util/generic/cast.h>

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
) {
    const int leafCount = buckets->ysize();

    using TBucketStats = std::pair<TVector<TDers>, TVector<double>>;
    const auto mapDocuments = [&](const NCB::TIndexRange<int>& range, TBucketStats* blockStats) {
        const auto* indicesData = indices.data();
        const auto* dersData = weightedDers.data();
        const auto* weightsData = weights.data();
        blockStats->first.resize(leafCount, TDers{/*Der1*/0.0, /*Der2*/0.0, /*Der3*/0.0});
        blockStats->second.resize(leafCount, 0.0);
        auto* blockDersData = blockStats->first.data();
        auto* blockWeightsData = blockStats->second.data();
        for (int docId = range.Begin; docId < range.End; ++docId) {
            TDers& currentDers = blockDersData[indicesData[docId]];
            currentDers.Der1 += dersData[docId].Der1;
            currentDers.Der2 += dersData[docId].Der2;
            blockWeightsData[indicesData[docId]] += weights.empty() ? 1.0f : weightsData[docId];
        }
    };
    const auto mergeBuckets = [&](TBucketStats* mergedStats, const TVector<TBucketStats>&& blocksStats) {
        for (const auto& blockStats : blocksStats) {
            for (int idx = 0; idx < leafCount; ++idx) {
                mergedStats->first[idx].Der1 += blockStats.first[idx].Der1;
                mergedStats->first[idx].Der2 += blockStats.first[idx].Der2;
                mergedStats->second[idx] += blockStats.second[idx];
            }
        }
    };
    const size_t begin = queriesInfo[queryStartIndex].Begin;
    const size_t end = queriesInfo[queryEndIndex - 1].End;
    NCB::TSimpleIndexRangesGenerator<int> rangeGenerator({IntegerCast<int>(begin), IntegerCast<int>(end)}, CeilDiv(IntegerCast<int>(end) - IntegerCast<int>(begin), CB_THREAD_LIMIT));
    TBucketStats bucketStats;
    NCB::MapMerge(localExecutor, rangeGenerator, mapDocuments, mergeBuckets, &bucketStats);

    if (estimationMethod == ELeavesEstimation::Newton) {
        UpdateBucketsForLeaves<ELeavesEstimation::Newton>(
            bucketStats.first,
            bucketStats.second,
            iteration,
            leafCount,
            buckets
        );
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        UpdateBucketsForLeaves<ELeavesEstimation::Gradient>(
            bucketStats.first,
            bucketStats.second,
            iteration,
            leafCount,
            buckets
        );
    }
}

