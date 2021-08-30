#include "approx_calcer_querywise.h"

#include <catboost/libs/helpers/map_merge.h>
#include <catboost/private/libs/index_range/index_range.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/cast.h>


void CalculateDersForQueries(
    const TVector<double>& approxes,
    const TVector<double>& approxesDelta,
    const TVector<float>& targets,
    const TVector<float>& weights,
    const TVector<TQueryInfo>& queriesInfo,
    const IDerCalcer& error,
    int queryStartIndex,
    int queryEndIndex,
    TArrayRef<TDers> approxDers,
    ui64 randomSeed,
    NPar::ILocalExecutor* localExecutor
) {
    if (!approxesDelta.empty()) {
        TVector<double> fullApproxes;
        fullApproxes.yresize(approxes.ysize());
        if (error.GetIsExpApprox()) {
            NPar::ParallelFor(
                *localExecutor,
                queriesInfo[queryStartIndex].Begin,
                queriesInfo[queryEndIndex - 1].End,
                [&](ui32 docId) {
                    fullApproxes[docId] = UpdateApprox</*StoreExpApprox*/true>(
                        approxes[docId],
                        approxesDelta[docId]
                    );
                });
        } else {
            NPar::ParallelFor(
                *localExecutor,
                queriesInfo[queryStartIndex].Begin,
                queriesInfo[queryEndIndex - 1].End,
                [&](ui32 docId) {
                    fullApproxes[docId] = UpdateApprox</*StoreExpApprox*/false>(
                        approxes[docId],
                        approxesDelta[docId]
                    );
                });
        }
        error.CalcDersForQueries(
            queryStartIndex,
            queryEndIndex,
            fullApproxes,
            targets,
            weights,
            queriesInfo,
            approxDers,
            randomSeed,
            localExecutor
        );
    } else {
        error.CalcDersForQueries(
            queryStartIndex,
            queryEndIndex,
            approxes,
            targets,
            weights,
            queriesInfo,
            approxDers,
            randomSeed,
            localExecutor
        );
    }
}

template <ELeavesEstimation estimationMethod>
static void AddMethodDersForLeaves(
    const TVector<TDers>& bucketDers,
    const TVector<double>& bucketWeights,
    bool updateWeight,
    int leafCount,
    TVector<TSum>* buckets
) {
    for (int leafId = 0; leafId < leafCount; ++leafId) {
        if (bucketWeights[leafId] > FLT_EPSILON) {
            AddMethodDer<estimationMethod>(
                bucketDers[leafId],
                bucketWeights[leafId],
                updateWeight,
                &(*buckets)[leafId]
            );
        }
    }
}

void AddLeafDersForQueries(
    const TVector<TDers>& weightedDers,
    const TVector<TIndexType>& indices,
    const TVector<float>& weights,
    const TVector<TQueryInfo>& queriesInfo,
    int queryStartIndex,
    int queryEndIndex,
    ELeavesEstimation estimationMethod,
    int recalcLeafWeights,
    TVector<TSum>* buckets,
    NPar::ILocalExecutor* localExecutor
) {
    const int leafCount = buckets->ysize();

    using TBucketStats = std::pair<TVector<TDers>, TVector<double>>;
    const auto mapDocuments = [&](const NCB::TIndexRange<int>& range, TBucketStats* blockStats) {
        const auto* indicesData = indices.data();
        const auto* dersData = weightedDers.data();
        const auto* weightsData = weights.empty() ? nullptr : weights.data();
        blockStats->first.resize(leafCount, TDers{/*Der1*/0.0, /*Der2*/0.0, /*Der3*/0.0});
        blockStats->second.resize(leafCount, 0.0);
        auto* blockDersData = blockStats->first.data();
        auto* blockWeightsData = blockStats->second.data();
        for (int docId = range.Begin; docId < range.End; ++docId) {
            TDers& currentDers = blockDersData[indicesData[docId]];
            currentDers.Der1 += dersData[docId].Der1;
            currentDers.Der2 += dersData[docId].Der2;
            blockWeightsData[indicesData[docId]] += weightsData == nullptr ? 1.0f : weightsData[docId];
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
    NCB::TSimpleIndexRangesGenerator<int> rangeGenerator(
        { IntegerCast<int>(begin), IntegerCast<int>(end) },
        CeilDiv(IntegerCast<int>(end) - IntegerCast<int>(begin), CB_THREAD_LIMIT));
    TBucketStats bucketStats;
    NCB::MapMerge(localExecutor, rangeGenerator, mapDocuments, mergeBuckets, &bucketStats);

    if (estimationMethod == ELeavesEstimation::Newton) {
        AddMethodDersForLeaves<ELeavesEstimation::Newton>(
            bucketStats.first,
            bucketStats.second,
            recalcLeafWeights,
            leafCount,
            buckets
        );
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        AddMethodDersForLeaves<ELeavesEstimation::Gradient>(
            bucketStats.first,
            bucketStats.second,
            recalcLeafWeights,
            leafCount,
            buckets
        );
    }
}

