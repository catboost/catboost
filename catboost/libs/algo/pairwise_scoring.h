#pragma once

#include "calc_score_cache.h"
#include "index_calcer.h"
#include "score_bin.h"
#include "split.h"

#include <catboost/libs/index_range/index_range.h>

#include <library/binsaver/bin_saver.h>

struct TBucketPairWeightStatistics {
    double SmallerBorderWeightSum = 0.0; // The weight sum of pair elements with smaller border.
    double GreaterBorderRightWeightSum = 0.0; // The weight sum of pair elements with greater border.

    void Add(const TBucketPairWeightStatistics& rhs) {
        SmallerBorderWeightSum += rhs.SmallerBorderWeightSum;
        GreaterBorderRightWeightSum += rhs.GreaterBorderRightWeightSum;
    }
    SAVELOAD(SmallerBorderWeightSum, GreaterBorderRightWeightSum);
};


struct TPairwiseStats {
    TVector<TVector<double>> DerSums; // [leafCount][bucketCount]
    TArray2D<TVector<TBucketPairWeightStatistics>> PairWeightStatistics; // [leafCount][leafCount][bucketCount]

    void Add(const TPairwiseStats& rhs);
    SAVELOAD(DerSums, PairWeightStatistics);
};


// TGetBucketFunc is of type ui32(ui32 docId)
template <class TGetBucketFunc>
inline TVector<TVector<double>> ComputeDerSums(
    TConstArrayRef<double> weightedDerivativesData,
    int leafCount,
    int bucketCount,
    const TVector<TIndexType>& leafIndices,
    TGetBucketFunc getBucketFunc,
    NCB::TIndexRange<int> docIndexRange
) {
    TVector<TVector<double>> derSums(leafCount, TVector<double>(bucketCount));

    for (int docId : docIndexRange.Iter()) {
        const ui32 leafIndex = leafIndices[docId];
        const ui32 bucketIndex = getBucketFunc((ui32)docId);
        derSums[leafIndex][bucketIndex] += weightedDerivativesData[docId];
    }
    return derSums;
}

// TGetBucketFunc is of type ui32(ui32 docId)
template <class TGetBucketFunc>
inline TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatistics(
    const TFlatPairsInfo& pairs,
    int leafCount,
    int bucketCount,
    const TVector<TIndexType>& leafIndices,
    TGetBucketFunc getBucketFunc,
    NCB::TIndexRange<int> pairIndexRange
) {
    TArray2D<TVector<TBucketPairWeightStatistics>> weightSums(leafCount, leafCount);
    weightSums.FillEvery(TVector<TBucketPairWeightStatistics>(bucketCount));
    for (size_t pairIdx : pairIndexRange.Iter()) {
        const auto winnerIdx = pairs[pairIdx].WinnerId;
        const auto loserIdx = pairs[pairIdx].LoserId;
        if (winnerIdx == loserIdx) {
            continue;
        }
        const size_t winnerBucketId = getBucketFunc(winnerIdx);
        const auto winnerLeafId = leafIndices[winnerIdx];
        const size_t loserBucketId = getBucketFunc(loserIdx);
        const auto loserLeafId = leafIndices[loserIdx];
        const float weight = pairs[pairIdx].Weight;
        if (winnerBucketId > loserBucketId) {
            weightSums[loserLeafId][winnerLeafId][loserBucketId].SmallerBorderWeightSum -= weight;
            weightSums[loserLeafId][winnerLeafId][winnerBucketId].GreaterBorderRightWeightSum -= weight;
        } else {
            weightSums[winnerLeafId][loserLeafId][winnerBucketId].SmallerBorderWeightSum -= weight;
            weightSums[winnerLeafId][loserLeafId][loserBucketId].GreaterBorderRightWeightSum -= weight;
        }
    }

    return weightSums;
}

void CalculatePairwiseScore(
    const TPairwiseStats& pairwiseStats,
    int bucketCount,
    ESplitType splitType,
    float l2DiagReg,
    float pairwiseBucketWeightPriorReg,
    TVector<TScoreBin>* scoreBins
);

