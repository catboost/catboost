#pragma once

#include "score_calcer.h"
#include "calc_score_cache.h"
#include "index_calcer.h"
#include "split.h"

struct TBucketPairWeightStatistics {
    double SmallerBorderWeightSum = 0.0; // The weight sum of pair elements with smaller border.
    double GreaterBorderRightWeightSum = 0.0; // The weight sum of pair elements with greater border.
};

TVector<TVector<double>> ComputeDerSums(
    TConstArrayRef<double> weightedDerivativesData,
    int leafCount,
    int bucketCount,
    const TVector<ui32>& leafIndices,
    const TVector<ui32>& bucketIndices
);

TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatistics(
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount,
    const TVector<ui32>& leafIndices,
    const TVector<ui32>& bucketIndices
);

void EvaluateBucketScores(
    const TVector<TVector<double>>& derSums,
    const TArray2D<TVector<TBucketPairWeightStatistics>>& pairWeightStatistics,
    int bucketCount,
    ESplitType splitType,
    float l2DiagReg,
    float pairwiseBucketWeightPriorReg,
    TVector<TScoreBin>* scoreBins
);

template<typename TFullIndexType>
inline void CalculatePairwiseScore(
    const TVector<TFullIndexType>& singleIdx,
    TConstArrayRef<double> weightedDerivativesData,
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount,
    ESplitType splitType,
    float l2DiagReg,
    float pairwiseBucketWeightPriorReg,
    TVector<TScoreBin>* scoreBins
) {
    const int docCount = singleIdx.ysize();
    TVector<ui32> leafIndices(docCount), bucketIndices(docCount);
    for(int docId = 0; docId < docCount; ++docId) {
        leafIndices[docId] = singleIdx[docId] / bucketCount;
        bucketIndices[docId] = singleIdx[docId] % bucketCount;
    }

    const TVector<TVector<double>> derSums = ComputeDerSums(weightedDerivativesData, leafCount, bucketCount, leafIndices, bucketIndices);
    const TArray2D<TVector<TBucketPairWeightStatistics>> pairWeightStatistics = ComputePairWeightStatistics(queriesInfo, leafCount, bucketCount, leafIndices, bucketIndices);
    EvaluateBucketScores(derSums, pairWeightStatistics, bucketCount, splitType, l2DiagReg, pairwiseBucketWeightPriorReg, scoreBins);
}
