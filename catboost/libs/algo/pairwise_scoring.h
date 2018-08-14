#pragma once

#include "score_calcer.h"
#include "calc_score_cache.h"
#include "index_calcer.h"
#include "split.h"

struct TBucketPairWeightStatistics {
    double SmallerBorderWeightSum = 0.0; // The weight sum of pair elements with smaller border.
    double GreaterBorderRightWeightSum = 0.0; // The weight sum of pair elements with greater border.
};

template<typename TFullIndexType>
TVector<TVector<double>> ComputeDerSums(
    TConstArrayRef<double> weightedDerivativesData,
    int leafCount,
    int bucketCount,
    const TVector<TFullIndexType>& singleIdx
);

template<typename TFullIndexType>
TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatistics(
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount,
    const TVector<TFullIndexType>& singleIdx
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
    const TVector<TVector<double>> derSums = ComputeDerSums(weightedDerivativesData, leafCount, bucketCount, singleIdx);
    const TArray2D<TVector<TBucketPairWeightStatistics>> pairWeightStatistics = ComputePairWeightStatistics(queriesInfo, leafCount, bucketCount, singleIdx);
    EvaluateBucketScores(derSums, pairWeightStatistics, bucketCount, splitType, l2DiagReg, pairwiseBucketWeightPriorReg, scoreBins);
}
