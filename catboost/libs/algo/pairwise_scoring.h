#pragma once

#include "score_calcer.h"
#include "calc_score_cache.h"
#include "index_calcer.h"
#include "split.h"

#include <catboost/libs/helpers/index_range.h>


struct TBucketPairWeightStatistics {
    double SmallerBorderWeightSum = 0.0; // The weight sum of pair elements with smaller border.
    double GreaterBorderRightWeightSum = 0.0; // The weight sum of pair elements with greater border.

    void Add(const TBucketPairWeightStatistics& rhs) {
        SmallerBorderWeightSum += rhs.SmallerBorderWeightSum;
        GreaterBorderRightWeightSum += rhs.GreaterBorderRightWeightSum;
    }
};


struct TPairwiseStats {
    TVector<TVector<double>> DerSums; // [leafCount][bucketCount]
    TArray2D<TVector<TBucketPairWeightStatistics>> PairWeightStatistics; // [leafCount][leafCount][bucketCount]

    void Add(const TPairwiseStats& rhs);
};


template<typename TFullIndexType>
TVector<TVector<double>> ComputeDerSums(
    TConstArrayRef<double> weightedDerivativesData,
    int leafCount,
    int bucketCount,
    const TVector<TFullIndexType>& singleIdx,
    NCB::TIndexRange docIndexRange
);

template<typename TFullIndexType>
TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatistics(
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount,
    const TVector<TFullIndexType>& singleIdx,
    NCB::TIndexRange queryIndexRange
);

void CalculatePairwiseScore(
    const TPairwiseStats& pairwiseStats,
    int bucketCount,
    ESplitType splitType,
    float l2DiagReg,
    float pairwiseBucketWeightPriorReg,
    TVector<TScoreBin>* scoreBins
);

