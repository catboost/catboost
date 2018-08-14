#include "pairwise_scoring.h"
#include "pairwise_leaves_calculation.h"

template<typename TFullIndexType>
inline static ui32 GetLeafIndex(TFullIndexType index, int bucketCount) {
    return index / bucketCount;
}

template<typename TFullIndexType>
inline static ui32 GetBucketIndex(TFullIndexType index, int bucketCount) {
    return index % bucketCount;
}

template<typename TFullIndexType>
TVector<TVector<double>> ComputeDerSums(
    TConstArrayRef<double> weightedDerivativesData,
    int leafCount,
    int bucketCount,
    const TVector<TFullIndexType>& singleIdx
) {
    TVector<TVector<double>> derSums(leafCount, TVector<double>(bucketCount));
    for (size_t docId = 0; docId < weightedDerivativesData.size(); ++docId) {
        const ui32 leafIndex = GetLeafIndex(singleIdx[docId], bucketCount);
        const ui32 bucketIndex = GetBucketIndex(singleIdx[docId], bucketCount);
        derSums[leafIndex][bucketIndex] += weightedDerivativesData[docId];
    }
    return derSums;
}

template
TVector<TVector<double>> ComputeDerSums<ui8>(
    TConstArrayRef<double> weightedDerivativesData,
    int leafCount,
    int bucketCount,
    const TVector<ui8>& singleIdx
);

template
TVector<TVector<double>> ComputeDerSums<ui16>(
    TConstArrayRef<double> weightedDerivativesData,
    int leafCount,
    int bucketCount,
    const TVector<ui16>& singleIdx
);

template
TVector<TVector<double>> ComputeDerSums<ui32>(
    TConstArrayRef<double> weightedDerivativesData,
    int leafCount,
    int bucketCount,
    const TVector<ui32>& singleIdx
);

template<typename TFullIndexType>
TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatistics(
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount,
    const TVector<TFullIndexType>& singleIdx
) {
    TArray2D<TVector<TBucketPairWeightStatistics>> pairWeightStatistics(leafCount, leafCount);
    pairWeightStatistics.FillEvery(TVector<TBucketPairWeightStatistics>(bucketCount));
    for (int queryId = 0; queryId < queriesInfo.ysize(); ++queryId) {
        const TQueryInfo& queryInfo = queriesInfo[queryId];
        const int begin = queryInfo.Begin;
        const int end = queryInfo.End;
        for (int docId = begin; docId < end; ++docId) {
            for (const auto& pair : queryInfo.Competitors[docId - begin]) {
                const int winnerBucketId = GetBucketIndex(singleIdx[docId], bucketCount);
                const int loserBucketId = GetBucketIndex(singleIdx[begin + pair.Id], bucketCount);
                const int winnerLeafId = GetLeafIndex(singleIdx[docId], bucketCount);
                const int loserLeafId = GetLeafIndex(singleIdx[begin + pair.Id], bucketCount);
                if (winnerBucketId == loserBucketId && winnerLeafId == loserLeafId) {
                    continue;
                }
                auto& bucketStatisticDirect = pairWeightStatistics[winnerLeafId][loserLeafId];
                auto& bucketStatisticReverse = pairWeightStatistics[loserLeafId][winnerLeafId];
                if (winnerBucketId > loserBucketId) {
                    bucketStatisticReverse[loserBucketId].SmallerBorderWeightSum -= pair.SampleWeight;
                    bucketStatisticReverse[winnerBucketId].GreaterBorderRightWeightSum -= pair.SampleWeight;
                } else {
                    bucketStatisticDirect[loserBucketId].GreaterBorderRightWeightSum -= pair.SampleWeight;
                    bucketStatisticDirect[winnerBucketId].SmallerBorderWeightSum -= pair.SampleWeight;
                }
            }
        }
    }
    return pairWeightStatistics;
}

template
TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatistics<ui8>(
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount,
    const TVector<ui8>& singleIdx
);

template
TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatistics<ui16>(
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount,
    const TVector<ui16>& singleIdx
);

template
TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatistics<ui32>(
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount,
    const TVector<ui32>& singleIdx
);

static double CalculateScore(const TVector<double>& avrg, const TVector<double>& sumDer, const TArray2D<double>& sumWeights) {
    double score = 0;
    for (int x = 0; x < sumDer.ysize(); ++x) {
        score += avrg[x] * sumDer[x];
        for (ui32 y = 0; y < sumDer.size(); ++y) {
            score -= 0.5 * avrg[x] * avrg[y] * sumWeights[x][y];
        }
    }
    return score;
}

void EvaluateBucketScores(
    const TVector<TVector<double>>& derSums,
    const TArray2D<TVector<TBucketPairWeightStatistics>>& pairWeightStatistics,
    int bucketCount,
    ESplitType splitType,
    float l2DiagReg,
    float pairwiseBucketWeightPriorReg,
    TVector<TScoreBin>* scoreBins
) {
    const int leafCount = derSums.ysize();
    TVector<double> derSum(2 * leafCount, 0.0);
    TArray2D<double> weightSum(2 * leafCount, 2 * leafCount);
    weightSum.FillZero();
    Y_ASSERT(splitType == ESplitType::OnlineCtr || splitType == ESplitType::FloatFeature);

    for (int leafId = 0; leafId < leafCount; ++leafId) {
        for (int bucketId = 0; bucketId < bucketCount; ++bucketId) {
            derSum[2 * leafId + 1] += derSums[leafId][bucketId];
        }
    }

    for (int y = 0; y < leafCount; ++y) {
        for (int x = y + 1; x < leafCount; ++x) {
            const TVector<TBucketPairWeightStatistics>& xy = pairWeightStatistics[x][y];
            const TVector<TBucketPairWeightStatistics>& yx = pairWeightStatistics[y][x];
            for (int bucketId = 0; bucketId < bucketCount; ++bucketId) {
                const double add = yx[bucketId].SmallerBorderWeightSum + xy[bucketId].SmallerBorderWeightSum;
                weightSum[2 * y + 1][2 * x + 1] += add;
                weightSum[2 * x + 1][2 * y + 1] += add;
                weightSum[2 * x + 1][2 * x + 1] -= add;
                weightSum[2 * y + 1][2 * y + 1] -= add;
            }
        }
    }

    for (int splitId = 0; splitId < bucketCount - 1; ++splitId) {
        for (int y = 0; y < leafCount; ++y) {
            const double derDelta = derSums[y][splitId];
            derSum[2 * y] += derDelta;
            derSum[2 * y + 1] -= derDelta;

            const double weightDelta = (pairWeightStatistics[y][y][splitId].SmallerBorderWeightSum -
                pairWeightStatistics[y][y][splitId].GreaterBorderRightWeightSum);
            weightSum[2 * y][2 * y + 1] += weightDelta;
            weightSum[2 * y + 1][2 * y] += weightDelta;
            weightSum[2 * y][2 * y] -= weightDelta;
            weightSum[2 * y + 1][2 * y + 1] -= weightDelta;
            for (int x = y + 1; x < leafCount; ++x) {
                const TBucketPairWeightStatistics& xy = pairWeightStatistics[x][y][splitId];
                const TBucketPairWeightStatistics& yx = pairWeightStatistics[y][x][splitId];

                const double w00Delta = xy.GreaterBorderRightWeightSum + yx.GreaterBorderRightWeightSum;
                const double w01Delta = xy.SmallerBorderWeightSum - xy.GreaterBorderRightWeightSum;
                const double w10Delta = yx.SmallerBorderWeightSum - yx.GreaterBorderRightWeightSum;
                const double w11Delta = -(xy.SmallerBorderWeightSum + yx.SmallerBorderWeightSum);

                weightSum[2 * x][2 * y] += w00Delta;
                weightSum[2 * y][2 * x] += w00Delta;
                weightSum[2 * x][2 * y + 1] += w01Delta;
                weightSum[2 * y + 1][2 * x] += w01Delta;
                weightSum[2 * x + 1][2 * y] += w10Delta;
                weightSum[2 * y][2 * x + 1] += w10Delta;
                weightSum[2 * x + 1][2 * y + 1] += w11Delta;
                weightSum[2 * y + 1][2 * x + 1] += w11Delta;

                weightSum[2 * y][2 * y] -= w00Delta + w10Delta;
                weightSum[2 * x][2 * x] -= w00Delta + w01Delta;
                weightSum[2 * x + 1][2 * x + 1] -= w10Delta + w11Delta;
                weightSum[2 * y + 1][2 * y + 1] -= w01Delta + w11Delta;
            }
        }

        const TVector<double> leafValues = CalculatePairwiseLeafValues(weightSum, derSum, l2DiagReg, pairwiseBucketWeightPriorReg);
        (*scoreBins)[splitId].D2 = 1.0;
        (*scoreBins)[splitId].DP = CalculateScore(leafValues, derSum, weightSum);
    }
}

