#include "pairwise_scoring.h"
#include "pairwise_leaves_calculation.h"

#include <util/generic/xrange.h>
#include <util/system/yassert.h>

#include <emmintrin.h>

void TPairwiseStats::Add(const TPairwiseStats& rhs) {
    Y_ASSERT(SplitEnsembleSpec == rhs.SplitEnsembleSpec);

    Y_ASSERT(DerSums.size() == rhs.DerSums.size());

    for (auto leafIdx : xrange(DerSums.size())) {
        auto& dst = DerSums[leafIdx];
        const auto& add = rhs.DerSums[leafIdx];

        Y_ASSERT(dst.size() == add.size());

        for (auto bucketIdx : xrange(dst.size())) {
            dst[bucketIdx] += add[bucketIdx];
        }
    }

    Y_ASSERT(PairWeightStatistics.GetXSize() == rhs.PairWeightStatistics.GetXSize());
    Y_ASSERT(PairWeightStatistics.GetYSize() == rhs.PairWeightStatistics.GetYSize());

    for (auto leafIdx1 : xrange(PairWeightStatistics.GetYSize())) {
        auto dst1 = PairWeightStatistics[leafIdx1];
        const auto add1 = rhs.PairWeightStatistics[leafIdx1];

        for (auto leafIdx2 : xrange(PairWeightStatistics.GetXSize())) {
            auto& dst2 = dst1[leafIdx2];
            const auto& add2 = add1[leafIdx2];

            Y_ASSERT(dst2.size() == add2.size());

            for (auto bucketIdx : xrange(dst2.size())) {
                dst2[bucketIdx].Add(add2[bucketIdx]);
            }
        }
    }
}


static inline double XmmHorizontalAdd(__m128d x) {
    return _mm_cvtsd_f64(_mm_add_pd(x, _mm_shuffle_pd(x, x, /*swap halves*/ 0x1)));
}

static inline __m128d XmmFusedMultiplyAdd(const double* x, const double* y, __m128d z) {
    return _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(x), _mm_loadu_pd(y)), z);
}

static inline __m128d XmmGather(const double* first, const double* second) {
    return _mm_loadh_pd(_mm_loadl_pd(_mm_undefined_pd(), first), second);
}

static double CalculateScore(const TVector<double>& avrg, const TVector<double>& sumDer, const TArray2D<double>& sumWeights) {
    const ui32 sumDerSize = sumDer.ysize();
    double score = 0;
    for (ui32 x = 0; x < sumDerSize; ++x) {
        const double* avrgData = avrg.data();
        const double* sumWeightsData = &sumWeights[x][0];
        __m128d subScore0 = _mm_setzero_pd();
        __m128d subScore2 = _mm_setzero_pd();
        for (ui32 y = 0; y + 4 <= sumDerSize; y += 4) {
            subScore0 = XmmFusedMultiplyAdd(avrgData + y + 0, sumWeightsData + y + 0, subScore0);
            subScore2 = XmmFusedMultiplyAdd(avrgData + y + 2, sumWeightsData + y + 2, subScore2);
        }
        double subScore = XmmHorizontalAdd(subScore0) + XmmHorizontalAdd(subScore2);
        for (ui32 y = sumDerSize & ~3u; y < sumDerSize; ++y) {
            subScore += avrgData[y] * sumWeightsData[y];
        }
        score += avrg[x] * (sumDer[x] - 0.5 * subScore);
    }
    return score;
}



inline static void UpdateWeightSumFromTotal(int y, int x, double total, TArray2D<double>* weightSum) {
    auto& weightSumRef = *weightSum;
    weightSumRef[2 * y + 1][2 * x + 1] += total;
    weightSumRef[2 * x + 1][2 * y + 1] += total;
    weightSumRef[2 * x + 1][2 * x + 1] -= total;
    weightSumRef[2 * y + 1][2 * y + 1] -= total;
}


inline static void UpdateWeightSumFromNonDiagStats(
    int y,
    int x,
    const TBucketPairWeightStatistics& xy,
    const TBucketPairWeightStatistics& yx,
    TArray2D<double>* weightSum
) {
    const double w00Delta = xy.GreaterBorderRightWeightSum + yx.GreaterBorderRightWeightSum;
    const double w01Delta = xy.SmallerBorderWeightSum - xy.GreaterBorderRightWeightSum;
    const double w10Delta = yx.SmallerBorderWeightSum - yx.GreaterBorderRightWeightSum;
    const double w11Delta = -(xy.SmallerBorderWeightSum + yx.SmallerBorderWeightSum);

    auto& weightSumRef = *weightSum;

    weightSumRef[2 * x][2 * y] += w00Delta;
    weightSumRef[2 * y][2 * x] += w00Delta;
    weightSumRef[2 * x][2 * y + 1] += w01Delta;
    weightSumRef[2 * y + 1][2 * x] += w01Delta;
    weightSumRef[2 * x + 1][2 * y] += w10Delta;
    weightSumRef[2 * y][2 * x + 1] += w10Delta;
    weightSumRef[2 * x + 1][2 * y + 1] += w11Delta;
    weightSumRef[2 * y + 1][2 * x + 1] += w11Delta;

    weightSumRef[2 * y][2 * y] -= w00Delta + w10Delta;
    weightSumRef[2 * x][2 * x] -= w00Delta + w01Delta;
    weightSumRef[2 * x + 1][2 * x + 1] -= w10Delta + w11Delta;
    weightSumRef[2 * y + 1][2 * y + 1] -= w01Delta + w11Delta;
}


void CalculatePairwiseScore(
    const TPairwiseStats& pairwiseStats,
    int bucketCount,
    float l2DiagReg,
    float pairwiseBucketWeightPriorReg,
    TVector<TScoreBin>* scoreBins
) {
    const auto& derSums = pairwiseStats.DerSums;
    const auto& pairWeightStatistics = pairwiseStats.PairWeightStatistics;

    const int leafCount = derSums.ysize();

    TArray2D<double> weightSum(2 * leafCount, 2 * leafCount);

    if (pairwiseStats.SplitEnsembleSpec.IsBinarySplitsPack) {
        const int binaryFeaturesCount = (int)GetValueBitCount(bucketCount - 1);

        scoreBins->yresize(binaryFeaturesCount);

        TVector<double> binDerSums;
        binDerSums.yresize(2 * leafCount);

        for (int binFeatureIdx = 0; binFeatureIdx < binaryFeaturesCount; ++binFeatureIdx) {
            binDerSums.assign(2 * leafCount, 0.0);

            for (int y = 0; y < leafCount; ++y) {
                for (int bucketIdx = 0; bucketIdx < bucketCount; ++bucketIdx) {
                    binDerSums[y * 2 + ((bucketIdx >> binFeatureIdx) & 1)] += derSums[y][bucketIdx];
                }
            }

            weightSum.FillZero();

            for (int y = 0; y < leafCount; ++y) {
                const double weightDelta =
                    (pairWeightStatistics[y][y][2 * binFeatureIdx].SmallerBorderWeightSum -
                     pairWeightStatistics[y][y][2 * binFeatureIdx].GreaterBorderRightWeightSum);
                weightSum[2 * y][2 * y + 1] += weightDelta;
                weightSum[2 * y + 1][2 * y] += weightDelta;
                weightSum[2 * y][2 * y] -= weightDelta;
                weightSum[2 * y + 1][2 * y + 1] -= weightDelta;

                for (int x = y + 1; x < leafCount; ++x) {
                    const TBucketPairWeightStatistics* xyData = pairWeightStatistics[x][y].data();
                    const TBucketPairWeightStatistics* yxData = pairWeightStatistics[y][x].data();

                    double total =
                        xyData[2 * binFeatureIdx].SmallerBorderWeightSum
                        + yxData[2 * binFeatureIdx].SmallerBorderWeightSum
                        + xyData[2 * binFeatureIdx + 1].SmallerBorderWeightSum
                        + yxData[2 * binFeatureIdx + 1].SmallerBorderWeightSum;

                    UpdateWeightSumFromTotal(y, x, total, &weightSum);

                    const TBucketPairWeightStatistics& xy = xyData[2 * binFeatureIdx];
                    const TBucketPairWeightStatistics& yx = yxData[2 * binFeatureIdx];

                    UpdateWeightSumFromNonDiagStats(y, x, xy, yx, &weightSum);
                }
            }

            const TVector<double> leafValues = CalculatePairwiseLeafValues(weightSum, binDerSums, l2DiagReg, pairwiseBucketWeightPriorReg);
            (*scoreBins)[binFeatureIdx].D2 = 1.0;
            (*scoreBins)[binFeatureIdx].DP = CalculateScore(leafValues, binDerSums, weightSum);
        }
    } else {
        scoreBins->yresize(bucketCount - 1);

        TVector<double> derSum(2 * leafCount, 0.0);
        weightSum.FillZero();

        for (int leafId = 0; leafId < leafCount; ++leafId) {
            for (int bucketIdx = 0; bucketIdx < bucketCount; ++bucketIdx) {
                derSum[2 * leafId + 1] += derSums[leafId][bucketIdx];
            }
        }

        for (int y = 0; y < leafCount; ++y) {
            for (int x = y + 1; x < leafCount; ++x) {
                const TBucketPairWeightStatistics* xyData = pairWeightStatistics[x][y].data();
                const TBucketPairWeightStatistics* yxData = pairWeightStatistics[y][x].data();
                __m128d totalXY0 = _mm_setzero_pd();
                __m128d totalXY2 = _mm_setzero_pd();
                __m128d totalYX0 = _mm_setzero_pd();
                __m128d totalYX2 = _mm_setzero_pd();
                for (int bucketId = 0; bucketId + 4 <= bucketCount; bucketId += 4) {
                    totalXY0 = _mm_add_pd(totalXY0, XmmGather(&xyData[bucketId + 0].SmallerBorderWeightSum, &xyData[bucketId + 1].SmallerBorderWeightSum));
                    totalXY2 = _mm_add_pd(totalXY2, XmmGather(&xyData[bucketId + 2].SmallerBorderWeightSum, &xyData[bucketId + 3].SmallerBorderWeightSum));
                    totalYX0 = _mm_add_pd(totalYX0, XmmGather(&yxData[bucketId + 0].SmallerBorderWeightSum, &yxData[bucketId + 1].SmallerBorderWeightSum));
                    totalYX2 = _mm_add_pd(totalYX2, XmmGather(&yxData[bucketId + 2].SmallerBorderWeightSum, &yxData[bucketId + 3].SmallerBorderWeightSum));
                }
                double total = XmmHorizontalAdd(totalXY0) + XmmHorizontalAdd(totalXY2) + XmmHorizontalAdd(totalYX0) + XmmHorizontalAdd(totalYX2);
                for (int bucketId = bucketCount & ~3u; bucketId < bucketCount; ++bucketId) {
                    total += xyData[bucketId].SmallerBorderWeightSum + yxData[bucketId].SmallerBorderWeightSum;
                }

                UpdateWeightSumFromTotal(y, x, total, &weightSum);
            }
        }

        Y_ASSERT(
            pairwiseStats.SplitEnsembleSpec.SplitType == ESplitType::OnlineCtr ||
            pairwiseStats.SplitEnsembleSpec.SplitType == ESplitType::FloatFeature);

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

                    UpdateWeightSumFromNonDiagStats(y, x, xy, yx, &weightSum);
                }
            }

            const TVector<double> leafValues = CalculatePairwiseLeafValues(weightSum, derSum, l2DiagReg, pairwiseBucketWeightPriorReg);
            (*scoreBins)[splitId].D2 = 1.0;
            (*scoreBins)[splitId].DP = CalculateScore(leafValues, derSum, weightSum);
        }
    }
}

