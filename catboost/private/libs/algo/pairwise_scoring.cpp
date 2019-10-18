#include "pairwise_scoring.h"


#include <catboost/private/libs/algo_helpers/pairwise_leaves_calculation.h>
#include <catboost/libs/helpers/short_vector_ops.h>

#include <util/generic/xrange.h>
#include <util/system/yassert.h>


using namespace NCB;


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


void TPairwiseScoreCalcer::CalculateScore(
    int splitIdx,
    TConstArrayRef<double> avrg,
    TConstArrayRef<double> sumDer,
    const TArray2D<double>& sumWeights
) {
    const ui32 sumDerSize = sumDer.ysize();
    double score = 0;
    for (ui32 x = 0; x < sumDerSize; ++x) {
        const double* avrgData = avrg.data();
        const double* sumWeightsData = &sumWeights[x][0];
        auto subScore0 = NSimdOps::MakeZeros();
        auto subScore2 = NSimdOps::MakeZeros();
        for (ui32 y = 0; y + 2 * NSimdOps::Size <= sumDerSize; y += 2 * NSimdOps::Size) {
            subScore0 = NSimdOps::FusedMultiplyAdd(
                avrgData + y + 0 * NSimdOps::Size,
                sumWeightsData + y + 0 * NSimdOps::Size,
                subScore0);
            subScore2 = NSimdOps::FusedMultiplyAdd(
                avrgData + y + 1 * NSimdOps::Size,
                sumWeightsData + y + 1 * NSimdOps::Size,
                subScore2);
        }
        double subScore = NSimdOps::HorizontalAdd(subScore0) + NSimdOps::HorizontalAdd(subScore2);
        for (ui32 y = sumDerSize - sumDerSize % (2 * NSimdOps::Size); y < sumDerSize; ++y) {
            subScore += avrgData[y] * sumWeightsData[y];
        }
        score += avrg[x] * (sumDer[x] - 0.5 * subScore);
    }
    Scores[splitIdx] = score;
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
    ui32 oneHotMaxSize,
    TPairwiseScoreCalcer* scoreCalcer
) {
    const auto& derSums = pairwiseStats.DerSums;
    const auto& pairWeightStatistics = pairwiseStats.PairWeightStatistics;

    const int leafCount = derSums.ysize();

    TArray2D<double> weightSum(2 * leafCount, 2 * leafCount);

    // TODO(ilyzhin): refactor this (extract common code to functions)
    switch (pairwiseStats.SplitEnsembleSpec.Type) {
        case ESplitEnsembleType::OneFeature:
            {
                scoreCalcer->SetSplitsCount(bucketCount - 1);

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
                        auto totalXY0 = NSimdOps::MakeZeros();
                        auto totalXY2 = NSimdOps::MakeZeros();
                        auto totalYX0 = NSimdOps::MakeZeros();
                        auto totalYX2 = NSimdOps::MakeZeros();
                        for (int bucketId = 0;
                             bucketId + 2 * static_cast<int>(NSimdOps::Size) <= bucketCount;
                             bucketId += 2 * NSimdOps::Size)
                        {
                            totalXY0 = NSimdOps::ElementwiseAdd(
                                totalXY0,
                                NSimdOps::Gather(
                                    &xyData[bucketId + 0].SmallerBorderWeightSum,
                                    &xyData[bucketId + 1].SmallerBorderWeightSum));
                            totalXY2 = NSimdOps::ElementwiseAdd(
                                totalXY2,
                                NSimdOps::Gather(
                                    &xyData[bucketId + 2].SmallerBorderWeightSum,
                                    &xyData[bucketId + 3].SmallerBorderWeightSum));
                            totalYX0 = NSimdOps::ElementwiseAdd(
                                totalYX0,
                                NSimdOps::Gather(
                                    &yxData[bucketId + 0].SmallerBorderWeightSum,
                                    &yxData[bucketId + 1].SmallerBorderWeightSum));
                            totalYX2 = NSimdOps::ElementwiseAdd(
                                totalYX2,
                                NSimdOps::Gather(
                                    &yxData[bucketId + 2].SmallerBorderWeightSum,
                                    &yxData[bucketId + 3].SmallerBorderWeightSum));
                        }
                        double total = NSimdOps::HorizontalAdd(totalXY0) + NSimdOps::HorizontalAdd(totalXY2)
                            + NSimdOps::HorizontalAdd(totalYX0) + NSimdOps::HorizontalAdd(totalYX2);
                        for (int bucketId = bucketCount - bucketCount % (2 * NSimdOps::Size);
                             bucketId < bucketCount;
                             ++bucketId)
                        {
                            total += xyData[bucketId].SmallerBorderWeightSum
                                + yxData[bucketId].SmallerBorderWeightSum;
                        }

                        UpdateWeightSumFromTotal(y, x, total, &weightSum);
                    }
                }

                Y_ASSERT(
                    pairwiseStats.SplitEnsembleSpec.OneSplitType == ESplitType::OnlineCtr ||
                    pairwiseStats.SplitEnsembleSpec.OneSplitType == ESplitType::FloatFeature);

                for (int splitId = 0; splitId < bucketCount - 1; ++splitId) {
                    for (int y = 0; y < leafCount; ++y) {
                        const double derDelta = derSums[y][splitId];
                        derSum[2 * y] += derDelta;
                        derSum[2 * y + 1] -= derDelta;

                        const double weightDelta = (pairWeightStatistics[y][y][splitId].SmallerBorderWeightSum
                            - pairWeightStatistics[y][y][splitId].GreaterBorderRightWeightSum);
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

                    const TVector<double> leafValues = CalculatePairwiseLeafValues(
                        weightSum,
                        derSum,
                        l2DiagReg,
                        pairwiseBucketWeightPriorReg);
                    scoreCalcer->CalculateScore(splitId, leafValues, derSum, weightSum);
                }
            }
            break;
        case ESplitEnsembleType::BinarySplits:
            {
                const int binaryFeaturesCount = (int)GetValueBitCount(bucketCount - 1);

                scoreCalcer->SetSplitsCount(binaryFeaturesCount);

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
                            (pairWeightStatistics[y][y][2 * binFeatureIdx].SmallerBorderWeightSum
                             - pairWeightStatistics[y][y][2 * binFeatureIdx].GreaterBorderRightWeightSum);
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

                    const TVector<double> leafValues = CalculatePairwiseLeafValues(
                        weightSum,
                        binDerSums,
                        l2DiagReg,
                        pairwiseBucketWeightPriorReg);
                    scoreCalcer->CalculateScore(binFeatureIdx, leafValues, binDerSums, weightSum);
                }
            }
            break;
        case ESplitEnsembleType::ExclusiveBundle:
            {
                scoreCalcer->SetSplitsCount(
                    CalcSplitsCount(pairwiseStats.SplitEnsembleSpec, bucketCount, oneHotMaxSize)
                );

                TVector<double> derSum;
                derSum.yresize(2 * leafCount);

                const auto& exclusiveFeaturesBundle = pairwiseStats.SplitEnsembleSpec.ExclusiveFeaturesBundle;

                TArray2D<double> derSumsForAllPositiveBucketsInPart(
                    exclusiveFeaturesBundle.Parts.size(),
                    leafCount); // [leafIdx][partIdx]
                derSumsForAllPositiveBucketsInPart.FillZero();

                TVector<double> derSumForAllBuckets; // [leafIdx]
                derSumForAllBuckets.yresize(leafCount);

                for (int leafId = 0; leafId < leafCount; ++leafId) {
                    derSumForAllBuckets[leafId] = derSums[leafId][bucketCount - 1];
                    for (auto bundlePartIdx : xrange(exclusiveFeaturesBundle.Parts.size())) {
                        const auto& bundlePart = exclusiveFeaturesBundle.Parts[bundlePartIdx];
                        auto& derSumForPart = derSumsForAllPositiveBucketsInPart[leafId][bundlePartIdx];
                        for (auto bucketIdx : bundlePart.Bounds.Iter()) {
                            derSumForPart += derSums[leafId][bucketIdx];
                        }
                        derSumForAllBuckets[leafId] += derSumForPart;
                    }
                }

                size_t srcBucketOffset = 0;
                size_t dstBinOffset = 0;
                for (auto bundlePartIdx : xrange(exclusiveFeaturesBundle.Parts.size())) {
                    const auto& bundlePart = exclusiveFeaturesBundle.Parts[bundlePartIdx];
                    if (!UseForCalcScores(bundlePart, oneHotMaxSize)) {
                        continue;
                    }

                    for (int y = 0; y < leafCount; ++y) {
                        derSum[2 * y] = derSumForAllBuckets[y]
                            - derSumsForAllPositiveBucketsInPart[y][bundlePartIdx];
                        derSum[2 * y + 1] = derSumsForAllPositiveBucketsInPart[y][bundlePartIdx];
                    }

                    weightSum.FillZero();

                    TBoundsInBundle boundsInBundle = bundlePart.Bounds;
                    auto bucketBegin = srcBucketOffset;
                    auto bucketEnd = srcBucketOffset + boundsInBundle.GetSize() + 1;

                    for (int y = 0; y < leafCount; ++y) {
                        for (int x = y + 1; x < leafCount; ++x) {
                            const TBucketPairWeightStatistics* xyData = pairWeightStatistics[x][y].data();
                            const TBucketPairWeightStatistics* yxData = pairWeightStatistics[y][x].data();
                            auto totalXY0 = NSimdOps::MakeZeros();
                            auto totalXY2 = NSimdOps::MakeZeros();
                            auto totalYX0 = NSimdOps::MakeZeros();
                            auto totalYX2 = NSimdOps::MakeZeros();

                            auto bucketId = bucketBegin;
                            for (;
                                 bucketId + 2 * NSimdOps::Size <= bucketEnd;
                                 bucketId += 2 * NSimdOps::Size)
                            {
                                totalXY0 = NSimdOps::ElementwiseAdd(
                                    totalXY0,
                                    NSimdOps::Gather(
                                        &xyData[bucketId + 0].SmallerBorderWeightSum,
                                        &xyData[bucketId + 1].SmallerBorderWeightSum));
                                totalXY2 = NSimdOps::ElementwiseAdd(
                                    totalXY2,
                                    NSimdOps::Gather(
                                        &xyData[bucketId + 2].SmallerBorderWeightSum,
                                        &xyData[bucketId + 3].SmallerBorderWeightSum));
                                totalYX0 = NSimdOps::ElementwiseAdd(
                                    totalYX0,
                                    NSimdOps::Gather(
                                        &yxData[bucketId + 0].SmallerBorderWeightSum,
                                        &yxData[bucketId + 1].SmallerBorderWeightSum));
                                totalYX2 = NSimdOps::ElementwiseAdd(
                                    totalYX2,
                                    NSimdOps::Gather(
                                        &yxData[bucketId + 2].SmallerBorderWeightSum,
                                        &yxData[bucketId + 3].SmallerBorderWeightSum));
                            }
                            double total = NSimdOps::HorizontalAdd(totalXY0)
                                + NSimdOps::HorizontalAdd(totalXY2)
                                + NSimdOps::HorizontalAdd(totalYX0)
                                + NSimdOps::HorizontalAdd(totalYX2);
                            for (; bucketId < bucketEnd; ++bucketId) {
                                total += xyData[bucketId].SmallerBorderWeightSum
                                    + yxData[bucketId].SmallerBorderWeightSum;
                            }

                            UpdateWeightSumFromTotal(y, x, total, &weightSum);
                        }
                    }

                    for (ui32 splitId = 0; splitId < boundsInBundle.GetSize(); ++splitId) {
                        auto bucketId = srcBucketOffset + splitId;
                        for (int y = 0; y < leafCount; ++y) {
                            if (splitId > 0) {
                                const double derDelta = derSums[y][bundlePart.Bounds.Begin + splitId - 1];
                                derSum[2 * y] += derDelta;
                                derSum[2 * y + 1] -= derDelta;
                            }

                            const double weightDelta =
                                (pairWeightStatistics[y][y][bucketId].SmallerBorderWeightSum
                                 - pairWeightStatistics[y][y][bucketId].GreaterBorderRightWeightSum);
                            weightSum[2 * y][2 * y + 1] += weightDelta;
                            weightSum[2 * y + 1][2 * y] += weightDelta;
                            weightSum[2 * y][2 * y] -= weightDelta;
                            weightSum[2 * y + 1][2 * y + 1] -= weightDelta;

                            for (int x = y + 1; x < leafCount; ++x) {
                                const TBucketPairWeightStatistics& xy = pairWeightStatistics[x][y][bucketId];
                                const TBucketPairWeightStatistics& yx = pairWeightStatistics[y][x][bucketId];

                                UpdateWeightSumFromNonDiagStats(y, x, xy, yx, &weightSum);
                            }
                        }

                        const TVector<double> leafValues = CalculatePairwiseLeafValues(
                            weightSum,
                            derSum,
                            l2DiagReg,
                            pairwiseBucketWeightPriorReg);
                        scoreCalcer->CalculateScore(
                            dstBinOffset + splitId,
                            leafValues,
                            derSum,
                            weightSum);
                    }

                    srcBucketOffset = bucketEnd;
                    dstBinOffset += boundsInBundle.GetSize();
                }
            }
            break;
        case ESplitEnsembleType::FeaturesGroup:
            {
                scoreCalcer->SetSplitsCount(
                    CalcSplitsCount(pairwiseStats.SplitEnsembleSpec, bucketCount, oneHotMaxSize)
                );
                const auto& group = pairwiseStats.SplitEnsembleSpec.FeaturesGroup;
                TVector<double> derSum;
                int bucketIdxOffset = 0;
                int splitIdxOffset = 0;
                for (const auto& part : group.Parts) {
                    derSum.assign(2 * leafCount, 0);
                    weightSum.FillZero();
                    for (int leafId = 0; leafId < leafCount; ++leafId) {
                        for (int bucketIdx = bucketIdxOffset; bucketIdx < bucketIdxOffset + static_cast<int>(part.BucketCount); ++bucketIdx) {
                            derSum[2 * leafId + 1] += derSums[leafId][bucketIdx];
                        }
                    }
                    for (int y = 0; y < leafCount; ++y) {
                        for (int x = y + 1; x < leafCount; ++x) {
                            const TBucketPairWeightStatistics* xyData = pairWeightStatistics[x][y].data();
                            const TBucketPairWeightStatistics* yxData = pairWeightStatistics[y][x].data();
                            auto totalXY0 = NSimdOps::MakeZeros();
                            auto totalXY2 = NSimdOps::MakeZeros();
                            auto totalYX0 = NSimdOps::MakeZeros();
                            auto totalYX2 = NSimdOps::MakeZeros();
                            for (int bucketId = bucketIdxOffset;
                                 bucketId + 2 * static_cast<int>(NSimdOps::Size) <= bucketIdxOffset + static_cast<int>(part.BucketCount);
                                 bucketId += 2 * NSimdOps::Size) {
                                totalXY0 = NSimdOps::ElementwiseAdd(
                                    totalXY0,
                                    NSimdOps::Gather(
                                        &xyData[bucketId + 0].SmallerBorderWeightSum,
                                        &xyData[bucketId + 1].SmallerBorderWeightSum));
                                totalXY2 = NSimdOps::ElementwiseAdd(
                                    totalXY2,
                                    NSimdOps::Gather(
                                        &xyData[bucketId + 2].SmallerBorderWeightSum,
                                        &xyData[bucketId + 3].SmallerBorderWeightSum));
                                totalYX0 = NSimdOps::ElementwiseAdd(
                                    totalYX0,
                                    NSimdOps::Gather(
                                        &yxData[bucketId + 0].SmallerBorderWeightSum,
                                        &yxData[bucketId + 1].SmallerBorderWeightSum));
                                totalYX2 = NSimdOps::ElementwiseAdd(
                                    totalYX2,
                                    NSimdOps::Gather(
                                        &yxData[bucketId + 2].SmallerBorderWeightSum,
                                        &yxData[bucketId + 3].SmallerBorderWeightSum));
                            }
                            double total =
                                NSimdOps::HorizontalAdd(totalXY0) + NSimdOps::HorizontalAdd(totalXY2)
                                + NSimdOps::HorizontalAdd(totalYX0) + NSimdOps::HorizontalAdd(totalYX2);
                            for (int bucketId = bucketIdxOffset + part.BucketCount - part.BucketCount % (2 * NSimdOps::Size);
                                 bucketId < bucketIdxOffset + bucketCount;
                                 ++bucketId) {
                                total += xyData[bucketId].SmallerBorderWeightSum
                                         + yxData[bucketId].SmallerBorderWeightSum;
                            }
                            UpdateWeightSumFromTotal(y, x, total, &weightSum);
                        }
                    }
                    for (int splitId = splitIdxOffset, bucketId = bucketIdxOffset; splitId < splitIdxOffset + static_cast<int>(part.BucketCount) - 1; ++splitId, ++bucketId) {
                        for (int y = 0; y < leafCount; ++y) {
                            const double derDelta = derSums[y][bucketId];
                            derSum[2 * y] += derDelta;
                            derSum[2 * y + 1] -= derDelta;
                            const double weightDelta = (
                                pairWeightStatistics[y][y][bucketId].SmallerBorderWeightSum
                                - pairWeightStatistics[y][y][bucketId].GreaterBorderRightWeightSum);
                            weightSum[2 * y][2 * y + 1] += weightDelta;
                            weightSum[2 * y + 1][2 * y] += weightDelta;
                            weightSum[2 * y][2 * y] -= weightDelta;
                            weightSum[2 * y + 1][2 * y + 1] -= weightDelta;
                            for (int x = y + 1; x < leafCount; ++x) {
                                const TBucketPairWeightStatistics& xy = pairWeightStatistics[x][y][bucketId];
                                const TBucketPairWeightStatistics& yx = pairWeightStatistics[y][x][bucketId];
                                UpdateWeightSumFromNonDiagStats(y, x, xy, yx, &weightSum);
                            }
                        }
                        const TVector<double> leafValues = CalculatePairwiseLeafValues(
                            weightSum,
                            derSum,
                            l2DiagReg,
                            pairwiseBucketWeightPriorReg);
                        scoreCalcer->CalculateScore(splitId, leafValues, derSum, weightSum);
                    }
                    bucketIdxOffset += part.BucketCount;
                    splitIdxOffset += part.BucketCount - 1;
                }
            }
            break;
    }
}

