#include "score_calcer.h"
#include "index_calcer.h"
#include "split.h"

struct TBucketStats {
    double SumWeightedDelta = 0;
    double SumWeight = 0;
    double SumDelta = 0;
    double Count = 0;

    void Add(const TBucketStats& other) {
        SumWeightedDelta += other.SumWeightedDelta;
        SumDelta += other.SumDelta;
        SumWeight += other.SumWeight;
        Count += other.Count;
    }

    void Remove(const TBucketStats& other) {
        SumWeightedDelta -= other.SumWeightedDelta;
        SumDelta -= other.SumDelta;
        SumWeight -= other.SumWeight;
        Count -= other.Count;
    }
};

static double CountDp(double avrg, const TBucketStats& leafStats) {
    return avrg * leafStats.SumWeightedDelta;
}

static double CountD2(double avrg, const TBucketStats& leafStats) {
    return avrg * avrg * leafStats.SumWeight;
}

struct TScoreBin {
    double DP, D2;

    TScoreBin()
        : DP(0)
        , D2(1e-100)
    {
    }

    double GetScore() const {
        return DP / sqrt(D2);
    }
};

static int GetSplitCount(const yvector<int>& splitsCount,
                         const yvector<yvector<int>>& oneHotValues,
                         const TSplitCandidate& split,
                         int ctrBorderCount) {
    if (split.Type == ESplitType::OnlineCtr) {
        return ctrBorderCount;
    } else if (split.Type == ESplitType::FloatFeature) {
        return splitsCount[split.FeatureIdx];
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        return oneHotValues[split.FeatureIdx].ysize();
    }
}

template<typename TFullIndexType>
yvector<double> CalcScoreImpl(const TAllFeatures& af,
                              const yvector<int>& splitsCount,
                              const TFold& fold,
                              const yvector<TIndexType>& indices,
                              const TSplitCandidate& split,
                              int depth,
                              int ctrBorderCount,
                              float l2Regularizer) {
    const int partCount = fold.BodyTailArr.ysize();
    const int docCount = fold.BodyTailArr[partCount - 1].TailFinish;

    yvector<TFullIndexType> singleIdx;
    singleIdx.yresize(docCount);

    const int splitCount = GetSplitCount(splitsCount, af.OneHotValues, split, ctrBorderCount);
    const int bucketCount = splitCount + 1;
    if (split.Type == ESplitType::OnlineCtr) {
        const ui8* ctrs = fold.GetCtr(split.Ctr.Projection).Feature[split.Ctr.CtrIdx]
                                                                   [split.Ctr.TargetBorderIdx]
                                                                   [split.Ctr.PriorIdx].data();
        for (int doc = 0; doc < docCount; ++doc) {
            singleIdx[doc] = indices[doc] * bucketCount + ctrs[doc];
        }
    } else if (split.Type == ESplitType::FloatFeature) {
        const int feature = split.FeatureIdx;
        for (int doc = 0; doc < docCount; ++doc) {
            const int originalDocIdx = fold.LearnPermutation[doc];
            singleIdx[doc] = indices[doc] * bucketCount + af.FloatHistograms[feature][originalDocIdx];
        }
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        const int feature = split.FeatureIdx;
        for (int doc = 0; doc < docCount; ++doc) {
            const int originalDocIdx = fold.LearnPermutation[doc];
            singleIdx[doc] = indices[doc] * bucketCount + af.CatFeaturesRemapped[feature][originalDocIdx];
        }
    }

    yvector<TScoreBin> scoreBin(bucketCount);

    const int approxDimension = fold.GetApproxDimension();
    const int leafCount = 1 << depth;
    for (const TFold::TBodyTail& bt : fold.BodyTailArr) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            yvector<TBucketStats> stats(leafCount * bucketCount);

            if (fold.LearnWeights.empty()) {
                for (int doc = 0; doc < bt.BodyFinish; ++doc) {
                    TBucketStats& leafStats = stats[singleIdx[doc]];
                    leafStats.SumDelta += bt.Derivatives[dim][doc];
                    leafStats.Count += 1;
                }
            } else {
                for (int doc = 0; doc < bt.BodyFinish; ++doc) {
                    TBucketStats& leafStats = stats[singleIdx[doc]];
                    leafStats.SumDelta += bt.Derivatives[dim][doc];
                    leafStats.Count += fold.LearnWeights[doc];
                }
            }

            for (int doc = bt.BodyFinish; doc < bt.TailFinish; ++doc) {
                TBucketStats& leafStats = stats[singleIdx[doc]];
                leafStats.SumWeightedDelta += bt.WeightedDer[dim][doc];
                leafStats.SumWeight += fold.SampleWeights[doc];
            }

            for (int leaf = 0; leaf < leafCount; ++leaf) {
                TBucketStats allStats;
                for (int bucket = 0; bucket < bucketCount; ++bucket) {
                    const TBucketStats& leafStats = stats[leaf * bucketCount + bucket];
                    allStats.Add(leafStats);
                }

                TBucketStats trueStats;
                TBucketStats falseStats;
                if (split.Type == ESplitType::OnlineCtr || split.Type == ESplitType::FloatFeature) {
                    trueStats = allStats;
                } else {
                    falseStats = allStats;
                }

                for (int splitIdx = 0; splitIdx < splitCount; ++splitIdx) {
                    if (split.Type == ESplitType::OnlineCtr || split.Type == ESplitType::FloatFeature) {
                        falseStats.Add(stats[leaf * bucketCount + splitIdx]);
                        trueStats.Remove(stats[leaf * bucketCount + splitIdx]);
                    } else {
                        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
                        if (splitIdx > 0) {
                            falseStats.Add(stats[leaf * bucketCount + splitIdx - 1]);
                        }
                        falseStats.Remove(stats[leaf * bucketCount + splitIdx]);
                        trueStats = stats[leaf * bucketCount + splitIdx];
                    }

                    double trueAvrg = CalcAverage(trueStats.SumDelta, trueStats.Count, l2Regularizer);
                    double falseAvrg = CalcAverage(falseStats.SumDelta, falseStats.Count, l2Regularizer);

                    scoreBin[splitIdx].DP += CountDp(trueAvrg, trueStats) + CountDp(falseAvrg, falseStats);
                    scoreBin[splitIdx].D2 += CountD2(trueAvrg, trueStats) + CountD2(falseAvrg, falseStats);
                }
            }
        }
    }

    yvector<double> result(splitCount);
    for (int splitIdx = 0; splitIdx < splitCount; ++splitIdx) {
        result[splitIdx] = scoreBin[splitIdx].GetScore();
    }
    return result;
}

yvector<double> CalcScore(const TAllFeatures& af,
                          const yvector<int>& splitsCount,
                          const TFold& fold,
                          const yvector<TIndexType>& indices,
                          const TSplitCandidate& split,
                          int depth,
                          int ctrBorderCount,
                          float l2Regularizer) {
    const int bucketIndexBits = GetValueBitCount(GetSplitCount(splitsCount, af.OneHotValues, split, ctrBorderCount) + 1) + depth + 1;
    if (bucketIndexBits <= 8) {
        return CalcScoreImpl<ui8>(af, splitsCount, fold, indices, split, depth, ctrBorderCount, l2Regularizer);
    } else if (bucketIndexBits <= 16) {
        return CalcScoreImpl<ui16>(af, splitsCount, fold, indices, split, depth, ctrBorderCount, l2Regularizer);
    } else if (bucketIndexBits <= 32) {
        return CalcScoreImpl<ui32>(af, splitsCount, fold, indices, split, depth, ctrBorderCount, l2Regularizer);
    } else {
        CB_ENSURE(false, "too deep or too much splitsCount for score calculation");
    }
}
