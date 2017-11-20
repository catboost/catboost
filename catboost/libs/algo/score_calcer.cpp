#include "calc_score_cache.h"
#include "index_calcer.h"
#include "score_calcer.h"
#include "split.h"

static double CountDp(double avrg, const TBucketStats& leafStats) {
    return avrg * leafStats.SumWeightedDelta;
}

static double CountD2(double avrg, const TBucketStats& leafStats) {
    return avrg * avrg * leafStats.SumWeight;
}

struct TScoreBin {
    double DP = 0, D2 = 1e-100;

    double GetScore() const {
        return DP / sqrt(D2);
    }
};

static int GetSplitCount(const TVector<int>& splitsCount,
                         const TVector<TVector<int>>& oneHotValues,
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

struct TStatsIndexer {
    const int BucketCount;
    TStatsIndexer(int bucketCount)
    : BucketCount(bucketCount)
    {
    }
    int CalcSize(int depth) const {
        return (1U << depth) * BucketCount;
    }
    int GetIndex(int leafIndex, int bucketIndex) const {
        return BucketCount * leafIndex + bucketIndex;
    }
};

static void UpdateScoreBin(const TBucketStats* stats, int leafCount, const TStatsIndexer& indexer, ESplitType splitType, float l2Regularizer, TVector<TScoreBin>* scoreBin) {
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        TBucketStats allStats{0, 0, 0, 0};
        for (int bucket = 0; bucket < indexer.BucketCount; ++bucket) {
            const TBucketStats& leafStats = stats[indexer.GetIndex(leaf, bucket)];
            allStats.Add(leafStats);
        }
        TBucketStats trueStats{0, 0, 0, 0};
        TBucketStats falseStats{0, 0, 0, 0};
        if (splitType == ESplitType::OnlineCtr || splitType == ESplitType::FloatFeature) {
            trueStats = allStats;
            for (int splitIdx = 0; splitIdx < indexer.BucketCount - 1; ++splitIdx) {
                falseStats.Add(stats[indexer.GetIndex(leaf, splitIdx)]);
                trueStats.Remove(stats[indexer.GetIndex(leaf, splitIdx)]);
                const double trueAvrg = CalcAverage(trueStats.SumDelta, trueStats.Count, l2Regularizer);
                const double falseAvrg = CalcAverage(falseStats.SumDelta, falseStats.Count, l2Regularizer);
                (*scoreBin)[splitIdx].DP += CountDp(trueAvrg, trueStats) + CountDp(falseAvrg, falseStats);
                (*scoreBin)[splitIdx].D2 += CountD2(trueAvrg, trueStats) + CountD2(falseAvrg, falseStats);
            }
        } else {
            Y_ASSERT(splitType == ESplitType::OneHotFeature);
            falseStats = allStats;
            for (int splitIdx = 0; splitIdx < indexer.BucketCount - 1; ++splitIdx) {
                if (splitIdx > 0) {
                    falseStats.Add(stats[indexer.GetIndex(leaf, splitIdx - 1)]);
                }
                falseStats.Remove(stats[indexer.GetIndex(leaf, splitIdx)]);
                trueStats = stats[indexer.GetIndex(leaf, splitIdx)];
                const double trueAvrg = CalcAverage(trueStats.SumDelta, trueStats.Count, l2Regularizer);
                const double falseAvrg = CalcAverage(falseStats.SumDelta, falseStats.Count, l2Regularizer);
                (*scoreBin)[splitIdx].DP += CountDp(trueAvrg, trueStats) + CountDp(falseAvrg, falseStats);
                (*scoreBin)[splitIdx].D2 += CountD2(trueAvrg, trueStats) + CountD2(falseAvrg, falseStats);
            }
        }
    }
}

template<typename TBucketIndexType, typename TFullIndexType>
static void SetSingleIndex(int docCount,
                           int permutationBlockSize,
                           const TStatsIndexer& indexer,
                           const TVector<TBucketIndexType>& bucketIndex,
                           const TIndexType* indices,
                           const int* docPermutation,
                           TVector<TFullIndexType>* singleIdx) {
    singleIdx->yresize(docCount);
    if (docPermutation == nullptr) {
        for (int doc = 0; doc < docCount; ++doc) {
            (*singleIdx)[doc] = indexer.GetIndex(indices[doc], bucketIndex[doc]);
        }
    } else if (permutationBlockSize != FoldPermutationBlockSizeNotSet) {
        const int blockCount = (docCount + permutationBlockSize - 1) / permutationBlockSize;
        int blockStart = 0;
        while (blockStart < docCount) {
            const int blockIdx = docPermutation[blockStart] / permutationBlockSize;
            const int nextBlockStart = blockStart + (blockIdx + 1 == blockCount ? docCount - blockIdx * permutationBlockSize : permutationBlockSize);
            const int originalBlockIdx = docPermutation[blockStart];
            for (int doc = blockStart; doc < nextBlockStart; ++doc) {
                const int originalDocIdx = originalBlockIdx + doc - blockStart;
                (*singleIdx)[doc] = indexer.GetIndex(indices[doc], bucketIndex[originalDocIdx]);
            }
            blockStart = nextBlockStart;
        }
    } else {
        for (int doc = 0; doc < docCount; ++doc) {
            const int originalDocIdx = docPermutation[doc];
            (*singleIdx)[doc] = indexer.GetIndex(indices[doc], bucketIndex[originalDocIdx]);
        }
    }
}

template<typename TFullIndexType>
static void BuildSingleIndex(int docCount,
                             const TAllFeatures& af,
                             const TFold& fold,
                             const TIndexType* indices,
                             const TSplitCandidate& split,
                             const int* learnPermutation,
                             const int* docSubset,
                             const TStatsIndexer& indexer,
                             TVector<TFullIndexType>* singleIdx) {
    const int effectiveBlockSize = docCount == fold.LearnPermutation.ysize() ? fold.PermutationBlockSize : FoldPermutationBlockSizeNotSet;
    if (split.Type == ESplitType::OnlineCtr) {
        SetSingleIndex(docCount, effectiveBlockSize, indexer, fold.GetCtr(split.Ctr.Projection).Feature[split.Ctr.CtrIdx][split.Ctr.TargetBorderIdx][split.Ctr.PriorIdx], indices, docSubset, singleIdx);
    } else if (split.Type == ESplitType::FloatFeature) {
        SetSingleIndex(docCount, effectiveBlockSize, indexer, af.FloatHistograms[split.FeatureIdx], indices, learnPermutation, singleIdx);
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        SetSingleIndex(docCount, effectiveBlockSize, indexer, af.CatFeaturesRemapped[split.FeatureIdx], indices, learnPermutation, singleIdx);
    }
}

template<typename TFullIndexType>
static void UpdateDeltaCount(const TVector<TFullIndexType>& singleIdx, const double* derivatives, const float* learnWeights, int docCount, TBucketStats* stats) {
    if (learnWeights == nullptr) {
        for (int doc = 0; doc < docCount; ++doc) {
            TBucketStats& leafStats = stats[singleIdx[doc]];
            leafStats.SumDelta += derivatives[doc];
            leafStats.Count += 1;
        }
    } else {
        for (int doc = 0; doc < docCount; ++doc) {
            TBucketStats& leafStats = stats[singleIdx[doc]];
            leafStats.SumDelta += derivatives[doc];
            leafStats.Count += learnWeights[doc];
        }
    }
}

template<typename TFullIndexType>
static void UpdateWeighted(const TVector<TFullIndexType>& singleIdx, const double* weightedDer, const float* sampleWeights, int docBegin, int docEnd, TBucketStats* stats) {
    for (int doc = docBegin; doc < docEnd; ++doc) {
        TBucketStats& leafStats = stats[singleIdx[doc]];
        leafStats.SumWeightedDelta += weightedDer[doc];
        leafStats.SumWeight += sampleWeights[doc];
    }
}

static void FixUpStats(int depth, const TStatsIndexer& indexer, bool selectedSplitValue, TBucketStats* stats) {
    const int halfOfStats = indexer.CalcSize(depth - 1);
    if (selectedSplitValue == true) {
        for (int statIdx = 0; statIdx < halfOfStats; ++statIdx) {
            stats[statIdx].Remove(stats[statIdx + halfOfStats]);
        }
    } else {
        for (int statIdx = 0; statIdx < halfOfStats; ++statIdx) {
            stats[statIdx].Remove(stats[statIdx + halfOfStats]);
            DoSwap(stats[statIdx], stats[statIdx + halfOfStats]);
        }
    }
}

template<typename TFullIndexType>
static TVector<double> CalcScoreImpl(const TAllFeatures& af,
                                     const TFold& fold,
                                     const TVector<TIndexType>& indices,
                                     const TSplitCandidate& split,
                                     const TFitParams& fitParams,
                                     const TStatsIndexer& indexer,
                                     int depth,
                                     int splitStatsCount,
                                     TBucketStats* splitStats) {
    TVector<TFullIndexType> singleIdx;
    BuildSingleIndex(indices.ysize(), af, fold, indices.data(), split, fold.LearnPermutation.data(), nullptr, indexer, &singleIdx);
    const int approxDimension = fold.GetApproxDimension();
    const int leafCount = 1 << depth;
    const float l2Regularizer = fitParams.L2LeafRegularizer;
    TVector<TScoreBin> scoreBin(indexer.BucketCount);
    for (int bodyTailIdx = 0; bodyTailIdx < fold.BodyTailArr.ysize(); ++bodyTailIdx) {
        const auto& bt = fold.BodyTailArr[bodyTailIdx];
        for (int dim = 0; dim < approxDimension; ++dim) {
            TBucketStats* stats = splitStats + (bodyTailIdx * approxDimension + dim) * splitStatsCount;
            Fill(stats, stats + indexer.CalcSize(depth), TBucketStats{0, 0, 0, 0});
            UpdateDeltaCount(singleIdx, bt.Derivatives[dim].data(), fold.LearnWeights.data(), bt.BodyFinish, stats);
            UpdateWeighted(singleIdx, bt.WeightedDer[dim].data(), fold.SampleWeights.data(), bt.BodyFinish, bt.TailFinish, stats);
            UpdateScoreBin(stats, leafCount, indexer, split.Type, l2Regularizer, &scoreBin);
        }
    }
    TVector<double> result(indexer.BucketCount - 1);
    for (int splitIdx = 0; splitIdx < indexer.BucketCount - 1; ++splitIdx) {
        result[splitIdx] = scoreBin[splitIdx].GetScore();
    }
    return result;
}

template<typename TFullIndexType>
static TVector<double> CachedCalcScoreImpl(const TAllFeatures& af,
                                           const TFold& fold,
                                           const TSmallestSplitSideFold& prevLevelData,
                                           const TSplitCandidate& split,
                                           const TFitParams& fitParams,
                                           const TStatsIndexer& indexer,
                                           int depth,
                                           int splitStatsCount,
                                           TBucketStats* splitStats) {
    Y_ASSERT(depth > 0);
    TVector<TFullIndexType> singleIdx;
    BuildSingleIndex(prevLevelData.DocCount, af, fold, prevLevelData.Indices.data(), split, prevLevelData.LearnPermutation.data(), prevLevelData.IndexInFold.data(), indexer, &singleIdx);
    const int approxDimension = prevLevelData.GetApproxDimension();
    const int leafCount = 1 << depth;
    const float l2Regularizer = fitParams.L2LeafRegularizer;
    TVector<TScoreBin> scoreBin(indexer.BucketCount);
    for (int bodyTailIdx = 0; bodyTailIdx < prevLevelData.BodyTailArr.ysize(); ++bodyTailIdx) {
        const auto& bt = prevLevelData.BodyTailArr[bodyTailIdx];
        for (int dim = 0; dim < approxDimension; ++dim) {
            TBucketStats* stats = splitStats + (bodyTailIdx * approxDimension + dim) * splitStatsCount;
            Fill(stats + indexer.CalcSize(depth - 1), stats + indexer.CalcSize(depth), TBucketStats{0, 0, 0, 0});
            UpdateDeltaCount(singleIdx, bt.Derivatives[dim].data(), prevLevelData.LearnWeights.data(), bt.BodyFinish, stats);
            UpdateWeighted(singleIdx, bt.WeightedDer[dim].data(), prevLevelData.SampleWeights.data(), bt.BodyFinish, bt.TailFinish, stats);
            FixUpStats(depth, indexer, prevLevelData.SmallestSplitSideValue, stats);
            UpdateScoreBin(stats, leafCount, indexer, split.Type, l2Regularizer, &scoreBin);
        }
    }
    TVector<double> result(indexer.BucketCount - 1);
    for (int splitIdx = 0; splitIdx < indexer.BucketCount - 1; ++splitIdx) {
        result[splitIdx] = scoreBin[splitIdx].GetScore();
    }
    return result;
}

TVector<double> CalcScore(const TAllFeatures& af,
                          const TVector<int>& splitsCount,
                          const TFold& fold,
                          const TVector<TIndexType>& indices,
                          const TSmallestSplitSideFold& prevLevelData,
                          const TFitParams& fitParams,
                          const TSplitCandidate& split,
                          int depth,
                          TStatsFromPrevTree* statsFromPrevTree) {
    const int splitCount = GetSplitCount(splitsCount, af.OneHotValues, split, fitParams.CtrParams.CtrBorderCount);
    const TStatsIndexer indexer(splitCount + 1);
    const int bucketIndexBits = GetValueBitCount(GetSplitCount(splitsCount, af.OneHotValues, split, fitParams.CtrParams.CtrBorderCount) + 1) + depth + 1;
    if (!AreStatsFromPrevTreeUsed(fitParams)) {
        TVector<TBucketStats> scratchSplitStats;
        scratchSplitStats.yresize(indexer.CalcSize(depth));
        if (bucketIndexBits <= 8) {
            return CalcScoreImpl<ui8>(af, fold, indices, split, fitParams, indexer, depth, /*splitStatsCount*/ 0, scratchSplitStats.data());
        } else if (bucketIndexBits <= 16) {
            return CalcScoreImpl<ui16>(af, fold, indices, split, fitParams, indexer, depth, /*splitStatsCount*/ 0, scratchSplitStats.data());
        } else if (bucketIndexBits <= 32) {
            return CalcScoreImpl<ui32>(af, fold, indices, split, fitParams, indexer, depth, /*splitStatsCount*/ 0, scratchSplitStats.data());
        }
    } else {
        const int splitStatsCount = indexer.CalcSize(fitParams.Depth);
        const int statsCount = fold.BodyTailArr.ysize() * fold.GetApproxDimension() * splitStatsCount;
        bool areStatsDirty;
        TVector<TBucketStats, TPoolAllocator>& splitStats = statsFromPrevTree->GetStats(split, statsCount, &areStatsDirty); // thread-safe access
        if (depth == 0 || areStatsDirty) {
            if (bucketIndexBits <= 8) {
                return CalcScoreImpl<ui8>(af, fold, indices, split, fitParams, indexer, depth, splitStatsCount, splitStats.data());
            } else if (bucketIndexBits <= 16) {
                return CalcScoreImpl<ui16>(af, fold, indices, split, fitParams, indexer, depth, splitStatsCount, splitStats.data());
            } else if (bucketIndexBits <= 32) {
                return CalcScoreImpl<ui32>(af, fold, indices, split, fitParams, indexer, depth, splitStatsCount, splitStats.data());
            }
        } else {
            if (bucketIndexBits <= 8) {
                return CachedCalcScoreImpl<ui8>(af, fold, prevLevelData, split, fitParams, indexer, depth, splitStatsCount, splitStats.data());
            } else if (bucketIndexBits <= 16) {
                return CachedCalcScoreImpl<ui16>(af, fold, prevLevelData, split, fitParams, indexer, depth, splitStatsCount, splitStats.data());
            } else if (bucketIndexBits <= 32) {
                return CachedCalcScoreImpl<ui32>(af, fold, prevLevelData, split, fitParams, indexer, depth, splitStatsCount, splitStats.data());
            }
        }
    }
    CB_ENSURE(false, "too deep or too much splitsCount for score calculation");
}
