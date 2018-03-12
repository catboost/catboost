#include "score_calcer.h"
#include "calc_score_cache.h"
#include "index_calcer.h"
#include "split.h"
#include <catboost/libs/options/defaults_helper.h>

#include <type_traits>

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
                         const TSplitCandidate& split) {
    if (split.Type == ESplitType::OnlineCtr) {
        return split.Ctr.BorderCount;
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

template<typename TIsPlainMode>
static void UpdateScoreBin(const TBucketStats* stats, int leafCount, const TStatsIndexer& indexer, ESplitType splitType, float l2Regularizer, TIsPlainMode isPlainMode, TVector<TScoreBin>* scoreBin) {
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
                double trueAvrg, falseAvrg;
                if (isPlainMode) {
                    trueAvrg = CalcAverage(trueStats.SumWeightedDelta, trueStats.SumWeight, l2Regularizer);
                    falseAvrg = CalcAverage(falseStats.SumWeightedDelta, falseStats.SumWeight, l2Regularizer);
                } else {
                    trueAvrg = CalcAverage(trueStats.SumDelta, trueStats.Count, l2Regularizer);
                    falseAvrg = CalcAverage(falseStats.SumDelta, falseStats.Count, l2Regularizer);
                }
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
                double trueAvrg, falseAvrg;
                if (isPlainMode) {
                    trueAvrg = CalcAverage(trueStats.SumWeightedDelta, trueStats.SumWeight, l2Regularizer);
                    falseAvrg = CalcAverage(falseStats.SumWeightedDelta, falseStats.SumWeight, l2Regularizer);
                } else {
                    trueAvrg = CalcAverage(trueStats.SumDelta, trueStats.Count, l2Regularizer);
                    falseAvrg = CalcAverage(falseStats.SumDelta, falseStats.Count, l2Regularizer);
                }
                (*scoreBin)[splitIdx].DP += CountDp(trueAvrg, trueStats) + CountDp(falseAvrg, falseStats);
                (*scoreBin)[splitIdx].D2 += CountD2(trueAvrg, trueStats) + CountD2(falseAvrg, falseStats);
            }
        }
    }
}

template<typename TBucketIndexType, typename TFullIndexType>
static void SetSingleIndex(const TCalcScoreFold& fold,
                           const TStatsIndexer& indexer,
                           const TVector<TBucketIndexType>& bucketIndex,
                           const size_t* docPermutation,
                           TVector<TFullIndexType>* singleIdx) {
    const size_t docCount = fold.GetDocCount();
    const size_t permBlockSize = fold.PermutationBlockSize;
    const TIndexType* indices = GetDataPtr(fold.Indices);

    singleIdx->yresize(docCount);
    if (docPermutation == nullptr || permBlockSize == docCount) {
        for (size_t doc = 0; doc < docCount; ++doc) {
            (*singleIdx)[doc] = indexer.GetIndex(indices[doc], bucketIndex[doc]);
        }
    } else if (permBlockSize > 1) {
        const size_t blockCount = (docCount + permBlockSize - 1) / permBlockSize;
        Y_ASSERT(docPermutation[0] / permBlockSize + 1 == blockCount || docPermutation[0] + permBlockSize - 1 == docPermutation[permBlockSize - 1]);
        size_t blockStart = 0;
        while (blockStart < docCount) {
            const size_t blockIdx = docPermutation[blockStart] / permBlockSize;
            const size_t nextBlockStart = blockStart + (blockIdx + 1 == blockCount ? docCount - blockIdx * permBlockSize : permBlockSize);
            const size_t originalBlockIdx = docPermutation[blockStart];
            for (size_t doc = blockStart; doc < nextBlockStart; ++doc) {
                const size_t originalDocIdx = originalBlockIdx + doc - blockStart;
                (*singleIdx)[doc] = indexer.GetIndex(indices[doc], bucketIndex[originalDocIdx]);
            }
            blockStart = nextBlockStart;
        }
    } else {
        for (size_t doc = 0; doc < docCount; ++doc) {
            const size_t originalDocIdx = docPermutation[doc];
            (*singleIdx)[doc] = indexer.GetIndex(indices[doc], bucketIndex[originalDocIdx]);
        }
    }
}

static inline const TOnlineCTR& GetCtr(const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs, const TProjection& proj) {
    static const constexpr size_t OnlineSingleCtrsIndex = 0;
    static const constexpr size_t OnlineCTRIndex = 1;
    return proj.HasSingleFeature() ? std::get<OnlineSingleCtrsIndex>(allCtrs).at(proj) : std::get<OnlineCTRIndex>(allCtrs).at(proj);
}

template<typename TFullIndexType>
static void BuildSingleIndex(const TCalcScoreFold& fold,
                             const TAllFeatures& af,
                             const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
                             const TSplitCandidate& split,
                             const TStatsIndexer& indexer,
                             TVector<TFullIndexType>* singleIdx) {
    if (split.Type == ESplitType::OnlineCtr) {
        const TCtr& ctr = split.Ctr;
        const size_t* docSubset = GetDataPtr(fold.IndexInFold);
        SetSingleIndex(fold, indexer, GetCtr(allCtrs, ctr.Projection).Feature[ctr.CtrIdx][ctr.TargetBorderIdx][ctr.PriorIdx], docSubset, singleIdx);
    } else if (split.Type == ESplitType::FloatFeature) {
        const size_t* learnPermutation = GetDataPtr(fold.LearnPermutation);
        SetSingleIndex(fold, indexer, af.FloatHistograms[split.FeatureIdx], learnPermutation, singleIdx);
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        const size_t* learnPermutation = GetDataPtr(fold.LearnPermutation);
        SetSingleIndex(fold, indexer, af.CatFeaturesRemapped[split.FeatureIdx], learnPermutation, singleIdx);
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

template<typename TFullIndexType, typename TIsCaching>
static TVector<double> CalcScoreImpl(const TIsCaching& isCaching,
                                     const TVector<TFullIndexType>& singleIdx,
                                     const TCalcScoreFold& fold,
                                     const TSplitCandidate& split,
                                     const NCatboostOptions::TCatBoostOptions& fitParams,
                                     const TStatsIndexer& indexer,
                                     int depth,
                                     int splitStatsCount,
                                     TBucketStats* splitStats) {
    Y_ASSERT(!isCaching || depth > 0);
    const int approxDimension = fold.GetApproxDimension();
    const int leafCount = 1 << depth;
    const float l2Regularizer = static_cast<const float>(fitParams.ObliviousTreeOptions->L2Reg);
    TVector<TScoreBin> scoreBin(indexer.BucketCount);
    for (int bodyTailIdx = 0; bodyTailIdx < fold.GetBodyTailCount(); ++bodyTailIdx) {
        const auto& bt = fold.BodyTailArr[bodyTailIdx];
        for (int dim = 0; dim < approxDimension; ++dim) {
            TBucketStats* stats = splitStats + (bodyTailIdx * approxDimension + dim) * splitStatsCount;
            if (isCaching) {
                Fill(stats + indexer.CalcSize(depth - 1), stats + indexer.CalcSize(depth), TBucketStats{0, 0, 0, 0});
            } else {
                Fill(stats, stats + indexer.CalcSize(depth), TBucketStats{0, 0, 0, 0});
            }

            const bool isPlainMode = IsPlainMode(fitParams.BoostingOptions->BoostingType);
            if (isPlainMode) {
                UpdateWeighted(singleIdx, GetDataPtr(bt.SampleWeightedDerivatives[dim]), GetDataPtr(fold.SampleWeights), 0, bt.TailFinish, stats);
            } else {
                UpdateDeltaCount(singleIdx, GetDataPtr(bt.WeightedDerivatives[dim]), GetDataPtr(fold.LearnWeights), bt.BodyFinish, stats);
                UpdateWeighted(singleIdx, GetDataPtr(bt.SampleWeightedDerivatives[dim]), GetDataPtr(fold.SampleWeights), bt.BodyFinish, bt.TailFinish, stats);
            }

            if (isCaching) {
                FixUpStats(depth, indexer, fold.SmallestSplitSideValue, stats);
            }
            if (isPlainMode) {
                UpdateScoreBin(stats, leafCount, indexer, split.Type, l2Regularizer, /*isPlainMode=*/std::true_type(), &scoreBin);
            } else {
                UpdateScoreBin(stats, leafCount, indexer, split.Type, l2Regularizer, /*isPlainMode=*/std::false_type(), &scoreBin);
            }
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
                          const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
                          const TCalcScoreFold& fold,
                          const TCalcScoreFold& prevLevelData,
                          const NCatboostOptions::TCatBoostOptions& fitParams,
                          const TSplitCandidate& split,
                          int depth,
                          TBucketStatsCache* statsFromPrevTree) {
    const int splitCount = GetSplitCount(splitsCount, af.OneHotValues, split);
    const TStatsIndexer indexer(splitCount + 1);
    const int bucketIndexBits = GetValueBitCount(GetSplitCount(splitsCount, af.OneHotValues, split) + 1) + depth + 1;

    decltype(auto) SelectCalcScoreImpl = [&] (auto isCaching, int bucketIndexBits, const TCalcScoreFold& fold, int splitStatsCount, auto* splitStats) {
        if (bucketIndexBits <= 8) {
            TVector<ui8> singleIdx;
            BuildSingleIndex(fold, af, allCtrs, split, indexer, &singleIdx);
            return CalcScoreImpl(isCaching, singleIdx, fold, split, fitParams, indexer, depth, splitStatsCount, GetDataPtr(*splitStats));
        } else if (bucketIndexBits <= 16) {
            TVector<ui16> singleIdx;
            BuildSingleIndex(fold, af, allCtrs, split, indexer, &singleIdx);
            return CalcScoreImpl(isCaching, singleIdx, fold, split, fitParams, indexer, depth, splitStatsCount, GetDataPtr(*splitStats));
        } else if (bucketIndexBits <= 32) {
            TVector<ui32> singleIdx;
            BuildSingleIndex(fold, af, allCtrs, split, indexer, &singleIdx);
            return CalcScoreImpl(isCaching, singleIdx, fold, split, fitParams, indexer, depth, splitStatsCount, GetDataPtr(*splitStats));
        }
        CB_ENSURE(false, "too deep or too much splitsCount for score calculation");
    };

    const auto& treeOptions = fitParams.ObliviousTreeOptions.Get();
    if (!IsSamplingPerTree(treeOptions)) {
        TVector<TBucketStats> scratchSplitStats;
        scratchSplitStats.yresize(indexer.CalcSize(depth));
        return SelectCalcScoreImpl(/*isCaching*/ std::false_type(), bucketIndexBits, fold, /*splitStatsCount*/ 0, &scratchSplitStats);
    } else {
        const int splitStatsCount = indexer.CalcSize(treeOptions.MaxDepth);
        const int statsCount = fold.GetBodyTailCount() * fold.GetApproxDimension() * splitStatsCount;
        bool areStatsDirty;
        TVector<TBucketStats, TPoolAllocator>& splitStats = statsFromPrevTree->GetStats(split, statsCount, &areStatsDirty); // thread-safe access
        if (depth == 0 || areStatsDirty) {
            return SelectCalcScoreImpl(/*isCaching*/ std::false_type(), bucketIndexBits, fold, splitStatsCount, &splitStats);
        } else {
            return SelectCalcScoreImpl(/*isCaching*/ std::true_type(), bucketIndexBits, prevLevelData, splitStatsCount, &splitStats);
        }
    }
    CB_ENSURE(false, "too deep or too much splitsCount for score calculation");
}
