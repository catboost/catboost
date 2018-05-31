#pragma once

#include "online_predictor.h"
#include "fold.h"
#include "online_ctr.h"
#include "rand_score.h"
#include "split.h"
#include "error_functions.h"
#include "calc_score_cache.h"

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>

// TODO(annaveronika): Currently this file has a bunch of structures and helper functions that are used for score calculation
// in local and distributed modes. This file needs to be refactored.

struct TFeatureScore {
    TSplit Split;
    int ScoreGroup;
    TRandomScore Score;

    size_t GetHash() const {
        size_t hashValue = Split.GetHash();
        hashValue = MultiHash(hashValue,
                              Score.StDev,
                              Score.Val,
                              ScoreGroup);
        return hashValue;
    }
};

// The class that stores final stats for a split and provides interface to calculate the deterministic score.
struct TScoreBin {
    double DP = 0, D2 = 1e-100;

    inline double GetScore() const {
        return DP / sqrt(D2);
    }
};

// Helper function that calculates deterministic scores given bins with statistics for each split.
inline TVector<double> GetScores(const TVector<TScoreBin>& scoreBin) {
    const int splitCount = scoreBin.ysize() - 1;
    TVector<double> scores(splitCount);
    for (int splitIdx = 0; splitIdx < splitCount; ++splitIdx) {
        scores[splitIdx] = scoreBin[splitIdx].GetScore();
    }
    return scores;
}

// Function that calculates score statistics for each split of a split candidate (candidate is a feature == all splits of this feature).
// This function does all the work - it calculates sums in buckets, gets real sums for splits and builds TScoreBin-s from that.
TVector<TScoreBin> CalcScore(
    const TAllFeatures& af,
    const TVector<int>& splitsCount,
    const std::tuple<const TOnlineCTRHash&,
    const TOnlineCTRHash&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TSplitCandidate& split,
    int depth,
    TBucketStatsCache* statsFromPrevTree);

// Statistics (sums for score calculation) are stored in an array. This class helps navigating in this array.
struct TStatsIndexer {
    const int BucketCount;
    explicit TStatsIndexer(int bucketCount)
    : BucketCount(bucketCount) {
    }
    int CalcSize(int depth) const {
        return (1U << depth) * BucketCount;
    }
    int GetIndex(int leafIndex, int bucketIndex) const {
        return BucketCount * leafIndex + bucketIndex;
    }
};

// A helper function that returns calculated ctr values for this projection (== feature or feature combination) from cache.
inline const TOnlineCTR& GetCtr(const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs, const TProjection& proj) {
    static const constexpr size_t OnlineSingleCtrsIndex = 0;
    static const constexpr size_t OnlineCTRIndex = 1;
    return proj.HasSingleFeature() ? std::get<OnlineSingleCtrsIndex>(allCtrs).at(proj) : std::get<OnlineCTRIndex>(allCtrs).at(proj);
}

// Helper function for calculating index of leaf for each document given a new split.
// Calculates indices when a permutation is given.
template<typename TBucketIndexType, typename TFullIndexType>
inline void SetSingleIndex(const TCalcScoreFold& fold,
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

// Calculate index of leaf for each document given a new split.
template<typename TFullIndexType>
inline void BuildSingleIndex(const TCalcScoreFold& fold,
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

// Update bootstraped sums on [docBegin, docEnd) in a bucket
template<typename TFullIndexType>
inline void UpdateWeighted(const TVector<TFullIndexType>& singleIdx, const double* weightedDer, const float* sampleWeights, int docBegin, int docEnd, TBucketStats* stats) {
    for (int doc = docBegin; doc < docEnd; ++doc) {
        TBucketStats& leafStats = stats[singleIdx[doc]];
        leafStats.SumWeightedDelta += weightedDer[doc];
        leafStats.SumWeight += sampleWeights[doc];
    }
}

// Update not bootstraped sums on [docBegin, docEnd) in a bucket
template<typename TFullIndexType>
inline void UpdateDeltaCount(const TVector<TFullIndexType>& singleIdx, const double* derivatives, const float* learnWeights, int docCount, TBucketStats* stats) {
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

// Calculate score numerator summand
inline double CountDp(double avrg, const TBucketStats& leafStats) {
    return avrg * leafStats.SumWeightedDelta;
}

// Calculate score denominator summand
inline double CountD2(double avrg, const TBucketStats& leafStats) {
    return avrg * avrg * leafStats.SumWeight;
}

// This function calculates resulting sums for each split given statistics that are calculated for each bucket of the histogram.
template<typename TIsPlainMode>
inline void UpdateScoreBin(const TBucketStats* stats, int leafCount, const TStatsIndexer& indexer, ESplitType splitType, float l2Regularizer, TIsPlainMode isPlainMode, TVector<TScoreBin>* scoreBin) {
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

// Helper function that returns how many splits has a split candidate.
int GetSplitCount(const TVector<int>& splitsCount,
                  const TVector<TVector<int>>& oneHotValues,
                  const TSplitCandidate& split);

inline void FixUpStats(int depth, const TStatsIndexer& indexer, bool selectedSplitValue, TBucketStats* stats) {
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
inline void CalcStatsKernel(const TIsCaching& isCaching,
                            const TVector<TFullIndexType>& singleIdx,
                            const TCalcScoreFold& fold,
                            bool isPlainMode,
                            const TStatsIndexer& indexer,
                            int depth,
                            const TCalcScoreFold::TBodyTail& bt,
                            int dim,
                            TBucketStats* stats) {
    Y_ASSERT(!isCaching || depth > 0);
    if (isCaching) {
        Fill(stats + indexer.CalcSize(depth - 1), stats + indexer.CalcSize(depth), TBucketStats{0, 0, 0, 0});
    } else {
        Fill(stats, stats + indexer.CalcSize(depth), TBucketStats{0, 0, 0, 0});
    }

    const bool hasPairwiseWeights = !bt.PairwiseWeights.empty();
    const float* weightsData = hasPairwiseWeights ? GetDataPtr(bt.PairwiseWeights) : GetDataPtr(fold.LearnWeights);
    const float* sampleWeightsData = hasPairwiseWeights ? GetDataPtr(bt.SamplePairwiseWeights) : GetDataPtr(fold.SampleWeights);
    if (isPlainMode) {
        UpdateWeighted(singleIdx, GetDataPtr(bt.SampleWeightedDerivatives[dim]), sampleWeightsData, 0, bt.TailFinish, stats);
    } else {
        UpdateDeltaCount(singleIdx, GetDataPtr(bt.WeightedDerivatives[dim]), weightsData, bt.BodyFinish, stats);
        UpdateWeighted(singleIdx, GetDataPtr(bt.SampleWeightedDerivatives[dim]), sampleWeightsData, bt.BodyFinish, bt.TailFinish, stats);
    }
    if (isCaching) {
        FixUpStats(depth, indexer, fold.SmallestSplitSideValue, stats);
    }
}
