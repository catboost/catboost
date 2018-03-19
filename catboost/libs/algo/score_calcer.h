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

struct TScoreBin {
    double DP = 0, D2 = 1e-100;

    inline double GetScore() const {
        return DP / sqrt(D2);
    }
};

inline TVector<double> GetScores(const TVector<TScoreBin>& scoreBin) {
    const int splitCount = scoreBin.ysize() - 1;
    TVector<double> scores(splitCount);
    for (int splitIdx = 0; splitIdx < splitCount; ++splitIdx) {
        scores[splitIdx] = scoreBin[splitIdx].GetScore();
    }
    return scores;
}

TVector<TScoreBin> CalcScore(
    const TAllFeatures& af,
    const TVector<int>& splitsCount,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TSplitCandidate& split,
    int depth,
    TBucketStatsCache* statsFromPrevTree);

struct TStats3D {
    TVector<TBucketStats> Stats; // [bodyTail & approxDim][leaf][bucket]
    int BucketCount;
    int MaxLeafCount;
    TStats3D() = default;
    TStats3D(const TVector<TBucketStats>& stats, int bucketCount, int maxLeafCount)
    : Stats(stats)
    , BucketCount(bucketCount)
    , MaxLeafCount(maxLeafCount)
    {
    }
    SAVELOAD(Stats, BucketCount, MaxLeafCount);
};

TStats3D CalcStats3D(
    const TAllFeatures& af,
    const TVector<int>& splitsCount,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TSplitCandidate& split,
    int depth,
    TBucketStatsCache* statsFromPrevTree);

TVector<TScoreBin> GetScoreBins(const TStats3D& stats, ESplitType splitType, int depth, const NCatboostOptions::TCatBoostOptions& fitParams);

