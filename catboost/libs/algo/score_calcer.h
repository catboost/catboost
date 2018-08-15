#pragma once

#include "calc_score_cache.h"
#include "fold.h"
#include "online_ctr.h"
#include "split.h"

#include <catboost/libs/data/quantized_features.h>
#include <catboost/libs/options/catboost_options.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>

#include <tuple>


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
// This function does all the work - it calculates sums in buckets, gets real sums for splits and (optionally - if scoreBins is non-null) builds TScoreBin-s from that.
void CalcStatsAndScores(
    const TAllFeatures& af,
    const TVector<int>& splitsCount,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,
    const TFold* initialFold,  // used only in score calculation, nullptr can be passed for stats (used in distibuted mode now)
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TSplitCandidate& split,
    int depth,
    NPar::TLocalExecutor* localExecutor,
    TBucketStatsCache* statsFromPrevTree,
    TStats3D* stats3d, // can be nullptr (and if PairwiseScoring must be), if so - don't return this data
    TVector<TScoreBin>* scoreBins // can be nullptr, if so - don't calc and return this data (used in dictributed mode now)
);

TVector<TScoreBin> GetScoreBins(
    const TStats3D& stats,
    ESplitType splitType,
    int depth,
    double sumAllWeights,
    int allDocCount,
    const NCatboostOptions::TCatBoostOptions& fitParams
);

