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

TVector<double> CalcScore(
    const TAllFeatures& af,
    const TVector<int>& splitsCount,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TSplitCandidate& split,
    int depth,
    TBucketStatsCache* statsFromPrevTree);
