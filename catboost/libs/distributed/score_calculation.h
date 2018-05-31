#pragma once

#include "data_types.h"

NCatboostDistributed::TStats3D CalcStats3D(
    const TAllFeatures& af,
    const TVector<int>& splitsCount,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TSplitCandidate& split,
    int depth,
    TBucketStatsCache* statsFromPrevTree);

TVector<TScoreBin> GetScoreBins(const NCatboostDistributed::TStats3D& stats,
                                ESplitType splitType,
                                int depth,
                                const NCatboostOptions::TCatBoostOptions& fitParams);
