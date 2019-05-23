#pragma once

#include "online_ctr.h"
#include "score_bin.h"

#include <catboost/libs/data_types/pair.h>

#include <util/generic/vector.h>

#include <tuple>


class TBucketStatsCache;
class TCalcScoreFold;
class TFold;
struct TPairwiseStats;
struct TSplitEnsemble;
struct TStats3D;

namespace NCatboostOptions {
    class TCatBoostOptions;
}

namespace NCB {
    class TQuantizedForCPUObjectsDataProvider;
}

namespace NPar {
    class TLocalExecutor;
}


// Function that calculates score statistics for each split of a split candidate
// (candidate is a feature == all splits of this feature).
// This function does all the work - it calculates sums in buckets, gets real sums for splits and
// (optionally - if scoreBins is non-null) builds TScoreBin-s from that.
void CalcStatsAndScores(
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,

    // used only in score calculation, nullptr can be passed for stats (used in distibuted mode now)
    const TFold* initialFold,
    const TFlatPairsInfo& pairs,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TSplitEnsemble& splitEnsemble,
    int depth,
    bool useTreeLevelCaching,
    NPar::TLocalExecutor* localExecutor,
    TBucketStatsCache* statsFromPrevTree,
    TStats3D* stats3d, // can be nullptr (and if PairwiseScoring must be), if so - don't return this data

    // can be nullptr (and if not PairwiseScoring must be), if so - don't return this data
    TPairwiseStats* pairwiseStats,

    // can be nullptr, if so - don't calc and return this data (used in dictributed mode now)
    TVector<TScoreBin>* scoreBins
);

TVector<TScoreBin> GetScoreBins(
    const TStats3D& stats,
    int depth,
    double sumAllWeights,
    int allDocCount,
    const NCatboostOptions::TCatBoostOptions& fitParams
);
