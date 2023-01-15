#pragma once

#include "online_ctr.h"
#include "score_calcers.h"

#include <catboost/private/libs/data_types/pair.h>

#include <util/generic/vector.h>

#include <tuple>


class TBucketStatsCache;
class TCalcScoreFold;
class TFold;
struct TPairwiseStats;
struct TCandidateInfo;
struct TStats3D;

namespace NCatboostOptions {
    class TCatBoostOptions;
}

namespace NCB {
    class TQuantizedObjectsDataProvider;
}

namespace NPar {
    class ILocalExecutor;
}


// Function that calculates score statistics for each split of a split candidate
// (candidate is a feature == all splits of this feature).
// This function does all the work - it calculates sums in buckets, gets real sums for splits and
// (optionally - if scoreCalcer is non-null) calculates scores.
void CalcStatsAndScores(
    const NCB::TQuantizedObjectsDataProvider& objectsDataProvider,
    const std::tuple<const TOnlineCtrBase&, const TOnlineCtrBase&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,

    // used only in score calculation, nullptr can be passed for stats (used in distibuted mode now)
    const TFold* initialFold,
    const TFlatPairsInfo& pairs,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TCandidateInfo& candidateInfo,
    int depth,
    bool useTreeLevelCaching,
    const TVector<int>& currTreeMonotonicConstraints,
    const TMap<ui32, int>& monotonicConstraints,
    NPar::ILocalExecutor* localExecutor,
    TBucketStatsCache* statsFromPrevTree,
    TStats3D* stats3d, // can be nullptr (and if PairwiseScoring must be), if so - don't return this data

    // can be nullptr (and if not PairwiseScoring must be), if so - don't return this data
    TPairwiseStats* pairwiseStats,

    // can be nullptr, if so - don't calc and return this data (used in dictributed mode now)
    IScoreCalcer* scoreCalcer
);

TVector<double> GetScores(
    const TStats3D& stats,
    int depth,
    double sumAllWeights,
    int allDocCount,
    const NCatboostOptions::TCatBoostOptions& fitParams
);
