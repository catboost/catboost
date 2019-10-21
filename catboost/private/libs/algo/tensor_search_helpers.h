#pragma once

#include "approx_calcer.h"
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>

#include <catboost/private/libs/algo_helpers/error_functions.h>
#include "rand_score.h"
#include "split.h"
#include "yetirank_helpers.h"

#include <catboost/libs/data/exclusive_feature_bundling.h>
#include <catboost/libs/data/packed_binary_features.h>
#include <catboost/private/libs/options/enums.h>

#include <library/binsaver/bin_saver.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>


class TCalcScoreFold;
struct TRestorableFastRng64;

namespace NCatboostOptions {
    class TCatBoostOptions;
}

namespace NCB {
    class TQuantizedForCPUObjectsDataProvider;
}

namespace NPar {
    class TLocalExecutor;
}


struct TCandidateInfo {
    TSplitEnsemble SplitEnsemble;
    TRandomScore BestScore;
    int BestBinId = -1;
    bool ShouldDropAfterScoreCalc = false;

public:
    SAVELOAD(SplitEnsemble, BestScore, BestBinId, ShouldDropAfterScoreCalc);

    TSplit GetBestSplit(
        const NCB::TQuantizedForCPUObjectsDataProvider& objectsData,
        ui32 oneHotMaxSize
    ) const;

    TSplit GetSplit(
        int binId,
        const NCB::TQuantizedForCPUObjectsDataProvider& objectsData,
        ui32 oneHotMaxSize
    ) const;
};

struct TCandidatesInfoList {
    TCandidatesInfoList() = default;
    explicit TCandidatesInfoList(const TCandidateInfo& oneCandidate) {
        Candidates.emplace_back(oneCandidate);
    }

    SAVELOAD(Candidates, ShouldDropCtrAfterCalc);

public:
    // All candidates here are either float or one-hot, or have the same
    // projection.
    // TODO(annaveronika): put projection out, because currently it's not clear.
    TVector<TCandidateInfo> Candidates;
    bool ShouldDropCtrAfterCalc = false;
};

using TCandidateList = TVector<TCandidatesInfoList>;

struct TCandidatesContext {
    ui32 OneHotMaxSize; // needed to select for which categorical features in bundles to calc stats
    TConstArrayRef<NCB::TExclusiveFeaturesBundle> BundlesMetaData;
    TConstArrayRef<NCB::TFeaturesGroup> FeaturesGroupsMetaData;

    TCandidateList CandidateList;
    TVector<TVector<ui32>> SelectedFeaturesInBundles; // [bundleIdx][inBundleIdx]
    TVector<NCB::TBinaryFeaturesPack> PerBinaryPackMasks;
    TVector<TVector<ui32>> SelectedFeaturesInGroups; // [groupIdx] -> {inGroupIdx_1, ..., inGroupIdx_k}
};


void Bootstrap(
    const NCatboostOptions::TCatBoostOptions& params,
    const TVector<TIndexType>& indices,
    const TVector<TVector<TVector<double>>>& leafValues,
    TFold* fold,
    TCalcScoreFold* sampledDocs,
    NPar::TLocalExecutor* localExecutor,
    TRestorableFastRng64* rand,
    bool shouldSortByLeaf = false,
    ui32 leavesCount = 0
);

THolder<IDerCalcer> BuildError(
    const NCatboostOptions::TCatBoostOptions& params,
    const TMaybe<TCustomObjectiveDescriptor>&
);

void CalcWeightedDerivatives(
    const IDerCalcer& error,
    int bodyTailIdx,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    TFold* takenFold,
    NPar::TLocalExecutor* localExecutor
);

void SetBestScore(
    ui64 randSeed,
    const TVector<TVector<double>>& allScores,
    double scoreStDev,
    const TCandidatesContext& candidatesContext, // candidates from it is not used, subcan
    TVector<TCandidateInfo>* subcandidates
);
