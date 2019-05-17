#pragma once

#include "approx_calcer.h"
#include "custom_objective_descriptor.h"
#include "error_functions.h"
#include "fold.h"
#include "rand_score.h"
#include "split.h"
#include "yetirank_helpers.h"

#include <catboost/libs/data_new/exclusive_feature_bundling.h>
#include <catboost/libs/data_new/packed_binary_features.h>
#include <catboost/libs/options/enums.h>

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

    TCandidateList CandidateList;
    TVector<TVector<ui32>> SelectedFeaturesInBundles; // [bundleIdx][inBundleIdx]
    TVector<NCB::TBinaryFeaturesPack> PerBinaryPackMasks;
};


void Bootstrap(
    const NCatboostOptions::TCatBoostOptions& params,
    const TVector<TIndexType>& indices,
    TFold* fold,
    TCalcScoreFold* sampledDocs,
    NPar::TLocalExecutor* localExecutor,
    TRestorableFastRng64* rand
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

template <bool StoreExpApprox>
inline void UpdateBodyTailApprox(const TVector<TVector<TVector<double>>>& approxDelta,
    double learningRate,
    NPar::TLocalExecutor* localExecutor,
    TFold* fold
) {
    const auto applyLearningRate = [=](TConstArrayRef<double> delta, TArrayRef<double> approx, size_t idx) {
        approx[idx] = UpdateApprox<StoreExpApprox>(
            approx[idx],
            ApplyLearningRate<StoreExpApprox>(delta[idx], learningRate)
        );
    };
    for (int bodyTailId = 0; bodyTailId < fold->BodyTailArr.ysize(); ++bodyTailId) {
        TFold::TBodyTail& bt = fold->BodyTailArr[bodyTailId];
        UpdateApprox(applyLearningRate, approxDelta[bodyTailId], &bt.Approx, localExecutor);
    }
}

void SetBestScore(
    ui64 randSeed,
    const TVector<TVector<double>>& allScores,
    double scoreStDev,
    const TCandidatesContext& candidatesContext, // candidates from it is not used, subcan
    TVector<TCandidateInfo>* subcandidates
);
