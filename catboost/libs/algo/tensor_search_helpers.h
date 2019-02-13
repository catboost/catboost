#pragma once

#include "split.h"
#include "rand_score.h"
#include "fold.h"
#include "calc_score_cache.h"
#include "error_functions.h"
#include "yetirank_helpers.h"
#include "approx_calcer.h"
#include "custom_objective_descriptor.h"

#include <catboost/libs/data_new/packed_binary_features.h>
#include <catboost/libs/options/enums.h>

#include <library/binsaver/bin_saver.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>


namespace NCB {
    class TQuantizedForCPUObjectsDataProvider;
}


struct TCandidateInfo {
    TSplitEnsemble SplitEnsemble;
    TRandomScore BestScore;
    int BestBinId = -1;
    bool ShouldDropAfterScoreCalc = false;
    SAVELOAD(SplitEnsemble, BestScore, BestBinId, ShouldDropAfterScoreCalc);

    TSplit GetBestSplit(const NCB::TQuantizedForCPUObjectsDataProvider& objectsData) const;
};

struct TCandidatesInfoList {
    TCandidatesInfoList() = default;
    explicit TCandidatesInfoList(const TCandidateInfo& oneCandidate) {
        Candidates.emplace_back(oneCandidate);
    }
    // All candidates here are either float or one-hot, or have the same
    // projection.
    // TODO(annaveronika): put projection out, because currently it's not clear.
    TVector<TCandidateInfo> Candidates;
    bool ShouldDropCtrAfterCalc = false;

    SAVELOAD(Candidates, ShouldDropCtrAfterCalc);
};

using TCandidateList = TVector<TCandidatesInfoList>;

void Bootstrap(const NCatboostOptions::TCatBoostOptions& params,
               const TVector<TIndexType>& indices,
               TFold* fold,
               TCalcScoreFold* sampledDocs,
               NPar::TLocalExecutor* localExecutor,
               TRestorableFastRng64* rand);

THolder<IDerCalcer> BuildError(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&);

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
    TConstArrayRef<NCB::TBinaryFeaturesPack> perPackMasks,
    TVector<TCandidateInfo>* subcandidates
);
