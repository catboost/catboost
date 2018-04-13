#pragma once

#include "split.h"
#include "rand_score.h"
#include "fold.h"
#include "calc_score_cache.h"
#include "error_functions.h"
#include "yetirank_helpers.h"
#include "approx_calcer.h"

#include <catboost/libs/options/enums.h>

#include <util/generic/vector.h>

#include <library/binsaver/bin_saver.h>
#include <library/dot_product/dot_product.h>
#include <library/threading/local_executor/local_executor.h>

inline double CalcScoreStDev(const TFold& ff) {
    double sum2 = 0, totalSum2Count = 0;
    for (const TFold::TBodyTail& bt : ff.BodyTailArr) {
        for (int dim = 0; dim < bt.WeightedDerivatives.ysize(); ++dim) {
            sum2 += DotProduct(bt.WeightedDerivatives[dim].data() + bt.BodyFinish, bt.WeightedDerivatives[dim].data() + bt.BodyFinish, bt.TailFinish - bt.BodyFinish);
        }
        totalSum2Count += bt.TailFinish - bt.BodyFinish;
    }
    return sqrt(sum2 / Max(totalSum2Count, DBL_EPSILON));
}

inline double CalcScoreStDevMult(int learnSampleCount, double modelLength) {
    double modelExpLength = log(learnSampleCount * 1.0);
    double modelLeft = exp(modelExpLength - modelLength);
    return modelLeft / (1 + modelLeft);
}

struct TCandidateInfo {
    TSplitCandidate SplitCandidate;
    TRandomScore BestScore;
    int BestBinBorderId = -1;
    bool ShouldDropAfterScoreCalc = false;
    SAVELOAD(SplitCandidate, BestScore, BestBinBorderId, ShouldDropAfterScoreCalc);
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

template <typename TError>
TError BuildError(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&) {
    return TError(IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()));
}
template <>
TCustomError BuildError<TCustomError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&);
template <>
TUserDefinedPerObjectError BuildError<TUserDefinedPerObjectError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&);
template <>
TUserDefinedQuerywiseError BuildError<TUserDefinedQuerywiseError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&);
template <>
TLogLinQuantileError BuildError<TLogLinQuantileError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&);
template<>
TQuantileError BuildError<TQuantileError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&);

template <typename TError>
inline void CalcWeightedDerivatives(
    const TError& error,
    int bodyTailIdx,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    TFold* takenFold,
    NPar::TLocalExecutor* localExecutor
) {
    TFold::TBodyTail& bt = takenFold->BodyTailArr[bodyTailIdx];
    const TVector<TVector<double>>& approx = bt.Approx;
    const TVector<float>& target = takenFold->LearnTarget;
    const TVector<float>& weight = takenFold->LearnWeights;
    TVector<TVector<double>>* weightedDerivatives = &bt.WeightedDerivatives;

    if (error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError) {
        TVector<TQueryInfo> recalculatedQueriesInfo;
        const bool isYetiRank = params.LossFunctionDescription->GetLossFunction() == ELossFunction::YetiRank;
        if (isYetiRank) {
            YetiRankRecalculation(*takenFold, bt, params, randomSeed, localExecutor, &recalculatedQueriesInfo, &bt.PairwiseWeights);
        }
        const TVector<TQueryInfo>& queriesInfo = isYetiRank ? recalculatedQueriesInfo : takenFold->LearnQueriesInfo;

        const int tailQueryFinish = bt.TailQueryFinish;
        TVector<TDers> ders((*weightedDerivatives)[0].ysize());
        error.CalcDersForQueries(0, tailQueryFinish, approx[0], target, weight, queriesInfo, &ders);
        for (int docId = 0; docId < ders.ysize(); ++docId) {
            (*weightedDerivatives)[0][docId] = ders[docId].Der1;
        }
    } else {
        const int tailFinish = bt.TailFinish;
        const int approxDimension = approx.ysize();
        NPar::TLocalExecutor::TExecRangeParams blockParams(0, tailFinish);
        blockParams.SetBlockSize(1000);

        Y_ASSERT(error.GetErrorType() == EErrorType::PerObjectError);
        if (approxDimension == 1) {
            localExecutor->ExecRange([&](int blockId) {
                const int blockOffset = blockId * blockParams.GetBlockSize();
                error.CalcFirstDerRange(blockOffset, Min<int>(blockParams.GetBlockSize(), tailFinish - blockOffset),
                    approx[0].data(),
                    nullptr, // no approx deltas
                    target.data(),
                    weight.data(),
                    (*weightedDerivatives)[0].data());
            }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            localExecutor->ExecRange([&](int blockId) {
                TVector<double> curApprox(approxDimension);
                TVector<double> curDelta(approxDimension);
                NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&](int z) {
                    for (int dim = 0; dim < approxDimension; ++dim) {
                        curApprox[dim] = approx[dim][z];
                    }
                    error.CalcDersMulti(curApprox, target[z], weight.empty() ? 1 : weight[z], &curDelta, nullptr);
                    for (int dim = 0; dim < approxDimension; ++dim) {
                        (*weightedDerivatives)[dim][z] = curDelta[dim];
                    }
                })(blockId);
            }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
        }
    }
}

template<bool StoreExpApprox>
inline void UpdateBodyTailApprox(const TVector<TVector<TVector<double>>>& approxDelta,
    double learningRate,
    NPar::TLocalExecutor* localExecutor,
    TFold* fold
) {
    const int approxDimension = fold->GetApproxDimension();
    for (int bodyTailId = 0; bodyTailId < fold->BodyTailArr.ysize(); ++bodyTailId) {
        TFold::TBodyTail& bt = fold->BodyTailArr[bodyTailId];
        for (int dim = 0; dim < approxDimension; ++dim) {
            const double* approxDeltaData = approxDelta[bodyTailId][dim].data();
            double* approxData = bt.Approx[dim].data();
            localExecutor->ExecRange(
                [=](int z) {
                    approxData[z] = UpdateApprox<StoreExpApprox>(
                        approxData[z],
                        ApplyLearningRate<StoreExpApprox>(approxDeltaData[z], learningRate)
                    );
                },
                NPar::TLocalExecutor::TExecRangeParams(0, bt.TailFinish).SetBlockSize(1000),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }
    }
}

void SetBestScore(ui64 randSeed, const TVector<TVector<double>>& allScores, double scoreStDev, TVector<TCandidateInfo>* subcandidates);
