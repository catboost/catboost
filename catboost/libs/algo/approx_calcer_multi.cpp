#include "approx_calcer_multi.h"

#include "approx_calcer_helpers.h"
#include "approx_updater_helpers.h"
#include "error_functions.h"
#include "gradient_walker.h"
#include "index_calcer.h"
#include "learn_context.h"
#include "online_predictor.h"

#include <library/threading/local_executor/local_executor.h>

inline static void AddDerNewtonMulti(
    const IDerCalcer& error,
    const TVector<double>& approx,
    float target,
    double weight,
    bool /*isUpdateWeight*/,
    TVector<double>* curDer,
    THessianInfo* curDer2,
    TSumMulti* curLeafDers
) {
    Y_ASSERT(curDer != nullptr && curDer2 != nullptr);
    error.CalcDersMulti(approx, target, weight, curDer, curDer2);
    curLeafDers->AddDerDer2(*curDer, *curDer2);
}

inline static void AddDerGradientMulti(
    const IDerCalcer& error,
    const TVector<double>& approx,
    float target,
    double weight,
    bool isUpdateWeight,
    TVector<double>* curDer,
    THessianInfo* /*curDer2*/,
    TSumMulti* curLeafDers
) {
    Y_ASSERT(curDer != nullptr);
    error.CalcDersMulti(approx, target, weight, curDer, nullptr);
    curLeafDers->AddDerWeight(*curDer, weight, isUpdateWeight);
}


void UpdateApproxDeltasMulti(
    bool storeExpApprox,
    const TVector<TIndexType>& indices,
    int docCount,
    NPar::TLocalExecutor* localExecutor,
    TVector<TVector<double>>* leafDeltas, //leafDeltas[dimension][leafId]
    TVector<TVector<double>>* approxDeltas
) {
    const auto indicesRef = MakeArrayRef(indices);
    if (storeExpApprox) {
        for (int dim = 0; dim < leafDeltas->ysize(); ++dim) {
            auto approxDeltaRef = MakeArrayRef((*approxDeltas)[dim]);
            auto leafDeltaRef = MakeArrayRef((*leafDeltas)[dim]);
            ExpApproxIf(/*storeExpApproxes*/true, leafDeltaRef);
            NPar::ParallelFor(
                *localExecutor,
                /*from*/0,
                /*to*/ docCount,
                /*body*/[=] (int z) {
                    approxDeltaRef[z] = UpdateApprox</*StoreExpApprox*/true>(
                        approxDeltaRef[z],
                        leafDeltaRef[indicesRef[z]]);
                }
            );
        }
    } else {
        for (int dim = 0; dim < leafDeltas->ysize(); ++dim) {
            auto approxDeltaRef = MakeArrayRef((*approxDeltas)[dim]);
            auto leafDeltaRef = MakeArrayRef((*leafDeltas)[dim]);
            ExpApproxIf(/*storeExpApproxes*/false, leafDeltaRef);
            NPar::ParallelFor(
                *localExecutor,
                /*from*/0,
                /*to*/ docCount,
                /*body*/[=] (int z) {
                    approxDeltaRef[z] = UpdateApprox</*StoreExpApprox*/false>(
                        approxDeltaRef[z],
                        leafDeltaRef[indicesRef[z]]);
                }
            );
        }
    }
}

void CalcApproxDeltaMulti(
    const TFold& fold,
    const TFold::TBodyTail& bt,
    int leafCount,
    const IDerCalcer& error,
    const TVector<TIndexType>& indices,
    TLearnContext* ctx,
    TVector<TVector<double>>* approxDelta,
    TVector<TVector<double>>* sumLeafDeltas
) {
    const auto& treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = treeLearnerOptions.LeavesEstimationIterations;
    const TVector<float>& target = fold.LearnTarget;
    const TVector<float>& weight = fold.GetLearnWeights();
    const int approxDimension = approxDelta->ysize();
    const ELeavesEstimation estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
    const float l2Regularizer = treeLearnerOptions.L2Reg;

    TVector<TSumMulti> leafDers(leafCount, TSumMulti(approxDimension, error.GetHessianType()));

    const auto leafUpdaterFunc = [&] (
        bool recalcLeafWeights,
        const TVector<TVector<double>>& approxDeltas,
        TVector<TVector<double>>* leafDeltas
    ) {
        CalcLeafDersMulti(
            indices,
            target,
            weight,
            bt.Approx,
            approxDeltas,
            error,
            bt.BodyFinish,
            recalcLeafWeights,
            estimationMethod,
            ctx->LocalExecutor,
            &leafDers
        );
        CalcLeafDeltasMulti(
            leafDers,
            estimationMethod,
            l2Regularizer,
            bt.BodySumWeight,
            bt.BodyFinish,
            leafDeltas
        );
    };

    const auto approxUpdaterFunc = [&] (
        const TVector<TVector<double>>& leafDeltas,
        TVector<TVector<double>>* approxDeltas
    ) {
        auto localLeafValues = leafDeltas;
        UpdateApproxDeltasMulti(
            error.GetIsExpApprox(),
            indices,
            bt.BodyFinish,
            ctx->LocalExecutor,
            &localLeafValues,
            approxDeltas
        );
        TVector<double> curApprox(approxDimension);
        TVector<double> curDelta(approxDimension);
        TVector<double> curDer(approxDimension);
        THessianInfo curDer2(approxDimension, error.GetHessianType());
        for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
            for (int dim = 0; dim < approxDimension; ++dim) {
                curApprox[dim] = UpdateApprox(error.GetIsExpApprox(), bt.Approx[dim][z], (*approxDeltas)[dim][z]);
            }
            TSumMulti& curLeafDers = leafDers[indices[z]];
            if (estimationMethod == ELeavesEstimation::Newton) {
                AddDerNewtonMulti(
                    error,
                    curApprox,
                    target[z],
                    weight.empty() ? 1 : weight[z],
                    /*isUpdateWeight*/true,
                    &curDer,
                    &curDer2,
                    &curLeafDers);
                CalcDeltaNewtonMulti(curLeafDers, l2Regularizer, bt.BodySumWeight, bt.BodyFinish, &curDelta);
            } else {
                Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
                AddDerGradientMulti(
                    error,
                    curApprox,
                    target[z],
                    weight.empty() ? 1 : weight[z],
                    /*isUpdateWeight*/true,
                    &curDer,
                    &curDer2,
                    &curLeafDers);
                CalcDeltaGradientMulti(curLeafDers, l2Regularizer, bt.BodySumWeight, bt.BodyFinish, &curDelta);
            }
            ExpApproxIf(error.GetIsExpApprox(), curDelta);
            for (int dim = 0; dim < approxDimension; ++dim) {
                (*approxDeltas)[dim][z] = UpdateApprox(error.GetIsExpApprox(), (*approxDeltas)[dim][z], curDelta[dim]);
            }
        }
    };

    bool haveBacktrackingObjective;
    double minimizationSign;
    TVector<THolder<IMetric>> lossFunction;
    CreateBacktrackingObjective(*ctx, &haveBacktrackingObjective, &minimizationSign, &lossFunction);

    const auto lossCalcerFunc = [&] (const TVector<TVector<double>>& approxDeltas) {
        TConstArrayRef<TQueryInfo> bodyTailQueryInfo(fold.LearnQueriesInfo.begin(), bt.BodyQueryFinish);
        TConstArrayRef<float> bodyTailTarget(fold.LearnTarget.begin(), bt.BodyFinish);
        const auto& additiveStats = EvalErrors(
            bt.Approx,
            approxDeltas,
            error.GetIsExpApprox(),
            bodyTailTarget,
            fold.GetLearnWeights(),
            bodyTailQueryInfo,
            *lossFunction[0],
            ctx->LocalExecutor
        );
        return minimizationSign * lossFunction[0]->GetFinalError(additiveStats);
    };

    const auto approxCopyFunc = [ctx] (const TVector<TVector<double>>& src, TVector<TVector<double>>* dst) {
        CopyApprox(src, dst, ctx->LocalExecutor);
    };

    GradientWalker(
        /*isTrivialWalker*/!haveBacktrackingObjective,
        gradientIterations,
        leafCount,
        ctx->LearnProgress->ApproxDimension,
        leafUpdaterFunc,
        approxUpdaterFunc,
        lossCalcerFunc,
        approxCopyFunc,
        approxDelta,
        sumLeafDeltas
    );
}

void CalcLeafDersMulti(
    const TVector<TIndexType>& indices,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDeltas,
    const IDerCalcer& error,
    int sampleCount,
    bool isUpdateWeight,
    ELeavesEstimation estimationMethod,
    NPar::TLocalExecutor* /*localExecutor*/,
    TVector<TSumMulti>* leafDers
) {
    const int approxDimension = approx.ysize();
    Y_ASSERT(approxDimension > 0);
    TVector<double> curApprox(approxDimension);
    TVector<double> curDer(approxDimension);
    THessianInfo curDer2(approxDimension, error.GetHessianType());
    for (auto& curLeafDers : *leafDers) {
        curLeafDers.SetZeroDers();
    }
    for (int z = 0; z < sampleCount; ++z) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            curApprox[dim] = approxDeltas.empty() ?
                approx[dim][z] :
                UpdateApprox(error.GetIsExpApprox(), approx[dim][z], approxDeltas[dim][z]);
        }
        TSumMulti& curLeafDers = (*leafDers)[indices[z]];
        if (estimationMethod == ELeavesEstimation::Newton) {
            AddDerNewtonMulti(
                error,
                curApprox,
                target[z],
                weight.empty() ? 1 : weight[z],
                isUpdateWeight,
                &curDer,
                &curDer2,
                &curLeafDers);
        } else {
            AddDerGradientMulti(
                error,
                curApprox,
                target[z],
                weight.empty() ? 1 : weight[z],
                isUpdateWeight,
                &curDer,
                &curDer2,
                &curLeafDers);
        }
    }
}

void CalcLeafDeltasMulti(
    const TVector<TSumMulti>& leafDers,
    ELeavesEstimation estimationMethod,
    float l2Regularizer,
    double sumAllWeights,
    int docCount,
    TVector<TVector<double>>* curLeafValues
) {
    const int leafCount = leafDers.ysize();
    TVector<double> curDelta;
    if (estimationMethod == ELeavesEstimation::Newton) {
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            CalcDeltaNewtonMulti(leafDers[leaf], l2Regularizer, sumAllWeights, docCount, &curDelta);
            for (int dim = 0; dim < curDelta.ysize(); ++dim) {
                (*curLeafValues)[dim][leaf] = curDelta[dim];
            }
        }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            CalcDeltaGradientMulti(leafDers[leaf], l2Regularizer, sumAllWeights, docCount, &curDelta);
            for (int dim = 0; dim < curDelta.ysize(); ++dim) {
                (*curLeafValues)[dim][leaf] = curDelta[dim];
            }
        }
    }
}

void CalcLeafValuesMulti(
    int leafCount,
    const IDerCalcer& error,
    const TFold& fold,
    const TVector<TIndexType>& indices,
    TLearnContext* ctx,
    TVector<TVector<double>>* sumLeafDeltas
) {
    const TFold::TBodyTail& bt = fold.BodyTailArr[0];
    const int approxDimension = fold.GetApproxDimension();
    const auto& treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = treeLearnerOptions.LeavesEstimationIterations;
    const ELeavesEstimation estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
    const float l2Regularizer = treeLearnerOptions.L2Reg;
    const TVector<float>& target = fold.LearnTarget;
    const TVector<float>& weight = fold.GetLearnWeights();
    const double sumWeight = fold.GetSumWeight();
    const int learnSampleCount = fold.GetLearnSampleCount();

    TVector<TVector<double>> approx;
    CopyApprox(bt.Approx, &approx, ctx->LocalExecutor);
    TVector<TSumMulti> leafDers(leafCount, TSumMulti(approxDimension, error.GetHessianType()));
    sumLeafDeltas->assign(approxDimension, TVector<double>(leafCount));

    const auto leafUpdaterFunc = [&] (
        bool recalcLeafWeights,
        const TVector<TVector<double>>& approxes,
        TVector<TVector<double>>* leafDeltas
    ) {
        CalcLeafDersMulti(
            indices,
            target,
            weight,
            approxes,
            /*approxDeltas*/ {},
            error,
            learnSampleCount,
            recalcLeafWeights,
            estimationMethod,
            ctx->LocalExecutor,
            &leafDers
        );

        CalcLeafDeltasMulti(
            leafDers,
            estimationMethod,
            l2Regularizer,
            sumWeight,
            learnSampleCount,
            leafDeltas
        );
    };

    const auto approxUpdaterFunc = [&] (
        const TVector<TVector<double>>& leafDeltas,
        TVector<TVector<double>>* approxes
    ) {
        auto localLeafValues = leafDeltas;
        UpdateApproxDeltasMulti(
            error.GetIsExpApprox(),
            indices,
            learnSampleCount,
            ctx->LocalExecutor,
            &localLeafValues,
            approxes
        );
    };

    auto& localExecutor = *ctx->LocalExecutor;
    bool haveBacktrackingObjective;
    double minimizationSign;
    TVector<THolder<IMetric>> lossFunction;
    CreateBacktrackingObjective(*ctx, &haveBacktrackingObjective, &minimizationSign, &lossFunction);

    const auto lossCalcerFunc = [&] (const TVector<TVector<double>>& approx) {
        const auto& additiveStats = EvalErrors(
            approx,
            /*approxDelta*/{},
            error.GetIsExpApprox(),
            fold.LearnTarget,
            fold.GetLearnWeights(),
            fold.LearnQueriesInfo,
            *lossFunction[0],
            &localExecutor
        );
        return minimizationSign * lossFunction[0]->GetFinalError(additiveStats);
    };

    const auto approxCopyFunc = [ctx] (const TVector<TVector<double>>& src, TVector<TVector<double>>* dst) {
        CopyApprox(src, dst, ctx->LocalExecutor);
    };

    GradientWalker(
        /*isTrivialWalker*/!haveBacktrackingObjective,
        gradientIterations,
        leafCount,
        ctx->LearnProgress->ApproxDimension,
        leafUpdaterFunc,
        approxUpdaterFunc,
        lossCalcerFunc,
        approxCopyFunc,
        &approx,
        sumLeafDeltas
    );
}
