#include "approx_delta_calcer_multi.h"

#include "approx_calcer_helpers.h"
#include "approx_updater_helpers.h"
#include "index_calcer.h"
#include "learn_context.h"

#include <catboost/private/libs/algo/approx_calcer/gradient_walker.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_multi_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/private/libs/algo_helpers/online_predictor.h>

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
    CB_ENSURE(!error.GetIsExpApprox(), "Multi-class does not support exponentiated approxes");

    const auto& treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = treeLearnerOptions.LeavesEstimationIterations;
    const TVector<float>& target = fold.LearnTarget[0];
    const TVector<float>& weight = fold.GetLearnWeights();
    const int approxDimension = approxDelta->ysize();
    const ELeavesEstimation estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
    const float l2Regularizer = treeLearnerOptions.L2Reg;

    TVector<TSumMulti> leafDers(leafCount, MakeZeroDers(approxDimension, estimationMethod, error.GetHessianType()));

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

    const bool useHessian = estimationMethod == ELeavesEstimation::Newton;
    const auto approxUpdaterFunc = [&] (
        TConstArrayRef<TVector<double>> leafDeltas,
        TVector<TVector<double>>* approxDeltas
    ) {
        UpdateApproxDeltasMulti(
            indices,
            bt.BodyFinish,
            leafDeltas,
            approxDeltas,
            ctx->LocalExecutor
        );
        TVector<double> curApprox(approxDimension);
        TVector<double> curDelta(approxDimension);
        TVector<double> curDer(approxDimension);
        THessianInfo curDer2(approxDimension * useHessian, error.GetHessianType());
        for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
            for (int dim = 0; dim < approxDimension; ++dim) {
                curApprox[dim] = bt.Approx[dim][z] + (*approxDeltas)[dim][z];
            }
            TSumMulti& curLeafDers = leafDers[indices[z]];
            if (useHessian) {
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
            for (int dim = 0; dim < approxDimension; ++dim) {
                (*approxDeltas)[dim][z] = (*approxDeltas)[dim][z] + curDelta[dim];
            }
        }
    };

    bool haveBacktrackingObjective;
    double minimizationSign;
    TVector<THolder<IMetric>> lossFunction;
    CreateBacktrackingObjective(*ctx, &haveBacktrackingObjective, &minimizationSign, &lossFunction);

    const auto lossCalcerFunc = [&] (const TVector<TVector<double>>& approxDeltas) {
        TConstArrayRef<TQueryInfo> bodyTailQueryInfo(fold.LearnQueriesInfo.begin(), bt.BodyQueryFinish);
        TConstArrayRef<float> bodyTailTarget(fold.LearnTarget[0].begin(), bt.BodyFinish);
        const auto& additiveStats = EvalErrors(
            bt.Approx,
            approxDeltas,
            /*isExpApprox*/false,
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

    GradientWalker</*IsLeafwise*/ false>(
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
