#include "approx_delta_calcer_multi.h"

#include "approx_calcer_helpers.h"
#include "approx_updater_helpers.h"
#include "index_calcer.h"
#include "learn_context.h"

#include <catboost/libs/helpers/matrix.h>
#include <catboost/libs/helpers/dispatch_generic_lambda.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/algo/approx_calcer/approx_calcer_multi.h>
#include <catboost/private/libs/algo/approx_calcer/eval_additive_metric_with_leaves.h>
#include <catboost/private/libs/algo/approx_calcer/gradient_walker.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_multi_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/private/libs/algo_helpers/langevin_utils.h>
#include <catboost/private/libs/algo_helpers/online_predictor.h>

#include <library/cpp/threading/local_executor/local_executor.h>

void CalcApproxDeltaMulti(
    const TFold& fold,
    const TFold::TBodyTail& bt,
    int leafCount,
    const IDerCalcer& error,
    const TVector<TIndexType>& indices,
    ui64 randomSeed,
    TLearnContext* ctx,
    TVector<TVector<double>>* approxDelta,
    TVector<TVector<double>>* sumLeafDeltas
) {
    CB_ENSURE(!error.GetIsExpApprox(), "Multi-class and multi-regression does not support exponentiated approxes");

    const auto multiError = dynamic_cast<const TMultiDerCalcer*>(&error);
    const auto& treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = treeLearnerOptions.LeavesEstimationIterations;
    const TVector<TVector<float>>& target = fold.LearnTarget;
    const TVector<float>& weight = fold.GetLearnWeights();
    const int approxDimension = approxDelta->ysize();
    const ELeavesEstimation estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
    const float l2Regularizer = treeLearnerOptions.L2Reg;
    const double scaledL2Regularizer = ScaleL2Reg(l2Regularizer, fold.GetSumWeight(), fold.GetLearnSampleCount());

    TVector<TSumMulti> leafDers(leafCount, MakeZeroDers(approxDimension, estimationMethod, error.GetHessianType()));

    const auto leafUpdaterFunc = [&] (
        bool recalcLeafWeights,
        const TVector<TVector<double>>& approxDeltas,
        TVector<TVector<double>>* leafDeltas
    ) {
        if (estimationMethod == ELeavesEstimation::Exact) {
            CalcExactLeafDeltasMulti(
                ctx->Params.LossFunctionDescription,
                indices,
                bt.BodyFinish,
                bt.Approx,
                To2DConstArrayRef<float>(target),
                weight,
                leafCount,
                ctx->LocalExecutor,
                leafDeltas);
            return;
        }
        CalcLeafDersMulti(
            indices,
            To2DConstArrayRef<float>(target),
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
        AddLangevinNoiseToLeafDerivativesSum(
            ctx->Params.BoostingOptions->DiffusionTemperature,
            ctx->Params.BoostingOptions->LearningRate,
            scaledL2Regularizer,
            randomSeed,
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

        auto updateApproxesImpl = [&](auto useHessian, auto isMultiTarget, auto useWeights) {
            for (int docIdx = bt.BodyFinish; docIdx < bt.TailFinish; ++docIdx) {
                const double w = useWeights ? weight[docIdx] : 1;
                for (auto dim : xrange(approxDimension)) {
                    curApprox[dim] = bt.Approx[dim][docIdx] + (*approxDeltas)[dim][docIdx];
                }
                TVector<float> curTarget(fold.LearnTarget.size());
                for (auto dim : xrange(curTarget.size())) {
                    curTarget[dim] = fold.LearnTarget[dim][docIdx];
                }
                TSumMulti& curLeafDers = leafDers[indices[docIdx]];
                if (useHessian) {
                    if (isMultiTarget) {
                        multiError->CalcDers(curApprox, curTarget, w, &curDer, &curDer2);
                    } else {
                        error.CalcDersMulti(curApprox, target[0][docIdx], w, &curDer, &curDer2);
                    }
                    curLeafDers.AddDerDer2(curDer, curDer2);
                    CalcDeltaNewtonMulti(curLeafDers, l2Regularizer, bt.BodySumWeight, bt.BodyFinish, &curDelta);
                } else {
                    Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
                    if (isMultiTarget) {
                        multiError->CalcDers(curApprox, curTarget, w, &curDer, /*der2*/nullptr);
                    } else {
                        error.CalcDersMulti(curApprox, target[0][docIdx], w, &curDer, /*der2*/nullptr);
                    }
                    curLeafDers.AddDerWeight(curDer, w, /*isUpdateWeights*/true);
                    CalcDeltaGradientMulti(curLeafDers, l2Regularizer, bt.BodySumWeight, bt.BodyFinish, &curDelta);
                }
                for (auto dim : xrange(approxDimension)) {
                    (*approxDeltas)[dim][docIdx] = (*approxDeltas)[dim][docIdx] + curDelta[dim];
                }
            }
        };

        DispatchGenericLambda(updateApproxesImpl, useHessian, /*isMultiTarget*/multiError != nullptr, !weight.empty());
    };

    bool haveBacktrackingObjective;
    double minimizationSign;
    TVector<THolder<IMetric>> lossFunction;
    CreateBacktrackingObjective(*ctx, &haveBacktrackingObjective, &minimizationSign, &lossFunction);

    TVector<TVector<double>> localSumLeafDeltas;
    if (sumLeafDeltas == nullptr) {
        ResizeRank2(approxDimension, leafCount, localSumLeafDeltas);
        sumLeafDeltas = &localSumLeafDeltas;
    }

    const auto lossCalcerFunc = [&] (const TVector<TVector<double>>& /*approxDeltas*/, const TVector<TVector<double>>& leafDeltas) {
        TConstArrayRef<TQueryInfo> bodyTailQueryInfo(fold.LearnQueriesInfo.begin(), bt.BodyQueryFinish);
        TVector<TVector<double>> localLeafDeltas(*sumLeafDeltas);
        AddElementwise(leafDeltas, &localLeafDeltas);
        const auto& additiveStats = EvalErrorsWithLeaves(
            To2DConstArrayRef<double>(bt.Approx),
            To2DConstArrayRef<double>(localLeafDeltas),
            indices,
            /*isExpApprox*/ false,
            To2DConstArrayRef<float>(fold.LearnTarget, 0, bt.BodyFinish),
            fold.GetLearnWeights(),
            bodyTailQueryInfo,
            *lossFunction[0],
            ctx->LocalExecutor);
        return minimizationSign * lossFunction[0]->GetFinalError(additiveStats);
    };

    FastGradientWalker(
        /*isTrivialWalker*/!haveBacktrackingObjective,
        gradientIterations,
        leafCount,
        ctx->LearnProgress->ApproxDimension,
        leafUpdaterFunc,
        approxUpdaterFunc,
        lossCalcerFunc,
        approxDelta,
        sumLeafDeltas
    );
}
