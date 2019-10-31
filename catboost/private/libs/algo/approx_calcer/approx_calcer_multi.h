#pragma once

#include "gradient_walker.h"

#include <catboost/private/libs/algo_helpers/approx_calcer_helpers.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_multi_helpers.h>
#include <catboost/private/libs/algo_helpers/approx_updater_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/private/libs/algo_helpers/leaf_statistics.h>
#include <catboost/private/libs/algo_helpers/online_predictor.h>
#include <catboost/private/libs/options/catboost_options.h>

#include <library/threading/local_executor/local_executor.h>


template <typename TStep>
void CalcLeafValuesMulti(
    const NCatboostOptions::TObliviousTreeLearnerOptions& learnerOptions,
    bool isLeafwise,
    int leafCount,
    const IDerCalcer& error,
    const TVector<TQueryInfo>& queryInfo,
    const TVector<TIndexType>& indices,
    TConstArrayRef<float> label,
    TConstArrayRef<float> weight,
    int approxDimension,
    double sumWeight,
    int learnSampleCount,
    int objectsInLeafCount,
    NCatboostOptions::TLossDescription metricDescriptions,
    NPar::TLocalExecutor* localExecutor,
    TVector<TStep>* sumLeafDeltas,
    TVector<TVector<double>>* approx
) {
    CB_ENSURE(!error.GetIsExpApprox(), "Multi-class does not support exponentiated approxes");

    int gradientIterations = learnerOptions.LeavesEstimationIterations;
    ELeavesEstimation estimationMethod = learnerOptions.LeavesEstimationMethod;
    float l2Regularizer = learnerOptions.L2Reg;

    TVector<TSumMulti> leafDers(isLeafwise ? 1 : leafCount, MakeZeroDers(approxDimension, estimationMethod, error.GetHessianType()));

    bool haveBacktrackingObjective;
    double minimizationSign;
    TVector<THolder<IMetric>> lossFunction;
    CreateBacktrackingObjective(
        metricDescriptions,
        learnerOptions,
        approxDimension,
        &haveBacktrackingObjective,
        &minimizationSign,
        &lossFunction
    );


    const auto leafUpdaterFunc = [&] (
        bool recalcLeafWeights,
        const TVector<TVector<double>>& approxes,
        auto leafDeltas
    ) {
        CalcLeafDersMulti(
            indices,
            label,
            weight,
            approxes,
            /*approxDeltas*/ {},
            error,
            isLeafwise ? objectsInLeafCount : learnSampleCount,
            recalcLeafWeights,
            estimationMethod,
            localExecutor,
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

    const auto approxUpdaterFunc = [&localExecutor, &indices, learnSampleCount, objectsInLeafCount, isLeafwise] (
        const auto& leafDeltas,
        TVector<TVector<double>>* approxes
    ) {
        if (!isLeafwise) {
            UpdateApproxDeltasMulti(
                indices,
                learnSampleCount,
                leafDeltas,
                approxes,
                localExecutor
            );
        } else {
            UpdateApproxDeltasMulti(
                {},
                objectsInLeafCount,
                leafDeltas,
                approxes,
                localExecutor
            );
        }
    };

    const auto lossCalcerFunc = [&] (const TVector<TVector<double>>& approx) {
        const auto& additiveStats = EvalErrors(
            approx,
            /*approxDelta*/{},
            /*isExpApprox*/false,
            label,
            weight,
            queryInfo,
            *lossFunction[0],
            localExecutor
        );
        return minimizationSign * lossFunction[0]->GetFinalError(additiveStats);
    };

    const auto approxCopyFunc = [localExecutor] (const TVector<TVector<double>>& src, TVector<TVector<double>>* dst) {
        CopyApprox(src, dst, localExecutor);
    };

    if (!isLeafwise) {
        GradientWalker</*IsLeafwise*/ false>(
            /*isTrivialWalker*/!haveBacktrackingObjective,
                               gradientIterations,
                               leafCount,
                               approxDimension,
                               leafUpdaterFunc,
                               approxUpdaterFunc,
                               lossCalcerFunc,
                               approxCopyFunc,
                               approx,
                               sumLeafDeltas
        );
    } else {
        GradientWalker</*IsLeafwise*/ true>(
            /*isTrivialWalker*/!haveBacktrackingObjective,
                               gradientIterations,
                               leafCount,
                               approxDimension,
                               leafUpdaterFunc,
                               approxUpdaterFunc,
                               lossCalcerFunc,
                               approxCopyFunc,
                               approx,
                               sumLeafDeltas
        );
    }
}
