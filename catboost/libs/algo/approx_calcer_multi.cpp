#include "approx_calcer_multi.h"

#include "approx_calcer_helpers.h"
#include "approx_updater_helpers.h"
#include "index_calcer.h"
#include "learn_context.h"

#include <catboost/libs/algo_helpers/error_functions.h>
#include <catboost/libs/algo_helpers/gradient_walker.h>
#include <catboost/libs/algo_helpers/online_predictor.h>

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
    const TVector<TIndexType>& indices,
    int docCount,
    NPar::TLocalExecutor* localExecutor,
    TVector<TVector<double>>* leafDeltas, //leafDeltas[dimension][leafId]
    TVector<TVector<double>>* approxDeltas
) {
    const auto indicesRef = MakeArrayRef(indices);
    for (int dim = 0; dim < leafDeltas->ysize(); ++dim) {
        auto approxDeltaRef = MakeArrayRef((*approxDeltas)[dim]);
        auto leafDeltaRef = MakeArrayRef((*leafDeltas)[dim]);
        NPar::ParallelFor(
            *localExecutor,
            /*from*/0,
            /*to*/ docCount,
            /*body*/[=] (int z) {
                approxDeltaRef[z] += leafDeltaRef[indicesRef[z]];
            }
        );
    }
}

inline static TSumMulti MakeZeroDers(
    int approxDimension,
    ELeavesEstimation estimationMethod,
    EHessianType hessianType
) {
    if (estimationMethod == ELeavesEstimation::Gradient) {
        return TSumMulti(approxDimension);
    } else {
        return TSumMulti(approxDimension, hessianType);
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
    CB_ENSURE(!error.GetIsExpApprox(), "Multi-class does not support exponentiated approxes");

    const auto& treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = treeLearnerOptions.LeavesEstimationIterations;
    const TVector<float>& target = fold.LearnTarget;
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
        const TVector<TVector<double>>& leafDeltas,
        TVector<TVector<double>>* approxDeltas
    ) {
        auto localLeafValues = leafDeltas;
        UpdateApproxDeltasMulti(
            indices,
            bt.BodyFinish,
            ctx->LocalExecutor,
            &localLeafValues,
            approxDeltas
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
        TConstArrayRef<float> bodyTailTarget(fold.LearnTarget.begin(), bt.BodyFinish);
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

static void AddDersRangeMulti(
    TConstArrayRef<TIndexType> leafIndices,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TVector<double>> approx, // [dimensionIdx][columnIdx]
    TConstArrayRef<TVector<double>> approxDeltas, // [dimensionIdx][columnIdx]
    const IDerCalcer& error,
    int rowBegin,
    int rowEnd,
    bool isUpdateWeight,
    TArrayRef<TSumMulti> leafDers // [dimensionIdx]
) {
    const int approxDimension = approx.size();
    const bool useHessian = !leafDers[0].SumDer2.Data.empty();
    THessianInfo curDer2(useHessian * approxDimension, error.GetHessianType());
    TVector<double> curDer(approxDimension);
    constexpr int UnrollMaxCount = 16;
    TVector<TVector<double>> curApprox(UnrollMaxCount, TVector<double>(approxDimension));
    for (int columnIdx = rowBegin; columnIdx < rowEnd; columnIdx += UnrollMaxCount) {
        const int unrollCount = Min(UnrollMaxCount, rowEnd - columnIdx);
        SumTransposedBlocks(columnIdx, columnIdx + unrollCount, approx, approxDeltas, MakeArrayRef(curApprox));
        for (int unrollIdx : xrange(unrollCount)) {
            error.CalcDersMulti(curApprox[unrollIdx], target[columnIdx + unrollIdx], weight.empty() ? 1 : weight[columnIdx + unrollIdx], &curDer, useHessian ? &curDer2 : nullptr);
            TSumMulti& curLeafDers = leafDers[leafIndices[columnIdx + unrollIdx]];
            if (useHessian) {
                curLeafDers.AddDerDer2(curDer, curDer2);
            } else {
                curLeafDers.AddDerWeight(curDer, weight.empty() ? 1 : weight[columnIdx + unrollIdx], isUpdateWeight);
            }
        }
    }
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
    NPar::TLocalExecutor* localExecutor,
    TVector<TSumMulti>* leafDers
) {
    const int approxDimension = approx.ysize();
    Y_ASSERT(approxDimension > 0);
    const auto leafCount = leafDers->size();
    for (auto& curLeafDers : *leafDers) {
        curLeafDers.SetZeroDers();
    }
    const auto& zeroDers = MakeZeroDers(approxDimension, estimationMethod, error.GetHessianType());
    const auto hessianSize = zeroDers.SumDer2.Data.size();
    NCB::MapMerge(
        localExecutor,
        NCB::TSimpleIndexRangesGenerator<int>(NCB::TIndexRange<int>(sampleCount), /*blockSize*/Max<ui32>(1000, hessianSize / CB_THREAD_LIMIT)),
        /*mapFunc*/[&](NCB::TIndexRange<int> partIndexRange, TVector<TSumMulti>* leafDers) {
            Y_ASSERT(!partIndexRange.Empty());
            leafDers->resize(leafCount, zeroDers);
            AddDersRangeMulti(
                indices,
                target,
                weight,
                approx, // [dimensionIdx][rowIdx]
                approxDeltas, // [dimensionIdx][rowIdx]
                error,
                partIndexRange.Begin,
                partIndexRange.End,
                isUpdateWeight,
                *leafDers // [dimensionIdx]
            );
        },
        /*mergeFunc*/[=](TVector<TSumMulti>* leafDers, TVector<TVector<TSumMulti>>&& addVector) {
            if (estimationMethod == ELeavesEstimation::Newton) {
                for (auto leafIdx : xrange(leafCount)) {
                    for (const auto& addItem : addVector) {
                        (*leafDers)[leafIdx].AddDerDer2(addItem[leafIdx].SumDer, addItem[leafIdx].SumDer2);
                    }
                }
            } else {
                for (auto leafIdx : xrange(leafCount)) {
                    for (const auto& addItem : addVector) {
                        (*leafDers)[leafIdx].AddDerWeight(addItem[leafIdx].SumDer, addItem[leafIdx].SumWeights, isUpdateWeight);
                    }
                }
            }
        },
        leafDers
    );
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
    CB_ENSURE(!error.GetIsExpApprox(), "Multi-class does not support exponentiated approxes");

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
    TVector<TSumMulti> leafDers(leafCount, MakeZeroDers(approxDimension, estimationMethod, error.GetHessianType()));

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
            /*isExpApprox*/false,
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

    sumLeafDeltas->assign(approxDimension, TVector<double>(leafCount));
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
