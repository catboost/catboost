#include "leafwise_approx_calcer.h"

#include "approx_calcer_multi.h"
#include "gradient_walker.h"
#include "eval_additive_metric_with_leaves.h"

#include <catboost/libs/helpers/dispatch_generic_lambda.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/algo/learn_context.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_helpers.h>
#include <catboost/private/libs/algo_helpers/approx_updater_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/private/libs/algo_helpers/online_predictor.h>
#include <catboost/private/libs/algo_helpers/pairwise_leaves_calculation.h>
#include <catboost/private/libs/options/loss_description.h>

#include <util/stream/output.h>

static void CalcLeafDer(
    TConstArrayRef<float> labels,
    TConstArrayRef<float> weights,
    TConstArrayRef<double> approxes,
    int objectsCount,
    const IDerCalcer& error,
    bool recalcLeafWeights,
    ELeavesEstimation estimationMethod,
    TArrayRef<TDers> weightedDers,
    TSum* leafDer,
    NPar::ILocalExecutor* localExecutor
) {
    NPar::ILocalExecutor::TExecRangeParams blockParams(0, objectsCount);
    blockParams.SetBlockCount(CB_THREAD_LIMIT);

    TVector<TDers> blockBucketDers(
        blockParams.GetBlockCount(),
        TDers{/*Der1*/ 0.0, /*Der2*/ 0.0, /*Der3*/ 0.0});
    // TODO(espetrov): Do not calculate sumWeights for Newton.
    // TODO(espetrov): Calculate sumWeights only on first iteration for Gradient, because on next iteration it
    //  is the same.
    // Check speedup on flights dataset.

    TVector<double> blockBucketSumWeights;
    blockBucketSumWeights.yresize(blockParams.GetBlockCount());
    bool useWeights = !weights.empty();
    localExecutor->ExecRangeWithThrow(
        [=, &error, &blockBucketDers, &blockBucketSumWeights](int blockId) {
            constexpr int innerBlockSize = APPROX_BLOCK_SIZE;
            const auto approxDers = MakeArrayRef(
                weightedDers.data() + innerBlockSize * blockId,
                innerBlockSize);

            const int blockStart = blockId * blockParams.GetBlockSize();
            const int nextBlockStart = Min(objectsCount, blockStart + blockParams.GetBlockSize());

            for (int innerBlockStart = blockStart;
                innerBlockStart < nextBlockStart;
                innerBlockStart += innerBlockSize
            ) {
                const int innerCount = Min(nextBlockStart - innerBlockStart, innerBlockSize);
                error.CalcDersRange(
                    0,
                    innerCount,
                    /*calcThirdDer=*/false,
                    approxes.data() + innerBlockStart,
                    nullptr,
                    labels.data() + innerBlockStart,
                    !useWeights ? nullptr : weights.data() + innerBlockStart,
                    approxDers.data());
                double der1 = 0;
                double der2 = 0;
                double blockWeight = 0;
                for (auto rowIdx : xrange(innerBlockStart, innerBlockStart + innerCount)) {
                    der1 += approxDers[rowIdx - innerBlockStart].Der1;
                    der2 += approxDers[rowIdx - innerBlockStart].Der2;
                    const double rowWeight = useWeights ? weights[rowIdx] : 1;
                    blockWeight += rowWeight;
                }
                blockBucketDers[blockId].Der1 = der1;
                blockBucketDers[blockId].Der2 = der2;
                blockBucketSumWeights[blockId] = blockWeight;
            }
        },
        0,
        blockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE);

    if (estimationMethod == ELeavesEstimation::Newton) {
        for (int blockId = 0; blockId < blockParams.GetBlockCount(); ++blockId) {
            if (blockBucketSumWeights[blockId] > FLT_EPSILON) { // empty weights
                AddMethodDer<ELeavesEstimation::Newton>(
                    blockBucketDers[blockId],
                    blockBucketSumWeights[blockId],
                    /* updateWeight */ false, // value doesn't matter
                    leafDer);
            }
        }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        for (int blockId = 0; blockId < blockParams.GetBlockCount(); ++blockId) {
            if (blockBucketSumWeights[blockId] > FLT_EPSILON) {
                AddMethodDer<ELeavesEstimation::Gradient>(
                    blockBucketDers[blockId],
                    blockBucketSumWeights[blockId],
                    recalcLeafWeights,
                    leafDer);
            }
        }
    }
}

inline double CalcLeafDelta(
    const TSum& leafDer,
    ELeavesEstimation estimationMethod,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount
) {
    if (estimationMethod == ELeavesEstimation::Newton) {
        return CalcMethodDelta<ELeavesEstimation::Newton>(
            leafDer,
            l2Regularizer,
            sumAllWeights,
            allDocCount);
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        return CalcMethodDelta<ELeavesEstimation::Gradient>(
            leafDer,
            l2Regularizer,
            sumAllWeights,
            allDocCount);
    }
}

template <bool StoreExpApprox, int VectorWidth>
inline void UpdateApproxKernel(double leafDelta, double* deltas) {
    Y_ASSERT(VectorWidth == 8);
    deltas[0] = UpdateApprox<StoreExpApprox>(deltas[0], leafDelta);
    deltas[1] = UpdateApprox<StoreExpApprox>(deltas[1], leafDelta);
    deltas[2] = UpdateApprox<StoreExpApprox>(deltas[2], leafDelta);
    deltas[3] = UpdateApprox<StoreExpApprox>(deltas[3], leafDelta);
    deltas[4] = UpdateApprox<StoreExpApprox>(deltas[4], leafDelta);
    deltas[5] = UpdateApprox<StoreExpApprox>(deltas[5], leafDelta);
    deltas[6] = UpdateApprox<StoreExpApprox>(deltas[6], leafDelta);
    deltas[7] = UpdateApprox<StoreExpApprox>(deltas[7], leafDelta);
}

static void UpdateApproxDeltas(
    bool storeExpApprox,
    int docCount,
    double leafDelta,
    NPar::ILocalExecutor* localExecutor,
    TArrayRef<double> deltas
) {
    if (storeExpApprox) {
        leafDelta = fast_exp(leafDelta);
    }

    NPar::ILocalExecutor::TExecRangeParams blockParams(0, docCount);
    blockParams.SetBlockSize(1000);
    const auto getUpdateApproxBlockLambda = [&](auto boolConst) -> std::function<void(int)> {
        return [=](int blockIdx) {
            const int blockStart = blockIdx * blockParams.GetBlockSize();
            const int nextBlockStart = Min<ui64>(blockStart + blockParams.GetBlockSize(), blockParams.LastId);
            int doc = blockStart;
            for (; doc < nextBlockStart; ++doc) {
                deltas[doc] = UpdateApprox<boolConst.value>(deltas[doc], leafDelta);
            }
        };
    };
    localExecutor->ExecRange(
        DispatchGenericLambda(getUpdateApproxBlockLambda, storeExpApprox),
        0,
        blockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void CalcExactLeafDeltas(
    const NCatboostOptions::TLossDescription& lossDescription,
    TConstArrayRef<float> labels,
    TConstArrayRef<float> weights,
    int objectsCount,
    TConstArrayRef<double> approxes,
    double& leafDelta
) {
    TVector<float> samples;
    samples.yresize(approxes.size());
    for (int i = 0; i < objectsCount; i++) {
        samples[i] = labels[i] - approxes[i];
    }
    leafDelta = *NCB::CalcOneDimensionalOptimumConstApprox(lossDescription, samples, weights);
}

static void CalcLeafValuesSimple(
    const IDerCalcer& error,
    TLeafStatistics* statistics,
    TLearnContext* ctx,
    TArrayRef<TDers> weightedDers
) {
    Y_ASSERT(error.GetErrorType() == EErrorType::PerObjectError);
    const auto& params = ctx->Params;
    NPar::ILocalExecutor* localExecutor = ctx->LocalExecutor;
    const auto& learnerOptions = params.ObliviousTreeOptions.Get();
    int gradientIterations = learnerOptions.LeavesEstimationIterations;
    ELeavesEstimation estimationMethod = learnerOptions.LeavesEstimationMethod;
    const float l2Regularizer = static_cast<const float>(params.ObliviousTreeOptions->L2Reg);

    const int scratchSize = APPROX_BLOCK_SIZE * CB_THREAD_LIMIT;
    Y_ASSERT(weightedDers.size() == scratchSize);

    TSum leafDer;

    Y_ASSERT(statistics->GetLabels().size() == 1);
    auto labels = statistics->GetLabels()[0];
    auto weights = statistics->GetWeights();

    const auto leafUpdaterFunc = [&](
        bool recalcLeafWeights,
        const TVector<TVector<double>>& approxes,
        TVector<TVector<double>>* leafDeltas
    ) {
        leafDer.SetZeroDers();

        if (estimationMethod == ELeavesEstimation::Exact) {
            CB_ENSURE(!params.BoostingOptions->ApproxOnFullHistory);
            CalcExactLeafDeltas(
                params.LossFunctionDescription,
                labels,
                statistics->GetSampleWeights(),
                statistics->GetObjectsCountInLeaf(),
                statistics->GetApprox(0),
                (*leafDeltas)[0][0]);
            return;
        }

        CalcLeafDer(
            labels,
            weights,
            approxes[0],
            statistics->GetObjectsCountInLeaf(),
            error,
            recalcLeafWeights,
            estimationMethod,
            weightedDers,
            &leafDer,
            localExecutor);
        (*leafDeltas)[0][0] = CalcLeafDelta(
            leafDer,
            estimationMethod,
            l2Regularizer,
            statistics->GetAllObjectsSumWeight(),
            statistics->GetLearnObjectsCount());
    };

    const auto approxUpdaterFunc = [&] (
        const TVector<TVector<double>>& leafDeltas,
        TVector<TVector<double>>* approxes
    ) {
        UpdateApproxDeltas(
            error.GetIsExpApprox(),
            statistics->GetObjectsCountInLeaf(),
            leafDeltas[0][0],
            localExecutor,
            (*approxes)[0]);
    };

    bool haveBacktrackingObjective;
    double minimizationSign;
    TVector<THolder<IMetric>> lossFunction;

    CreateBacktrackingObjective(
        params.MetricOptions->ObjectiveMetric,
        ctx->EvalMetricDescriptor,
        learnerOptions,
        /*approxDimension*/ 1,
        &haveBacktrackingObjective,
        &minimizationSign,
        &lossFunction);


    const auto lossCalcerFunc = [&](const TVector<TVector<double>>& approx, const TVector<TVector<double>>& leafDeltas) {
        const auto& additiveStats = EvalErrorsWithLeaves(
            To2DConstArrayRef<double>(approx),
            To2DConstArrayRef<double>(leafDeltas),
            /*indices*/{},
            error.GetIsExpApprox(),
            {labels},
            weights,
            /*queryInfo*/{},
            *lossFunction[0],
            localExecutor);
        return minimizationSign * lossFunction[0]->GetFinalError(additiveStats);
    };

    const auto leafValuesRef = statistics->GetLeafValuesRef();
    TVector<TVector<double>> leafValues(1, TVector<double>(1)); // [approxDim][0]
    leafValues[0][0] = leafValuesRef->at(0);
    FastGradientWalker(
        /*isTrivialWalker*/ !haveBacktrackingObjective,
        gradientIterations,
        /*leafCount*/ 1,
        /*approxDimension*/ 1,
        leafUpdaterFunc,
        approxUpdaterFunc,
        lossCalcerFunc,
        statistics->GetApproxRef(),
        &leafValues);
    leafValuesRef->at(0) = leafValues[0][0];
}

void CalcLeafValues(
    const IDerCalcer& error,
    TLeafStatistics* statistics,
    TLearnContext* ctx,
    TArrayRef<TDers> weightedDers
) {
    const auto& params = ctx->Params;

    Y_ASSERT(error.GetErrorType() == EErrorType::PerObjectError);
    Y_ASSERT(!params.BoostingOptions->ApproxOnFullHistory);
    Y_ASSERT(
        params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Exact ||
        params.BoostingOptions->BoostingType == EBoostingType::Plain);

    if (statistics->GetObjectsCountInLeaf() == 0) {
        return;
    }

    const bool isMultiTarget = dynamic_cast<const TMultiDerCalcer*>(&error) != nullptr;
    if (statistics->GetApproxDimension() == 1 && !isMultiTarget) {
        CalcLeafValuesSimple(
            error,
            statistics,
            ctx,
            weightedDers);
    } else {
        const auto leafValuesRef = statistics->GetLeafValuesRef();
        const auto approxDim = statistics->GetApproxDimension();
        TVector<TVector<double>> leafValues(approxDim, TVector<double>(1)); // [approxDim][0]
        for (auto dim : xrange(approxDim)) {
            leafValues[dim][0] = leafValuesRef->at(dim);
        }
        CalcLeafValuesMulti(
            /*leafCount*/ 1,
            error,
            /*queryInfo*/ {},
            /*indices*/ {},
            To2DConstArrayRef<float>(statistics->GetLabels()),
            statistics->GetWeights(),
            statistics->GetAllObjectsSumWeight(),
            statistics->GetLearnObjectsCount(),
            statistics->GetObjectsCountInLeaf(),
            ctx,
            &leafValues,
            statistics->GetApproxRef()
        );
        for (auto dim : xrange(approxDim)) {
            leafValuesRef->at(dim) = leafValues[dim][0];
        }
    }
}

void AssignLeafValues(
    const TVector<TLeafStatistics>& leafStatistics,
    TVector<TVector<double>>* treeValues
) {
    int approxDimensionSize = leafStatistics[0].GetApproxDimension();
    treeValues->resize(approxDimensionSize, TVector<double>(leafStatistics.ysize()));
    for (const auto& statistics : leafStatistics) {
        auto treeLeaves = statistics.GetLeafValues();
        for (int dimIdx = 0; dimIdx < approxDimensionSize; ++dimIdx) {
            (*treeValues)[dimIdx][statistics.GetLeafIdx()] = treeLeaves[dimIdx];
        }
    }
}
