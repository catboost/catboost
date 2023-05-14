#include "approx_calcer_multi.h"

#include "gradient_walker.h"
#include "eval_additive_metric_with_leaves.h"

#include <catboost/libs/helpers/dispatch_generic_lambda.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_helpers.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_multi_helpers.h>
#include <catboost/private/libs/algo_helpers/approx_updater_helpers.h>
#include <catboost/private/libs/algo_helpers/langevin_utils.h>
#include <catboost/private/libs/algo_helpers/leaf_statistics.h>
#include <catboost/private/libs/algo_helpers/online_predictor.h>

void CalcExactLeafDeltasMulti(
    const NCatboostOptions::TLossDescription& lossDescription,
    const TVector<TIndexType>& indices,
    const size_t sampleCount,
    const TVector<TVector<double>>& approxes,
    TConstArrayRef<TConstArrayRef<float>> targets,
    TConstArrayRef<float> weight,
    size_t leafCount,
    NPar::ILocalExecutor* localExecutor,
    TVector<TVector<double>>* leafDeltas
) {
    CB_ENSURE(targets.size() == 1, "Exact for multi quantile is not supported for multi target");
    CB_ENSURE(indices.size() > 0 || leafCount == 1, "Need leaf indices if leaf count > 1");
    TVector<TVector<float>> leafWeights(leafCount);
    if (!weight.empty()) {
        for (auto& weights : leafWeights) {
            weights.reserve(sampleCount / leafCount);
        }
        const auto setWeights = [&] (auto haveLeaves) {
            for (auto i : xrange(sampleCount)) {
                leafWeights[GetIf(haveLeaves, indices, i, 0)].emplace_back(weight[i]);
            }
        };
        DispatchGenericLambda(setWeights, leafCount > 1);
    }

    auto params = lossDescription.GetLossParamsMap();
    const auto alpha = NCatboostOptions::GetAlphaMultiQuantile(params);

    const auto quantileCount = approxes.size();
    Y_ASSERT(quantileCount == leafDeltas->size());

    NPar::ParallelFor(
        *localExecutor,
        0,
        quantileCount,
        [&] (int quantile) {
            TVector<TVector<float>> leafSamples(leafCount); // [leaf][sample]
            for (auto i : xrange(leafCount)) {
                leafSamples[i].reserve(leafWeights[i].size());
            }
            const auto setSamples = [&] (auto haveLeaves) {
                TConstArrayRef<double> approx(approxes[quantile]);
                TConstArrayRef<float> target(targets[0]);
                TConstArrayRef<TIndexType> indicesRef(indices);
                TArrayRef<TVector<float>> samples(leafSamples);
                for (auto i : xrange(sampleCount)) {
                    samples[GetIf(haveLeaves, indicesRef, i, 0)].emplace_back(target[i] - approx[i]);
                }
            };
            DispatchGenericLambda(setSamples, leafCount > 1);

            NCatboostOptions::TLossDescription quantileDescription;
            quantileDescription.LossFunction = ELossFunction::Quantile;
            if (params.contains("delta")) {
                quantileDescription.LossParams->Put("delta", params.at("delta"));
            }
            quantileDescription.LossParams->Put("alpha", ToString(alpha[quantile]));

            TArrayRef<double> leafDelta((*leafDeltas)[quantile]);
            Y_ASSERT(leafCount == leafDelta.size());
            for (auto i : xrange(leafCount)) {
                leafDelta[i] = *NCB::CalcOneDimensionalOptimumConstApprox(
                    quantileDescription,
                    leafSamples[i],
                    leafWeights[i]);
            }
    });
}


void CalcLeafValuesMulti(
    int leafCount,
    const IDerCalcer& error,
    const TVector<TQueryInfo>& queryInfo,
    const TVector<TIndexType>& indices,
    const TVector<TConstArrayRef<float>>& label,
    TConstArrayRef<float> weight,
    double sumWeight,
    int l2RegSampleCount,
    int sampleCount,
    TLearnContext* ctx,
    TVector<TVector<double>>* sumLeafDeltas, // [dim][leafIdx]
    TVector<TVector<double>>* approx
) {
    CB_ENSURE(!error.GetIsExpApprox(), "Multi-class does not support exponentiated approxes");

    const auto& params = ctx->Params;
    const int approxDimension = ctx->LearnProgress->ApproxDimension;
    const auto& metricDescriptions = ctx->Params.MetricOptions->ObjectiveMetric;
    TRestorableFastRng64* rng = &ctx->LearnProgress->Rand;
    NPar::ILocalExecutor* localExecutor = ctx->LocalExecutor;

    const auto& learnerOptions = params.ObliviousTreeOptions.Get();
    int gradientIterations = learnerOptions.LeavesEstimationIterations;
    ELeavesEstimation estimationMethod = learnerOptions.LeavesEstimationMethod;
    float l2Regularizer = learnerOptions.L2Reg;

    TVector<TSumMulti> leafDers(leafCount, MakeZeroDers(approxDimension, estimationMethod, error.GetHessianType()));

    bool haveBacktrackingObjective;
    double minimizationSign;
    TVector<THolder<IMetric>> lossFunction;
    CreateBacktrackingObjective(
        metricDescriptions,
        ctx->EvalMetricDescriptor,
        learnerOptions,
        approxDimension,
        &haveBacktrackingObjective,
        &minimizationSign,
        &lossFunction);


    const auto leafUpdaterFunc = [&] (
        bool recalcLeafWeights,
        const TVector<TVector<double>>& approxes,
        TVector<TVector<double>>* leafDeltas
    ) {
        if (estimationMethod == ELeavesEstimation::Exact) {
            CalcExactLeafDeltasMulti(
                params.LossFunctionDescription,
                indices,
                sampleCount,
                approxes,
                label,
                weight,
                leafCount,
                localExecutor,
                leafDeltas);
            return;
        }

        CalcLeafDersMulti(
            indices,
            label,
            weight,
            approxes,
            /*approxDeltas*/ {},
            error,
            sampleCount,
            recalcLeafWeights,
            estimationMethod,
            localExecutor,
            &leafDers);

        if (params.BoostingOptions->Langevin) {
            if (estimationMethod == ELeavesEstimation::Gradient) {
                AddLangevinNoiseToLeafDerivativesSum(
                    params.BoostingOptions->DiffusionTemperature,
                    params.BoostingOptions->LearningRate,
                    ScaleL2Reg(l2Regularizer, sumWeight, l2RegSampleCount),
                    rng->GenRand(),
                    &leafDers);
            } else if (estimationMethod == ELeavesEstimation::Newton) {
                AddLangevinNoiseToLeafNewtonSum(
                    params.BoostingOptions->DiffusionTemperature,
                    params.BoostingOptions->LearningRate,
                    ScaleL2Reg(l2Regularizer, sumWeight, l2RegSampleCount),
                    rng->GenRand(),
                    &leafDers);
            }
        }

        CalcLeafDeltasMulti(
            leafDers,
            estimationMethod,
            l2Regularizer,
            sumWeight,
            l2RegSampleCount,
            leafDeltas);
    };

    const auto approxUpdaterFunc = [&] (
        const TVector<TVector<double>>& leafDeltas,
        TVector<TVector<double>>* approxes
    ) {
        UpdateApproxDeltasMulti(
            indices,
            sampleCount,
            leafDeltas,
            approxes,
            localExecutor);
    };

    const auto lossCalcerFunc = [&](const TVector<TVector<double>>& approx, const TVector<TVector<double>>& leafDeltas) {
        const auto& additiveStats = EvalErrorsWithLeaves(
            To2DConstArrayRef<double>(approx),
            To2DConstArrayRef<double>(leafDeltas),
            indices,
            /*isExpApprox*/false,
            label,
            weight,
            queryInfo,
            *lossFunction[0],
            localExecutor);
        return minimizationSign * lossFunction[0]->GetFinalError(additiveStats);
    };

    FastGradientWalker(
        /*isTrivialWalker*/!haveBacktrackingObjective,
        gradientIterations,
        leafCount,
        approxDimension,
        leafUpdaterFunc,
        approxUpdaterFunc,
        lossCalcerFunc,
        approx,
        sumLeafDeltas);
}
