#include "approx_calcer_helpers.h"

static inline double GetMinimizeSign(const THolder<IMetric>& metric) {
    EMetricBestValue bestMetric;
    float ignoredBestValue;
    metric->GetBestValue(&bestMetric, &ignoredBestValue);
    switch (bestMetric) {
        case EMetricBestValue::Min: {
            return 1.0;
        }
        case EMetricBestValue::Max: {
            return -1.0;
        }
        default: {
            Y_UNREACHABLE();
        }
    }
}

void CreateBacktrackingObjective(
    const TLearnContext& ctx,
    bool* haveBacktrackingObjective,
    double* minimizationSign,
    TVector<THolder<IMetric>>* lossFunction
) {
    const int dimensionCount = ctx.LearnProgress->ApproxDimension;
    const auto& treeOptions = ctx.Params.ObliviousTreeOptions.Get();
    *haveBacktrackingObjective = treeOptions.LeavesEstimationIterations.Get() > 1
        && treeOptions.LeavesEstimationBacktrackingType != ELeavesEstimationStepBacktracking::No;
    if (*haveBacktrackingObjective) {
        *lossFunction = CreateMetricFromDescription(ctx.Params.MetricOptions->ObjectiveMetric, dimensionCount);
        *minimizationSign = GetMinimizeSign((*lossFunction)[0]);
    }
}
