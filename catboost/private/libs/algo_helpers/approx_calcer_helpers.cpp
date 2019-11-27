#include "approx_calcer_helpers.h"


void CreateBacktrackingObjectiveImpl(
    int dimensionCount,
    int leavesEstimationIterations,
    ELeavesEstimationStepBacktracking leavesEstimationBacktrackingType,
    const NCatboostOptions::TLossDescription& objectiveMetric,
    bool* haveBacktrackingObjective,
    double* minimizationSign,
    TVector<THolder<IMetric>>* lossFunction
) {
    *haveBacktrackingObjective = leavesEstimationIterations > 1
                                 && leavesEstimationBacktrackingType != ELeavesEstimationStepBacktracking::No;
    if (*haveBacktrackingObjective) {
        *lossFunction = CreateMetricFromDescription(objectiveMetric, dimensionCount);
        *minimizationSign = GetMinimizeSign((*lossFunction)[0]);
    }
}

void CreateBacktrackingObjective(
    NCatboostOptions::TLossDescription metricDescriptions,
    const NCatboostOptions::TObliviousTreeLearnerOptions& treeOptions,
    int approxDimension,
    bool* haveBacktrackingObjective,
    double* minimizationSign,
    TVector<THolder<IMetric>>* lossFunction
) {
    CreateBacktrackingObjectiveImpl(
        approxDimension,
        int(treeOptions.LeavesEstimationIterations.Get()),
        treeOptions.LeavesEstimationBacktrackingType,
        metricDescriptions,
        haveBacktrackingObjective,
        minimizationSign,
        lossFunction);
}

double GetMinimizeSign(const THolder<IMetric>& metric) {
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
