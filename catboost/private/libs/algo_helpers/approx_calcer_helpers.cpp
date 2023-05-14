#include "approx_calcer_helpers.h"


void CreateBacktrackingObjectiveImpl(
    int dimensionCount,
    int leavesEstimationIterations,
    ELeavesEstimationStepBacktracking leavesEstimationBacktrackingType,
    const NCatboostOptions::TLossDescription& objectiveMetric,
    const TMaybe<TCustomMetricDescriptor>& customMetric,
    bool* haveBacktrackingObjective,
    double* minimizationSign,
    TVector<THolder<IMetric>>* lossFunction
) {
    *haveBacktrackingObjective = leavesEstimationIterations > 1
                                 && leavesEstimationBacktrackingType != ELeavesEstimationStepBacktracking::No;
    if (*haveBacktrackingObjective) {
        if (objectiveMetric.LossFunction == ELossFunction::PythonUserDefinedPerObject) {
            CB_ENSURE(customMetric, "PythonUserDefinedPerObject requires a Python metric class");
            lossFunction->resize(0);
            lossFunction->emplace_back(MakeCustomMetric(*customMetric));
            *minimizationSign = customMetric->IsMaxOptimalFunc(customMetric->CustomData) ? -1.0 : 1.0;
        } else {
            *lossFunction = CreateMetricFromDescription(objectiveMetric, dimensionCount);
            *minimizationSign = GetMinimizeSign((*lossFunction)[0]);
        }
    }
}

void CreateBacktrackingObjective(
    NCatboostOptions::TLossDescription metricDescriptions,
    const TMaybe<TCustomMetricDescriptor>& customMetric,
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
        customMetric,
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
            CB_ENSURE(false, "Unexpected best metric value type");
        }
    }
}
