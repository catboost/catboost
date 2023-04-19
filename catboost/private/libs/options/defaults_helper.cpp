#include "defaults_helper.h"

#include "enums.h"
#include "enum_helpers.h"

#include <catboost/libs/logging/logging.h>

#include <util/generic/set.h>


void UpdateMetricPeriodOption(
    const NCatboostOptions::TCatBoostOptions& trainOptions,
    NCatboostOptions::TOutputFilesOptions* outputOptions
) {
    if (outputOptions->IsMetricPeriodSet()) {
        return;
    }
    if (trainOptions.GetTaskType() == ETaskType::CPU) {
        return;
    }

    const auto& metricOptions = trainOptions.MetricOptions;
    TSet<ELossFunction> cpuOnlyMetrics;
    if (!HasGpuImplementation(metricOptions->EvalMetric->GetLossFunction())) {
        cpuOnlyMetrics.insert(metricOptions->EvalMetric->GetLossFunction());
    }
    if (!HasGpuImplementation(metricOptions->ObjectiveMetric->GetLossFunction())) {
        cpuOnlyMetrics.insert(metricOptions->ObjectiveMetric->GetLossFunction());
    }
    for (const auto& metric : metricOptions->CustomMetrics.Get()) {
        if (!HasGpuImplementation(metric.GetLossFunction())) {
            cpuOnlyMetrics.insert(metric.GetLossFunction());
        }
    }
    if (cpuOnlyMetrics.size() > 0) {
        constexpr ui32 HeavyMetricPeriod = 5;
        const auto someMetric = *cpuOnlyMetrics.begin();
        cpuOnlyMetrics.erase(someMetric);
        CATBOOST_WARNING_LOG << "Default metric period is " << HeavyMetricPeriod << " because " << ToString(someMetric);
        for (auto metric : cpuOnlyMetrics) {
            CATBOOST_WARNING_LOG << ", " << ToString(metric);
        }
        CATBOOST_WARNING_LOG << " is/are not implemented for GPU" << Endl;
        outputOptions->SetMetricPeriod(HeavyMetricPeriod);
    }
}
