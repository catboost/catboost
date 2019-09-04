#include "metric_options.h"
#include "json_helper.h"

NCatboostOptions::TMetricOptions::TMetricOptions()
    : EvalMetric("eval_metric", TLossDescription())
    , ObjectiveMetric("objective_metric", TLossDescription())
    , CustomMetrics("custom_metrics", TVector<TLossDescription>()) {
}

void NCatboostOptions::TMetricOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &EvalMetric, &ObjectiveMetric, &CustomMetrics);
    CB_ENSURE(EvalMetric.Get().GetLossFunction() != ELossFunction::CtrFactor, ToString(ELossFunction::CtrFactor) << " cannot be used for overfitting detection or selecting best iteration on validation");
}

void NCatboostOptions::TMetricOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, EvalMetric, ObjectiveMetric, CustomMetrics);
}

bool NCatboostOptions::TMetricOptions::operator==(const TMetricOptions& rhs) const {
    return std::tie(EvalMetric, ObjectiveMetric, CustomMetrics) == std::tie(rhs.EvalMetric, rhs.ObjectiveMetric, rhs.CustomMetrics);
}

bool NCatboostOptions::TMetricOptions::operator!=(const TMetricOptions& rhs) const {
    return !(rhs == *this);
}


bool IsValidForObjectiveOrEvalMetric(
    const ELossFunction objective,
    const NCatboostOptions::TMetricOptions& metricOptions,
    std::function<bool(ELossFunction)> predicate)
{
    return predicate(objective) ||
        (objective == ELossFunction::PythonUserDefinedPerObject &&
            metricOptions.EvalMetric.IsSet() && predicate(metricOptions.EvalMetric->GetLossFunction()));
}


bool IsMultiClassOnly(
    const ELossFunction lossFunction,
    const NCatboostOptions::TMetricOptions& metricOptions)
{
    return IsValidForObjectiveOrEvalMetric(lossFunction, metricOptions, IsMultiClassOnlyMetric);
}


void IterateOverObjectiveAndMetrics(
    const NCatboostOptions::TLossDescription& objective,
    const NCatboostOptions::TMetricOptions& metricOptions,
    std::function<void(const NCatboostOptions::TLossDescription&)>&& visitor)
{
    visitor(objective);
    if (metricOptions.EvalMetric.IsSet()) {
        visitor(metricOptions.EvalMetric);
    }
    if (metricOptions.CustomMetrics.IsSet()) {
        for (const auto& customMetric : metricOptions.CustomMetrics.Get()) {
            visitor(customMetric);
        }
    }
}


bool IsAnyOfObjectiveOrMetrics(
    const NCatboostOptions::TLossDescription& objective,
    const NCatboostOptions::TMetricOptions& metricOptions,
    std::function<bool(ELossFunction)> predicate)
{
    bool result = false;

    IterateOverObjectiveAndMetrics(
        objective,
        metricOptions,
        [&] (const NCatboostOptions::TLossDescription& lossDescription) {
            if (predicate(lossDescription.GetLossFunction())) {
                result = true;
            }
        });

    return result;
}

void ValidateIsMetricCalculationSupported(
    ELossFunction metric,
    ELossFunction lossFunction,
    ETaskType taskType
) {
    CB_ENSURE(
        taskType != ETaskType::CPU || lossFunction != ELossFunction::CrossEntropy || metric != ELossFunction::AUC,
        ToString(metric) << " calculation on " << ToString(taskType)
        << " doesn't supported if loss function is " << ToString(lossFunction) << "."
    );
}
