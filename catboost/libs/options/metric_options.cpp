#include "metric_options.h"
#include "json_helper.h"

NCatboostOptions::TMetricOptions::TMetricOptions()
    : EvalMetric("eval_metric", TLossDescription())
    , CustomMetrics("custom_metrics", TVector<TLossDescription>()) {
}

void NCatboostOptions::TMetricOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &EvalMetric, &CustomMetrics);
    CB_ENSURE(EvalMetric.Get().GetLossFunction() != ELossFunction::CtrFactor, ToString(ELossFunction::CtrFactor) << " cannot be used for overfitting detection or selecting best iteration on validation");
}

void NCatboostOptions::TMetricOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, EvalMetric, CustomMetrics);
}

bool NCatboostOptions::TMetricOptions::operator==(const TMetricOptions& rhs) const {
    return std::tie(EvalMetric, CustomMetrics) == std::tie(rhs.EvalMetric, rhs.CustomMetrics);
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


bool IsMultiClass(
    const ELossFunction lossFunction,
    const NCatboostOptions::TMetricOptions& metricOptions)
{
    return IsValidForObjectiveOrEvalMetric(lossFunction, metricOptions, IsMultiClassMetric);
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
