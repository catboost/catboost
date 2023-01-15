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
        (IsUserDefined(objective) &&
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

static ELossFunction GetMetric(const NCatboostOptions::TLossDescription& lossDescription) {
    const auto lossFunction = lossDescription.GetLossFunction();
    if (lossFunction != ELossFunction::Combination) {
        return lossFunction;
    }
    return GetMetricFromCombination(lossDescription.GetLossParamsMap());
}

void CheckMetrics(const NCatboostOptions::TMetricOptions& metrics) {
    ELossFunction referenceLoss;
    if (metrics.EvalMetric.IsSet()) {
        referenceLoss = GetMetric(metrics.EvalMetric.Get());
    } else if (metrics.ObjectiveMetric.IsSet()) {
        referenceLoss = GetMetric(metrics.ObjectiveMetric.Get());
    } else {
        referenceLoss = ELossFunction::RMSE;
    }
    CheckMetric(GetMetric(metrics.EvalMetric.Get()), referenceLoss);
    CheckMetric(GetMetric(metrics.ObjectiveMetric.Get()), referenceLoss);
    for (const auto& loss : metrics.CustomMetrics.Get()) {
        CheckMetric(GetMetric(loss), referenceLoss);
    }
}

