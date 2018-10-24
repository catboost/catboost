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

bool IsMultiClass(
    const ELossFunction lossFunction,
    const NCatboostOptions::TMetricOptions& metricOptions)
{
    return IsMultiClassMetric(lossFunction) ||
        (lossFunction == ELossFunction::Custom &&
            metricOptions.EvalMetric.IsSet() &&
            IsMultiClassMetric(metricOptions.EvalMetric->GetLossFunction()));
}
