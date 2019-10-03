#include "check_train_options.h"
#include "plain_options_helper.h"
#include "catboost_options.h"

void CheckFitParams(
    const NJson::TJsonValue& plainOptions,
    const TCustomObjectiveDescriptor* objectiveDescriptor,
    const TCustomMetricDescriptor* evalMetricDescriptor
) {
    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputJsonOptions;
    NCatboostOptions::PlainJsonToOptions(plainOptions, &catBoostJsonOptions, &outputJsonOptions);
    auto options = NCatboostOptions::LoadOptions(catBoostJsonOptions);

    if (options.LossFunctionDescription->GetLossFunction() == ELossFunction::PythonUserDefinedPerObject) {
        CB_ENSURE(objectiveDescriptor != nullptr, "Error: provide objective descriptor for custom loss");
    }

    if (options.MetricOptions->EvalMetric.IsSet() && options.MetricOptions->EvalMetric->GetLossFunction() == ELossFunction::PythonUserDefinedPerObject) {
        CB_ENSURE(evalMetricDescriptor != nullptr, "Error: provide eval metric descriptor for custom eval metric");
    }
}
