#include "catboost_options.h"
#include "check_train_options.h"
#include "monotone_constraints.h"
#include "plain_options_helper.h"

void CheckFitParams(
    const NJson::TJsonValue& plainOptions,
    const TCustomObjectiveDescriptor* objectiveDescriptor,
    const TCustomMetricDescriptor* evalMetricDescriptor
) {
    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputJsonOptions;
    NCatboostOptions::PlainJsonToOptions(plainOptions, &catBoostJsonOptions, &outputJsonOptions);
    // Monotone constraints should be correctly converted to canonical format before loading.
    // Because of absent information about feature layout here, just skip check of these constraints.
    if (catBoostJsonOptions["tree_learner_options"].Has("monotone_constraints")) {
        catBoostJsonOptions["tree_learner_options"].EraseValue("monotone_constraints");
    }
    auto options = NCatboostOptions::LoadOptions(catBoostJsonOptions);

    if (IsUserDefined(options.LossFunctionDescription->GetLossFunction())) {
        CB_ENSURE(objectiveDescriptor != nullptr, "Error: provide objective descriptor for custom loss");
    }

    if (options.MetricOptions->EvalMetric.IsSet() && IsUserDefined(options.MetricOptions->EvalMetric->GetLossFunction())) {
        CB_ENSURE(evalMetricDescriptor != nullptr, "Error: provide eval metric descriptor for custom eval metric");
    }

    const auto& penaltiesOptions = options.ObliviousTreeOptions->FeaturePenalties;
    // Feature penalties params should be correctly converted to canonical format before loading.
    // Because of absent information about feature layout here, we can only check common coefficient param.
    if (penaltiesOptions.IsSet() && penaltiesOptions->PenaltiesCoefficient.IsSet()) {
        CB_ENSURE(penaltiesOptions->PenaltiesCoefficient >= 0, "Error: penalties coefficient should be nonnegative");
    }
}
