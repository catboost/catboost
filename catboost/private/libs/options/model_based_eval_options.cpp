#include "model_based_eval_options.h"
#include "json_helper.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/logging_level.h>

#include <util/string/builder.h>

NCatboostOptions::TModelBasedEvalOptions::TModelBasedEvalOptions(ETaskType /*taskType*/)
    : FeaturesToEvaluate("features_to_evaluate", TVector<TVector<ui32>>())
    , BaselineModelSnapshot("baseline_model_snapshot", "baseline_model_snapshot")
    , Offset("offset", 1000)
    , ExperimentCount("experiment_count", 200)
    , ExperimentSize("experiment_size", 5)
    , UseEvaluatedFeaturesInBaselineModel("use_evaluated_features_in_baseline_model", false)
{
}

void NCatboostOptions::TModelBasedEvalOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(
        options,
        &FeaturesToEvaluate,
        &BaselineModelSnapshot,
        &Offset,
        &ExperimentCount,
        &ExperimentSize,
        &UseEvaluatedFeaturesInBaselineModel
    );

    Validate();
}

void NCatboostOptions::TModelBasedEvalOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(
        options,
        FeaturesToEvaluate,
        BaselineModelSnapshot,
        Offset,
        ExperimentCount,
        ExperimentSize,
        UseEvaluatedFeaturesInBaselineModel
    );
}

bool NCatboostOptions::TModelBasedEvalOptions::operator==(const TModelBasedEvalOptions& rhs) const {
    const auto& options = std::tie(
        FeaturesToEvaluate,
        BaselineModelSnapshot,
        Offset,
        ExperimentCount,
        ExperimentSize,
        UseEvaluatedFeaturesInBaselineModel
    );
    const auto& rhsOptions = std::tie(
        rhs.FeaturesToEvaluate,
        rhs.BaselineModelSnapshot,
        rhs.Offset,
        rhs.ExperimentCount,
        rhs.ExperimentSize,
        rhs.UseEvaluatedFeaturesInBaselineModel
    );
    return options == rhsOptions;
}

bool NCatboostOptions::TModelBasedEvalOptions::operator!=(const TModelBasedEvalOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TModelBasedEvalOptions::Validate() const {
    CB_ENSURE(ExperimentCount * ExperimentSize <= Offset, "Offset must be greater than or equal to ExperimentCount * ExperimentSize");
}

TString NCatboostOptions::GetExperimentName(ui32 featureSetIdx, ui32 experimentIdx) {
    return TStringBuilder() << "feature_set" << featureSetIdx << "_fold" << experimentIdx;
}
