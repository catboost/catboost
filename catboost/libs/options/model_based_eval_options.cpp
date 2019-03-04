#include "model_based_eval_options.h"
#include "json_helper.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/logging_level.h>

NCatboostOptions::TModelBasedEvalOptions::TModelBasedEvalOptions(ETaskType /*taskType*/)
    : FeaturesToEvaluate("features_to_evaluate", TVector<TVector<ui32>>())
    , BaselineModelSnapshot("baseline_model_snapshot", "baseline_model_snapshot")
    , Offset("offset", 1000)
    , ExperimentCount("experiment_count", 200)
    , ExperimentSize("experiment_size", 5)
{
}

void NCatboostOptions::TModelBasedEvalOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &FeaturesToEvaluate, &BaselineModelSnapshot, &Offset, &ExperimentCount, &ExperimentSize);

    Validate();
}

void NCatboostOptions::TModelBasedEvalOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, FeaturesToEvaluate, BaselineModelSnapshot, Offset, ExperimentCount, ExperimentSize);
}

bool NCatboostOptions::TModelBasedEvalOptions::operator==(const TModelBasedEvalOptions& rhs) const {
    return std::tie(FeaturesToEvaluate, BaselineModelSnapshot, Offset, ExperimentCount, ExperimentSize) ==
        std::tie(rhs.FeaturesToEvaluate, rhs.BaselineModelSnapshot, rhs.Offset, rhs.ExperimentCount, rhs.ExperimentSize);
}

bool NCatboostOptions::TModelBasedEvalOptions::operator!=(const TModelBasedEvalOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TModelBasedEvalOptions::Validate() const {
    CB_ENSURE(ExperimentCount * ExperimentSize <= Offset, "Offset must be greater than or equal to ExperimentCount * ExperimentSize");
}

TString NCatboostOptions::TModelBasedEvalOptions::GetExperimentName(ui32 featureSetIdx, ui32 experimentIdx) const {
    return "feature_set" + ToString(featureSetIdx) + "_fold" + ToString(experimentIdx);
}
