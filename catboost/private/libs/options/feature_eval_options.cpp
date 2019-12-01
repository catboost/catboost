#include "feature_eval_options.h"
#include "json_helper.h"

NCatboostOptions::TFeatureEvalOptions::TFeatureEvalOptions()
    : FeaturesToEvaluate("features_to_evaluate", TVector<TVector<ui32>>())
    , FeatureEvalMode("feature_eval_mode", NCB::EFeatureEvalMode::OneVsNone)
    , EvalFeatureFileName("eval_feature_file", "")
    , Offset("offset", 0)
    , FoldCount("fold_count", 0)
    , FoldSizeUnit("fold_size_unit", ESamplingUnit::Object)
    , FoldSize("fold_size", 0)
{
}

void NCatboostOptions::TFeatureEvalOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(
        options, &FeaturesToEvaluate, &FeatureEvalMode, &EvalFeatureFileName,
        &Offset, &FoldCount, &FoldSizeUnit, &FoldSize);
}

void NCatboostOptions::TFeatureEvalOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(
        options, FeaturesToEvaluate, FeatureEvalMode, EvalFeatureFileName,
        Offset, FoldCount, FoldSizeUnit, FoldSize);
}

bool NCatboostOptions::TFeatureEvalOptions::operator==(const TFeatureEvalOptions& rhs) const {
    const auto& options = std::tie(
        FeaturesToEvaluate, FeatureEvalMode,
        Offset, FoldCount, FoldSizeUnit, FoldSize);
    const auto& rhsOptions = std::tie(
        rhs.FeaturesToEvaluate, rhs.FeatureEvalMode,
        rhs.Offset, rhs.FoldCount, rhs.FoldSizeUnit, rhs.FoldSize);
    return options == rhsOptions;
}

bool NCatboostOptions::TFeatureEvalOptions::operator!=(const TFeatureEvalOptions& rhs) const {
    return !(rhs == *this);
}
