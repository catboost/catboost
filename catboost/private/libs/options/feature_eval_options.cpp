#include "feature_eval_options.h"
#include "json_helper.h"

#include <cmath>

NCatboostOptions::TFeatureEvalOptions::TFeatureEvalOptions()
    : FeaturesToEvaluate("features_to_evaluate", TVector<TVector<ui32>>())
    , FeatureEvalMode("feature_eval_mode", NCB::EFeatureEvalMode::OneVsNone)
    , EvalFeatureFileName("eval_feature_file", "")
    , ProcessorsUsageFileName("processors_usage_file", "")
    , Offset("offset", 0)
    , FoldCount("fold_count", 0)
    , FoldSizeUnit("fold_size_unit", ESamplingUnit::Object)
    , FoldSize("fold_size", 0)
    , RelativeFoldSize("relative_fold_size", 0.0f)
    , TimeSplitQuantile("timesplit_quantile", 0.5)
{
}

void NCatboostOptions::TFeatureEvalOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(
        options, &FeaturesToEvaluate, &FeatureEvalMode, &EvalFeatureFileName, &ProcessorsUsageFileName,
        &Offset, &FoldCount, &FoldSizeUnit, &FoldSize, &RelativeFoldSize, &TimeSplitQuantile);
}

void NCatboostOptions::TFeatureEvalOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(
        options, FeaturesToEvaluate, FeatureEvalMode, EvalFeatureFileName, ProcessorsUsageFileName,
        Offset, FoldCount, FoldSizeUnit, FoldSize, RelativeFoldSize, TimeSplitQuantile);
}

bool NCatboostOptions::TFeatureEvalOptions::operator==(const TFeatureEvalOptions& rhs) const {
    const auto& options = std::tie(
        FeaturesToEvaluate, FeatureEvalMode,
        Offset, FoldCount, FoldSizeUnit, FoldSize, RelativeFoldSize, TimeSplitQuantile);
    const auto& rhsOptions = std::tie(
        rhs.FeaturesToEvaluate, rhs.FeatureEvalMode,
        rhs.Offset, rhs.FoldCount, rhs.FoldSizeUnit, rhs.FoldSize, rhs.RelativeFoldSize, rhs.TimeSplitQuantile);
    return options == rhsOptions;
}

bool NCatboostOptions::TFeatureEvalOptions::operator!=(const TFeatureEvalOptions& rhs) const {
    return !(rhs == *this);
}
