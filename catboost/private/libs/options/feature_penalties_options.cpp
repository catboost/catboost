#include "feature_penalties_options.h"

#include "json_helper.h"
#include "parse_per_feature_options.h"

using namespace NCB;
using namespace NJson;
using namespace NCatboostOptions;

namespace NCatboostOptions {
    void NCatboostOptions::TFeaturePenaltiesOptions::Load(const NJson::TJsonValue& options) {
        CheckedLoad(options, &FeatureWeights, &PenaltiesCoefficient, &FirstFeatureUsePenalty, &PerObjectFeaturePenalty);
    }

    void NCatboostOptions::TFeaturePenaltiesOptions::Save(NJson::TJsonValue* options) const {
        SaveFields(options, FeatureWeights, PenaltiesCoefficient, FirstFeatureUsePenalty, PerObjectFeaturePenalty);
    }

    static constexpr auto nonnegativeFloatRegex = TStringBuf("([+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)");

    static void LeaveOnlyNonTrivialOptions(const float defaultValue, TJsonValue* penaltiesJsonOptions) {
        TJsonValue nonTrivialOptions(EJsonValueType::JSON_MAP);
        const auto& optionsRefMap = penaltiesJsonOptions->GetMapSafe();
        for (const auto& [feature, option] : optionsRefMap) {
            if (option.GetDoubleRobust() != defaultValue) {
                nonTrivialOptions[feature] = option;
            }
        }
        *penaltiesJsonOptions = nonTrivialOptions;
    }

    static void ConvertFeaturePenaltiesToCanonicalFormat(
        const TStringBuf optionName,
        const float defaultValue,
        NJson::TJsonValue* featurePenaltiesJsonOptions
    ) {
        ConvertFeatureOptionsToCanonicalFormat<float>(optionName, nonnegativeFloatRegex, featurePenaltiesJsonOptions);
        LeaveOnlyNonTrivialOptions(defaultValue, featurePenaltiesJsonOptions);
    }

    void ConvertAllFeaturePenaltiesToCanonicalFormat(NJson::TJsonValue* penaltiesOptions) {
        TJsonValue& penaltiesRef = *penaltiesOptions;

        if (penaltiesRef.Has("feature_weights")) {
            ConvertFeaturePenaltiesToCanonicalFormat(
                "feature_weights",
                DEFAULT_FEATURE_WEIGHT,
                &penaltiesRef["feature_weights"]
            );
        }
        if (penaltiesRef.Has("first_feature_use_penalties")) {
            ConvertFeaturePenaltiesToCanonicalFormat(
                "first_feature_use_penalties",
                DEFAULT_FEATURE_PENALTY,
                &penaltiesRef["first_feature_use_penalties"]
            );
        }
        if (penaltiesRef.Has("per_object_feature_penalties")) {
            ConvertFeaturePenaltiesToCanonicalFormat(
                "per_object_feature_penalties",
                DEFAULT_FEATURE_PENALTY,
                &penaltiesRef["per_object_feature_penalties"]
            );
        }
    }

    static void ValidateFeatureSinglePenaltiesOption(const TPerFeaturePenalty& options, const TString& featureName) {
        for (auto [featureIdx, value] : options) {
            CB_ENSURE(value >= 0, "Values in " << featureName << " should be nonnegative. Got: " << featureIdx << ":" << value);
        }
    }

    void ValidateFeaturePenaltiesOptions(const TFeaturePenaltiesOptions& options) {
        const TPerFeaturePenalty& featureWeights = options.FeatureWeights.Get();
        if (!featureWeights.empty()) {
            ValidateFeatureSinglePenaltiesOption(featureWeights, "feature_weights");
        }
        const TPerFeaturePenalty& firstFeatureUsePenalties = options.FirstFeatureUsePenalty.Get();
        if (!firstFeatureUsePenalties.empty()) {
            ValidateFeatureSinglePenaltiesOption(firstFeatureUsePenalties, "first_feature_use_penalties");
        }
    }

    TVector<float> ExpandFeatureWeights(const TFeaturePenaltiesOptions& options, size_t featureCount) {
        TVector<float> featureWeights(featureCount, 1.0f);
        for (const auto& [feature, weight] : options.FeatureWeights.Get()) {
            CB_ENSURE(
                feature < featureCount,
                "Feature index " << feature << " exceeds feature count " << featureCount);
            featureWeights[feature] = weight;
        }
        return featureWeights;
    }
}
