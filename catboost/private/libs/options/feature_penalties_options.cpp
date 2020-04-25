#include "feature_penalties_options.h"

#include "json_helper.h"
#include "parse_per_feature_options.h"

using namespace NCB;
using namespace NJson;
using namespace NCatboostOptions;

namespace NCatboostOptions {
    void NCatboostOptions::TFeaturePenaltiesOptions::Load(const NJson::TJsonValue& options) {
        CheckedLoad(options, &FeatureWeights, &PenaltiesCoefficient, &FirstFeatureUsePenalty);
    }

    void NCatboostOptions::TFeaturePenaltiesOptions::Save(NJson::TJsonValue* options) const {
        SaveFields(options, FeatureWeights, PenaltiesCoefficient, FirstFeatureUsePenalty);
    }

    static constexpr auto nonnegativeFloatRegex = AsStringBuf("([+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)");

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

    void ConvertAllFeaturePenaltiesToCanonicalFormat(NJson::TJsonValue* catBoostJsonOptions) {
        auto& treeOptions = (*catBoostJsonOptions)["tree_learner_options"];
        if (!treeOptions.Has("penalties")) {
            return;
        }

        TJsonValue& penaltiesRef = treeOptions["penalties"];

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
    }
}
