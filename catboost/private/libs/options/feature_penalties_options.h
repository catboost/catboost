#pragma once

#include "enums.h"
#include "option.h"
#include "unimplemented_aware_option.h"

#include <util/generic/map.h>

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    constexpr float DEFAULT_FEATURE_WEIGHT = 1;
    constexpr float DEFAULT_FEATURE_PENALTY = 0;

    using TPerFeaturePenalty = TMap<ui32, float>;

    struct TFeaturePenaltiesOptions {
    public:
        TFeaturePenaltiesOptions()
            : FeatureWeights("feature_weights", {})
            , PenaltiesCoefficient("penalties_coefficient", 1, ETaskType::CPU)
            , FirstFeatureUsePenalty("first_feature_use_penalties", {}, ETaskType::CPU)
            , PerObjectFeaturePenalty("per_object_feature_penalties", {}, ETaskType::CPU)
        {
        }

        bool operator==(const TFeaturePenaltiesOptions& rhs) const {
            return std::tie(FeatureWeights, PenaltiesCoefficient, FirstFeatureUsePenalty, PerObjectFeaturePenalty) ==
                std::tie(rhs.FeatureWeights, PenaltiesCoefficient, FirstFeatureUsePenalty, PerObjectFeaturePenalty);
        }

        bool operator!=(const TFeaturePenaltiesOptions& rhs) const {
            return !(rhs == *this);
        }

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        TOption<TPerFeaturePenalty> FeatureWeights;
        TCpuOnlyOption<float> PenaltiesCoefficient;
        TCpuOnlyOption<TPerFeaturePenalty> FirstFeatureUsePenalty;
        TCpuOnlyOption<TPerFeaturePenalty> PerObjectFeaturePenalty;
    };

    void ConvertAllFeaturePenaltiesToCanonicalFormat(NJson::TJsonValue* catBoostJsonOptions);

    void ValidateFeaturePenaltiesOptions(const TFeaturePenaltiesOptions& options);

    TVector<float> ExpandFeatureWeights(const TFeaturePenaltiesOptions& options, size_t featureCount);
}
