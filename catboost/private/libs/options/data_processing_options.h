#pragma once

#include "option.h"
#include "enums.h"
#include "binarization_options.h"
#include "text_processing_options.h"
#include "embedding_processing_options.h"
#include "unimplemented_aware_option.h"

#include <catboost/libs/helpers/sparse_array.h>

#include <library/cpp/grid_creator/binarization.h>
#include <library/cpp/json/json_value.h>

#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NCatboostOptions {
    struct TDataProcessingOptions {
        explicit TDataProcessingOptions(ETaskType type);

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        bool operator==(const TDataProcessingOptions& rhs) const;
        bool operator!=(const TDataProcessingOptions& rhs) const;

        void Validate() const;

        TOption<TVector<ui32>> IgnoredFeatures;
        TOption<bool> HasTimeFlag;
        TOption<bool> AllowConstLabel;
        TOption<TMaybe<float>> TargetBorder;
        TOption<TBinarizationOptions> FloatFeaturesBinarization;
        TOption<TMap<ui32, TBinarizationOptions>> PerFloatFeatureQuantization;
        TOption<TTextProcessingOptions> TextProcessingOptions;
        TOption<TEmbeddingProcessingOptions> EmbeddingProcessingOptions;
        TOption<ui32> ClassesCount;
        TOption<TVector<float>> ClassWeights;
        TOption<EAutoClassWeightsType> AutoClassWeights;
        TOption<TVector<NJson::TJsonValue>> ClassLabels; // can be Integers, Floats or Strings

        TOption<float> DevDefaultValueFractionToEnableSparseStorage; // 0 means sparse storage is disabled
        TOption<NCB::ESparseArrayIndexingType> DevSparseArrayIndexingType;
        TOption<bool> ForceUnitAutoPairWeights;
        TOption<float> EvalFraction;

        TGpuOnlyOption<EGpuCatFeaturesStorage> GpuCatFeaturesStorage;
        TCpuOnlyOption<bool> DevLeafwiseScoring;
        TCpuOnlyOption<bool> DevGroupFeatures;
    private:
        void SetPerFeatureMissingSettingToCommonValues();
    };

    constexpr float GetDefaultTargetBorder() {
        return 0.5;
    }

    constexpr float GetDefaultPredictionBorder() {
        return 0.5;
    }
}
