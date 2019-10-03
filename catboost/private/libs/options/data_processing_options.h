#pragma once

#include "option.h"
#include "enums.h"
#include "binarization_options.h"
#include "text_processing_options.h"
#include "unimplemented_aware_option.h"

#include <catboost/libs/helpers/sparse_array.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NJson {
    class TJsonValue;
}

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
        TOption<ui32> ClassesCount;
        TOption<TVector<float>> ClassWeights;
        TOption<TVector<TString>> ClassNames;

        TOption<float> DevDefaultValueFractionToEnableSparseStorage; // 0 means sparse storage is disabled
        TOption<NCB::ESparseArrayIndexingType> DevSparseArrayIndexingType;

        TGpuOnlyOption<EGpuCatFeaturesStorage> GpuCatFeaturesStorage;
    private:
        void SetPerFeatureMissingSettingToCommonValues();
    };

    constexpr float GetDefaultTargetBorder() {
        return 0.5;
    }
}
