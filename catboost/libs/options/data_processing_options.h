#pragma once

#include "option.h"
#include "enums.h"
#include "binarization_options.h"
#include "unimplemented_aware_option.h"

#include <library/grid_creator/binarization.h>

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

        TOption<TVector<ui32>> IgnoredFeatures;
        TOption<bool> HasTimeFlag;
        TOption<bool> AllowConstLabel;
        TOption<TBinarizationOptions> FloatFeaturesBinarization;
        TOption<ui32> ClassesCount;
        TOption<TVector<float>> ClassWeights;
        TOption<TVector<TString>> ClassNames;
        TGpuOnlyOption<EGpuCatFeaturesStorage> GpuCatFeaturesStorage;
    };
}
