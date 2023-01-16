#pragma once

#include "enums.h"
#include "option.h"

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    class TFeaturesSelectOptions {
    public:
        TFeaturesSelectOptions();

        bool operator==(const TFeaturesSelectOptions& rhs) const;
        bool operator!=(const TFeaturesSelectOptions& rhs) const;

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        void CheckAndUpdateSteps();

    public:
        TOption<TVector<ui32>> FeaturesForSelect;
        TOption<int> NumberOfFeaturesToSelect;
        TOption<int> Steps;
        TOption<bool> TrainFinalModel;
        TOption<TString> ResultPath;
        TOption<NCB::EFeaturesSelectionAlgorithm> Algorithm;
        TOption<ECalcTypeShapValues> ShapCalcType;
    };
}
