#pragma once

#include "enums.h"
#include "option.h"

#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    struct TFeatureEvalOptions {
        explicit TFeatureEvalOptions();

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        bool operator==(const TFeatureEvalOptions& rhs) const;
        bool operator!=(const TFeatureEvalOptions& rhs) const;

        TOption<TVector<TVector<ui32>>> FeaturesToEvaluate;
        TOption<NCB::EFeatureEvalMode> FeatureEvalMode;
        TOption<TString> EvalFeatureFileName;
        TOption<TString> ProcessorsUsageFileName;
        TOption<ui32> Offset;
        TOption<ui32> FoldCount;
        TOption<ESamplingUnit> FoldSizeUnit;
        TOption<ui32> FoldSize;
        TOption<float> RelativeFoldSize;
        TOption<double> TimeSplitQuantile;
    };
}
