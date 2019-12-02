#pragma once

#include "enums.h"
#include "option.h"
#include "unimplemented_aware_option.h"

#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    struct TModelBasedEvalOptions {
        explicit TModelBasedEvalOptions(ETaskType taskType);

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        bool operator==(const TModelBasedEvalOptions& rhs) const;
        bool operator!=(const TModelBasedEvalOptions& rhs) const;

        void Validate() const;

        TOption<TVector<TVector<ui32>>> FeaturesToEvaluate;
        TOption<TString> BaselineModelSnapshot;
        TOption<int> Offset;
        TOption<int> ExperimentCount;
        TOption<int> ExperimentSize;
        TOption<bool> UseEvaluatedFeaturesInBaselineModel;
    };

    TString GetExperimentName(ui32 featureSetIdx, ui32 foldIdx);
}
