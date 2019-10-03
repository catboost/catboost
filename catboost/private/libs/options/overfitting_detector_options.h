#pragma once

#include "enums.h"
#include "option.h"

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    class TOverfittingDetectorOptions {
    public:
        TOverfittingDetectorOptions();

        bool operator==(const TOverfittingDetectorOptions& rhs) const;
        bool operator!=(const TOverfittingDetectorOptions& rhs) const;

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        void Validate() const;

        TOption<float> AutoStopPValue;
        TOption<EOverfittingDetectorType> OverfittingDetectorType;
        TOption<int> IterationsWait;
    };
}
