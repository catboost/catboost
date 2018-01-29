#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/option.h>
#include <catboost/libs/options/unimplemented_aware_option.h>
#include <library/json/json_value.h>

namespace NCatboostOptions {
    struct TSystemOptions {
        explicit TSystemOptions(ETaskType taskType);

        void Load(const NJson::TJsonValue& options);
        void Save(NJson::TJsonValue* options) const;

        bool operator==(const TSystemOptions& rhs) const;
        bool operator!=(const TSystemOptions& rhs) const;

        void Validate() const;

        TOption<ui32> NumThreads;
        TCpuOnlyOption<ui64> CpuUsedRamLimit;
        TGpuOnlyOption<TString> Devices;
        TGpuOnlyOption<double> GpuRamPart;
        TGpuOnlyOption<ui64> PinnedMemorySize;
    };
}
