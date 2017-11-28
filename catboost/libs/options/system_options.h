#pragma once

#include "option.h"
#include "json_helper.h"

#include <catboost/libs/logging/logging_level.h>
#include <util/system/types.h>
#include <library/json/json_value.h>

namespace NCatboostOptions {
    struct TSystemOptions {
        explicit TSystemOptions(ETaskType taskType)
            : NumThreads("thread_count", 8)
            , CpuUsedRamLimit("used_ram_limit", Max<ui64>(), taskType)
            , Devices("devices", "-1", taskType)
            , GpuRamPart("gpu_ram_part", 0.95, taskType)
            , PinnedMemorySize("pinned_memory_bytes", 104857600, taskType)
        {
            CpuUsedRamLimit.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
            Devices.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
            GpuRamPart.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
            PinnedMemorySize.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options, &NumThreads, &CpuUsedRamLimit, &Devices, &GpuRamPart);
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, NumThreads, CpuUsedRamLimit, Devices, GpuRamPart);
        }

        bool operator==(const TSystemOptions& rhs) const {
            return std::tie(NumThreads, CpuUsedRamLimit, Devices, GpuRamPart) ==
                   std::tie(rhs.NumThreads, rhs.CpuUsedRamLimit, rhs.Devices,
                            rhs.GpuRamPart);
        }

        bool operator!=(const TSystemOptions& rhs) const {
            return !(rhs == *this);
        }

        void Validate() const {
            CB_ENSURE(NumThreads > 0, "thread count should be positive");
            CB_ENSURE(GpuRamPart.GetUnchecked() > 0 && GpuRamPart.GetUnchecked() <= 1.0, "GPU ram part should be in (0, 1]");
        }

        TOption<ui32> NumThreads;
        TCpuOnlyOption<ui64> CpuUsedRamLimit;
        TGpuOnlyOption<TString> Devices;
        TGpuOnlyOption<double> GpuRamPart;
        TGpuOnlyOption<ui64> PinnedMemorySize;
    };
}
