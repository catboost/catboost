#pragma once

#include "enums.h"
#include "option.h"
#include "unimplemented_aware_option.h"

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    struct TSystemOptions {
        explicit TSystemOptions(ETaskType taskType);

        void Load(const NJson::TJsonValue& options);
        void Save(NJson::TJsonValue* options) const;

        bool operator==(const TSystemOptions& rhs) const;
        bool operator!=(const TSystemOptions& rhs) const;

        void Validate() const;

        TOption<ui32> NumThreads;
        TOption<TString> CpuUsedRamLimit;
        TGpuOnlyOption<TString> Devices;
        TGpuOnlyOption<double> GpuRamPart;
        TGpuOnlyOption<TString> PinnedMemorySize;

        TCpuOnlyOption<ENodeType> NodeType;
        TCpuOnlyOption<TString> FileWithHosts;
        TCpuOnlyOption<ui32> NodePort;

        static ui32 GetUnusedNodePort() { return 0; }
        bool IsMaster() const;
        bool IsSingleHost() const;
        bool IsWorker() const;
    };
}

ui64 ParseMemorySizeDescription(TStringBuf memSizeDescription);
