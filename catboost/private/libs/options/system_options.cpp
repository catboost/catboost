#include "system_options.h"

#include <catboost/private/libs/options/json_helper.h>

#include <util/charset/utf8.h>
#include <util/string/cast.h>
#include <util/system/info.h>

using namespace NCatboostOptions;

TSystemOptions::TSystemOptions(ETaskType taskType)
    : NumThreads("thread_count", NSystemInfo::CachedNumberOfCpus())
    , CpuUsedRamLimit("used_ram_limit", {})
    , Devices("devices", "-1", taskType)
    , GpuRamPart("gpu_ram_part", 0.95, taskType)
    , PinnedMemorySize("pinned_memory_bytes", "104857600", taskType)
    , NodeType("node_type", ENodeType::SingleHost, taskType)
    , FileWithHosts("file_with_hosts", "hosts.txt", taskType)
    , NodePort("node_port", GetUnusedNodePort(), taskType)
{
    Devices.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
    GpuRamPart.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
    PinnedMemorySize.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
}

void TSystemOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &NumThreads, &CpuUsedRamLimit, &Devices, &GpuRamPart, &PinnedMemorySize, &NodeType, &FileWithHosts, &NodePort);
}

void TSystemOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, NumThreads, CpuUsedRamLimit, Devices, GpuRamPart, PinnedMemorySize, NodeType, FileWithHosts, NodePort);
}

bool TSystemOptions::operator==(const TSystemOptions& rhs) const {
    return std::tie(NumThreads, CpuUsedRamLimit, Devices,
                    GpuRamPart, PinnedMemorySize, NodeType, FileWithHosts, NodePort) ==
           std::tie(rhs.NumThreads, rhs.CpuUsedRamLimit, rhs.Devices,
                    rhs.GpuRamPart, rhs.PinnedMemorySize, rhs.NodeType, rhs.FileWithHosts, rhs.NodePort);
}

bool TSystemOptions::operator!=(const TSystemOptions& rhs) const {
    return !(rhs == *this);
}

void TSystemOptions::Validate() const {
    CB_ENSURE(NumThreads > 0, "thread count should be positive");
    CB_ENSURE(GpuRamPart.GetUnchecked() > 0 && GpuRamPart.GetUnchecked() <= 1.0, "GPU ram part should be in (0, 1]");
    ParseMemorySizeDescription(CpuUsedRamLimit.Get());
    ParseMemorySizeDescription(PinnedMemorySize.GetUnchecked());
}

bool TSystemOptions::IsMaster() const {
    return NodeType == ENodeType::Master;
}

bool TSystemOptions::IsSingleHost() const {
    return NodeType == ENodeType::SingleHost;
}

bool TSystemOptions::IsWorker() const {
    return NodeType == ENodeType::Master && FileWithHosts->empty();
}

static bool IsInfinity(const TStringBuf value) {
    static const TStringBuf examples[] = {
        "",
        "no", "none",
        "off",
        "inf", "infinity",
        "unlim", "unlimited"
    };
    for (const auto example : examples) {
        if (example == value) {
            return true;
        }
    }

    return false;
}

ui64 ParseMemorySizeDescription(const TStringBuf description) {
    char* suffixBegin = nullptr;
    const double number = StrToD(description.begin(), description.end(), &suffixBegin);
    if (suffixBegin > description.begin() && number >= 0) {
        // `number` is valid
        const auto suffix = to_lower(TString(suffixBegin, description.end()));
        if (suffix == "tb") {
            return static_cast<ui64>(number * (1ll << 40));
        } else if (suffix == "gb") {
            return static_cast<ui64>(number * (1ll << 30));
        } else if (suffix == "mb") {
            return static_cast<ui64>(number * (1ll << 20));
        } else if (suffix == "kb") {
            return static_cast<ui64>(number * (1ll << 10));
        } else if (suffix == "b" || suffix.empty()) {
            return static_cast<ui64>(number);
        }
    } else {
        if (IsInfinity(ToLowerUTF8(description))) {
            return Max<ui64>();
        }
    }
    CB_ENSURE(false, "incomprehensible memory size description: " << description);
}
