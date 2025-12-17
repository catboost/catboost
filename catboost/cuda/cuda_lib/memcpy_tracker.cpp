#include "memcpy_tracker.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/string.h>
#include <util/stream/format.h>
#include <util/stream/str.h>
#include <util/string/cast.h>
#include <util/system/backtrace.h>
#include <util/system/env.h>

#include <cstdint>

#if !defined(_win_)
#include <dlfcn.h>
#endif

namespace {
    static bool ParseBoolEnv(const TString& value, bool defaultValue) {
        if (value.empty()) {
            return defaultValue;
        }
        if ((value == "1") || (value == "true") || (value == "TRUE") || (value == "yes") || (value == "YES")) {
            return true;
        }
        if ((value == "0") || (value == "false") || (value == "FALSE") || (value == "no") || (value == "NO")) {
            return false;
        }
        return defaultValue;
    }

    static TString FormatBackTraceWithModuleOffsets() {
        void* frames[64];
        const size_t frameCount = BackTrace(frames, 64);

        TStringStream ss;
        for (size_t i = 0; i < frameCount; ++i) {
#if !defined(_win_)
            Dl_info info;
            if (dladdr(frames[i], &info) && info.dli_fname && info.dli_fbase) {
                const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
                const auto addr = reinterpret_cast<uintptr_t>(frames[i]);
                ss << info.dli_fname << "+" << Hex(addr - base, HF_ADDX);
                if (info.dli_sname) {
                    ss << " " << info.dli_sname;
                }
                ss << "\n";
            } else {
                ss << Hex(reinterpret_cast<uintptr_t>(frames[i]), HF_ADDX) << "\n";
            }
#else
            ss << Hex(reinterpret_cast<uintptr_t>(frames[i]), HF_ADDX) << "\n";
#endif
        }
        return ss.Str();
    }
}

NCudaLib::TMemcpyTracker& NCudaLib::TMemcpyTracker::Instance() {
    static TMemcpyTracker tracker;
    return tracker;
}

void NCudaLib::TMemcpyTracker::ResetConfig() {
    Configured.store(false);
}

void NCudaLib::TMemcpyTracker::ResetStats() {
    HostToHostBytes.store(0);
    HostToDeviceBytes.store(0);
    DeviceToHostBytes.store(0);
    DeviceToDeviceBytes.store(0);
    UnknownBytes.store(0);
}

NCudaLib::TMemcpyStats NCudaLib::TMemcpyTracker::GetStats() const {
    return {
        HostToHostBytes.load(),
        HostToDeviceBytes.load(),
        DeviceToHostBytes.load(),
        DeviceToDeviceBytes.load(),
        UnknownBytes.load()
    };
}

void NCudaLib::TMemcpyTracker::EnsureConfigured() {
    if (Configured.load()) {
        return;
    }

    {
        std::lock_guard<std::mutex> guard(ConfigLock);
        if (Configured.load()) {
            return;
        }

        const bool strictNoD2H = ParseBoolEnv(GetEnv("CATBOOST_CUDA_STRICT_NO_D2H", ""), false);
        StrictNoD2H.store(strictNoD2H);

        // Enabled by default only in tests/benchmarks, but strict mode implies enabled tracking.
        const bool enabled = ParseBoolEnv(GetEnv("CATBOOST_CUDA_MEMCPY_TRACK", ""), false) || strictNoD2H;
        Enabled.store(enabled);

        ui64 limit = 0;
        const auto limitStr = GetEnv("CATBOOST_CUDA_D2H_BYTES_LIMIT", "");
        const bool hasLimit = !limitStr.empty();
        if (hasLimit) {
            limit = FromString<ui64>(limitStr);
        }
        HasDeviceToHostBytesLimit.store(hasLimit);
        DeviceToHostBytesLimit.store(limit);

        ui64 singleLimit = 0;
        const auto singleLimitStr = GetEnv("CATBOOST_CUDA_D2H_SINGLE_BYTES_LIMIT", "");
        const bool hasSingleLimit = !singleLimitStr.empty();
        if (hasSingleLimit) {
            singleLimit = FromString<ui64>(singleLimitStr);
        }
        HasDeviceToHostSingleBytesLimit.store(hasSingleLimit);
        DeviceToHostSingleBytesLimit.store(singleLimit);

        Configured.store(true);
    }
}

void NCudaLib::TMemcpyTracker::RecordMemcpyAsync(const void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    EnsureConfigured();
    if (!Enabled.load()) {
        return;
    }

    EMemcpyDirection direction = EMemcpyDirection::Unknown;
    switch (kind) {
        case cudaMemcpyHostToHost:
            direction = EMemcpyDirection::HostToHost;
            break;
        case cudaMemcpyHostToDevice:
            direction = EMemcpyDirection::HostToDevice;
            break;
        case cudaMemcpyDeviceToHost:
            direction = EMemcpyDirection::DeviceToHost;
            break;
        case cudaMemcpyDeviceToDevice:
            direction = EMemcpyDirection::DeviceToDevice;
            break;
        case cudaMemcpyDefault:
            direction = ClassifyMemcpyDefault(dst, src);
            break;
        default:
            direction = EMemcpyDirection::Unknown;
            break;
    }

    Record(direction, static_cast<ui64>(bytes));
}

void NCudaLib::TMemcpyTracker::Record(EMemcpyDirection direction, ui64 bytes) {
    if (bytes == 0) {
        return;
    }

    switch (direction) {
        case EMemcpyDirection::HostToHost:
            HostToHostBytes.fetch_add(bytes);
            break;
        case EMemcpyDirection::HostToDevice:
            HostToDeviceBytes.fetch_add(bytes);
            break;
        case EMemcpyDirection::DeviceToHost: {
            const ui64 newTotal = DeviceToHostBytes.fetch_add(bytes) + bytes;
            if (StrictNoD2H.load()) {
                const bool hasSingleLimit = HasDeviceToHostSingleBytesLimit.load();
                const bool hasLimit = HasDeviceToHostBytesLimit.load();

                const ui64 singleLimit = DeviceToHostSingleBytesLimit.load();
                const ui64 limit = DeviceToHostBytesLimit.load();

                const bool singleLimitViolated = hasSingleLimit && (bytes > singleLimit);
                const bool limitViolated = hasLimit && (newTotal > limit);
                const bool anyD2HForbidden = !hasSingleLimit && !hasLimit;

                if (singleLimitViolated || limitViolated || anyD2HForbidden) {
                    const auto backtrace = FormatBackTraceWithModuleOffsets();
                    CB_ENSURE(
                        false,
                        "CATBOOST_CUDA_STRICT_NO_D2H=1: detected device->host copy, cumulative bytes="
                            << newTotal
                            << ", total_limit=" << (hasLimit ? ToString(limit) : TString("unset"))
                            << ", single_limit=" << (hasSingleLimit ? ToString(singleLimit) : TString("unset"))
                            << ", last_copy_bytes=" << bytes
                            << "\nBacktrace:\n" << backtrace
                    );
                }
            }
            break;
        }
        case EMemcpyDirection::DeviceToDevice:
            DeviceToDeviceBytes.fetch_add(bytes);
            break;
        case EMemcpyDirection::Unknown:
            UnknownBytes.fetch_add(bytes);
            break;
    }
}

NCudaLib::EMemcpyDirection NCudaLib::TMemcpyTracker::ClassifyMemcpyDefault(const void* dst, const void* src) const {
    const bool dstIsDevice = IsDevicePointer(dst);
    const bool srcIsDevice = IsDevicePointer(src);

    if (!dstIsDevice && !srcIsDevice) {
        return EMemcpyDirection::HostToHost;
    }
    if (dstIsDevice && !srcIsDevice) {
        return EMemcpyDirection::HostToDevice;
    }
    if (!dstIsDevice && srcIsDevice) {
        return EMemcpyDirection::DeviceToHost;
    }
    if (dstIsDevice && srcIsDevice) {
        return EMemcpyDirection::DeviceToDevice;
    }
    return EMemcpyDirection::Unknown;
}

bool NCudaLib::TMemcpyTracker::IsDevicePointer(const void* ptr) {
    if (ptr == nullptr) {
        return false;
    }

    cudaPointerAttributes attributes;
    const auto status = cudaPointerGetAttributes(&attributes, ptr);
    if (status != cudaSuccess) {
        // Most likely a host pointer; clear the error state.
        cudaGetLastError();
        return false;
    }

#if (CUDART_VERSION >= 10000)
    return (attributes.type == cudaMemoryTypeDevice) || (attributes.type == cudaMemoryTypeManaged);
#else
    return (attributes.memoryType == cudaMemoryTypeDevice);
#endif
}
