#pragma once

#include <cuda_runtime_api.h>

#include <util/system/types.h>

#include <atomic>
#include <mutex>

namespace NCudaLib {

    enum class EMemcpyDirection {
        Unknown,
        HostToHost,
        HostToDevice,
        DeviceToHost,
        DeviceToDevice
    };

    struct TMemcpyStats {
        ui64 HostToHostBytes = 0;
        ui64 HostToDeviceBytes = 0;
        ui64 DeviceToHostBytes = 0;
        ui64 DeviceToDeviceBytes = 0;
        ui64 UnknownBytes = 0;

        ui64 TotalBytes() const {
            return HostToHostBytes + HostToDeviceBytes + DeviceToHostBytes + DeviceToDeviceBytes + UnknownBytes;
        }
    };

    // Lightweight optional instrumentation for memcpy calls initiated by CatBoost.
    // Intended for tests/benchmarks and guarded by env vars.
    class TMemcpyTracker final {
    public:
        static TMemcpyTracker& Instance();

        // Re-reads env vars on next Record call.
        void ResetConfig();

        void ResetStats();
        TMemcpyStats GetStats() const;

        // Hook points used by CatBoost wrappers around cudaMemcpyAsync.
        void RecordMemcpyAsync(const void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);

    private:
        TMemcpyTracker() = default;
        void EnsureConfigured();

        void Record(EMemcpyDirection direction, ui64 bytes);
        EMemcpyDirection ClassifyMemcpyDefault(const void* dst, const void* src) const;

        static bool IsDevicePointer(const void* ptr);

    private:
        mutable std::atomic<bool> Configured{false};
        mutable std::mutex ConfigLock;

        std::atomic<bool> Enabled{false};
        std::atomic<bool> StrictNoD2H{false};
        // Strict-no-D2H can be configured either as a cumulative budget or as a per-memcpy limit.
        // If StrictNoD2H is true and neither limit is explicitly set, any D2H copy fails.
        std::atomic<bool> HasDeviceToHostBytesLimit{false};
        std::atomic<ui64> DeviceToHostBytesLimit{0};
        std::atomic<bool> HasDeviceToHostSingleBytesLimit{false};
        std::atomic<ui64> DeviceToHostSingleBytesLimit{0};

        std::atomic<ui64> HostToHostBytes{0};
        std::atomic<ui64> HostToDeviceBytes{0};
        std::atomic<ui64> DeviceToHostBytes{0};
        std::atomic<ui64> DeviceToDeviceBytes{0};
        std::atomic<ui64> UnknownBytes{0};
    };

}
