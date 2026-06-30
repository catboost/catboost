#include "per_cpu.h"

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>

#include <cstdio>
#endif

#include <mutex>

namespace NYT::NRseq {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

int ParsePossibleCpuCount(std::string_view list)
{
    // The list enumerates CPU id ranges (e.g. "0-3,8-11"); the highest id + 1 is nr_cpu_ids,
    // the exclusive upper bound for the rseq cpu_id. This differs from _SC_NPROCESSORS_CONF
    // (a popcount) on sparse topologies: "0-3,8-11" yields 12 here but a count of 8.
    int maxId = -1;
    for (size_t index = 0; index < list.size();) {
        if (list[index] < '0' || list[index] > '9') {
            ++index;
            continue;
        }
        int value = 0;
        while (index < list.size() && list[index] >= '0' && list[index] <= '9') {
            value = value * 10 + (list[index] - '0');
            ++index;
        }
        if (value > maxId) {
            maxId = value;
        }
    }
    return maxId >= 0 ? maxId + 1 : -1;
}

#if defined(__linux__)

//! Reads /sys/devices/system/cpu/possible and returns nr_cpu_ids, or -1 if it cannot be read.
static int TryReadPossibleCpuCount()
{
    auto* file = std::fopen("/sys/devices/system/cpu/possible", "re");
    if (!file) {
        return -1;
    }
    char buffer[256] = {};
    size_t size = std::fread(buffer, 1, sizeof(buffer) - 1, file);
    std::fclose(file);
    return ParsePossibleCpuCount(std::string_view(buffer, size));
}

#endif

// Published by GetCpuCount(); see the declaration in per_cpu-inl.h. Defaults to 0 so the
// fast path's bounds check sends every update to the safe atomic fallback until the size is
// known.
constinit int CpuCount = 0;

int GetFallbackCpuId()
{
#if defined(__linux__)
    int cpuId = ::sched_getcpu();
    if (cpuId < 0) {
        return 0;
    }
    int cpuCount = GetCpuCount();
    // Defensive: keep the index in range even if a CPU came online beyond the configured
    // count. On the fallback path slots are touched atomically, so a shared slot is safe.
    return cpuId < cpuCount ? cpuId : cpuId % cpuCount;
#else
    return 0;
#endif
}

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

int GetCpuCount()
{
    static std::once_flag OnceFlag;
    std::call_once(OnceFlag, [] {
        int cpuCount = 1;
#if defined(__linux__)
        // The fast path indexes the slot array by the raw rseq cpu_id, so size to the highest
        // CPU id the kernel can report plus one (nr_cpu_ids), not merely the number of CPUs;
        // the possible-CPU bitmap gives this exact bound, and then every cpu_id is in range.
        if (int possible = NDetail::TryReadPossibleCpuCount(); possible > 0) {
            cpuCount = possible;
        } else {
            // Bitmap unavailable (e.g. /sys masked in a container): _SC_NPROCESSORS_CONF is a
            // count, not a cpu_id bound, so on a sparse topology it may be smaller than some
            // cpu_id. The fast path's bounds check then routes those CPUs to the clamped
            // atomic fallback -- still memory-safe, though a clamped slot may mix atomic and
            // non-atomic writes (at worst a lost counter update on such exotic setups).
            int configured = static_cast<int>(::sysconf(_SC_NPROCESSORS_CONF));
            cpuCount = configured > 0 ? configured : 1;
        }
#endif
        // Publish for the fast-path bounds check before any update can index the array.
        NDetail::CpuCount = cpuCount;
    });
    return NDetail::CpuCount;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq
