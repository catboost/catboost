#include "clock.h"

#include <util/system/hp_timer.h>

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/misc/tls.h>

#if defined(_x86_64_) && defined(_linux_)
#include <algorithm>
#include <cstring>
#include <iterator>
#include <optional>
#include <string_view>

#include <cpuid.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include <linux/perf_event.h>

#include <sys/mman.h>
#include <sys/syscall.h>
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// Re-calibrate every 1B CPU ticks.
constexpr auto CalibrationCpuPeriod = 1'000'000'000;

struct TCalibrationState
{
    TCpuInstant CpuInstant;
    TInstant Instant;
};

namespace {

#if defined(_x86_64_) && defined(_linux_)

bool IsSaneTicksPerSecond(double ticksPerSecond)
{
    return ticksPerSecond >= 5e8 && ticksPerSecond <= 1e10;
}

bool HasInvariantTsc()
{
    ui32 eax, ebx, ecx, edx;
    if (!__get_cpuid(0x80000007, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    return edx & (1u << 8);
}

std::optional<double> TryGetTicksPerSecondFromCpuid()
{
    ui32 eax, ebx, ecx, edx;
    // Zero eax/ebx/ecx means "not enumerated".
    if (!__get_cpuid(0x15, &eax, &ebx, &ecx, &edx) || eax == 0 || ebx == 0 || ecx == 0) {
        return std::nullopt;
    }
    return static_cast<double>(ecx) * ebx / eax;
}

bool IsRunningUnderHypervisor()
{
    ui32 eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return ecx & (1u << 31);
}

std::optional<double> TryGetTicksPerSecondFromHypervisor()
{
    ui32 eax, ebx, ecx, edx;
    __cpuid(0x40000000, eax, ebx, ecx, edx);
    char signature[12];
    std::memcpy(signature + 0, &ebx, 4);
    std::memcpy(signature + 4, &ecx, 4);
    std::memcpy(signature + 8, &edx, 4);
    // Leaf 0x40000010 (TSC frequency in kHz) is defined by VMware and adopted by KVM.
    if (std::memcmp(signature, "KVMKVMKVM\0\0\0", 12) != 0 &&
        std::memcmp(signature, "VMwareVMware", 12) != 0)
    {
        return std::nullopt;
    }
    if (eax < 0x40000010) {
        return std::nullopt;
    }

    __cpuid(0x40000010, eax, ebx, ecx, edx);
    if (eax == 0) {
        return std::nullopt;
    }
    return eax * 1'000.0;
}

bool IsSeccompActive()
{
    int fd = ::open("/proc/self/status", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return true;
    }

    char buffer[8192];
    ssize_t size = 0;
    while (size < std::ssize(buffer) - 1) {
        auto bytesRead = ::read(fd, buffer + size, sizeof(buffer) - 1 - size);
        if (bytesRead <= 0) {
            break;
        }
        size += bytesRead;
    }
    ::close(fd);
    buffer[size] = '\0';

    constexpr std::string_view Key = "\nSeccomp:";
    const auto* line = std::strstr(buffer, Key.data());
    if (!line) {
        return true;
    }
    line += Key.size();
    while (*line == ' ' || *line == '\t') {
        ++line;
    }
    return *line != '0';
}

std::optional<double> TryGetTicksPerSecondFromPerf()
{
    // A seccomp filter may kill the process on perf_event_open.
    if (IsSeccompActive()) {
        return std::nullopt;
    }

    perf_event_attr attr{};
    attr.size = sizeof(attr);
    attr.type = PERF_TYPE_SOFTWARE;
    attr.config = PERF_COUNT_SW_DUMMY;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    auto fd = static_cast<int>(syscall(SYS_perf_event_open, &attr, /*pid*/ 0, /*cpu*/ -1, /*groupFd*/ -1, PERF_FLAG_FD_CLOEXEC));
    if (fd < 0) {
        return std::nullopt;
    }

    auto pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto* page = mmap(nullptr, pageSize, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (page == MAP_FAILED) {
        return std::nullopt;
    }

    const auto* header = static_cast<const perf_event_mmap_page*>(page);
    ui32 sequence;
    bool capUserTime;
    ui32 timeMult;
    ui16 timeShift;
    do {
        sequence = __atomic_load_n(&header->lock, __ATOMIC_ACQUIRE);
        capUserTime = header->cap_user_time;
        timeMult = header->time_mult;
        timeShift = header->time_shift;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
    } while ((sequence & 1) || __atomic_load_n(&header->lock, __ATOMIC_RELAXED) != sequence);
    munmap(page, pageSize);

    if (!capUserTime || timeMult == 0 || timeShift >= 64) {
        return std::nullopt;
    }
    return 1e9 * static_cast<double>(1ull << timeShift) / timeMult;
}

i64 GetRawMonotonicNanoseconds()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec * 1'000'000'000 + ts.tv_nsec;
}

double MeasureTicksPerSecondOnce()
{
    constexpr i64 WarmupNanoseconds = 20'000;
    constexpr i64 WindowNanoseconds = 1'000'000;

    // The first clock reads after idling are slow and would skew the starting point.
    auto warmupStartNanoseconds = GetRawMonotonicNanoseconds();
    while (GetRawMonotonicNanoseconds() - warmupStartNanoseconds < WarmupNanoseconds) { }

    auto startNanoseconds = GetRawMonotonicNanoseconds();
    auto startTicks = GetApproximateCpuInstant();
    i64 endNanoseconds;
    TCpuInstant endTicks;
    // Read ticks right after each clock read so both ends are measured alike.
    do {
        endNanoseconds = GetRawMonotonicNanoseconds();
        endTicks = GetApproximateCpuInstant();
    } while (endNanoseconds - startNanoseconds < WindowNanoseconds);

    return static_cast<double>(endTicks - startTicks) * 1e9 / (endNanoseconds - startNanoseconds);
}

std::optional<double> TryMeasureTicksPerSecond()
{
    constexpr int AttemptCount = 3;
    constexpr int SampleCount = 3;

    for (int attempt = 0; attempt < AttemptCount; ++attempt) {
        double samples[SampleCount];
        for (auto& sample : samples) {
            sample = MeasureTicksPerSecondOnce();
        }
        std::ranges::sort(samples);
        auto median = samples[SampleCount / 2];
        if (IsSaneTicksPerSecond(median)) {
            return median;
        }
    }
    return std::nullopt;
}

#endif

double ComputeTicksPerSecond()
{
#if defined(_x86_64_) && defined(_linux_)
    auto filterSane = [] (std::optional<double> ticksPerSecond) {
        return ticksPerSecond && IsSaneTicksPerSecond(*ticksPerSecond)
            ? ticksPerSecond
            : std::nullopt;
    };

    if (HasInvariantTsc()) {
        // Under a hypervisor, leaf 0x15 may describe the host rather than the guest TSC.
        auto ticksPerSecond = filterSane(IsRunningUnderHypervisor()
            ? TryGetTicksPerSecondFromHypervisor()
            : TryGetTicksPerSecondFromCpuid());
        if (ticksPerSecond) {
            return *ticksPerSecond;
        }
    }
    if (auto ticksPerSecond = filterSane(TryGetTicksPerSecondFromPerf())) {
        return *ticksPerSecond;
    }
    if (auto ticksPerSecond = TryMeasureTicksPerSecond()) {
        return *ticksPerSecond;
    }
#endif
    return NHPTimer::GetCyclesPerSecond();
}

} // namespace

double GetMicrosecondsToTicks()
{
    static const auto MicrosecondsToTicks = ComputeTicksPerSecond() / 1'000'000;
    return MicrosecondsToTicks;
}

double GetTicksToMicroseconds()
{
    static const auto TicksToMicroseconds = 1.0 / GetMicrosecondsToTicks();
    return TicksToMicroseconds;
}

YT_PREVENT_TLS_CACHING TCalibrationState GetCalibrationState(TCpuInstant cpuInstant)
{
    thread_local TCalibrationState State;

    if (State.CpuInstant + CalibrationCpuPeriod < cpuInstant) {
        State.CpuInstant = cpuInstant;
        State.Instant = TInstant::Now();
    }

    return State;
}

TCalibrationState GetCalibrationState()
{
    // The calibration "now" only needs to be approximate, so a non-serializing rdtsc suffices.
    return GetCalibrationState(GetApproximateCpuInstant());
}

TDuration CpuDurationToDuration(TCpuDuration cpuDuration, double ticksToMicroseconds)
{
    // TDuration is unsigned and thus does not support negative values.
    if (cpuDuration < 0) {
        return TDuration::Zero();
    }
    return TDuration::MicroSeconds(static_cast<ui64>(cpuDuration * ticksToMicroseconds));
}

TCpuDuration DurationToCpuDuration(TDuration duration, double microsecondsToTicks)
{
    return static_cast<TCpuDuration>(duration.MicroSeconds() * microsecondsToTicks);
}

TInstant GetInstant()
{
    auto cpuInstant = GetCpuInstant();
    auto state = GetCalibrationState(cpuInstant);
    YT_ASSERT(cpuInstant >= state.CpuInstant);
    return state.Instant + CpuDurationToDuration(cpuInstant - state.CpuInstant, GetTicksToMicroseconds());
}

TDuration CpuDurationToDuration(TCpuDuration cpuDuration)
{
    return CpuDurationToDuration(cpuDuration, GetTicksToMicroseconds());
}

TCpuDuration DurationToCpuDuration(TDuration duration)
{
    return DurationToCpuDuration(duration, GetMicrosecondsToTicks());
}

TInstant CpuInstantToInstant(TCpuInstant cpuInstant)
{
    // TDuration is unsigned and does not support negative values,
    // thus we consider two cases separately.
    auto state = GetCalibrationState();
    return cpuInstant >= state.CpuInstant
        ? state.Instant + CpuDurationToDuration(cpuInstant - state.CpuInstant, GetTicksToMicroseconds())
        : state.Instant - CpuDurationToDuration(state.CpuInstant - cpuInstant, GetTicksToMicroseconds());
}

TCpuInstant InstantToCpuInstant(TInstant instant)
{
    // See above.
    auto state = GetCalibrationState();
    return instant >= state.Instant
        ? state.CpuInstant + DurationToCpuDuration(instant - state.Instant, GetMicrosecondsToTicks())
        : state.CpuInstant - DurationToCpuDuration(state.Instant - instant, GetMicrosecondsToTicks());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
