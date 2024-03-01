#include "clock.h"

#include <util/system/hp_timer.h>

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/misc/tls.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// Re-calibrate every 1B CPU ticks.
constexpr auto CalibrationCpuPeriod = 1'000'000'000;

struct TCalibrationState
{
    TCpuInstant CpuInstant;
    TInstant Instant;
};

double GetMicrosecondsToTicks()
{
    static const auto MicrosecondsToTicks = static_cast<double>(NHPTimer::GetCyclesPerSecond()) / 1'000'000;
    return MicrosecondsToTicks;
}

double GetTicksToMicroseconds()
{
    static const auto TicksToMicroseconds = 1.0 / GetMicrosecondsToTicks();
    return TicksToMicroseconds;
}

TCalibrationState GetCalibrationState(TCpuInstant cpuInstant)
{
    YT_THREAD_LOCAL(TCalibrationState) State;

    auto& state = GetTlsRef(State);

    if (state.CpuInstant + CalibrationCpuPeriod < cpuInstant) {
        state.CpuInstant = cpuInstant;
        state.Instant = TInstant::Now();
    }

    return state;
}

TCalibrationState GetCalibrationState()
{
    return GetCalibrationState(GetCpuInstant());
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
