#pragma once

#include "defaults.h"

namespace NHPTimer {
    using STime = i64;
    // May delay for ~50ms to compute frequency
    double GetSeconds(const STime& a) noexcept;
    // Returns the current time
    void GetTime(STime* pTime) noexcept;
    // Returns the time passed since *pTime, and writes the current time into *pTime.
    double GetTimePassed(STime* pTime) noexcept;
    // Get TSC frequency, may delay for ~50ms to compute frequency
    double GetClockRate() noexcept;
    // same as GetClockRate, but in integer
    ui64 GetCyclesPerSecond() noexcept;
}

struct THPTimer {
    THPTimer() noexcept {
        Reset();
    }
    void Reset() noexcept {
        NHPTimer::GetTime(&Start);
    }
    double Passed() const noexcept {
        NHPTimer::STime tmp = Start;
        return NHPTimer::GetTimePassed(&tmp);
    }
    double PassedReset() noexcept {
        return NHPTimer::GetTimePassed(&Start);
    }

private:
    NHPTimer::STime Start;
};
