#include "hp_timer.h"

#include <util/generic/algorithm.h>
#include <util/generic/singleton.h>
#include <util/datetime/cputimer.h>

using namespace NHPTimer;

namespace {
    struct TFreq {
        inline TFreq()
            : Freq(InitHPTimer())
            , Rate(1.0 / Freq)
            , CyclesPerSecond(static_cast<ui64>(Rate))
        {
        }

        static inline const TFreq& Instance() {
            return *SingletonWithPriority<TFreq, 1>();
        }

        static double EstimateCPUClock() {
            for (;;) {
                ui64 startCycle = 0;
                ui64 startMS = 0;

                for (;;) {
                    startMS = MicroSeconds();
                    startCycle = GetCycleCount();

                    ui64 n = MicroSeconds();

                    if (n - startMS < 100) {
                        break;
                    }
                }

                Sleep(TDuration::MicroSeconds(5000));

                ui64 finishCycle = 0;
                ui64 finishMS = 0;

                for (;;) {
                    finishMS = MicroSeconds();

                    if (finishMS - startMS < 100) {
                        continue;
                    }

                    finishCycle = GetCycleCount();

                    ui64 n = MicroSeconds();

                    if (n - finishMS < 100) {
                        break;
                    }
                }
                if (startMS < finishMS && startCycle < finishCycle) {
                    return (finishCycle - startCycle) * 1000000.0 / (finishMS - startMS);
                }
            }
        }

        static double InitHPTimer() {
            const size_t N_VEC = 9;

            double vec[N_VEC];

            for (auto& i : vec) {
                i = EstimateCPUClock();
            }

            Sort(vec, vec + N_VEC);

            return 1.0 / vec[N_VEC / 2];
        }

        inline double GetSeconds(const STime& a) const {
            return static_cast<double>(a) * Freq;
        }

        inline double GetClockRate() const {
            return Rate;
        }

        inline ui64 GetCyclesPerSecond() const {
            return CyclesPerSecond;
        }

        const double Freq;
        const double Rate;
        const ui64 CyclesPerSecond;
    };
} // namespace

double NHPTimer::GetSeconds(const STime& a) noexcept {
    return TFreq::Instance().GetSeconds(a);
}

double NHPTimer::GetClockRate() noexcept {
    return TFreq::Instance().GetClockRate();
}

ui64 NHPTimer::GetCyclesPerSecond() noexcept {
    return TFreq::Instance().GetCyclesPerSecond();
}

void NHPTimer::GetTime(STime* pTime) noexcept {
    *pTime = GetCycleCount();
}

double NHPTimer::GetTimePassed(STime* pTime) noexcept {
    STime old(*pTime);

    *pTime = GetCycleCount();

    return GetSeconds(*pTime - old);
}
