#pragma once

#include <util/datetime/base.h>
#include <util/system/yield.h>

class TSpinWaitHelper {
public:
    template <class TFunc>
    static void Wait(const TDuration& duration,
                     TFunc&& isComplete) {
        const ui32 fastIters = 10000;
        ui32 iter = 0;
        ui64 sleepTime = Min<ui64>(10, duration.NanoSeconds());

        TInstant start = TInstant::Now();
        while ((TInstant::Now() - start) < duration) {
            if (isComplete()) {
                break;
            }

            if (iter < fastIters) {
                SchedYield();
                ++iter;
            } else {
                NanoSleep(sleepTime);
            }
        }
    }
};
