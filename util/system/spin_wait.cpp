#include "spin_wait.h"
#include "yield.h"
#include "compat.h"
#include "spinlock.h"

#include <util/digest/numeric.h>
#include <util/generic/utility.h>

#include <atomic>

namespace {
    unsigned RandomizeSleepTime(unsigned t) noexcept {
        static std::atomic<unsigned> counter = 0;
        const unsigned rndNum = IntHash(++counter);

        return (t * 4 + (rndNum % t) * 2) / 5;
    }

    //arbitrary values
    constexpr unsigned MIN_SLEEP_TIME = 500;
    constexpr unsigned MAX_SPIN_COUNT = 0x7FF;
}

TSpinWait::TSpinWait() noexcept
    : T(MIN_SLEEP_TIME)
    , C(0)
{
}

void TSpinWait::Sleep() noexcept {
    ++C;

    if (C == MAX_SPIN_COUNT) {
        ThreadYield();
    } else if ((C & MAX_SPIN_COUNT) == 0) {
        usleep(RandomizeSleepTime(T));

        T = Min<unsigned>(T * 3 / 2, 20000);
    } else {
        SpinLockPause();
    }
}
