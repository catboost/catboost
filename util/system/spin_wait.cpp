#include "spin_wait.h"
#include "yield.h"
#include "compat.h"
#include "thread.h"
#include "spinlock.h"

#include <util/digest/numeric.h>
#include <util/generic/utility.h>

template <class T>
static inline T RandomizeSleepTime(T t) noexcept {
    static TAtomic counter = 0;
    const T rndNum = IntHash((T)AtomicIncrement(counter));

    return (t * (T)4 + (rndNum % t) * (T)2) / (T)5;
}

//arbitrary values
#define MIN_SLEEP_TIME 500
#define MAX_SPIN_COUNT 0x7FF

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

        T = Min<unsigned>((T * 3) / 2, 20000);
    } else {
        SpinLockPause();
    }
}
