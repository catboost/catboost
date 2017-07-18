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

void AcquireAdaptiveLockSlow(TAtomic* lock) {
    unsigned t = MIN_SLEEP_TIME;
    unsigned c = 0;

    while (!AtomicTryAndTryLock(lock)) {
        ++c;

        if (c == MAX_SPIN_COUNT) {
            ThreadYield();
        } else if ((c & MAX_SPIN_COUNT) == 0) {
            usleep(RandomizeSleepTime(t));

            t = Min<unsigned>((t * 3) / 2, 20000);
        } else {
            SpinLockPause();
        }
    }
}
