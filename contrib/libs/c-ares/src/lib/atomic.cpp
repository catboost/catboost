#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/spin_wait.h>

#include "atomic.h"

EXTERN_C void acquire_lock(atomic_t *lock)
{
    if (!AtomicTryLock(lock)) {
        TSpinWait sw;

        while (!AtomicTryAndTryLock(lock)) {
            sw.Sleep();
        }
    }
}

EXTERN_C void release_lock(atomic_t *lock)
{
    AtomicUnlock(lock);
}
