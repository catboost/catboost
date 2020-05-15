#include "hot_swap.h"

#include <util/system/spinlock.h>

namespace NHotSwapPrivate {
    void TWriterLock::Acquire() noexcept {
        AtomicIncrement(ReadersCount);
    }

    void TWriterLock::Release() noexcept {
        AtomicDecrement(ReadersCount);
    }

    void TWriterLock::WaitAllReaders() const noexcept {
        TAtomicBase cnt = AtomicGet(ReadersCount);
        while (cnt > 0) {
            SpinLockPause();
            cnt = AtomicGet(ReadersCount);
            Y_ASSERT(cnt >= 0);
        }
    }

}
