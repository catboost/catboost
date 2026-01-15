#include "hot_swap.h"

#include <util/system/spinlock.h>

namespace NHotSwapPrivate {
    void TWriterLock::Acquire() noexcept {
        ++ReadersCount;
    }

    void TWriterLock::Release() noexcept {
        --ReadersCount;
    }

    void TWriterLock::WaitAllReaders() const noexcept {
        for (i32 cnt = ReadersCount.load(); cnt > 0;) {
            SpinLockPause();
            cnt = ReadersCount.load();
            Y_ASSERT(cnt >= 0);
        }
    }

}
