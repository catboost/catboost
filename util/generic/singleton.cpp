#include "singleton.h"

#include <util/system/spinlock.h>
#include <util/system/thread.h>
#include <util/system/sanitizers.h>

#include <cstring>

namespace {
    static inline bool MyAtomicTryLock(std::atomic<size_t>& a, size_t v) noexcept {
        size_t zero = 0;
        return a.compare_exchange_strong(zero, v);
    }

    static inline bool MyAtomicTryAndTryLock(std::atomic<size_t>& a, size_t v) noexcept {
        return a.load(std::memory_order_acquire) == 0 && MyAtomicTryLock(a, v);
    }

    static inline size_t MyThreadId() noexcept {
        const size_t ret = TThread::CurrentThreadId();

        if (ret) {
            return ret;
        }

        //clash almost impossible, ONLY if we have threads with ids 0 and 1!
        return 1;
    }
}

void NPrivate::FillWithTrash(void* ptr, size_t len) {
#if defined(NDEBUG)
    Y_UNUSED(ptr);
    Y_UNUSED(len);
#else
    if constexpr (NSan::TSanIsOn()) {
        Y_UNUSED(ptr);
        Y_UNUSED(len);
    } else {
        memset(ptr, 0xBA, len);
    }
#endif
}

void NPrivate::LockRecursive(std::atomic<size_t>& lock) noexcept {
    const size_t id = MyThreadId();

    Y_ABORT_UNLESS(lock.load(std::memory_order_acquire) != id, "recursive singleton initialization");

    if (!MyAtomicTryLock(lock, id)) {
        TSpinWait sw;

        do {
            sw.Sleep();
        } while (!MyAtomicTryAndTryLock(lock, id));
    }
}

void NPrivate::UnlockRecursive(std::atomic<size_t>& lock) noexcept {
    Y_ABORT_UNLESS(lock.load(std::memory_order_acquire) == MyThreadId(), "unlock from another thread?!?!");
    lock.store(0);
}
