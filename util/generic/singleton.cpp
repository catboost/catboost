#include "singleton.h"

#include <util/system/spinlock.h>
#include <util/system/thread.h>
#include <util/system/sanitizers.h>

#include <cstring>

namespace {
    static inline bool MyAtomicTryLock(TAtomic& a, TAtomicBase v) noexcept {
        return AtomicCas(&a, v, 0);
    }

    static inline bool MyAtomicTryAndTryLock(TAtomic& a, TAtomicBase v) noexcept {
        return (AtomicGet(a) == 0) && MyAtomicTryLock(a, v);
    }

    static inline TAtomicBase MyThreadId() noexcept {
        const TAtomicBase ret = TThread::CurrentThreadId();

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

void NPrivate::LockRecursive(TAtomic& lock) noexcept {
    const TAtomicBase id = MyThreadId();

    Y_VERIFY(AtomicGet(lock) != id, "recursive singleton initialization");

    if (!MyAtomicTryLock(lock, id)) {
        TSpinWait sw;

        do {
            sw.Sleep();
        } while (!MyAtomicTryAndTryLock(lock, id));
    }
}

void NPrivate::UnlockRecursive(TAtomic& lock) noexcept {
    Y_VERIFY(AtomicGet(lock) == MyThreadId(), "unlock from another thread?!?!");
    AtomicUnlock(&lock);
}
