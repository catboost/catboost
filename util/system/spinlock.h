#pragma once

#include "atomic.h"
#include "spin_wait.h"

class TSpinLockBase {
protected:
    inline TSpinLockBase() noexcept {
        AtomicSet(Val_, 0);
    }

public:
    inline bool IsLocked() const noexcept {
        return AtomicGet(Val_);
    }

    inline bool TryAcquire() noexcept {
        return AtomicTryLock(&Val_);
    }

    inline bool try_lock() noexcept {
        return TryAcquire();
    }

protected:
    TAtomic Val_;
};

static inline void SpinLockPause() {
#if defined(__GNUC__) && (defined(_i386_) || defined(_x86_64_))
    __asm __volatile("pause");
#endif
}

static inline void AcquireSpinLock(TAtomic* l) {
    if (!AtomicTryLock(l)) {
        do {
            SpinLockPause();
        } while (!AtomicTryAndTryLock(l));
    }
}

static inline void ReleaseSpinLock(TAtomic* l) {
    AtomicUnlock(l);
}

/*
 * You should almost always use TAdaptiveLock instead of TSpinLock
 */
class TSpinLock: public TSpinLockBase {
public:
    inline void Release() noexcept {
        ReleaseSpinLock(&Val_);
    }

    inline void Acquire() noexcept {
        AcquireSpinLock(&Val_);
    }

    inline void unlock() noexcept {
        Release();
    }

    inline void lock() noexcept {
        Acquire();
    }
};

static inline void AcquireAdaptiveLock(TAtomic* l) {
    if (!AtomicTryLock(l)) {
        TSpinWait sw;

        while (!AtomicTryAndTryLock(l)) {
            sw.Sleep();
        }
    }
}

static inline void ReleaseAdaptiveLock(TAtomic* l) {
    AtomicUnlock(l);
}

class TAdaptiveLock: public TSpinLockBase {
public:
    inline void Release() noexcept {
        ReleaseAdaptiveLock(&Val_);
    }

    inline void Acquire() noexcept {
        AcquireAdaptiveLock(&Val_);
    }

    inline void unlock() noexcept {
        Release();
    }

    inline void lock() noexcept {
        Acquire();
    }
};

#include "guard.h"

template <>
struct TCommonLockOps<TAtomic> {
    static inline void Acquire(TAtomic* v) noexcept {
        AcquireAdaptiveLock(v);
    }

    static inline bool TryAcquire(TAtomic* v) noexcept {
        return AtomicTryLock(v);
    }

    static inline void Release(TAtomic* v) noexcept {
        ReleaseAdaptiveLock(v);
    }
};
