#pragma once

#include "platform.h"
#include "spin_wait.h"

#include <atomic>

class TSpinLockBase {
protected:
    TSpinLockBase() = default;

    // These were unearthed in IGNIETFERRO-1105
    // Need to get rid of them separately
    TSpinLockBase(const TSpinLockBase& other)
        : Val_(other.Val_.load())
    {
    }

    TSpinLockBase& operator=(const TSpinLockBase& other)
    {
        Val_.store(other.Val_);
        return *this;
    }

public:
    inline bool IsLocked() const noexcept {
        return Val_.load();
    }

    inline bool TryAcquire() noexcept {
        intptr_t zero = 0;
        return Val_.compare_exchange_strong(zero, 1);
    }

    inline void Release() noexcept {
        Val_.store(0, std::memory_order_release);
    }

    inline bool try_lock() noexcept {
        return TryAcquire();
    }

    inline void unlock() noexcept {
        Release();
    }

protected:
    std::atomic<intptr_t> Val_{0};
};

static inline void SpinLockPause() {
#if defined(__GNUC__)
    #if defined(_i386_) || defined(_x86_64_)
    __asm __volatile("pause");
    #elif defined(_arm64_)
    __asm __volatile("yield" ::
                         : "memory");
    #endif
#endif
}

/*
 * You should almost always use TAdaptiveLock instead of TSpinLock
 */
class TSpinLock: public TSpinLockBase {
public:
    using TSpinLockBase::TSpinLockBase;

    inline void Acquire() noexcept {
        intptr_t zero = 0;
        if (Val_.compare_exchange_strong(zero, 1)) {
            return;
        }
        do {
            SpinLockPause();
            zero = 0;
        } while (Val_.load(std::memory_order_acquire) != 0 ||
                 !Val_.compare_exchange_strong(zero, 1));
    }

    inline void lock() noexcept {
        Acquire();
    }
};

/**
 * TAdaptiveLock almost always should be used instead of TSpinLock.
 * It also should be used instead of TMutex for short-term locks.
 * This usually means that the locked code should not use syscalls,
 * since almost every syscall:
 *   - might run unpredictably long and the waiting thread will waste a lot of CPU
 *   - takes considerable amount of time, so you should not care about the mutex performance
 */
class TAdaptiveLock: public TSpinLockBase {
public:
    using TSpinLockBase::TSpinLockBase;

    void Acquire() noexcept {
        intptr_t zero = 0;
        if (Val_.compare_exchange_strong(zero, 1)) {
            return;
        }

        TSpinWait sw;

        for (;;) {
            zero = 0;
            if (Val_.load(std::memory_order_acquire) == 0 &&
                Val_.compare_exchange_strong(zero, 1)) {
                break;
            }
            sw.Sleep();
        }
    }

    inline void lock() noexcept {
        Acquire();
    }
};
