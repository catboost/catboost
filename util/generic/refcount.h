#pragma once

#include <util/system/guard.h>
#include <util/system/atomic.h>
#include <util/system/defaults.h>
#include <util/system/yassert.h>

template <class TCounterCheckPolicy>
class TSimpleCounterTemplate: public TCounterCheckPolicy {
    using TCounterCheckPolicy::Check;

public:
    inline TSimpleCounterTemplate(long initial = 0) noexcept
        : Counter_(initial)
    {
    }

    inline ~TSimpleCounterTemplate() {
        Check();
    }

    inline TAtomicBase Add(TAtomicBase d) noexcept {
        Check();
        return Counter_ += d;
    }

    inline TAtomicBase Inc() noexcept {
        return Add(1);
    }

    inline TAtomicBase Sub(TAtomicBase d) noexcept {
        Check();
        return Counter_ -= d;
    }

    inline TAtomicBase Dec() noexcept {
        return Sub(1);
    }

    inline bool TryWeakInc() noexcept {
        if (!Counter_) {
            return false;
        }

        Inc();
        Y_ASSERT(Counter_ != 0);

        return true;
    }

    inline TAtomicBase Val() const noexcept {
        return Counter_;
    }

private:
    TAtomicBase Counter_;
};

class TNoCheckPolicy {
protected:
    inline void Check() const {
    }
};

#if defined(SIMPLE_COUNTER_THREAD_CHECK)

#include <util/system/thread.i>

class TCheckPolicy {
public:
    inline TCheckPolicy() {
        ThreadId = SystemCurrentThreadId();
    }

protected:
    inline void Check() const {
        Y_VERIFY(ThreadId == SystemCurrentThreadId(), "incorrect usage of TSimpleCounter");
    }

private:
    size_t ThreadId;
};
#else
using TCheckPolicy = TNoCheckPolicy;
#endif

// Use this one if access from multiple threads to your pointer is an error and you want to enforce thread checks
using TSimpleCounter = TSimpleCounterTemplate<TCheckPolicy>;
// Use this one if you do want to share the pointer between threads, omit thread checks and do the synchronization yourself
using TExplicitSimpleCounter = TSimpleCounterTemplate<TNoCheckPolicy>;

template <class TCounterCheckPolicy>
struct TCommonLockOps<TSimpleCounterTemplate<TCounterCheckPolicy>> {
    static inline void Acquire(TSimpleCounterTemplate<TCounterCheckPolicy>* t) noexcept {
        t->Inc();
    }

    static inline void Release(TSimpleCounterTemplate<TCounterCheckPolicy>* t) noexcept {
        t->Dec();
    }
};

class TAtomicCounter {
public:
    inline TAtomicCounter(long initial = 0) noexcept
        : Counter_(initial)
    {
    }

    inline ~TAtomicCounter() = default;

    inline TAtomicBase Add(TAtomicBase d) noexcept {
        return AtomicAdd(Counter_, d);
    }

    inline TAtomicBase Inc() noexcept {
        return Add(1);
    }

    inline TAtomicBase Sub(TAtomicBase d) noexcept {
        return AtomicSub(Counter_, d);
    }

    inline TAtomicBase Dec() noexcept {
        return Sub(1);
    }

    inline TAtomicBase Val() const noexcept {
        return AtomicGet(Counter_);
    }

    inline bool TryWeakInc() noexcept {
        while (true) {
            intptr_t curValue = Counter_;

            if (!curValue) {
                return false;
            }

            intptr_t newValue = curValue + 1;
            Y_ASSERT(newValue != 0);

            if (AtomicCas(&Counter_, newValue, curValue)) {
                return true;
            }
        }
    }

private:
    TAtomic Counter_;
};

template <>
struct TCommonLockOps<TAtomicCounter> {
    static inline void Acquire(TAtomicCounter* t) noexcept {
        t->Inc();
    }

    static inline void Release(TAtomicCounter* t) noexcept {
        t->Dec();
    }
};
