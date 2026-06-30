#pragma once

#include <util/system/guard.h>
#include <util/system/defaults.h>
#include <util/system/yassert.h>

#include <atomic>

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

    inline intptr_t Add(intptr_t d) noexcept {
        Check();
        return Counter_ += d;
    }

    inline intptr_t Inc() noexcept {
        return Add(1);
    }

    inline intptr_t Sub(intptr_t d) noexcept {
        Check();
        return Counter_ -= d;
    }

    inline intptr_t Dec() noexcept {
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

    inline intptr_t Val() const noexcept {
        return Counter_;
    }

private:
    intptr_t Counter_;
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
        Y_ABORT_UNLESS(ThreadId == SystemCurrentThreadId(), "incorrect usage of TSimpleCounter");
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

    TAtomicCounter(const TAtomicCounter& other)
        : Counter_(other.Counter_.load())
    {
    }

    TAtomicCounter& operator=(const TAtomicCounter& other) {
        Counter_.store(other.Counter_.load());
        return *this;
    }

    inline ~TAtomicCounter() = default;

    inline intptr_t Add(intptr_t d) noexcept {
        return Counter_ += d;
    }

    inline intptr_t Inc() noexcept {
        return Add(1);
    }

    inline intptr_t Sub(intptr_t d) noexcept {
        return Counter_ -= d;
    }

    inline intptr_t Dec() noexcept {
        return Sub(1);
    }

    inline intptr_t Val() const noexcept {
        return Counter_.load();
    }

    inline bool TryWeakInc() noexcept {
        for (auto curValue = Counter_.load(std::memory_order_acquire);;) {
            if (!curValue) {
                return false;
            }

            if (Counter_.compare_exchange_weak(curValue, curValue + 1)) {
                return true;
            }
        }
    }

private:
    std::atomic<intptr_t> Counter_;
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
