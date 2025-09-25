#ifndef EVENT_COUNT_INL_H_
#error "Direct inclusion of this file is not allowed, include event_count.h"
// For the sake of sane code completion.
#include "event_count.h"
#endif
#undef EVENT_COUNT_INL_H_

#include <library/cpp/yt/assert/assert.h>

#include "futex.h"

#include <errno.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

inline void TEventCount::NotifyOne()
{
    NotifyMany(1);
}

inline void TEventCount::NotifyAll()
{
    NotifyMany(std::numeric_limits<int>::max());
}

inline void TEventCount::NotifyMany(int count)
{
    // The order is important: Epoch is incremented before Waiters is checked.
    // prepareWait() increments Waiters before checking Epoch, so it is
    // impossible to miss a wakeup.
#ifndef _linux_
    TGuard<TMutex> guard(Mutex_);
#endif

    ui64 prev = Value_.fetch_add(AddEpoch, std::memory_order::acq_rel);
    if (Y_UNLIKELY((prev & WaiterMask) != 0)) {
#ifdef _linux_
        FutexWake(
            reinterpret_cast<int*>(&Value_) + 1, // assume little-endian architecture
            count);
#else
        if (count == 1) {
            ConditionVariable_.Signal();
        } else {
            ConditionVariable_.BroadCast();
        }
#endif
    }
}

inline TEventCount::TCookie TEventCount::PrepareWait()
{
    ui64 value = Value_.load(std::memory_order::acquire);
    return TCookie(static_cast<ui32>(value >> EpochShift));
}

inline void TEventCount::CancelWait()
{ }

inline bool TEventCount::Wait(TCookie cookie, TInstant deadline)
{
    Value_.fetch_add(AddWaiter, std::memory_order::acq_rel);

    bool result = true;
#ifdef _linux_
    while ((Value_.load(std::memory_order::acquire) >> EpochShift) == cookie.Epoch_) {
        auto timeout = deadline - TInstant::Now();

        auto futexResult = FutexWait(
            reinterpret_cast<int*>(&Value_) + 1, // assume little-endian architecture
            cookie.Epoch_,
            timeout);

        if (futexResult != 0 && errno == ETIMEDOUT) {
            result = false;
            break;
        }
    }
#else
    TGuard<TMutex> guard(Mutex_);
    if ((Value_.load(std::memory_order::acquire) >> EpochShift) == cookie.Epoch_) {
        result = ConditionVariable_.WaitD(Mutex_, deadline);
    }
#endif
    ui64 prev = Value_.fetch_add(SubWaiter, std::memory_order::seq_cst);
    YT_ASSERT((prev & WaiterMask) != 0);
    return result;
}

inline bool TEventCount::Wait(TCookie cookie, TDuration timeout)
{
    return Wait(cookie, timeout.ToDeadLine());
}

template <class TCondition>
bool TEventCount::Await(TCondition&& condition, TInstant deadline)
{
    if (condition()) {
        // Fast path.
        return true;
    }

    // condition() is the only thing that may throw, everything else is
    // noexcept, so we can hoist the try/catch block outside of the loop
    try {
        for (;;) {
            auto cookie = PrepareWait();
            if (condition()) {
                CancelWait();
                break;
            }
            if (!Wait(cookie, deadline)) {
                return false;
            }
        }
    } catch (...) {
        CancelWait();
        throw;
    }
    return true;
}

template <class TCondition>
bool TEventCount::Await(TCondition&& condition, TDuration timeout)
{
    return Await(std::forward<TCondition>(condition), timeout.ToDeadLine());
}
////////////////////////////////////////////////////////////////////////////////

inline void TEvent::NotifyOne()
{
    Set_.store(true, std::memory_order::release);
    EventCount_.NotifyOne();
}

inline void TEvent::NotifyAll()
{
    Set_.store(true, std::memory_order::release);
    EventCount_.NotifyAll();
}

inline bool TEvent::Test() const
{
    return Set_.load(std::memory_order::acquire);
}

inline bool TEvent::Wait(TInstant deadline)
{
    return EventCount_.Await(
        [&] {
            return Set_.load(std::memory_order::acquire);
        },
        deadline);
}

inline bool TEvent::Wait(TDuration timeout)
{
    return Wait(timeout.ToDeadLine());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
