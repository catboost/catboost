#pragma once
#ifndef SPIN_LOCK_INL_H_
#error "Direct inclusion of this file is not allowed, include spin_lock.h"
// For the sake of sane code completion.
#include "spin_lock.h"
#endif
#undef SPIN_LOCK_INL_H_

#include "spin_wait.h"

#include <library/cpp/yt/assert/assert.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

inline void TSpinLock::Acquire() noexcept
{
    if (TryAcquire()) {
        return;
    }

    AcquireSlow();
}

inline void TSpinLock::Release() noexcept
{
#ifdef NDEBUG
    Value_.store(UnlockedValue, std::memory_order::release);
#else
    YT_ASSERT(Value_.exchange(UnlockedValue, std::memory_order::release) != UnlockedValue);
#endif
    NDetail::RecordSpinLockReleased();
}

inline bool TSpinLock::IsLocked() const noexcept
{
    return Value_.load(std::memory_order::relaxed) != UnlockedValue;
}

inline bool TSpinLock::TryAcquire() noexcept
{
    auto expectedValue = UnlockedValue;
#ifdef YT_ENABLE_SPIN_LOCK_OWNERSHIP_TRACKING
    auto newValue = GetSequentialThreadId();
#else
    auto newValue = LockedValue;
#endif

    bool acquired = Value_.compare_exchange_weak(
        expectedValue,
        newValue,
        std::memory_order::acquire,
        std::memory_order::relaxed);
    NDetail::RecordSpinLockAcquired(acquired);
    return acquired;
}

inline bool TSpinLock::TryAndTryAcquire() noexcept
{
    auto value = Value_.load(std::memory_order::relaxed);
#ifdef YT_ENABLE_SPIN_LOCK_OWNERSHIP_TRACKING
    YT_ASSERT(value != GetSequentialThreadId());
#endif
    if (value != UnlockedValue) {
        return false;
    }
    return TryAcquire();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

