#ifndef RECURSIVE_SPIN_LOCK_INL_H_
#error "Direct inclusion of this file is not allowed, include recursive_spinlock.h"
// For the sake of sane code completion.
#include "recursive_spin_lock.h"
#endif
#undef RECURSIVE_SPIN_LOCK_INL_H_

#include "spin_wait.h"

#include <library/cpp/yt/assert/assert.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

inline void TRecursiveSpinLock::Acquire() noexcept
{
    if (TryAcquire()) {
        return;
    }
    AcquireSlow();
}

inline bool TRecursiveSpinLock::TryAcquire() noexcept
{
    auto currentThreadId = GetSequentialThreadId();
    auto oldValue = Value_.load();
    auto oldRecursionDepth = oldValue & RecursionDepthMask;
    if (oldRecursionDepth > 0 && (oldValue >> ThreadIdShift) != currentThreadId) {
        return false;
    }
    auto newValue = (oldRecursionDepth + 1) | (static_cast<TValue>(currentThreadId) << ThreadIdShift);

    bool acquired = Value_.compare_exchange_weak(oldValue, newValue);
    NDetail::RecordSpinLockAcquired(acquired);
    return acquired;
}

inline void TRecursiveSpinLock::Release() noexcept
{
#ifndef NDEBUG
    auto value = Value_.load();
    YT_ASSERT((value & RecursionDepthMask) > 0);
    YT_ASSERT((value >> ThreadIdShift) == GetSequentialThreadId());
#endif
    --Value_;
    NDetail::RecordSpinLockReleased();
}

inline bool TRecursiveSpinLock::IsLocked() const noexcept
{
    auto value = Value_.load();
    return (value & RecursionDepthMask) > 0;
}

inline bool TRecursiveSpinLock::IsLockedByCurrentThread() const noexcept
{
    auto value = Value_.load();
    return (value & RecursionDepthMask) > 0 && (value >> ThreadIdShift) == GetSequentialThreadId();
}

inline bool TRecursiveSpinLock::TryAndTryAcquire() noexcept
{
    auto value = Value_.load(std::memory_order::relaxed);
    auto recursionDepth = value & RecursionDepthMask;
    if (recursionDepth > 0 && (value >> ThreadIdShift) != GetSequentialThreadId()) {
        return false;
    }
    return TryAcquire();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

