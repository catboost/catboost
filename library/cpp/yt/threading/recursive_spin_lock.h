#pragma once

#include "public.h"
#include "spin_lock_base.h"
#include "spin_lock_count.h"

#include <library/cpp/yt/system/thread_id.h>

#include <util/system/types.h>

#include <atomic>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

//! A counterpart of #TSpinLock that can be acquired from a single thread multiple times.
class TRecursiveSpinLock
    : public TSpinLockBase
{
public:
    using TSpinLockBase::TSpinLockBase;

    void Acquire() noexcept;
    bool TryAcquire() noexcept;

    void Release() noexcept;

    bool IsLocked() const noexcept;
    bool IsLockedByCurrentThread() const noexcept;

private:
    // Bits  0..31: recursion depth; if zero then the lock is not taken,
    //              thread id can be arbitrary
    // Bits 32..63: id of the thread owning the lock
    using TValue = ui64;
    std::atomic<TValue> Value_ = 0;

    static constexpr int ThreadIdShift = 32;
    static constexpr TValue RecursionDepthMask = (1ULL << ThreadIdShift) - 1;

    static_assert(sizeof(TSequentialThreadId) == 4);

    bool TryAndTryAcquire() noexcept;
    void AcquireSlow() noexcept;
};

REGISTER_TRACKED_SPIN_LOCK_CLASS(TRecursiveSpinLock)

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

#define RECURSIVE_SPIN_LOCK_INL_H_
#include "recursive_spin_lock-inl.h"
#undef RECURSIVE_SPIN_LOCK_INL_H_
