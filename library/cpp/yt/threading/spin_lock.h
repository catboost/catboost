#pragma once

#include "public.h"
#include "spin_lock_base.h"
#include "spin_lock_count.h"

#include <library/cpp/yt/misc/port.h>

#include <library/cpp/yt/system/thread_id.h>

#include <library/cpp/yt/memory/public.h>

#include <util/system/src_location.h>
#include <util/system/types.h>

#include <atomic>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

//! A slightly modified version of TAdaptiveLock.
/*!
 *  The lock is unfair.
 */
class TSpinLock
    : public TSpinLockBase
{
public:
    using TSpinLockBase::TSpinLockBase;

    //! Acquires the lock.
    void Acquire() noexcept;

    //! Tries acquiring the lock.
    //! Returns |true| on success.
    bool TryAcquire() noexcept;

    //! Releases the lock.
    void Release() noexcept;

    //! Returns true if the lock is taken.
    /*!
     *  This is inherently racy.
     *  Only use for debugging and diagnostic purposes.
     */
    bool IsLocked() const noexcept;

private:
#ifdef YT_ENABLE_SPIN_LOCK_OWNERSHIP_TRACKING
    using TValue = TSequentialThreadId;
    static constexpr TValue UnlockedValue = InvalidSequentialThreadId;
#else
    using TValue = ui32;
    static constexpr TValue UnlockedValue = 0;
    static constexpr TValue LockedValue = 1;
#endif

    std::atomic<TValue> Value_ = UnlockedValue;

    bool TryAndTryAcquire() noexcept;

    void AcquireSlow() noexcept;
};

REGISTER_TRACKED_SPIN_LOCK_CLASS(TSpinLock)

////////////////////////////////////////////////////////////////////////////////

//! A variant of TSpinLock occupying the whole cache line.
class alignas(CacheLineSize) TPaddedSpinLock
    : public TSpinLock
{ };

REGISTER_TRACKED_SPIN_LOCK_CLASS(TPaddedSpinLock)

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

#define SPIN_LOCK_INL_H_
#include "spin_lock-inl.h"
#undef SPIN_LOCK_INL_H_
