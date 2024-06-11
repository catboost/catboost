#pragma once

#include "public.h"
#include "spin_lock.h"

#include <util/system/src_location.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

//! Wraps TSpinLock and additionally acquires a global fork lock (in read mode)
//! preventing concurrent forks from happening.
class TForkAwareSpinLock
{
public:
    TForkAwareSpinLock(const TForkAwareSpinLock&) = delete;
    TForkAwareSpinLock& operator =(const TForkAwareSpinLock&) = delete;

    constexpr TForkAwareSpinLock() = default;

    // TODO(babenko): make use of location.
    explicit constexpr TForkAwareSpinLock(const ::TSourceLocation& /*location*/)
    { }

    void Acquire() noexcept;
    bool TryAcquire() noexcept;
    void Release() noexcept;

    bool IsLocked() const noexcept;

private:
    ::NYT::NThreading::TSpinLock SpinLock_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
