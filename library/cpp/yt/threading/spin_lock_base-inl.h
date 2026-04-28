#pragma once
#ifndef SPIN_LOCK_BASE_INL_H_
#error "Direct inclusion of this file is not allowed, include spin_lock_base.h"
// For the sake of sane code completion.
#include "spin_lock_base.h"
#endif
#undef SPIN_LOCK_BASE_INL_H_

#include <library/cpp/yt/misc/source_location.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

inline constexpr TSpinLockBase::TSpinLockBase()
    : Location_({}, -1)
{ }

inline constexpr TSpinLockBase::TSpinLockBase(const ::TSourceLocation& location)
    : Location_(location)
{ }

////////////////////////////////////////////////////////////////////////////////

template <class TLock, auto LocationLite>
TSpinLockInplace<TLock, LocationLite>::TSpinLockInplace()
    : TLock(::TSourceLocation{LocationLite.FileName, LocationLite.Line})
{ }

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
