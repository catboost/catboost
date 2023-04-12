#pragma once
#ifndef SPIN_LOCK_BASE_INL_H_
#error "Direct inclusion of this file is not allowed, include spin_lock_base.h"
// For the sake of sane code completion.
#include "spin_lock_base.h"
#endif
#undef SPIN_LOCK_BASE_INL_H_

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

inline constexpr TSpinLockBase::TSpinLockBase()
    : Location_({}, -1)
{ }

inline constexpr TSpinLockBase::TSpinLockBase(const ::TSourceLocation& location)
    : Location_(location)
{ }

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

