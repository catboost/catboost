#pragma once

#include <util/system/src_location.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

class TSpinLockBase
{
public:
    constexpr TSpinLockBase();
    explicit constexpr TSpinLockBase(const ::TSourceLocation& location);

protected:
    const ::TSourceLocation Location_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

#define SPIN_LOCK_BASE_INL_H_
#include "spin_lock_base-inl.h"
#undef SPIN_LOCK_BASE_INL_H_
