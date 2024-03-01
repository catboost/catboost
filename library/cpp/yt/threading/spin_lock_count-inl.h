#pragma once
#ifndef SPIN_LOCK_COUNT_INL_H_
#error "Direct inclusion of this file is not allowed, include spin_lock_count.h"
#endif
#undef SPIN_LOCK_COUNT_INL_H_

#include "public.h"
#include "spin_lock_base.h"

#ifdef NDEBUG
#include <util/system/compiler.h>
#endif

#include <cstdint>

namespace NYT::NThreading::NPrivate {

////////////////////////////////////////////////////////////////////////////////

#ifdef NDEBUG

Y_FORCE_INLINE void RecordSpinLockAcquired([[maybe_unused]] bool isAcquired = true)
{ }

Y_FORCE_INLINE void RecordSpinLockReleased()
{ }

Y_FORCE_INLINE void VerifyNoSpinLockAffinity()
{ }

#else

void RecordSpinLockAcquired(bool isAcquired = true);
void RecordSpinLockReleased();
void VerifyNoSpinLockAffinity();

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading::NPrivate
