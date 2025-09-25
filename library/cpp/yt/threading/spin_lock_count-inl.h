#pragma once
#ifndef SPIN_LOCK_COUNT_INL_H_
#error "Direct inclusion of this file is not allowed, include spin_lock_count.h"
#endif
#undef SPIN_LOCK_COUNT_INL_H_

#include <util/system/compiler.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

#ifdef NDEBUG

Y_FORCE_INLINE void RecordSpinLockAcquired()
{ }

Y_FORCE_INLINE void RecordSpinLockReleased()
{ }

#else

void RecordSpinLockAcquired();
void RecordSpinLockReleased();

#endif

Y_FORCE_INLINE void MaybeRecordSpinLockAcquired(bool acquired)
{
    if (acquired) {
        RecordSpinLockAcquired();
    }
}

} // namespace NDetail

#ifdef NDEBUG

Y_FORCE_INLINE int GetActiveSpinLockCount()
{
    return 0;
}

Y_FORCE_INLINE void VerifyNoSpinLockAffinity()
{ }

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
