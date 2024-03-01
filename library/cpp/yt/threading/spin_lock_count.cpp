
#include "spin_lock_count.h"

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/misc/tls.h>

#include <util/system/types.h>

namespace NYT::NThreading::NPrivate {

#ifndef NDEBUG

////////////////////////////////////////////////////////////////////////////////

YT_THREAD_LOCAL(i64) ActiveSpinLockCount = 0;

////////////////////////////////////////////////////////////////////////////////

void RecordSpinLockAcquired(bool isAcquired)
{
    if (isAcquired) {
        ActiveSpinLockCount++;
    }
}

void RecordSpinLockReleased()
{
    YT_VERIFY(ActiveSpinLockCount > 0);
    ActiveSpinLockCount--;
}

void VerifyNoSpinLockAffinity()
{
    YT_VERIFY(ActiveSpinLockCount == 0);
}

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading::NPrivate

