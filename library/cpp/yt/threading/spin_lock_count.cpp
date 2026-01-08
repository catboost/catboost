
#include "spin_lock_count.h"

#include <library/cpp/yt/assert/assert.h>

#include <library/cpp/yt/misc/tls.h>

#include <util/system/types.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

#ifndef NDEBUG

namespace NDetail {

YT_DEFINE_THREAD_LOCAL(i64, ActiveSpinLockCount, 0);

void RecordSpinLockAcquired()
{
    ActiveSpinLockCount()++;
}

void RecordSpinLockReleased()
{
    YT_VERIFY(ActiveSpinLockCount() > 0);
    ActiveSpinLockCount()--;
}

} // namespace NDetail

int GetActiveSpinLockCount()
{
    return NDetail::ActiveSpinLockCount();
}

void VerifyNoSpinLockAffinity()
{
    YT_VERIFY(NDetail::ActiveSpinLockCount() == 0);
}

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

