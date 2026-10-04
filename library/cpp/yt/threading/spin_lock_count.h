#pragma once

// TODO(babenko): Drop this shim; include library/cpp/yt/system/spin_lock_count.h instead.

#include "public.h"

#include <library/cpp/yt/system/spin_lock_count.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

using ::NYT::CTracedSpinLock;
using ::NYT::GetActiveSpinLockCount;
using ::NYT::VerifyNoSpinLockAffinity;

namespace NDetail {

using ::NYT::NDetail::MaybeRecordSpinLockAcquired;
using ::NYT::NDetail::RecordSpinLockAcquired;
using ::NYT::NDetail::RecordSpinLockReleased;

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
