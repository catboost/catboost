#pragma once

// TODO(babenko): Drop this shim; include library/cpp/yt/system/spin_lock_base.h instead.

#include <library/cpp/yt/system/spin_lock_base.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

using ::NYT::TSpinLockBase;
using ::NYT::TSpinLockInplace;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
