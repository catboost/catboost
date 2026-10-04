#pragma once

// TODO(babenko): Drop this shim; include library/cpp/yt/system/spin_lock.h instead.

#include "public.h"
#include "spin_lock_base.h"
#include "spin_lock_count.h"
#include "spin_wait.h"

#include <library/cpp/yt/memory/public.h>

#include <library/cpp/yt/system/spin_lock.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

using ::NYT::TPaddedSpinLock;
using ::NYT::TSpinLock;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
