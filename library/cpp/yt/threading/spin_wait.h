#pragma once

// TODO(babenko): Drop this shim; include library/cpp/yt/system/spin_wait.h instead.

#include "spin_wait_hook.h"

#include <library/cpp/yt/system/spin_wait.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

using ::NYT::TSpinWait;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
