#pragma once

// TODO(babenko): Drop this shim; include library/cpp/yt/system/futex.h instead.

#include <library/cpp/yt/system/futex.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

#ifdef _linux_

using ::NYT::FutexWait;
using ::NYT::FutexWake;

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
