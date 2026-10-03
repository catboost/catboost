#pragma once

// TODO(babenko): Drop this shim; include library/cpp/yt/system/event_count.h instead.

#include "futex.h"

#include <library/cpp/yt/system/event_count.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

using ::NYT::TEvent;
using ::NYT::TEventCount;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
