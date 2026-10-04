#pragma once

// TODO(babenko): Drop this shim; include library/cpp/yt/system/spin_wait_hook.h instead.

#include <library/cpp/yt/system/spin_wait_hook.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

using ::NYT::ESpinLockActivityKind;
using ::NYT::InvokeSpinWaitSlowPathHooks;
using ::NYT::RegisterSpinWaitSlowPathHook;
using ::NYT::TSpinWaitSlowPathHook;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
