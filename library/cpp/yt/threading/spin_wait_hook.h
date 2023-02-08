#pragma once

#include <library/cpp/yt/cpu_clock/clock.h>

#include <library/cpp/yt/misc/enum.h>

#include <util/system/src_location.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

DEFINE_ENUM(ESpinLockActivityKind,
    (Read)
    (Write)
    (ReadWrite)
);

using TSpinWaitSlowPathHook = void(*)(
    TCpuDuration cpuDelay,
    const ::TSourceLocation& location,
    ESpinLockActivityKind activityKind);

void RegisterSpinWaitSlowPathHook(TSpinWaitSlowPathHook hook);
void InvokeSpinWaitSlowPathHooks(
    TCpuDuration cpuDelay,
    const ::TSourceLocation& location,
    ESpinLockActivityKind activityKind);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
