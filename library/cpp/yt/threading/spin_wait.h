#pragma once

#include "spin_wait_hook.h"

#include <library/cpp/yt/cpu_clock/clock.h>

#include <util/datetime/base.h>

#include <util/system/src_location.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

class TSpinWait
{
public:
    TSpinWait(
        const ::TSourceLocation& location,
        ESpinLockActivityKind activityKind);
    ~TSpinWait();

    void Wait();

private:
    const ::TSourceLocation Location_;
    const ESpinLockActivityKind ActivityKind_;

    int SpinIteration_ = 0;
    int SleepIteration_ = 0;

    TCpuInstant SlowPathStartInstant_ = -1;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
