#pragma once
#ifndef CLOCK_INL_H_
#error "Direct inclusion of this file is not allowed, include clock.h"
// For the sake of sane code completion.
#include "clock.h"
#endif

#include <util/system/datetime.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

inline TCpuInstant GetCpuInstant()
{
    return static_cast<TCpuInstant>(GetCycleCount());
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
