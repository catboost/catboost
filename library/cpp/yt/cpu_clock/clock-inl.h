#pragma once
#ifndef CLOCK_INL_H_
#error "Direct inclusion of this file is not allowed, include clock.h"
// For the sake of sane code completion.
#include "clock.h"
#endif

#include <util/system/datetime.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE TCpuInstant GetCpuInstant()
{
    return static_cast<TCpuInstant>(GetCycleCount());
}

Y_FORCE_INLINE TCpuInstant GetApproximateCpuInstant()
{
#if defined(_x86_64_)
    ui32 hi, lo;
    __asm__ __volatile__("rdtsc"
                        : "=a"(lo), "=d"(hi));
    return static_cast<TCpuInstant>(lo) | (static_cast<TCpuInstant>(hi) << 32);
#else
    return GetCpuInstant();
#endif
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
