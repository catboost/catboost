#pragma once

#include "public.h"

#include <util/datetime/base.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Returns the current processor clock (typically obtained via |rdtscp| instruction).
TCpuInstant GetCpuInstant();

//! Returns the approximate current processor clock (obtained via |rdtsc| instruction).
TCpuInstant GetApproximateCpuInstant();

//! Returns the current time (obtained via #GetCpuInstant).
TInstant GetInstant();

//! Converts a number of processor ticks into a regular duration.
TDuration CpuDurationToDuration(TCpuDuration cpuDuration);

//! Converts a regular duration into the number of processor ticks.
TCpuDuration DurationToCpuDuration(TDuration duration);

//! Converts a processor clock into the regular time instant.
TInstant CpuInstantToInstant(TCpuInstant cpuInstant);

//! Converts a regular time instant into the processor clock.
TCpuInstant InstantToCpuInstant(TInstant instant);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define CLOCK_INL_H_
#include "clock-inl.h"
#undef CLOCK_INL_H_
