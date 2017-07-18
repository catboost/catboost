#pragma once

#include <sys/types.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define AcquireAdaptiveLockSlow AllocAcquireAdaptiveLockSlow
#define SchedYield AllocSchedYield
#define ThreadYield AllocThreadYield
#define NSystemInfo NAllocSystemInfo

#ifdef _MSC_VER
#   define __restrict__ __restrict
#   define JEMALLOC_EXPORT
#endif

#if defined(__cplusplus)
};
#endif
