#pragma once

#include <util/system/defaults.h>

typedef volatile intptr_t spinlock_t;

#define SPIN_L AllocAcquireAdaptiveLock
#define SPIN_U AllocReleaseAdaptiveLock

#define    _SPINLOCK_INITIALIZER 0
#define _SPINUNLOCK(_lck)     SPIN_U(_lck)
#define    _SPINLOCK(_lck)       SPIN_L(_lck)

#if defined(__cplusplus)
extern "C" {
#endif
    void SPIN_L(spinlock_t* lock);
    void SPIN_U(spinlock_t* lock);
#if defined(__cplusplus)
};
#endif
