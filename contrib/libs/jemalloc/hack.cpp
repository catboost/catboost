#include "hack.h"

#include <util/system/yield.cpp>
#include <util/system/spinlock.cpp>

#include "spinlock.h"

void SPIN_L(spinlock_t* l) {
    AcquireAdaptiveLock(l);
}

void SPIN_U(spinlock_t* l) {
    ReleaseAdaptiveLock(l);
}
