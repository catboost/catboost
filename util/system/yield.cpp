#include "platform.h"

#ifdef _win_
    #include "winint.h"
    #include <process.h>
#else
    #include <sched.h>
#endif

void SchedYield() noexcept {
#if defined(_unix_)
    sched_yield();
#else
    Sleep(0);
#endif
}

void ThreadYield() noexcept {
#if defined(_freebsd_)
    pthread_yield();
#else
    SchedYield();
#endif
}
