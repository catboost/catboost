#include "futex.h"

#ifdef _linux_
    #include <linux/futex.h>

    #include <sys/time.h>
    #include <sys/syscall.h>
#endif

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

#ifdef _linux_

namespace {

int Futex(
    int* uaddr,
    int op,
    int val,
    const timespec* timeout,
    int* uaddr2,
    int val3)
{
    return ::syscall(SYS_futex, uaddr, op, val, timeout, uaddr2, val3);
}

} // namespace

int FutexWait(int* addr, int value, TDuration timeout)
{
    struct timespec timeoutSpec;
    if (timeout != TDuration::Max()) {
        timeoutSpec.tv_sec = timeout.Seconds();
        timeoutSpec.tv_nsec = (timeout - TDuration::Seconds(timeout.Seconds())).MicroSeconds() * 1000;
    }

    return Futex(
        addr,
        FUTEX_WAIT_PRIVATE,
        value,
        timeout != TDuration::Max() ? &timeoutSpec : nullptr,
        nullptr,
        0);
}

int FutexWake(int* addr, int count)
{
    return Futex(
        addr,
        FUTEX_WAKE_PRIVATE,
        count,
        nullptr,
        nullptr,
        0);
}

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
