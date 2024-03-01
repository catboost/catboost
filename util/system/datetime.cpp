#include "datetime.h"
#include "yassert.h"
#include "platform.h"
#include "cpu_id.h"

#include <util/datetime/systime.h>

#include <ctime>
#include <cerrno>

#ifdef _darwin_
    #include <AvailabilityMacros.h>
    #if defined(MAC_OS_X_VERSION_10_12) && MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_12
        #define Y_HAS_CLOCK_GETTIME
    #endif
#elif defined(_linux_) || defined(_freebsd_) || defined(_cygwin_)
    #define Y_HAS_CLOCK_GETTIME
#endif

static ui64 ToMicroSeconds(const struct timeval& tv) {
    return (ui64)tv.tv_sec * 1000000 + (ui64)tv.tv_usec;
}

#if defined(_win_)
static ui64 ToMicroSeconds(const FILETIME& ft) {
    return (((ui64)ft.dwHighDateTime << 32) + (ui64)ft.dwLowDateTime) / (ui64)10;
}
#elif defined(Y_HAS_CLOCK_GETTIME)
static ui64 ToMicroSeconds(const struct timespec& ts) {
    return (ui64)ts.tv_sec * 1000000 + (ui64)ts.tv_nsec / 1000;
}
#endif

ui64 MicroSeconds() noexcept {
    struct timeval tv;
    gettimeofday(&tv, nullptr);

    return ToMicroSeconds(tv);
}

ui64 ThreadCPUUserTime() noexcept {
#if defined(_win_)
    FILETIME creationTime, exitTime, kernelTime, userTime;
    GetThreadTimes(GetCurrentThread(), &creationTime, &exitTime, &kernelTime, &userTime);
    return ToMicroSeconds(userTime);
#else
    return 0;
#endif
}

ui64 ThreadCPUSystemTime() noexcept {
#if defined(_win_)
    FILETIME creationTime, exitTime, kernelTime, userTime;
    GetThreadTimes(GetCurrentThread(), &creationTime, &exitTime, &kernelTime, &userTime);
    return ToMicroSeconds(kernelTime);
#else
    return 0;
#endif
}

ui64 ThreadCPUTime() noexcept {
#if defined(_win_)
    FILETIME creationTime, exitTime, kernelTime, userTime;
    GetThreadTimes(GetCurrentThread(), &creationTime, &exitTime, &kernelTime, &userTime);
    return ToMicroSeconds(userTime) + ToMicroSeconds(kernelTime);
#elif defined(Y_HAS_CLOCK_GETTIME)
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ToMicroSeconds(ts);
#else
    return 0;
#endif
}

ui32 Seconds() noexcept {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec;
}

void NanoSleep(ui64 ns) noexcept {
#if defined(_win_)
    Sleep(ns / 1000000);
#else
    const ui64 NS = 1000 * 1000 * 1000;
    struct timespec req;
    req.tv_sec = ns / NS;
    req.tv_nsec = ns % NS;
    struct timespec left;
    while (nanosleep(&req, &left) < 0) {
        Y_ASSERT(errno == EINTR);
        req = left;
    }
#endif
}

#if defined(_x86_)
namespace NPrivate {
    bool HaveRdtscpImpl() {
        return NX86::HaveRDTSCP();
    }
}
#endif

#ifdef Y_HAS_CLOCK_GETTIME
    #undef Y_HAS_CLOCK_GETTIME
#endif
