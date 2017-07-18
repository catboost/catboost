#ifndef _LIBCPP_SUPPORT_WIN32_TIME_WIN32_H
#define _LIBCPP_SUPPORT_WIN32_TIME_WIN32_H

#if !defined(_LIBCPP_MSVCRT)
#error "This header complements Microsoft's C Runtime library, and should not be included otherwise."
#else // _LIBCPP_MSVCRT

#include <time.h>

struct timezone;

#ifndef _WINSOCKAPI_
struct timeval {
    long tv_sec;
    long tv_usec;
};
#endif

#if _MSC_VER >= 1900
#define _TIMESPEC_DEFINED
#endif

#ifndef _TIMESPEC_DEFINED
struct timespec {
    time_t tv_sec;
    long tv_nsec;
};
#define _TIMESPEC_DEFINED
#endif // _TIMESPEC_DEFINED

typedef	int	clockid_t;
static const clockid_t CLOCK_MONOTONIC = 0;

int gettimeofday(struct timeval * tp, struct timezone * tzp);

int clock_gettime(clockid_t clock_id, struct timespec *tv);

int nanosleep(const struct timespec *req, struct timespec *rem);

#endif // _LIBCPP_MSVCRT

#endif // _LIBCPP_SUPPORT_WIN32_TIME_WIN32_H
