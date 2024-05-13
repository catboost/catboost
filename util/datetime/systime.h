#pragma once

#include <util/system/platform.h>
#include <util/generic/string.h>

#include <ctime>

// timegm and gmtime_r versions that don't need access to filesystem or a big stack
time_t TimeGM(const struct tm* t);
struct tm* GmTimeR(const time_t* timer, struct tm* tmbuf);
// safe version of ctime, convinient version of ctime_r
TString CTimeR(const time_t* timer);

#ifdef _win_
    #include <util/system/winint.h>
    #include <winsock2.h>

// Convert FILETIME to timeval - seconds and microseconds.
void FileTimeToTimeval(const FILETIME* ft, struct timeval* tv);

// Convert FILETIME to timespec - seconds and nanoseconds.
void FileTimeToTimespec(const FILETIME& ft, struct timespec* ts);

// obtains the current time, expressed as seconds and microseconds since 00:00 UTC, January 1, 1970
int gettimeofday(struct timeval* tp, void*);

// thou should not mix these with non-_r functions
tm* localtime_r(const time_t* clock, tm* result);
tm* gmtime_r(const time_t* clock, tm* result);
char* ctime_r(const time_t* clock, char* buf);

inline time_t timegm(struct tm* t) {
    return TimeGM(t);
}

char* strptime(const char* buf, const char* fmt, struct tm* tm); // strptime.cpp
#else
    #include <sys/time.h>
#endif

#ifndef timersub
    #define timersub(tvp, uvp, vvp)                           \
        do {                                                  \
            (vvp)->tv_sec = (tvp)->tv_sec - (uvp)->tv_sec;    \
            (vvp)->tv_usec = (tvp)->tv_usec - (uvp)->tv_usec; \
            if ((vvp)->tv_usec < 0) {                         \
                (vvp)->tv_sec--;                              \
                (vvp)->tv_usec += 1000000;                    \
            }                                                 \
        } while (0)
#endif
