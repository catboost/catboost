#include "systime.h"

#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/generic/string.h>

#include <string.h>

#ifdef _win_

void FileTimeToTimeval(const FILETIME* ft, timeval* tv) {
    const i64 NANOINTERVAL = LL(116444736000000000); // Number of 100 nanosecond units from 1/1/1601 to 1/1/1970
    union {
        ui64 ft_scalar;
        FILETIME ft_struct;
    } nt_time;
    nt_time.ft_struct = *ft;
    tv->tv_sec = (long)((nt_time.ft_scalar - NANOINTERVAL) / LL(10000000));
    tv->tv_usec = (i32)((nt_time.ft_scalar / LL(10)) % LL(1000000));
}

int gettimeofday(timeval* tp, void*) {
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    FileTimeToTimeval(&ft, tp);
    return 0;
}

tm* localtime_r(const time_t* clock, tm* result) {
    tzset();
    tm* res = localtime(clock);
    if (res) {
        memcpy(result, res, sizeof(tm));
        return result;
    }
    return 0;
}

tm* gmtime_r(const time_t* clock, tm* result) {
    return gmtime_s(result, clock) == 0 ? result : 0;
}

char* ctime_r(const time_t* clock, char* buf) {
    char* res = ctime(clock);
    if (res) {
        memcpy(buf, res, 26);
        return buf;
    }
    return 0;
}

#endif /* _win_ */

#define YEAR0 1900
#define EPOCH_YR 1970
#define SECS_DAY (24L * 60L * 60L)
#define LEAPYEAR(year) (!((year) % 4) && (((year) % 100) || !((year) % 400)))
#define YEARSIZE(year) (LEAPYEAR(year) ? 366 : 365)
#define FOURCENTURIES (400 * 365 + 100 - 3)

//! Inverse of gmtime: converts struct tm to time_t, assuming the data
//! in tm is UTC rather than local timezone. This implementation
//! returns the number of seconds since 1970-01-01, converted to time_t.
//! @note this code adopted from
//!       http://osdir.com/ml/web.wget.patches/2005-07/msg00010.html
//!       Subject: A more robust timegm - msg#00010
time_t TimeGM(const struct tm* t) {
    static const unsigned short int month_to_days[][13] = {
        {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},
        {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335}};

    // Only handles years after 1970
    if (t->tm_year < 70)
        return (time_t)-1;

    int days = 365 * (t->tm_year - 70);
    // Take into account the leap days between 1970 and YEAR-1
    days += (t->tm_year - 1 - 68) / 4 - ((t->tm_year - 1) / 100) + ((t->tm_year - 1 + 300) / 400);

    if (t->tm_mon < 0 || t->tm_mon >= 12)
        return (time_t)-1;
    days += month_to_days[LEAPYEAR(1900 + t->tm_year)][t->tm_mon];
    days += t->tm_mday - 1;

    unsigned long secs = days * 86400ul + t->tm_hour * 3600 + t->tm_min * 60 + t->tm_sec;
    return (time_t)secs;
}

struct tm* GmTimeR(const time_t* timer, struct tm* tmbuf) {
    static const int _ytab[2][12] = {
        {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
        {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};

    i64 time = static_cast<i64>(*timer);

    ui64 dayclock, dayno;
    int year = EPOCH_YR;

    if (time < 0) {
        ui64 shift = (ui64)(-time - 1) / ((ui64)FOURCENTURIES * SECS_DAY) + 1;
        time += shift * ((ui64)FOURCENTURIES * SECS_DAY);
        year -= shift * 400;
    }

    dayclock = (ui64)time % SECS_DAY;
    dayno = (ui64)time / SECS_DAY;

    year += 400 * (dayno / FOURCENTURIES);
    dayno = dayno % FOURCENTURIES;

    tmbuf->tm_sec = dayclock % 60;
    tmbuf->tm_min = (dayclock % 3600) / 60;
    tmbuf->tm_hour = dayclock / 3600;
    tmbuf->tm_wday = (dayno + 4) % 7; // Day 0 was a thursday
    while (dayno >= (ui64)YEARSIZE(year)) {
        dayno -= YEARSIZE(year);
        ++year;
    }
    tmbuf->tm_year = year - YEAR0;
    tmbuf->tm_yday = dayno;
    tmbuf->tm_mon = 0;
    while (dayno >= (ui64)_ytab[LEAPYEAR(year)][tmbuf->tm_mon]) {
        dayno -= _ytab[LEAPYEAR(year)][tmbuf->tm_mon];
        ++tmbuf->tm_mon;
    }
    tmbuf->tm_mday = dayno + 1;
    tmbuf->tm_isdst = 0;
#ifndef _win_
    tmbuf->tm_gmtoff = 0;
    tmbuf->tm_zone = (char*)"UTC";
#endif

    return tmbuf;
}

TString CTimeR(const time_t* timer) {
    char sTime[32];
    sTime[0] = 0;
    ctime_r(timer, &sTime[0]);
    return sTime;
}
