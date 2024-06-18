#include "systime.h"

#include <util/system/yassert.h>
#include <util/system/defaults.h>

#ifdef _win_

namespace {
    // Number of 100 nanosecond units from 1/1/1601 to 1/1/1970
    constexpr ui64 NUMBER_OF_100_NANO_BETWEEN_1601_1970 =
        ULL(116444736000000000);
    constexpr ui64 NUMBER_OF_100_NANO_IN_SECOND = ULL(10000000);

    union TFTUnion {
        ui64 FTScalar;
        FILETIME FTStruct;
    };
} // namespace

void FileTimeToTimeval(const FILETIME* ft, timeval* tv) {
    Y_ASSERT(ft);
    Y_ASSERT(tv);
    TFTUnion ntTime;
    ntTime.FTStruct = *ft;
    ntTime.FTScalar -= NUMBER_OF_100_NANO_BETWEEN_1601_1970;
    tv->tv_sec =
        static_cast<long>(ntTime.FTScalar / NUMBER_OF_100_NANO_IN_SECOND);
    tv->tv_usec = static_cast<long>(
        (ntTime.FTScalar % NUMBER_OF_100_NANO_IN_SECOND) / LL(10));
}

void FileTimeToTimespec(const FILETIME& ft, struct timespec* ts) {
    Y_ASSERT(ts);
    TFTUnion ntTime;
    ntTime.FTStruct = ft;
    ntTime.FTScalar -= NUMBER_OF_100_NANO_BETWEEN_1601_1970;
    ts->tv_sec =
        static_cast<time_t>(ntTime.FTScalar / NUMBER_OF_100_NANO_IN_SECOND);
    ts->tv_nsec = static_cast<long>(
        (ntTime.FTScalar % NUMBER_OF_100_NANO_IN_SECOND) * LL(100));
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

namespace {
    constexpr int STRUCT_TM_BASE_YEAR = 1900;
    constexpr int UNIX_TIME_BASE_YEAR = 1970;
    constexpr long SECONDS_PER_DAY = (24L * 60L * 60L);

    constexpr bool IsLeapYear(int year) {
        if (year % 4 != 0) {
            return false;
        }
        if (year % 100 != 0) {
            return true;
        }
        return year % 400 == 0;
    }

    constexpr ui16 DAYS_IN_YEAR = 365;
    constexpr ui16 DAYS_IN_LEAP_YEAR = 366;

    constexpr ui16 YearSize(int year) {
        return IsLeapYear(year) ? DAYS_IN_LEAP_YEAR : DAYS_IN_YEAR;
    }

    constexpr ui64 FOUR_CENTURIES = (400 * 365 + 100 - 3);

    constexpr ui16 MONTH_TO_DAYS[12] = {
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

    constexpr ui16 MONTH_TO_DAYS_LEAP[12] = {
        0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};

    template <ui8 DaysInFeb>
    constexpr int DayOfYearToMonth(ui64& day) {
        Y_ASSERT(day >= 0);
        Y_ASSERT(day < 366);

        constexpr ui8 JanDays = 31;
        constexpr ui8 FebDays = JanDays + DaysInFeb;
        constexpr ui8 MarDays = FebDays + 31;
        constexpr ui8 AprDays = MarDays + 30;
        constexpr ui8 MayDays = AprDays + 31;
        constexpr ui8 JunDays = MayDays + 30;
        constexpr ui8 JulDays = JunDays + 31;
        constexpr ui16 AugDays = JulDays + 31;
        constexpr ui16 SepDays = AugDays + 30;
        constexpr ui16 OctDays = SepDays + 31;
        constexpr ui16 NovDays = OctDays + 30;

        // hard-coded binary search
        // this approach is faster that lookup in array using std::lower_bound()
        // GmTimeR takes ~40 cycles vs ~60 cycles using std::lower_bound version
        if (day < JunDays) {
            if (day < MarDays) {
                if (day < JanDays) {
                    return 0;
                } else if (day < FebDays) {
                    day -= JanDays;
                    return 1;
                } else {
                    day -= FebDays;
                    return 2;
                }
            } else {
                if (day < AprDays) {
                    day -= MarDays;
                    return 3;
                } else if (day < MayDays) {
                    day -= AprDays;
                    return 4;
                } else {
                    day -= MayDays;
                    return 5;
                }
            }
        } else {
            if (day < SepDays) {
                if (day < JulDays) {
                    day -= JunDays;
                    return 6;
                } else if (day < AugDays) {
                    day -= JulDays;
                    return 7;
                } else {
                    day -= AugDays;
                    return 8;
                }
            } else {
                if (day < OctDays) {
                    day -= SepDays;
                    return 9;
                } else if (day < NovDays) {
                    day -= OctDays;
                    return 10;
                } else {
                    day -= NovDays;
                    return 11;
                }
            }
        }
    }

    class TDayNoToYearLookupTable {
    private:
        static constexpr int TableSize = 128;
        // lookup table for years in [1970, 1970 + 128 = 2098] range
        ui16 DaysSinceEpoch[TableSize] = {};

    public:
        constexpr TDayNoToYearLookupTable() {
            DaysSinceEpoch[0] = YearSize(UNIX_TIME_BASE_YEAR);

            for (int year = UNIX_TIME_BASE_YEAR + 1; year < UNIX_TIME_BASE_YEAR + TableSize; ++year) {
                DaysSinceEpoch[year - UNIX_TIME_BASE_YEAR] = DaysSinceEpoch[year - UNIX_TIME_BASE_YEAR - 1] + YearSize(year);
            }
        }

        // lookup year by days since epoch, decrement day counter to the corresponding amount of days.
        // The method returns the last year in the table, if year is too big
        int GetYear(ui64& days) const {
            size_t year = std::upper_bound(DaysSinceEpoch, Y_ARRAY_END(DaysSinceEpoch), days) - Y_ARRAY_BEGIN(DaysSinceEpoch);
            if (year > 0) {
                days -= DaysSinceEpoch[year - 1];
            }

            return year + UNIX_TIME_BASE_YEAR;
        }
    };

    constexpr TDayNoToYearLookupTable DAYS_TO_YEAR_LOOKUP;
}

//! Inverse of gmtime: converts struct tm to time_t, assuming the data
//! in tm is UTC rather than local timezone. This implementation
//! returns the number of seconds since 1970-01-01, converted to time_t.
//! @note this code adopted from
//!       http://osdir.com/ml/web.wget.patches/2005-07/msg00010.html
//!       Subject: A more robust timegm - msg#00010
time_t TimeGM(const struct tm* t) {
    // Only handles years after 1970
    if (Y_UNLIKELY(t->tm_year < 70)) {
        return (time_t)-1;
    }

    int days = 365 * (t->tm_year - 70);
    // Take into account the leap days between 1970 and YEAR-1
    days += (t->tm_year - 1 - 68) / 4 - ((t->tm_year - 1) / 100) + ((t->tm_year - 1 + 300) / 400);

    if (Y_UNLIKELY(t->tm_mon < 0 || t->tm_mon >= 12)) {
        return (time_t)-1;
    }
    if (IsLeapYear(1900 + t->tm_year)) {
        days += MONTH_TO_DAYS_LEAP[t->tm_mon];
    } else {
        days += MONTH_TO_DAYS[t->tm_mon];
    }

    days += t->tm_mday - 1;

    unsigned long secs = days * 86400ul + t->tm_hour * 3600 + t->tm_min * 60 + t->tm_sec;
    return (time_t)secs;
}

struct tm* GmTimeR(const time_t* timer, struct tm* tmbuf) {
    i64 time = static_cast<i64>(*timer);

    ui64 dayclock, dayno;
    int year = UNIX_TIME_BASE_YEAR;

    if (Y_UNLIKELY(time < 0)) {
        ui64 shift = (ui64)(-time - 1) / (FOUR_CENTURIES * SECONDS_PER_DAY) + 1;
        time += shift * (FOUR_CENTURIES * SECONDS_PER_DAY);
        year -= shift * 400;
    }

    dayclock = (ui64)time % SECONDS_PER_DAY;
    dayno = (ui64)time / SECONDS_PER_DAY;

    if (Y_UNLIKELY(dayno >= FOUR_CENTURIES)) {
        year += 400 * (dayno / FOUR_CENTURIES);
        dayno = dayno % FOUR_CENTURIES;
    }

    tmbuf->tm_sec = dayclock % 60;
    tmbuf->tm_min = (dayclock % 3600) / 60;
    tmbuf->tm_hour = dayclock / 3600;
    tmbuf->tm_wday = (dayno + 4) % 7; // Day 0 was a thursday

    if (Y_LIKELY(year == UNIX_TIME_BASE_YEAR)) {
        year = DAYS_TO_YEAR_LOOKUP.GetYear(dayno);
    }

    for (;;) {
        const ui16 yearSize = YearSize(year);
        if (dayno < yearSize) {
            break;
        }
        dayno -= yearSize;
        ++year;
    }

    tmbuf->tm_year = year - STRUCT_TM_BASE_YEAR;
    tmbuf->tm_yday = dayno;
    tmbuf->tm_mon = IsLeapYear(year)
                        ? DayOfYearToMonth<29>(dayno)
                        : DayOfYearToMonth<28>(dayno);
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
