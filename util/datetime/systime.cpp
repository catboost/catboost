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
    constexpr ui64 SECONDS_PER_DAY = (24L * 60L * 60L);

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

    constexpr ui32 FOUR_CENTURY_YEARS = 400;

    constexpr ui32 LeapYearCount(ui32 years) {
        return years / 4 - years / 100 + years / 400;
    }

    constexpr ui32 FOUR_CENTURY_DAYS = FOUR_CENTURY_YEARS * DAYS_IN_YEAR + LeapYearCount(FOUR_CENTURY_YEARS);

    constexpr int FindYearWithin4Centuries(ui32& dayno) {
        Y_ASSERT(dayno < FOUR_CENTURY_DAYS);
        ui32 years = dayno / DAYS_IN_YEAR;

        const ui32 diff = years * DAYS_IN_YEAR + LeapYearCount(years);

        if (diff <= dayno) {
            dayno -= diff;
        } else {
            dayno -= diff - YearSize(static_cast<int>(years));
            --years;
        }

        return static_cast<int>(years);
    }

    constexpr ui16 MONTH_TO_DAYS[12] = {
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

    constexpr ui16 MONTH_TO_DAYS_LEAP[12] = {
        0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};

    constexpr int DayOfYearToMonth(ui32& yearDay, const bool leapYear) {
        if (yearDay >= 31 + 28 + leapYear) {
            yearDay += 2 - leapYear;
        }
        const ui32 month = (yearDay * 67 + 35) >> 11;
        yearDay -= (month * 489 + 8) >> 4;
        return static_cast<int>(month);
    }

    class TDayNoToYearLookupTable {
        static constexpr int TableSize = 128;
        // lookup table for years in [StartYear, StartYear + TableSize] range
        ui16 DaysSinceEpoch[TableSize] = {};

    public:
        static constexpr int StartYear = 1970;
        static constexpr int StartDays = (StartYear - UNIX_TIME_BASE_YEAR) * DAYS_IN_YEAR + LeapYearCount(StartYear - 1) - LeapYearCount(UNIX_TIME_BASE_YEAR - 1);
        static constexpr i64 MinTimestamp = StartDays * static_cast<i64>(SECONDS_PER_DAY);
        static constexpr i64 MaxTimestamp = MinTimestamp + static_cast<i64>(TableSize) * DAYS_IN_LEAP_YEAR * SECONDS_PER_DAY - 1;
        constexpr TDayNoToYearLookupTable() {
            ui16 daysAccumulated = 0;

            for (int year = StartYear; year < StartYear + TableSize; ++year) {
                daysAccumulated += YearSize(year);
                DaysSinceEpoch[year - StartYear] = daysAccumulated;
            }
        }

        // lookup year by days since epoch, decrement day counter to the corresponding amount of days.
        // The method returns the last year in the table, if year is too big
        int FindYear(ui32& days) const {
            const ui32 yearIndex = days / DAYS_IN_LEAP_YEAR;

            // we can miss by at most 1 year
            Y_ASSERT(yearIndex < TableSize);
            if (const auto diff = DaysSinceEpoch[yearIndex]; diff <= days) {
                days -= diff;
                return static_cast<int>(yearIndex + StartYear + 1);
            }

            if (yearIndex > 0) {
                days -= DaysSinceEpoch[yearIndex - 1];
            }

            return static_cast<int>(yearIndex + StartYear);
        }
    };

    constexpr TDayNoToYearLookupTable DAYS_TO_YEAR_LOOKUP;
} // namespace

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
    tm* resut = tmbuf;
    int dayClock;
    ui32 daysRemaining;
    bool isLeapYear;

    if (time >= TDayNoToYearLookupTable::MinTimestamp && time <= TDayNoToYearLookupTable::MaxTimestamp)
    {
        dayClock = static_cast<int>(time % SECONDS_PER_DAY);
        daysRemaining = time / SECONDS_PER_DAY;
        tmbuf->tm_wday = static_cast<int>((daysRemaining + 4) % 7); // Day 0 was a thursday
        daysRemaining -= TDayNoToYearLookupTable::StartDays;
        const int year = DAYS_TO_YEAR_LOOKUP.FindYear(daysRemaining);
        isLeapYear = IsLeapYear(year);
        tmbuf->tm_year = year - STRUCT_TM_BASE_YEAR;
    } else {
        i64 year = UNIX_TIME_BASE_YEAR;

        if (Y_UNLIKELY(time < 0)) {
            const ui64 shift = (ui64)(-time - 1) / (static_cast<ui64>(FOUR_CENTURY_DAYS) * SECONDS_PER_DAY) + 1;
            time += static_cast<i64>(shift * FOUR_CENTURY_DAYS * SECONDS_PER_DAY);
            year -= static_cast<i64>(shift * FOUR_CENTURY_YEARS);
        }

        dayClock = static_cast<int>(time % SECONDS_PER_DAY);
        ui64 dayNo = (ui64)time / SECONDS_PER_DAY;
        tmbuf->tm_wday = (dayNo + 4) % 7; // Day 0 was a thursday

        if (int shiftYears = (year - 1) % FOUR_CENTURY_YEARS; shiftYears != 0) {
            if (shiftYears < 0) {
                shiftYears += FOUR_CENTURY_YEARS;
            }
            year -= shiftYears;
            dayNo += shiftYears * DAYS_IN_YEAR + LeapYearCount(shiftYears);
        }

        if (Y_UNLIKELY(dayNo >= FOUR_CENTURY_DAYS)) {
            year += FOUR_CENTURY_YEARS * (dayNo / FOUR_CENTURY_DAYS);
            dayNo = dayNo % FOUR_CENTURY_DAYS;
        }

        daysRemaining = dayNo;
        const int yearDiff = FindYearWithin4Centuries(daysRemaining);
        year += yearDiff;
        isLeapYear = IsLeapYear(yearDiff + 1);
        tmbuf->tm_year = static_cast<int>(year - STRUCT_TM_BASE_YEAR);

        // check year overflow
        if (Y_UNLIKELY(year - STRUCT_TM_BASE_YEAR != tmbuf->tm_year)) {
            resut = nullptr;
        }
    }

    tmbuf->tm_sec = dayClock % 60;
    tmbuf->tm_min = (dayClock % 3600) / 60;
    tmbuf->tm_hour = dayClock / 3600;

    tmbuf->tm_yday = static_cast<int>(daysRemaining);
    tmbuf->tm_mon = DayOfYearToMonth(daysRemaining, isLeapYear);
    tmbuf->tm_mday = static_cast<int>(daysRemaining + 1);
    tmbuf->tm_isdst = 0;
#ifndef _win_
    tmbuf->tm_gmtoff = 0;
    tmbuf->tm_zone = (char*)"UTC";
#endif

    return resut;
}

TString CTimeR(const time_t* timer) {
    char sTime[32];
    sTime[0] = 0;
    ctime_r(timer, &sTime[0]);
    return sTime;
}
