#pragma once

#include <util/generic/algorithm.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/generic/ymath.h>
#include <util/datetime/base.h>

#include <cstdlib>

#include <time.h>

namespace NDatetime {
    extern const ui32 MonthDays[2][12];        // !leapYear; !!leapYear
    extern const ui32 MonthDaysNewYear[2][13]; // !leapYear; !!leapYear

    inline ui32 YearDaysAD(ui32 year) {
        year = Max<ui32>(year, 1) - 1; //1 AD comes straight after 1 BC, no 0 AD
        return year * 365 + year / 4 - year / 100 + year / 400;
    }

    inline bool LeapYearAD(ui32 year) {
        return (!(year % 4) && (year % 100)) || !(year % 400);
    }

    inline ui32 YDayFromMonthAndDay(ui32 month /*0 - based*/, ui32 mday /*1 - based*/, bool isleap) {
        return MonthDaysNewYear[isleap][Min(month, (ui32)11u)] + mday - 1;
    }

    void YDayToMonthAndDay(ui32 yday /*0 - based*/, bool isleap, ui32* month /*0 - based*/, ui32* mday /*1 - based*/);

    struct TSimpleTM {
        enum EField {
            F_NONE = 0,
            F_SEC,
            F_MIN,
            F_HOUR,
            F_DAY,
            F_MON,
            F_YEAR
        };

        i32 GMTOff = 0; // -43200 - 50400 seconds
        ui16 Year = 0;  // from 1900
        ui16 YDay = 0;  // 0-365
        ui8 Mon = 0;    // 0-11
        ui8 MDay = 0;   // 1-31
        ui8 WDay = 0;   // 0-6
        ui8 Hour = 0;   // 0-23
        ui8 Min = 0;    // 0-59
        ui8 Sec = 0;    // 0-60 - doesn't care for leap seconds. Most of the time it's ok.
        i8 IsDst = 0;   // -1/0/1
        bool IsLeap = false;

    public:
        static TSimpleTM New(time_t t = 0, i32 gmtoff = 0, i8 isdst = 0);
        static TSimpleTM NewLocal(time_t = 0);

        static TSimpleTM New(const struct tm&);

        static TSimpleTM CurrentUTC();

        TSimpleTM() = default;

        TSimpleTM(ui32 year, ui32 mon, ui32 day, ui32 h = 0, ui32 m = 0, ui32 s = 0) {
            Zero(*this);
            SetRealDate(year, mon, day, h, m, s);
        }

        // keeps the object consistent
        TSimpleTM& Add(EField f, i32 amount = 1);

        TString ToString(const char* fmt = "%a, %d %b %Y %H:%M:%S %z") const;

        TSimpleTM& ToUTC() {
            return *this = New(AsTimeT());
        }

        bool IsUTC() const {
            return !IsDst && !GMTOff;
        }

        time_t AsTimeT() const;

        operator time_t() const {
            return AsTimeT();
        }

        struct tm AsStructTmLocal() const;

        struct tm AsStructTmUTC() const;

        operator struct tm() const {
            return AsStructTmLocal();
        }

        ui32 RealYear() const {
            return ui32(Year + 1900);
        }

        ui32 RealMonth() const {
            return ui32(Mon + 1);
        }

        TSimpleTM& SetRealDate(ui32 year, ui32 mon, ui32 mday, ui32 hour = -1, ui32 min = -1, ui32 sec = -1, i32 isdst = 0);

        // regenerates all fields from Year, MDay, Hour, Min, Sec, IsDst, GMTOffset
        TSimpleTM& RegenerateFields();

        friend bool operator==(const TSimpleTM& a, const TSimpleTM& b) {
            return a.AsTimeT() == b.AsTimeT();
        }

        friend bool operator==(const TSimpleTM& s, const struct tm& t) {
            return s == New(t);
        }

        friend bool operator==(const struct tm& t, const TSimpleTM& s) {
            return s == t;
        }

        friend bool operator!=(const TSimpleTM& a, const TSimpleTM& b) {
            return !(a == b);
        }

        friend bool operator!=(const TSimpleTM& s, const struct tm& t) {
            return !(s == t);
        }

        friend bool operator!=(const struct tm& t, const TSimpleTM& s) {
            return s != t;
        }
    };
}

inline TString date2str(const time_t date) {
    struct tm dateTm;
    memset(&dateTm, 0, sizeof(dateTm));
    localtime_r(&date, &dateTm);
    char buf[9];
    strftime(buf, sizeof(buf), "%Y%m%d", &dateTm);
    return TString(buf);
}

inline time_t str2date(const TString& dateStr) {
    struct tm dateTm;
    memset(&dateTm, 0, sizeof(tm));
    strptime(dateStr.data(), "%Y%m%d", &dateTm);
    return mktime(&dateTm);
}

// checks whether time2 > time1 and close enough to it
inline bool AreTimesSeqAndClose(time_t time1, time_t time2, time_t closeInterval = 10) {
    return (time2 - time1) <= closeInterval;
}

// checks whether time2 and time1 are close enough
inline bool AreTimesClose(time_t time1, time_t time2, time_t closeInterval = 10) {
    return std::abs(time2 - time1) <= closeInterval;
}

////////////////////////////////

struct TMonth {
    ui16 Year;
    ui8 Month;

    TMonth(ui16 year = 0, ui8 month = 0)
        : Year(year)
        , Month(month)
    {
    }

    TMonth operator-(ui16 n) {
        if (n <= Month) {
            return TMonth(Year, Month - (ui8)n);
        } else {
            n -= Month;
            return (n % 12) ? TMonth(Year - 1 - (n / 12), 12 - (n % 12)) : TMonth(Year - (n / 12), 0);
        }
    }
};

Y_DECLARE_PODTYPE(NDatetime::TSimpleTM);
