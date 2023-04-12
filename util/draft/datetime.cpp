#include "datetime.h"

#include <util/ysaveload.h>

#include <util/system/fasttime.h>
#include <util/datetime/base.h>
#include <util/datetime/systime.h>
#include <util/stream/output.h>
#include <util/stream/mem.h>
#include <util/string/cast.h>
#include <util/string/printf.h>

namespace NDatetime {
    const ui32 MonthDays[2][12] = {
        {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}, //nleap
        {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}  //leap
    };

    const ui32 MonthDaysNewYear[2][13] = {
        {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365}, //nleap
        {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}  //leap
    };

    void YDayToMonthAndDay(ui32 yday, bool isleap, ui32* month, ui32* mday) {
        const ui32* begin = MonthDaysNewYear[isleap] + 1;
        const ui32* end = begin + 12;
        // [31, ..., 365] or [31, ..., 366] (12 elements)

        const ui32* pos = UpperBound(begin, end, yday);
        Y_ENSURE(pos != end, "day no. " << yday << " does not exist in " << (isleap ? "leap" : "non-leap") << " year");

        *month = pos - begin;
        *mday = yday - *(pos - 1) + 1;

        Y_ASSERT((*month < 12) && (1 <= *mday) && (*mday <= MonthDays[isleap][*month]));
    }

    struct TTimeData {
        i32 IsDst = 0;
        i32 GMTOff = 0;

        TTimeData(time_t t) {
            struct ::tm tt;
            ::localtime_r(&t, &tt);
#ifndef _win_
            GMTOff = tt.tm_gmtoff;
#else
            TIME_ZONE_INFORMATION tz;
            switch (GetTimeZoneInformation(&tz)) {
                case TIME_ZONE_ID_UNKNOWN:
                    GMTOff = tz.Bias * -60;
                    break;
                case TIME_ZONE_ID_STANDARD:
                    GMTOff = (tz.Bias + tz.StandardBias) * -60;
                    break;
                case TIME_ZONE_ID_DAYLIGHT:
                    GMTOff = (tz.Bias + tz.DaylightBias) * -60;
                    break;
                default:
                    break;
            }
#endif
            IsDst = tt.tm_isdst;
        }
    };

    TSimpleTM TSimpleTM::CurrentUTC() {
        return New((time_t)TInstant::MicroSeconds(InterpolatedMicroSeconds()).Seconds());
    }

    TSimpleTM TSimpleTM::New(time_t t, i32 gmtoff, i8 isdst) {
        time_t tt = t + gmtoff + isdst * 3600;
        struct tm tmSys;
        Zero(tmSys);
        GmTimeR(&tt, &tmSys);
        tmSys.tm_isdst = isdst;
#ifndef _win_
        tmSys.tm_gmtoff = gmtoff;
#endif

        return New(tmSys);
    }

    TSimpleTM TSimpleTM::NewLocal(time_t t) {
        TTimeData d(t);
        return New(t, d.GMTOff, d.IsDst);
    }

    TSimpleTM TSimpleTM::New(const struct tm& t) {
        TSimpleTM res;
        res.IsDst = t.tm_isdst;
        res.Sec = t.tm_sec;
        res.Min = t.tm_min;
        res.Hour = t.tm_hour;
        res.WDay = t.tm_wday;
        res.Mon = t.tm_mon;
        res.MDay = t.tm_mday;
        res.Year = t.tm_year;
        res.YDay = t.tm_yday;
        res.IsLeap = LeapYearAD(res.Year + 1900);
#ifndef _win_
        res.GMTOff = t.tm_gmtoff;
#endif
        return res;
    }

    TSimpleTM& TSimpleTM::SetRealDate(ui32 year, ui32 mon, ui32 mday, ui32 hour, ui32 min, ui32 sec, i32 isdst) {
        mday = ::Max<ui32>(mday, 1);
        mon = ::Min<ui32>(::Max<ui32>(mon, 1), 12);
        year = ::Max<ui32>(year, 1900);

        IsLeap = LeapYearAD(year);
        Year = year - 1900;
        Mon = mon - 1;
        MDay = ::Min<ui32>(mday, MonthDays[IsLeap][Mon]);
        Hour = Max<ui32>() == hour ? Hour : ::Min<ui32>(hour, 23);
        Min = Max<ui32>() == min ? Min : ::Min<ui32>(min, 59);
        Sec = Max<ui32>() == sec ? Sec : ::Min<ui32>(sec, 60);
        IsDst = isdst;

        return RegenerateFields();
    }

    TSimpleTM& TSimpleTM::RegenerateFields() {
        return *this = New(AsTimeT(), GMTOff, IsDst);
    }

    TSimpleTM& TSimpleTM::Add(EField f, i32 amount) {
        if (!amount) {
            return *this;
        }

        switch (f) {
            default:
                return *this;
            case F_DAY:
                amount *= 24;
                [[fallthrough]];
            case F_HOUR:
                amount *= 60;
                [[fallthrough]];
            case F_MIN:
                amount *= 60;
                [[fallthrough]];
            case F_SEC: {
                return *this = New(AsTimeT() + amount, GMTOff, IsDst);
            }
            case F_YEAR: {
                i32 y = amount + (i32)Year;
                y = ::Min<i32>(Max<i32>(y, 0), 255 /*max year*/);

                // YDay may correspond to different MDay if it's March or greater and the years have different leap status
                if (Mon > 1) {
                    YDay += (i32)LeapYearAD(RealYear()) - (i32)LeapYearAD(RealYear());
                }

                Year = y;
                IsLeap = LeapYearAD(RealYear());
                return RegenerateFields();
            }
            case F_MON: {
                i32 m = amount + Mon;
                i32 y = (m < 0 ? (-12 + m) : m) / 12;
                m = m - y * 12;

                if (y) {
                    Add(F_YEAR, y);
                }

                if (m >= 0 && m < 12) {
                    MDay = ::Min<ui32>(MonthDays[IsLeap][m], MDay);
                    Mon = m;
                }

                return RegenerateFields();
            }
        }
    }

    TString TSimpleTM::ToString(const char* fmt) const {
        struct tm t = *this;
        return Strftime(fmt, &t);
    }

    time_t TSimpleTM::AsTimeT() const {
        struct tm t = AsStructTmLocal();
        return TimeGM(&t) - GMTOff - IsDst * 3600;
    }

    struct tm TSimpleTM::AsStructTmUTC() const {
        struct tm res;
        Zero(res);
        time_t t = AsTimeT();
        return *GmTimeR(&t, &res);
    }

    struct tm TSimpleTM::AsStructTmLocal() const {
        struct tm t;
        Zero(t);
        t.tm_isdst = IsDst;
        t.tm_sec = Sec;
        t.tm_min = Min;
        t.tm_hour = Hour;
        t.tm_wday = WDay;
        t.tm_mon = Mon;
        t.tm_mday = MDay;
        t.tm_year = Year;
        t.tm_yday = YDay;
#ifndef _win_
        t.tm_gmtoff = GMTOff;
#endif
        return t;
    }
}

template <>
void In<TMonth>(IInputStream& in, TMonth& t) {
    char buf[4];
    LoadPodArray(&in, buf, 4);
    t.Year = FromString<ui16>(buf, 4);
    LoadPodArray(&in, buf, 2);
    t.Month = ui8(FromString<ui16>(buf, 2)) - 1;
}

template <>
void Out<TMonth>(IOutputStream& o, const TMonth& t) {
    o << t.Year << Sprintf("%.2hu", (ui16)(t.Month + 1));
}

template <>
TMonth FromStringImpl<TMonth, char>(const char* s, size_t len) {
    TMonth res;
    TMemoryInput in(s, len);
    in >> res;
    return res;
}
