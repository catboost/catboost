#pragma once

#include "systime.h"

#include <util/system/platform.h>
#include <util/system/datetime.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/ylimits.h>
#include <util/generic/utility.h>
#include <util/generic/typetraits.h>
#include <util/generic/yexception.h>

#include <ctime>
#include <cstdio>

#include <time.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // conversion from 'time_t' to 'long', possible loss of data
#endif                          // _MSC_VER

// Microseconds since epoch
class TInstant;

// Duration is microseconds. Could be used to store timeouts, for example.
class TDuration;

/// Current time
static inline TInstant Now() noexcept;

/// Use Now() method to obtain current time instead of *Seconds() unless you understand what are you doing.

class TDateTimeParseException: public yexception {
};

const int DATE_BUF_LEN = 4 + 2 + 2 + 1; // [YYYYMMDD*]

inline void sprint_date(char* buf, const struct tm& theTm) {
    sprintf(buf, "%04d%02d%02d", theTm.tm_year + 1900, theTm.tm_mon + 1, theTm.tm_mday);
}

constexpr long seconds(const struct tm& theTm) {
    return 60 * (60 * theTm.tm_hour + theTm.tm_min) + theTm.tm_sec;
}

void sprint_date(char* buf, time_t when, long* sec = nullptr);
void sprint_gm_date(char* buf, time_t when, long* sec = nullptr);
bool sscan_date(const char* date, struct tm& theTm);

const int DATE_8601_LEN = 21; // strlen("YYYY-MM-DDThh:mm:ssZ") = 20 + '\0'

size_t FormatDate8601(char* buf, size_t len, time_t when);

inline void sprint_date8601(char* buf, time_t when) {
    buf[FormatDate8601(buf, 64, when)] = 0;
}

bool ParseISO8601DateTime(const char* date, time_t& utcTime);
bool ParseISO8601DateTime(const char* date, size_t dateLen, time_t& utcTime);
bool ParseRFC822DateTime(const char* date, time_t& utcTime);
bool ParseRFC822DateTime(const char* date, size_t dateLen, time_t& utcTime);
bool ParseHTTPDateTime(const char* date, time_t& utcTime);
bool ParseHTTPDateTime(const char* date, size_t dateLen, time_t& utcTime);
bool ParseX509ValidityDateTime(const char* date, time_t& utcTime);
bool ParseX509ValidityDateTime(const char* date, size_t dateLen, time_t& utcTime);

constexpr long TVdiff(timeval r1, timeval r2) {
    return (1000000 * (r2.tv_sec - r1.tv_sec) + (r2.tv_usec - r1.tv_usec));
}

TString Strftime(const char* format, const struct tm* tm);

template <class S>
class TTimeBase {
public:
    using TValue = ui64;
    constexpr TTimeBase() noexcept
        : Value_(0)
    {
    }

    constexpr TTimeBase(const TValue& value) noexcept
        : Value_(value)
    {
    }

    constexpr TTimeBase(const struct timeval& tv) noexcept
        : Value_(tv.tv_sec * (ui64)1000000 + tv.tv_usec)
    {
    }

    constexpr TValue GetValue() const noexcept {
        return Value_;
    }

    constexpr double SecondsFloat() const noexcept {
        return Value_ * (1 / 1000000.0);
    }

    constexpr TValue MicroSeconds() const noexcept {
        return Value_;
    }

    constexpr TValue MilliSeconds() const noexcept {
        return MicroSeconds() / 1000;
    }

    constexpr TValue Seconds() const noexcept {
        return MilliSeconds() / 1000;
    }

    constexpr TValue Minutes() const noexcept {
        return Seconds() / 60;
    }

    constexpr TValue Hours() const noexcept {
        return Minutes() / 60;
    }

    constexpr TValue Days() const noexcept {
        return Hours() / 24;
    }

    constexpr TValue NanoSeconds() const noexcept {
        return MicroSeconds() >= (Max<TValue>() / (TValue)1000) ? Max<TValue>() : MicroSeconds() * (TValue)1000;
    }

    constexpr ui32 MicroSecondsOfSecond() const noexcept {
        return MicroSeconds() % (TValue)1000000;
    }

    constexpr ui32 MilliSecondsOfSecond() const noexcept {
        return MicroSecondsOfSecond() / (TValue)1000;
    }

    constexpr ui32 NanoSecondsOfSecond() const noexcept {
        return MicroSecondsOfSecond() * (TValue)1000;
    }

    constexpr explicit operator bool() const noexcept {
        return Value_;
    }

protected:
    TValue Value_;
};

namespace NDateTimeHelpers {
    template <typename T>
    struct TPrecisionHelper {
        using THighPrecision = ui64;
    };

    template <>
    struct TPrecisionHelper<float> {
        using THighPrecision = double;
    };

    template <>
    struct TPrecisionHelper<double> {
        using THighPrecision = double;
    };
}

class TDuration: public TTimeBase<TDuration> {
    using TBase = TTimeBase<TDuration>;

public:
    constexpr TDuration() noexcept {
    }

    //better use static constructors
    constexpr explicit TDuration(TValue value) noexcept
        : TBase(value)
    {
    }

    constexpr TDuration(const struct timeval& tv) noexcept
        : TBase(tv)
    {
    }

    static constexpr TDuration MicroSeconds(ui64 us) noexcept {
        return TDuration(us);
    }

    /* noexcept(false) as conversion from T might throw, for example FromString("abc") */
    template <typename T>
    static constexpr TDuration MilliSeconds(T ms) noexcept(false) {
        return MicroSeconds((ui64)(typename NDateTimeHelpers::TPrecisionHelper<T>::THighPrecision(ms) * 1000));
    }

    using TBase::Days;
    using TBase::Hours;
    using TBase::Minutes;
    using TBase::Seconds;
    using TBase::MilliSeconds;
    using TBase::MicroSeconds;

    /// DeadLineFromTimeOut
    inline TInstant ToDeadLine() const;
    constexpr TInstant ToDeadLine(TInstant now) const;

    static constexpr TDuration Max() noexcept {
        return TDuration(::Max<TValue>());
    }

    static constexpr TDuration Zero() noexcept {
        return TDuration();
    }

    /* noexcept(false) as conversion from T might throw, for example FromString("abc") */
    template <typename T>
    static constexpr TDuration Seconds(T s) noexcept(false) {
        return MilliSeconds(typename NDateTimeHelpers::TPrecisionHelper<T>::THighPrecision(s) * 1000);
    }

    static constexpr TDuration Minutes(ui64 m) noexcept {
        return Seconds(m * 60);
    }

    static constexpr TDuration Hours(ui64 h) noexcept {
        return Minutes(h * 60);
    }

    static constexpr TDuration Days(ui64 d) noexcept {
        return Hours(d * 24);
    }

    /// parses strings like 10s, 15ms, 15.05s, 20us, or just 25 (s). See parser_ut.cpp for details
    static TDuration Parse(const TStringBuf input);

    static bool TryParse(const TStringBuf input, TDuration& result);

    // note global Out method is defined for TDuration, so it could be written to IOutputStream as text

    template <class T>
    inline TDuration& operator+=(const T& t) noexcept {
        return (*this = (*this + t));
    }

    template <class T>
    inline TDuration& operator-=(const T& t) noexcept {
        return (*this = (*this - t));
    }

    template <class T>
    inline TDuration& operator*=(const T& t) noexcept {
        return (*this = (*this * t));
    }

    template <class T>
    inline TDuration& operator/=(const T& t) noexcept {
        return (*this = (*this / t));
    }

    TString ToString() const;
};

Y_DECLARE_PODTYPE(TDuration);

/// TInstant and TDuration are guaranteed to have same precision
class TInstant: public TTimeBase<TInstant> {
    using TBase = TTimeBase<TInstant>;

public:
    constexpr TInstant() {
    }

    //better use static constructors
    constexpr explicit TInstant(TValue value)
        : TBase(value)
    {
    }

    constexpr TInstant(const struct timeval& tv)
        : TBase(tv)
    {
    }

    static inline TInstant Now() {
        return TInstant::MicroSeconds(::MicroSeconds());
    }

    using TBase::Days;
    using TBase::Hours;
    using TBase::Minutes;
    using TBase::Seconds;
    using TBase::MilliSeconds;
    using TBase::MicroSeconds;

    static constexpr TInstant Max() noexcept {
        return TInstant(::Max<TValue>());
    }

    static constexpr TInstant Zero() noexcept {
        return TInstant();
    }

    /// us since epoch
    static constexpr TInstant MicroSeconds(ui64 us) noexcept {
        return TInstant(us);
    }

    /// ms since epoch
    static constexpr TInstant MilliSeconds(ui64 ms) noexcept {
        return MicroSeconds(ms * 1000);
    }

    /// seconds since epoch
    static constexpr TInstant Seconds(ui64 s) noexcept {
        return MilliSeconds(s * 1000);
    }

    /// minutes since epoch
    static constexpr TInstant Minutes(ui64 m) noexcept {
        return Seconds(m * 60);
    }

    /// hours since epoch
    static constexpr TInstant Hours(ui64 h) noexcept {
        return Minutes(h * 60);
    }

    /// days since epoch
    static constexpr TInstant Days(ui64 d) noexcept {
        return Hours(d * 24);
    }

    constexpr time_t TimeT() const noexcept {
        return (time_t)Seconds();
    }

    inline struct timeval TimeVal() const noexcept {
        struct timeval tv;
        ::Zero(tv);
        tv.tv_sec = TimeT();
        tv.tv_usec = MicroSecondsOfSecond();
        return tv;
    }

    inline struct tm* LocalTime(struct tm* tm) const noexcept {
        time_t clock = Seconds();
        return localtime_r(&clock, tm);
    }

    inline struct tm* GmTime(struct tm* tm) const noexcept {
        time_t clock = Seconds();
        return GmTimeR(&clock, tm);
    }

    /**
     * Formats the instant using the UTC time zone, with microsecond precision.
     *
     * @returns An ISO 8601 formatted string, e.g. '2015-11-21T23:30:27.991669Z'.
     * @note Global Out method is defined to TInstant, so it can be written as text to IOutputStream.
     */
    TString ToString() const;

    /**
     * Formats the instant using the UTC time zone, with second precision.
     *
     * @returns An ISO 8601 formatted string, e.g. '2015-11-21T23:30:27Z'.
     */
    TString ToStringUpToSeconds() const;

    /**
     * Formats the instant using the system time zone, with microsecond precision.
     *
     * @returns An ISO 8601 formatted string, e.g. '2015-11-22T04:30:27.991669+0500'.
     */
    TString ToStringLocal() const;

    /**
     * Formats the instant using the system time zone, with second precision.
     *
     * @returns An ISO 8601 formatted string, e.g. '2015-11-22T04:30:27+0500'.
     */
    TString ToStringLocalUpToSeconds() const;

    static TInstant ParseIso8601(const TStringBuf);
    static TInstant ParseRfc822(const TStringBuf);
    static TInstant ParseHttp(const TStringBuf);
    static TInstant ParseX509Validity(const TStringBuf);

    static bool TryParseIso8601(const TStringBuf input, TInstant& instant);
    static bool TryParseRfc822(const TStringBuf input, TInstant& instant);
    static bool TryParseHttp(const TStringBuf input, TInstant& instant);
    static bool TryParseX509(const TStringBuf input, TInstant& instant);

    template <class T>
    inline TInstant& operator+=(const T& t) noexcept {
        return (*this = (*this + t));
    }

    template <class T>
    inline TInstant& operator-=(const T& t) noexcept {
        return (*this = (*this - t));
    }
};

Y_DECLARE_PODTYPE(TInstant);

namespace NPrivate {
    template <bool PrintUpToSeconds>
    struct TPrintableLocalTime {
        TInstant MomentToPrint;

        constexpr explicit TPrintableLocalTime(TInstant momentToPrint)
            : MomentToPrint(momentToPrint)
        {
        }
    };
}

/** @name Helpers for printing local times to `IOutputStream`s.
 *        The FormatLocal* functions create an opaque object that, when written to
 *        a `IOutputStream`, outputs this instant as an ISO 8601 formatted string
 *        using the system time zone.
 *
 *  @note The only reason behind this set of functions is to avoid excessive
 *        allocations when you directly print the local time to a stream.
 *
 *        If you need something beyond just printing the value or your code
 *        is not performance-critical, feel free to use the corresponding
 *        TInstant::ToString*() functions.
 */
///@{
/// @see TInstant::ToStringLocal()
::NPrivate::TPrintableLocalTime<false> FormatLocal(TInstant instant);
/// @see TInstant::ToStringLocalUpToSeconds()
::NPrivate::TPrintableLocalTime<true> FormatLocalUpToSeconds(TInstant instant);
///@}

template <class S>
static constexpr bool operator<(const TTimeBase<S>& l, const TTimeBase<S>& r) noexcept {
    return l.GetValue() < r.GetValue();
}

template <class S>
static constexpr bool operator<=(const TTimeBase<S>& l, const TTimeBase<S>& r) noexcept {
    return l.GetValue() <= r.GetValue();
}

template <class S>
static constexpr bool operator==(const TTimeBase<S>& l, const TTimeBase<S>& r) noexcept {
    return l.GetValue() == r.GetValue();
}

template <class S>
static constexpr bool operator!=(const TTimeBase<S>& l, const TTimeBase<S>& r) noexcept {
    return l.GetValue() != r.GetValue();
}

template <class S>
static constexpr bool operator>(const TTimeBase<S>& l, const TTimeBase<S>& r) noexcept {
    return l.GetValue() > r.GetValue();
}

template <class S>
static constexpr bool operator>=(const TTimeBase<S>& l, const TTimeBase<S>& r) noexcept {
    return l.GetValue() >= r.GetValue();
}

namespace NDateTimeHelpers {
    template <typename T>
    static constexpr T SumWithSaturation(T a, T b) {
        static_assert(!std::numeric_limits<T>::is_signed, "expect !std::numeric_limits<T>::is_signed");

        return Max<T>() - a < b ? Max<T>() : a + b;
    }

    template <typename T>
    static constexpr T DiffWithSaturation(T a, T b) {
        static_assert(!std::numeric_limits<T>::is_signed, "expect !std::numeric_limits<T>::is_signed");

        return a < b ? 0 : a - b;
    }
}

constexpr TDuration operator-(const TInstant& l, const TInstant& r) noexcept {
    return TDuration(::NDateTimeHelpers::DiffWithSaturation(l.GetValue(), r.GetValue()));
}

constexpr TInstant operator+(const TInstant& i, const TDuration& d) noexcept {
    return TInstant(::NDateTimeHelpers::SumWithSaturation(i.GetValue(), d.GetValue()));
}

constexpr TInstant operator-(const TInstant& i, const TDuration& d) noexcept {
    return TInstant(::NDateTimeHelpers::DiffWithSaturation(i.GetValue(), d.GetValue()));
}

constexpr TDuration operator-(const TDuration& l, const TDuration& r) noexcept {
    return TDuration(::NDateTimeHelpers::DiffWithSaturation(l.GetValue(), r.GetValue()));
}

constexpr TDuration operator+(const TDuration& l, const TDuration& r) noexcept {
    return TDuration(::NDateTimeHelpers::SumWithSaturation(l.GetValue(), r.GetValue()));
}

template <class T>
inline TDuration operator*(const TDuration& d, const T& t) noexcept {
    Y_ASSERT(t >= T());
    Y_ASSERT(t == T() || Max<TDuration::TValue>() / t >= d.GetValue());
    return TDuration(d.GetValue() * t);
}

template <class T, std::enable_if_t<!std::is_same<TDuration, T>::value, int> = 0>
constexpr TDuration operator/(const TDuration& d, const T& t) noexcept {
    return TDuration(d.GetValue() / t);
}

constexpr double operator/(const TDuration& x, const TDuration& y) noexcept {
    return static_cast<double>(x.GetValue()) / static_cast<double>(y.GetValue());
}

inline TInstant TDuration::ToDeadLine() const {
    return ToDeadLine(TInstant::Now());
}

constexpr TInstant TDuration::ToDeadLine(TInstant now) const {
    return now + *this;
}

void Sleep(TDuration duration);
void SleepUntil(TInstant instant);

static inline TInstant Now() noexcept {
    return TInstant::Now();
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER
