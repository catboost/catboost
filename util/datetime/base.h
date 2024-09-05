#pragma once

#include "systime.h"

#include <util/str_stl.h>
#include <util/system/platform.h>
#include <util/system/datetime.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/ylimits.h>
#include <util/generic/utility.h>
#include <util/generic/typetraits.h>
#include <util/generic/yexception.h>

#include <chrono>

#if defined(__cpp_lib_three_way_comparison)
    #include <compare>
#endif

#include <ctime>
#include <cstdio>
#include <ratio>

#include <time.h>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4244) // conversion from 'time_t' to 'long', possible loss of data
#endif                              // _MSC_VER

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

constexpr long seconds(const struct tm& theTm) {
    return 60 * (60 * theTm.tm_hour + theTm.tm_min) + theTm.tm_sec;
}

void sprint_gm_date(char* buf, time_t when, long* sec = nullptr);
bool sscan_date(const char* date, struct tm& theTm);

const int DATE_8601_LEN = 21; // strlen("YYYY-MM-DDThh:mm:ssZ") = 20 + '\0'

size_t FormatDate8601(char* buf, size_t len, time_t when);

inline void sprint_date8601(char* buf, time_t when) {
    buf[FormatDate8601(buf, 64, when)] = 0;
}

bool ParseISO8601DateTimeDeprecated(const char* date, time_t& utcTime);
bool ParseISO8601DateTimeDeprecated(const char* date, size_t dateLen, time_t& utcTime);
bool ParseRFC822DateTimeDeprecated(const char* date, time_t& utcTime);
bool ParseRFC822DateTimeDeprecated(const char* date, size_t dateLen, time_t& utcTime);
bool ParseHTTPDateTimeDeprecated(const char* date, time_t& utcTime);
bool ParseHTTPDateTimeDeprecated(const char* date, size_t dateLen, time_t& utcTime);
bool ParseX509ValidityDateTimeDeprecated(const char* date, time_t& utcTime);
bool ParseX509ValidityDateTimeDeprecated(const char* date, size_t dateLen, time_t& utcTime);

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

// Use functions below instead of sprint_date (check IGNIETFERRO-892 for details)
void DateToString(char* buf, const struct tm& theTm);
void DateToString(char* buf, time_t when, long* sec = nullptr);
TString DateToString(const struct tm& theTm);
TString DateToString(time_t when, long* sec = nullptr);
// Year in format "YYYY", throws an exception if year not in range [0, 9999]
TString YearToString(const struct tm& theTm);
TString YearToString(time_t when);

template <class S>
class TTimeBase {
public:
    using TValue = ui64;

protected:
    constexpr TTimeBase(const TValue& value) noexcept
        : Value_(value)
    {
    }

public:
    constexpr TTimeBase() noexcept
        : Value_(0)
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

    constexpr double MillisecondsFloat() const noexcept {
        return Value_ * (1 / 1000.0);
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
    TValue Value_; // microseconds count
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
} // namespace NDateTimeHelpers

class TDuration: public TTimeBase<TDuration> {
    using TBase = TTimeBase<TDuration>;

private:
    /**
     * private construct from microseconds
     */
    constexpr explicit TDuration(TValue value) noexcept
        : TBase(value)
    {
    }

public:
    constexpr TDuration() noexcept {
    }

    constexpr TDuration(const struct timeval& tv) noexcept
        : TBase(tv)
    {
    }

    /**
     * TDuration is compatible with std::chrono::duration:
     *   it can be constructed and compared with std::chrono::duration.
     * But there are two significant and dangerous differences between them:
     *   1) TDuration is never negative and use saturation between 0 and maximum value.
     *      std::chrono::duration can be negative and can overflow.
     *   2) TDuration uses integer number of microseconds.
     *      std::chrono::duration is flexible, can be integer of floating point,
     *      can have different precisions.
     * So when casted from std::chrono::duration to TDuration value is clamped and rounded.
     * In arithmetic operations std::chrono::duration argument is only rounded,
     *   result is TDuration and it clamped and rounded.
     * In comparisons std::chrono::duration argument is rounded.
     */
    template <typename T, typename TRatio>
    constexpr TDuration(std::chrono::duration<T, TRatio> duration) noexcept {
        static_assert(
            std::ratio_greater_equal<TRatio, std::micro>::value &&
                (!std::is_floating_point<T>::value || std::ratio_greater<TRatio, std::micro>::value),
            "Extremely likely it is loss of precision, because TDuration stores microseconds. "
            "Cast you duration explicitly to microseconds if you really need it.");

        if (duration.count() < 0) {
            *this = TDuration::Zero(); // clamp from the bottom
        } else {
            if
#if !defined(__NVCC__)
                constexpr
#endif
                /* if [constexpr] */ (std::ratio_greater<TRatio, std::micro>::value || std::is_floating_point<T>::value) {
                // clamp from the top
                using TCommonDuration = std::chrono::duration<typename std::common_type<T, TValue>::type, TRatio>;
                constexpr auto maxDuration = std::chrono::duration<TValue, std::micro>(::Max<TValue>());
                if (std::chrono::duration_cast<TCommonDuration>(duration) >= std::chrono::duration_cast<TCommonDuration>(maxDuration)) {
                    *this = TDuration::Max();
                    return;
                }
            }
            const TValue us = std::chrono::duration_cast<std::chrono::duration<TValue, std::micro>>(duration).count();
            *this = TDuration::MicroSeconds(us);
        }
    }

    static constexpr TDuration FromValue(TValue value) noexcept {
        return TDuration(value);
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
    using TBase::MicroSeconds;
    using TBase::MilliSeconds;
    using TBase::Minutes;
    using TBase::Seconds;

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
    [[nodiscard]] static TDuration Parse(const TStringBuf input);

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

template <>
struct THash<TDuration> {
    size_t operator()(const TDuration& key) const {
        return THash<TDuration::TValue>()(key.GetValue());
    }
};

/// TInstant and TDuration are guaranteed to have same precision
class TInstant: public TTimeBase<TInstant> {
    using TBase = TTimeBase<TInstant>;

private:
    /**
     * private construct from microseconds since epoch
     */
    constexpr explicit TInstant(TValue value) noexcept
        : TBase(value)
    {
    }

public:
    constexpr TInstant() noexcept {
    }

    constexpr TInstant(const struct timeval& tv) noexcept
        : TBase(tv)
    {
    }

    static constexpr TInstant FromValue(TValue value) noexcept {
        return TInstant(value);
    }

    static inline TInstant Now() {
        return TInstant::MicroSeconds(::MicroSeconds());
    }

    using TBase::Days;
    using TBase::Hours;
    using TBase::MicroSeconds;
    using TBase::MilliSeconds;
    using TBase::Minutes;
    using TBase::Seconds;

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
     * Formats the instant using the UTC time zone.
     *
     * @returns An RFC822 formatted string, e.g. 'Sun, 06 Nov 1994 08:49:37 GMT'.
     */
    TString ToRfc822String() const;

    /**
     * Formats the instant using the UTC time zone, with second precision.
     *
     * @returns An ISO 8601 formatted string, e.g. '2015-11-21T23:30:27Z'.
     */
    TString ToStringUpToSeconds() const;

    /**
     * Formats the instant using the system time zone, with microsecond precision.
     *
     * @returns An ISO 8601 / RFC 3339 formatted string,
     * e.g. '2015-11-22T04:30:27.991669+05:00'.
     */
    TString ToIsoStringLocal() const;

    /**
     * Formats the instant using the system time zone, with microsecond precision.
     *
     * @returns A semi-ISO 8601 formatted string with timezone without colon,
     * e.g. '2015-11-22T04:30:27.991669+0500'.
     */
    TString ToStringLocal() const;

    /**
     * Formats the instant using the system time zone.
     *
     * @returns An RFC822 formatted string, e.g. 'Sun, 06 Nov 1994 08:49:37 MSK'.
     */
    TString ToRfc822StringLocal() const;

    /**
     * Formats the instant using the system time zone, with second precision.
     *
     * @returns An ISO 8601 / RFC 3339 formatted string,
     * e.g. '2015-11-22T04:30:27+05:00'.
     */
    TString ToIsoStringLocalUpToSeconds() const;

    /**
     * Formats the instant using the system time zone, with second precision.
     *
     * @returns A semi-ISO 8601 formatted string with timezone without colon,
     * e.g. '2015-11-22T04:30:27+0500'.
     */
    TString ToStringLocalUpToSeconds() const;

    TString FormatLocalTime(const char* format) const noexcept;
    TString FormatGmTime(const char* format) const noexcept;

    /// See #TryParseIso8601.
    static TInstant ParseIso8601(TStringBuf);
    /// See #TryParseRfc822.
    static TInstant ParseRfc822(TStringBuf);
    /// See #TryParseHttp.
    static TInstant ParseHttp(TStringBuf);
    /// See #TryParseX509.
    static TInstant ParseX509Validity(TStringBuf);

    /// ISO 8601 Representation of Dates and Times
    ///
    /// @link https://www.iso.org/standard/40874.html Description of format.
    static bool TryParseIso8601(TStringBuf input, TInstant& instant);

    /// RFC 822 Date and Time specification
    ///
    /// @link https://tools.ietf.org/html/rfc822#section-5 Description of format.
    static bool TryParseRfc822(TStringBuf input, TInstant& instant);

    /// RFC 2616 3.3.1 Full Date
    ///
    /// @link https://tools.ietf.org/html/rfc2616#section-3.3.1 Description of format.
    static bool TryParseHttp(TStringBuf input, TInstant& instant);

    /// X.509 certificate validity time (see rfc5280 4.1.2.5.*)
    ///
    /// @link https://tools.ietf.org/html/rfc5280#section-4.1.2.5 Description of format.
    static bool TryParseX509(TStringBuf input, TInstant& instant);

    static TInstant ParseIso8601Deprecated(TStringBuf);
    static TInstant ParseRfc822Deprecated(TStringBuf);
    static TInstant ParseHttpDeprecated(TStringBuf);
    static TInstant ParseX509ValidityDeprecated(TStringBuf);

    static bool TryParseIso8601Deprecated(TStringBuf input, TInstant& instant);
    static bool TryParseRfc822Deprecated(TStringBuf input, TInstant& instant);
    static bool TryParseHttpDeprecated(TStringBuf input, TInstant& instant);
    static bool TryParseX509Deprecated(TStringBuf input, TInstant& instant);

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

template <>
struct THash<TInstant> {
    size_t operator()(const TInstant& key) const {
        return THash<TInstant::TValue>()(key.GetValue());
    }
};

namespace NPrivate {
    template <bool PrintUpToSeconds, bool iso>
    struct TPrintableLocalTime {
        TInstant MomentToPrint;

        constexpr explicit TPrintableLocalTime(TInstant momentToPrint)
            : MomentToPrint(momentToPrint)
        {
        }
    };
} // namespace NPrivate

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
/// @see TInstant::ToIsoStringLocal()
::NPrivate::TPrintableLocalTime<false, true> FormatIsoLocal(TInstant instant);
/// @see TInstant::ToStringLocal()
::NPrivate::TPrintableLocalTime<false, false> FormatLocal(TInstant instant);
/// @see TInstant::ToIsoStringLocalUpToSeconds()
::NPrivate::TPrintableLocalTime<true, true> FormatIsoLocalUpToSeconds(TInstant instant);
/// @see TInstant::ToStringLocalUpToSeconds()
::NPrivate::TPrintableLocalTime<true, false> FormatLocalUpToSeconds(TInstant instant);
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
} // namespace NDateTimeHelpers

constexpr TDuration operator-(const TInstant& l, const TInstant& r) noexcept {
    return TDuration::FromValue(::NDateTimeHelpers::DiffWithSaturation(l.GetValue(), r.GetValue()));
}

constexpr TInstant operator+(const TInstant& i, const TDuration& d) noexcept {
    return TInstant::FromValue(::NDateTimeHelpers::SumWithSaturation(i.GetValue(), d.GetValue()));
}

constexpr TInstant operator-(const TInstant& i, const TDuration& d) noexcept {
    return TInstant::FromValue(::NDateTimeHelpers::DiffWithSaturation(i.GetValue(), d.GetValue()));
}

constexpr TDuration operator-(const TDuration& l, const TDuration& r) noexcept {
    return TDuration::FromValue(::NDateTimeHelpers::DiffWithSaturation(l.GetValue(), r.GetValue()));
}

constexpr TDuration operator+(const TDuration& l, const TDuration& r) noexcept {
    return TDuration::FromValue(::NDateTimeHelpers::SumWithSaturation(l.GetValue(), r.GetValue()));
}

template <typename T, typename TRatio>
constexpr bool operator==(const TDuration& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return r.count() >= 0 && l == TDuration(r);
}

#if defined(__cpp_lib_three_way_comparison)

template <typename T, typename TRatio>
constexpr std::strong_ordering operator<=>(const TDuration& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    if (r.count() < 0) {
        return std::strong_ordering::greater;
    }
    return l.GetValue() <=> TDuration(r).GetValue();
}

#else

template <typename T, typename TRatio>
constexpr bool operator<(const TDuration& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return r.count() >= 0 && l < TDuration(r);
}

template <typename T, typename TRatio>
constexpr bool operator<=(const TDuration& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return r.count() >= 0 && l <= TDuration(r);
}

template <typename T, typename TRatio>
constexpr bool operator!=(const TDuration& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return !(l == r);
}

template <typename T, typename TRatio>
constexpr bool operator>(const TDuration& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return r.count() < 0 || l > TDuration(r);
}

template <typename T, typename TRatio>
constexpr bool operator>=(const TDuration& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return r.count() < 0 || l >= TDuration(r);
}

template <typename T, typename TRatio>
constexpr bool operator<(const std::chrono::duration<T, TRatio>& l, const TDuration& r) noexcept {
    return r > l;
}

template <typename T, typename TRatio>
constexpr bool operator<=(const std::chrono::duration<T, TRatio>& l, const TDuration& r) noexcept {
    return r >= l;
}

template <typename T, typename TRatio>
constexpr bool operator==(const std::chrono::duration<T, TRatio>& l, const TDuration& r) noexcept {
    return r == l;
}

template <typename T, typename TRatio>
constexpr bool operator!=(const std::chrono::duration<T, TRatio>& l, const TDuration& r) noexcept {
    return r != l;
}

template <typename T, typename TRatio>
constexpr bool operator>(const std::chrono::duration<T, TRatio>& l, const TDuration& r) noexcept {
    return r < l;
}

template <typename T, typename TRatio>
constexpr bool operator>=(const std::chrono::duration<T, TRatio>& l, const TDuration& r) noexcept {
    return r >= l;
}

#endif

template <typename T, typename TRatio>
constexpr TDuration operator+(const TDuration& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return r < r.zero() ? l - TDuration(-r) : l + TDuration(r);
}

template <typename T, typename TRatio>
constexpr TDuration operator+(const std::chrono::duration<T, TRatio>& l, const TDuration& r) noexcept {
    return r + l;
}

template <typename T, typename TRatio>
constexpr TDuration operator-(const TDuration& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return l + (-r);
}

template <typename T, typename TRatio>
constexpr TDuration operator-(const std::chrono::duration<T, TRatio>& l, const TDuration& r) noexcept {
    return TDuration(l) - r;
}

template <typename T, typename TRatio>
constexpr TInstant operator+(const TInstant& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return r < r.zero() ? l - TDuration(-r) : l + TDuration(r);
}

template <typename T, typename TRatio>
constexpr TInstant operator-(const TInstant& l, const std::chrono::duration<T, TRatio>& r) noexcept {
    return l + (-r);
}

template <class T>
inline TDuration operator*(TDuration d, T t) noexcept {
    Y_ASSERT(t >= T());
    Y_ASSERT(t == T() || Max<TDuration::TValue>() / t >= d.GetValue());
    return TDuration::FromValue(d.GetValue() * t);
}

template <>
inline TDuration operator*(TDuration d, double t) noexcept {
    Y_ASSERT(t >= 0 && MaxFloor<TDuration::TValue>() >= d.GetValue() * t);
    return TDuration::FromValue(d.GetValue() * t);
}

template <>
inline TDuration operator*(TDuration d, float t) noexcept {
    return d * static_cast<double>(t);
}

template <class T>
inline TDuration operator*(T t, TDuration d) noexcept {
    return d * t;
}

template <class T, std::enable_if_t<!std::is_same<TDuration, T>::value, int> = 0>
constexpr TDuration operator/(const TDuration& d, const T& t) noexcept {
    return TDuration::FromValue(d.GetValue() / t);
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
