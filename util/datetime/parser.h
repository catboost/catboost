#pragma once

// probably you do not need to include this file directly, use "util/datetime/base.h"

#include "base.h"

struct TDateTimeFields {
    TDateTimeFields() {
        Zero(*this);
        ZoneOffsetMinutes = 0;
        Hour = 0;
    }

    ui32 Year;
    ui32 Month;       // 1..12
    ui32 Day;         // 1 .. 31
    ui32 Hour;        // 0 .. 23
    ui32 Minute;      // 0 .. 59
    ui32 Second;      // 0 .. 60
    ui32 MicroSecond; // 0 .. 999999
    i32 ZoneOffsetMinutes;

    void SetLooseYear(ui32 year) {
        if (year < 60)
            year += 100;
        if (year < 160)
            year += 1900;
        Year = year;
    }

    bool IsOk() const noexcept {
        if (Year < 1970)
            return false;
        if (Month < 1 || Month > 12)
            return false;

        unsigned int maxMonthDay = 31;
        if (Month == 4 || Month == 6 || Month == 9 || Month == 11) {
            maxMonthDay = 30;
        } else if (Month == 2) {
            if (Year % 4 == 0 && (Year % 100 != 0 || Year % 400 == 0))
                // leap year
                maxMonthDay = 29;
            else
                maxMonthDay = 28;
        }
        if (Day > maxMonthDay)
            return false;

        if (Hour > 23)
            return false;

        if (Minute > 59)
            return false;

        // handle leap second which is explicitly allowed by ISO 8601:2004(E) $2.2.2
        // https://datatracker.ietf.org/doc/html/rfc3339#section-5.6
        if (Second > 60)
            return false;

        if (MicroSecond > 999999)
            return false;

        if (Year == 1970 && Month == 1 && Day == 1) {
            if ((i64)(3600 * Hour + 60 * Minute + Second) < (60 * ZoneOffsetMinutes))
                return false;
        }

        return true;
    }

    TInstant ToInstant(TInstant defaultValue) const {
        time_t tt = ToTimeT(-1);
        if (tt == -1)
            return defaultValue;
        return TInstant::Seconds(tt) + TDuration::MicroSeconds(MicroSecond);
    }

    time_t ToTimeT(time_t defaultValue) const {
        if (!IsOk())
            return defaultValue;
        struct tm tm;
        Zero(tm);
        tm.tm_year = Year - 1900;
        tm.tm_mon = Month - 1;
        tm.tm_mday = Day;
        tm.tm_hour = Hour;
        tm.tm_min = Minute;
        tm.tm_sec = Second;
        time_t tt = TimeGM(&tm);
        if (tt == -1)
            return defaultValue;
        return tt - ZoneOffsetMinutes * 60;
    }
};

class TDateTimeParserBase {
public:
    const TDateTimeFields& GetDateTimeFields() const {
        return DateTimeFields;
    }

protected:
    TDateTimeFields DateTimeFields;
    int cs; //for ragel
    int Sign;
    int I;
    int Dc;

protected:
    TDateTimeParserBase()
        : DateTimeFields()
        , cs(0)
        , Sign(0)
        , I(0xDEADBEEF) // to guarantee unittest break if ragel code is incorrect
        , Dc(0xDEADBEEF)
    {
    }

    inline TInstant GetResult(int firstFinalState, TInstant defaultValue) const {
        if (cs < firstFinalState)
            return defaultValue;
        return DateTimeFields.ToInstant(defaultValue);
    }
};

#define DECLARE_PARSER(CLASS)                            \
    struct CLASS: public TDateTimeParserBase {           \
        CLASS();                                         \
        bool ParsePart(const char* input, size_t len);   \
        TInstant GetResult(TInstant defaultValue) const; \
    };

DECLARE_PARSER(TIso8601DateTimeParser)
DECLARE_PARSER(TRfc822DateTimeParser)
DECLARE_PARSER(THttpDateTimeParser)
DECLARE_PARSER(TX509ValidityDateTimeParser)
DECLARE_PARSER(TX509Validity4yDateTimeParser)

#undef DECLARE_PARSER

struct TDurationParser {
    int cs;

    ui64 I;
    ui32 Dc;

    i32 MultiplierPower; // 6 for seconds, 0 for microseconds, -3 for nanoseconds
    i32 Multiplier;
    ui64 IntegerPart;
    ui32 FractionPart;
    ui32 FractionDigits;

    TDurationParser();
    bool ParsePart(const char* input, size_t len);
    TDuration GetResult(TDuration defaultValue) const;
};

/**
Depcrecated cause of default hour offset (+4 hours)
@see IGNIETFERRO-823
*/
struct TDateTimeFieldsDeprecated {
    TDateTimeFieldsDeprecated() {
        Zero(*this);
        ZoneOffsetMinutes = (i32)TDuration::Hours(4).Minutes(); // legacy code
        Hour = 11;
    }

    ui32 Year;
    ui32 Month;       // 1..12
    ui32 Day;         // 1 .. 31
    ui32 Hour;        // 0 .. 23
    ui32 Minute;      // 0 .. 59
    ui32 Second;      // 0 .. 60
    ui32 MicroSecond; // 0 .. 999999
    i32 ZoneOffsetMinutes;

    void SetLooseYear(ui32 year) {
        if (year < 60)
            year += 100;
        if (year < 160)
            year += 1900;
        Year = year;
    }

    bool IsOk() const noexcept {
        if (Year < 1970)
            return false;
        if (Month < 1 || Month > 12)
            return false;

        unsigned int maxMonthDay = 31;
        if (Month == 4 || Month == 6 || Month == 9 || Month == 11) {
            maxMonthDay = 30;
        } else if (Month == 2) {
            if (Year % 4 == 0 && (Year % 100 != 0 || Year % 400 == 0))
                // leap year
                maxMonthDay = 29;
            else
                maxMonthDay = 28;
        }
        if (Day > maxMonthDay)
            return false;

        if (Hour > 23)
            return false;

        if (Minute > 59)
            return false;

        if (Second > 60)
            return false;

        if (MicroSecond > 999999)
            return false;

        if (Year == 1970 && Month == 1 && Day == 1) {
            if ((i64)(3600 * Hour + 60 * Minute + Second) < (60 * ZoneOffsetMinutes))
                return false;
        }

        return true;
    }

    TInstant ToInstant(TInstant defaultValue) const {
        time_t tt = ToTimeT(-1);
        if (tt == -1)
            return defaultValue;
        return TInstant::Seconds(tt) + TDuration::MicroSeconds(MicroSecond);
    }

    time_t ToTimeT(time_t defaultValue) const {
        if (!IsOk())
            return defaultValue;
        struct tm tm;
        Zero(tm);
        tm.tm_year = Year - 1900;
        tm.tm_mon = Month - 1;
        tm.tm_mday = Day;
        tm.tm_hour = Hour;
        tm.tm_min = Minute;
        tm.tm_sec = Second;
        time_t tt = TimeGM(&tm);
        if (tt == -1)
            return defaultValue;
        return tt - ZoneOffsetMinutes * 60;
    }
};

class TDateTimeParserBaseDeprecated {
public:
    const TDateTimeFieldsDeprecated& GetDateTimeFields() const {
        return DateTimeFields;
    }

protected:
    TDateTimeFieldsDeprecated DateTimeFields;
    int cs; //for ragel
    int Sign;
    int I;
    int Dc;

protected:
    TDateTimeParserBaseDeprecated()
        : DateTimeFields()
        , cs(0)
        , Sign(0)
        , I(0xDEADBEEF) // to guarantee unittest break if ragel code is incorrect
        , Dc(0xDEADBEEF)
    {
    }

    inline TInstant GetResult(int firstFinalState, TInstant defaultValue) const {
        if (cs < firstFinalState)
            return defaultValue;
        return DateTimeFields.ToInstant(defaultValue);
    }
};

#define DECLARE_PARSER(CLASS)                            \
    struct CLASS: public TDateTimeParserBaseDeprecated { \
        CLASS();                                         \
        bool ParsePart(const char* input, size_t len);   \
        TInstant GetResult(TInstant defaultValue) const; \
    };

DECLARE_PARSER(TIso8601DateTimeParserDeprecated)
DECLARE_PARSER(TRfc822DateTimeParserDeprecated)
DECLARE_PARSER(THttpDateTimeParserDeprecated)
DECLARE_PARSER(TX509ValidityDateTimeParserDeprecated)
DECLARE_PARSER(TX509Validity4yDateTimeParserDeprecated)

#undef DECLARE_PARSER
