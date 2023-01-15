#pragma once

#include <util/stream/output.h>
#include <util/stream/input.h>
#include <util/generic/string.h>
#include <util/datetime/constants.h>

#include <ctime>

// XXX: uses system calls for trivial things. may be very slow therefore.

time_t GetDateStart(time_t ts);

// Local date (without time zone)
class TDate {
    // XXX: wrong: must store number of days since epoch
    time_t Timestamp;

public:
    TDate()
        : Timestamp(0)
    {
    }

    // XXX: wrong. Should be replace with two methods: TodayGmt() and TodayLocal()
    static TDate Today() {
        return TDate(time(nullptr));
    }

    TDate(const char* yyyymmdd);
    TDate(const TString& yyyymmdd);
    TDate(unsigned year, unsigned month, unsigned monthDay); // month from 01, monthDay from 01
    TDate(const TString& date, const TString& format);

    explicit TDate(time_t t);

    time_t GetStart() const {
        return Timestamp;
    }

    time_t GetStartUTC() const;

    TString ToStroka(const char* format = "%Y%m%d") const;

    TDate& operator++() {
        Timestamp = GetDateStart(Timestamp + 3 * (SECONDS_IN_DAY / 2));
        return *this;
    }

    TDate& operator--() {
        Timestamp = GetDateStart(Timestamp - SECONDS_IN_DAY / 2);
        return *this;
    }

    TDate& operator+=(unsigned days) {
        Timestamp = GetDateStart(Timestamp + days * SECONDS_IN_DAY + SECONDS_IN_DAY / 2);
        return *this;
    }

    TDate& operator-=(unsigned days) {
        Timestamp = GetDateStart(Timestamp - days * SECONDS_IN_DAY + SECONDS_IN_DAY / 2);
        return *this;
    }

    TDate operator+(unsigned days) const {
        return TDate(Timestamp + days * SECONDS_IN_DAY + SECONDS_IN_DAY / 2);
    }

    TDate operator-(unsigned days) const {
        return TDate(Timestamp - days * SECONDS_IN_DAY + SECONDS_IN_DAY / 2);
    }

    unsigned GetWeekDay() const; // days since Sunday

    unsigned GetYear() const;
    unsigned GetMonth() const;    // from 01
    unsigned GetMonthDay() const; // from 01

    friend bool operator<(const TDate& left, const TDate& right);
    friend bool operator>(const TDate& left, const TDate& right);
    friend bool operator<=(const TDate& left, const TDate& right);
    friend bool operator>=(const TDate& left, const TDate& right);
    friend bool operator==(const TDate& left, const TDate& right);
    friend int operator-(const TDate& left, const TDate& right);

    friend IInputStream& operator>>(IInputStream& left, TDate& right);
    friend IOutputStream& operator<<(IOutputStream& left, const TDate& right);
};

Y_DECLARE_PODTYPE(TDate);

inline bool operator<(const TDate& left, const TDate& right) {
    return left.Timestamp < right.Timestamp;
}

inline bool operator>(const TDate& left, const TDate& right) {
    return left.Timestamp > right.Timestamp;
}

inline bool operator<=(const TDate& left, const TDate& right) {
    return left.Timestamp <= right.Timestamp;
}

inline bool operator>=(const TDate& left, const TDate& right) {
    return left.Timestamp >= right.Timestamp;
}

inline bool operator==(const TDate& left, const TDate& right) {
    return left.Timestamp == right.Timestamp;
}

inline int operator-(const TDate& left, const TDate& right) {
    if (left < right) {
        return -(right - left);
    }
    return static_cast<int>((left.Timestamp + SECONDS_IN_DAY / 2 - right.Timestamp) / SECONDS_IN_DAY);
}

inline IInputStream& operator>>(IInputStream& left, TDate& right) {
    TString stroka;
    left >> stroka;
    TDate date(stroka.c_str());
    right = date;
    return left;
}

inline IOutputStream& operator<<(IOutputStream& left, const TDate& right) {
    return left << right.ToStroka();
}
