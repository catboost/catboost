#include "date.h"

#include <util/string/cast.h>
#include <util/generic/yexception.h>
#include <util/datetime/base.h>

time_t GetDateStart(time_t ts) {
    tm dateTm;
    memset(&dateTm, 0, sizeof(tm));
    localtime_r(&ts, &dateTm);

    dateTm.tm_isdst = -1;

    dateTm.tm_sec = 0;
    dateTm.tm_min = 0;
    dateTm.tm_hour = 0;
    return mktime(&dateTm);
}

static time_t ParseDate(const char* date, const char* format) {
    tm dateTm;
    memset(&dateTm, 0, sizeof(tm));
    if (!strptime(date, format, &dateTm)) {
        ythrow yexception() << "Invalid date string and format: " << date << ", " << format;
    }
    return mktime(&dateTm);
}

static time_t ParseDate(const char* dateStr) {
    if (strlen(dateStr) != 8) {
        ythrow yexception() << "Invalid date string: " << dateStr;
    }

    return ParseDate(dateStr, "%Y%m%d");
}

template <>
TDate FromStringImpl<TDate>(const char* data, size_t len) {
    return TDate(ParseDate(TString(data, len).data()));
}

TDate::TDate(const char* yyyymmdd)
    : Timestamp(GetDateStart(ParseDate(yyyymmdd)))
{
}

TDate::TDate(const TString& yyyymmdd)
    : Timestamp(GetDateStart(ParseDate(yyyymmdd.c_str())))
{
}

TDate::TDate(time_t ts)
    : Timestamp(GetDateStart(ts))
{
}

TDate::TDate(const TString& date, const TString& format)
    : Timestamp(GetDateStart(ParseDate(date.data(), format.data())))
{
}

TDate::TDate(unsigned year, unsigned month, unsigned monthDay) {
    tm dateTm;
    Zero(dateTm);
    dateTm.tm_year = year - 1900;
    dateTm.tm_mon = month - 1;
    dateTm.tm_mday = monthDay;
    dateTm.tm_isdst = -1;
    Timestamp = mktime(&dateTm);
    if (Timestamp == (time_t)-1) {
        ythrow yexception() << "Invalid TDate args:(" << year << ',' << month << ',' << monthDay << ')';
    }
}

time_t TDate::GetStartUTC() const {
    tm dateTm;
    localtime_r(&Timestamp, &dateTm);
    dateTm.tm_isdst = -1;
    dateTm.tm_sec = 0;
    dateTm.tm_min = 0;
    dateTm.tm_hour = 0;
    return TimeGM(&dateTm);
}

TString TDate::ToStroka(const char* format) const {
    tm dateTm;
    localtime_r(&Timestamp, &dateTm);
    return Strftime(format, &dateTm);
}

unsigned TDate::GetWeekDay() const {
    tm dateTm;
    localtime_r(&Timestamp, &dateTm);
    return (unsigned)dateTm.tm_wday;
}

unsigned TDate::GetYear() const {
    tm dateTm;
    localtime_r(&Timestamp, &dateTm);
    return ((unsigned)dateTm.tm_year) + 1900;
}

unsigned TDate::GetMonth() const {
    tm dateTm;
    localtime_r(&Timestamp, &dateTm);
    return ((unsigned)dateTm.tm_mon) + 1;
}

unsigned TDate::GetMonthDay() const {
    tm dateTm;
    localtime_r(&Timestamp, &dateTm);
    return (unsigned)dateTm.tm_mday;
}
