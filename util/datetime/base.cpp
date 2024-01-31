#include "base.h"

#include <util/string/cast.h>
#include <util/stream/output.h>
#include <util/stream/mem.h>
#include <util/system/compat.h>
#include <util/memory/tempbuf.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>

TString Strftime(const char* format, const struct tm* tm) {
    size_t size = Max<size_t>(strlen(format) * 2 + 1, 107);
    for (;;) {
        TTempBuf buf(size);
        int r = strftime(buf.Data(), buf.Size(), format, tm);
        if (r != 0) {
            return TString(buf.Data(), r);
        }
        size *= 2;
    }
}

template <>
TDuration FromStringImpl<TDuration, char>(const char* s, size_t len) {
    return TDuration::Parse(TStringBuf(s, len));
}

template <>
bool TryFromStringImpl<TDuration, char>(const char* s, size_t len, TDuration& result) {
    return TDuration::TryParse(TStringBuf(s, len), result);
}

namespace {
    template <size_t N>
    struct TPad {
        int I;
    };

    template <size_t N>
    inline TPad<N> Pad(int i) {
        return {i};
    }

    inline IOutputStream& operator<<(IOutputStream& o, const TPad<2>& p) {
        if (p.I < 10) {
            if (p.I >= 0) {
                o << '0';
            }
        }

        return o << p.I;
    }

    inline IOutputStream& operator<<(IOutputStream& o, const TPad<4>& p) {
        if (p.I < 1000) {
            if (p.I >= 0) {
                if (p.I < 10) {
                    o << '0' << '0' << '0';
                } else if (p.I < 100) {
                    o << '0' << '0';
                } else {
                    o << '0';
                }
            }
        }

        return o << p.I;
    }

    inline IOutputStream& operator<<(IOutputStream& o, const TPad<6>& p) {
        if (p.I < 100000) {
            if (p.I >= 0) {
                if (p.I < 10) {
                    o << '0' << '0' << '0' << '0' << '0';
                } else if (p.I < 100) {
                    o << '0' << '0' << '0' << '0';
                } else if (p.I < 1000) {
                    o << '0' << '0' << '0';
                } else if (p.I < 10000) {
                    o << '0' << '0';
                } else {
                    o << '0';
                }
            }
        }

        return o << p.I;
    }

    void WriteMicroSecondsToStream(IOutputStream& os, ui32 microSeconds) {
        os << '.' << Pad<6>(microSeconds);
    }

    void WriteTmToStream(IOutputStream& os, const struct tm& theTm) {
        os << Pad<4>(theTm.tm_year + 1900) << '-' << Pad<2>(theTm.tm_mon + 1) << '-' << Pad<2>(theTm.tm_mday) << 'T'
           << Pad<2>(theTm.tm_hour) << ':' << Pad<2>(theTm.tm_min) << ':' << Pad<2>(theTm.tm_sec);
    }

    template <bool PrintUpToSeconds, bool iso>
    void WritePrintableLocalTimeToStream(IOutputStream& os, const ::NPrivate::TPrintableLocalTime<PrintUpToSeconds, iso>& timeToPrint) {
        const TInstant& momentToPrint = timeToPrint.MomentToPrint;
        struct tm localTime;
        momentToPrint.LocalTime(&localTime);
        WriteTmToStream(os, localTime);
        if (!PrintUpToSeconds) {
            WriteMicroSecondsToStream(os, momentToPrint.MicroSecondsOfSecond());
        }
#ifndef _win_
        i64 utcOffsetInMinutes = localTime.tm_gmtoff / 60;
#else
        TIME_ZONE_INFORMATION tz;
        if (GetTimeZoneInformation(&tz) == TIME_ZONE_ID_INVALID) {
            ythrow TSystemError() << "Failed to get the system time zone";
        }
        i64 utcOffsetInMinutes = -tz.Bias;
#endif
        if (utcOffsetInMinutes == 0) {
            os << 'Z';
        } else {
            if (utcOffsetInMinutes < 0) {
                os << '-';
                utcOffsetInMinutes = -utcOffsetInMinutes;
            } else {
                os << '+';
            }
            os << Pad<2>(utcOffsetInMinutes / 60);
            if (iso) {
                os << ':';
            }
            os << Pad<2>(utcOffsetInMinutes % 60);
        }
    }
}

template <>
void Out<TDuration>(IOutputStream& os, TTypeTraits<TDuration>::TFuncParam duration) {
    os << duration.Seconds();
    WriteMicroSecondsToStream(os, duration.MicroSecondsOfSecond());
    os << 's';
}

template <>
void Out<TInstant>(IOutputStream& os, TTypeTraits<TInstant>::TFuncParam instant) {
    char buf[64];
    auto len = FormatDate8601(buf, sizeof(buf), instant.TimeT());

    // shouldn't happen due to current implementation of FormatDate8601() and GmTimeR()
    Y_ENSURE(len, TStringBuf("Out<TInstant>: year does not fit into an integer"));

    os.Write(buf, len - 1 /* 'Z' */);
    WriteMicroSecondsToStream(os, instant.MicroSecondsOfSecond());
    os << 'Z';
}

template <>
void Out<::NPrivate::TPrintableLocalTime<false, false>>(IOutputStream& os, TTypeTraits<::NPrivate::TPrintableLocalTime<false, false>>::TFuncParam localTime) {
    WritePrintableLocalTimeToStream(os, localTime);
}

template <>
void Out<::NPrivate::TPrintableLocalTime<false, true>>(IOutputStream& os, TTypeTraits<::NPrivate::TPrintableLocalTime<false, true>>::TFuncParam localTime) {
    WritePrintableLocalTimeToStream(os, localTime);
}

template <>
void Out<::NPrivate::TPrintableLocalTime<true, false>>(IOutputStream& os, TTypeTraits<::NPrivate::TPrintableLocalTime<true, false>>::TFuncParam localTime) {
    WritePrintableLocalTimeToStream(os, localTime);
}

template <>
void Out<::NPrivate::TPrintableLocalTime<true, true>>(IOutputStream& os, TTypeTraits<::NPrivate::TPrintableLocalTime<true, true>>::TFuncParam localTime) {
    WritePrintableLocalTimeToStream(os, localTime);
}

TString TDuration::ToString() const {
    return ::ToString(*this);
}

TString TInstant::ToString() const {
    return ::ToString(*this);
}

TString TInstant::ToRfc822String() const {
    return FormatGmTime("%a, %d %b %Y %H:%M:%S GMT");
}

TString TInstant::ToStringUpToSeconds() const {
    char buf[64];
    auto len = FormatDate8601(buf, sizeof(buf), TimeT());
    if (!len) {
        ythrow yexception() << "TInstant::ToStringUpToSeconds: year does not fit into an integer";
    }
    return TString(buf, len);
}

TString TInstant::ToIsoStringLocal() const {
    return ::ToString(FormatIsoLocal(*this));
}

TString TInstant::ToStringLocal() const {
    return ::ToString(FormatLocal(*this));
}

TString TInstant::ToRfc822StringLocal() const {
    return FormatLocalTime("%a, %d %b %Y %H:%M:%S %Z");
}

TString TInstant::ToIsoStringLocalUpToSeconds() const {
    return ::ToString(FormatIsoLocalUpToSeconds(*this));
}

TString TInstant::ToStringLocalUpToSeconds() const {
    return ::ToString(FormatLocalUpToSeconds(*this));
}

TString TInstant::FormatLocalTime(const char* format) const noexcept {
    struct tm theTm;
    LocalTime(&theTm);
    return Strftime(format, &theTm);
}

TString TInstant::FormatGmTime(const char* format) const noexcept {
    struct tm theTm;
    GmTime(&theTm);
    return Strftime(format, &theTm);
}

::NPrivate::TPrintableLocalTime<false, true> FormatIsoLocal(TInstant instant) {
    return ::NPrivate::TPrintableLocalTime<false, true>(instant);
}

::NPrivate::TPrintableLocalTime<false, false> FormatLocal(TInstant instant) {
    return ::NPrivate::TPrintableLocalTime<false, false>(instant);
}

::NPrivate::TPrintableLocalTime<true, true> FormatIsoLocalUpToSeconds(TInstant instant) {
    return ::NPrivate::TPrintableLocalTime<true, true>(instant);
}

::NPrivate::TPrintableLocalTime<true, false> FormatLocalUpToSeconds(TInstant instant) {
    return ::NPrivate::TPrintableLocalTime<true, false>(instant);
}

void Sleep(TDuration duration) {
    NanoSleep(duration.NanoSeconds());
}

void sprint_gm_date(char* buf, time_t when, long* sec) {
    struct tm theTm;
    ::Zero(theTm);
    GmTimeR(&when, &theTm);
    DateToString(buf, theTm);
    if (sec) {
        *sec = seconds(theTm);
    }
}

void DateToString(char* buf, const struct tm& theTm) {
    Y_ENSURE(0 <= theTm.tm_year + 1900 && theTm.tm_year + 1900 <= 9999, "invalid year " + ToString(theTm.tm_year + 1900) + ", year should be in range [0, 9999]");

    snprintf(buf, DATE_BUF_LEN, "%04d%02d%02d", theTm.tm_year + 1900, theTm.tm_mon + 1, theTm.tm_mday);
}

void DateToString(char* buf, time_t when, long* sec) {
    struct tm theTm;
    localtime_r(&when, &theTm);

    DateToString(buf, theTm);

    if (sec) {
        *sec = seconds(theTm);
    }
}

TString DateToString(const struct tm& theTm) {
    char buf[DATE_BUF_LEN];
    DateToString(buf, theTm);
    return buf;
}

TString DateToString(time_t when, long* sec) {
    char buf[DATE_BUF_LEN];
    DateToString(buf, when, sec);
    return buf;
}

TString YearToString(const struct tm& theTm) {
    Y_ENSURE(0 <= theTm.tm_year + 1900 && theTm.tm_year + 1900 <= 9999, "invalid year " + ToString(theTm.tm_year + 1900) + ", year should be in range [0, 9999]");
    char buf[16];
    snprintf(buf, 16, "%04d", theTm.tm_year + 1900);
    return buf;
}

TString YearToString(time_t when) {
    struct tm theTm;
    localtime_r(&when, &theTm);

    return YearToString(theTm);
}

bool sscan_date(const char* date, struct tm& theTm) {
    int year, mon, mday;
    if (sscanf(date, "%4d%2d%2d", &year, &mon, &mday) != 3) {
        return false;
    }
    theTm.tm_year = year - 1900;
    theTm.tm_mon = mon - 1;
    theTm.tm_mday = mday;
    return true;
}

size_t FormatDate8601(char* buf, size_t len, time_t when) {
    struct tm theTm;
    struct tm* ret = GmTimeR(&when, &theTm);

    if (ret) {
        TMemoryOutput out(buf, len);

        WriteTmToStream(out, theTm);
        out << 'Z';

        return out.Buf() - buf;
    }

    return 0;
}

void SleepUntil(TInstant instant) {
    TInstant now = TInstant::Now();
    if (instant <= now) {
        return;
    }
    TDuration duration = instant - now;
    Sleep(duration);
}
