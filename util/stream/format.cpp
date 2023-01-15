#include "format.h"
#include "output.h"

#include <util/generic/ymath.h>
#include <util/string/cast.h>

namespace NFormatPrivate {
    static inline i64 Round(double value) {
        double res1 = floor(value);
        double res2 = ceil(value);
        return (value - res1 < res2 - value) ? (i64)res1 : (i64)res2;
    }

    static inline IOutputStream& PrintDoubleShortly(IOutputStream& os, const double& d) {
        // General case: request 3 significant digits
        // Side-effect: allows exponential representation
        EFloatToStringMode mode = PREC_NDIGITS;
        int ndigits = 3;

        if (IsValidFloat(d) && Abs(d) < 1e12) {
            // For reasonably-sized finite values, it's better to avoid
            // exponential representation.
            // Use compact fixed representation and determine
            // precision based on magnitude.
            mode = PREC_POINT_DIGITS_STRIP_ZEROES;
            if (i64(Abs(d) * 100) < 1000) {
                ndigits = 2;
            } else if (i64(Abs(d) * 10) < 1000) {
                ndigits = 1;
            } else {
                ndigits = 0;
            }
        }

        return os << Prec(d, mode, ndigits);
    }
}

template <>
void Out<NFormatPrivate::THumanReadableSize>(IOutputStream& stream, const NFormatPrivate::THumanReadableSize& value) {
    ui64 base = value.Format == SF_BYTES ? 1024 : 1000;
    ui64 base2 = base * base;
    ui64 base3 = base * base2;
    ui64 base4 = base * base3;

    double v = value.Value;
    if (v < 0) {
        stream << "-";
        v = -v;
    }

    if (v < base) {
        NFormatPrivate::PrintDoubleShortly(stream, v);
    } else if (v < base2) {
        NFormatPrivate::PrintDoubleShortly(stream, v / (double)base) << 'K';
    } else if (v < base3) {
        NFormatPrivate::PrintDoubleShortly(stream, v / (double)base2) << 'M';
    } else if (v < base4) {
        NFormatPrivate::PrintDoubleShortly(stream, v / (double)base3) << 'G';
    } else {
        NFormatPrivate::PrintDoubleShortly(stream, v / (double)base4) << 'T';
    }

    if (value.Format == SF_BYTES) {
        if (v < base) {
            stream << "B";
        } else {
            stream << "iB";
        }
    }
}

template <>
void Out<NFormatPrivate::THumanReadableDuration>(IOutputStream& os, const NFormatPrivate::THumanReadableDuration& hr) {
    TTempBuf buf;
    TMemoryOutput ss(buf.Data(), buf.Size());

    do {
        ui64 microSeconds = hr.Value.MicroSeconds();
        if (microSeconds < 1000) {
            ss << microSeconds << "us";
            break;
        }
        if (microSeconds < 1000 * 1000) {
            NFormatPrivate::PrintDoubleShortly(ss, (double)microSeconds / 1000.0) << "ms";
            break;
        }

        double seconds = (double)(hr.Value.MilliSeconds()) / 1000.0;
        if (seconds < 60) {
            NFormatPrivate::PrintDoubleShortly(ss, seconds) << 's';
            break;
        }

        ui64 s = NFormatPrivate::Round(seconds * 1000 + 0.5) / 1000;

        ui64 m = s / 60;
        s = s % 60;

        ui64 h = m / 60;
        m = m % 60;

        ui64 d = h / 24;
        h = h % 24;

        ui64 times[] = {d, h, m, s};
        char names[] = {'d', 'h', 'm', 's'};
        bool first = true;

        for (size_t i = 0; i < Y_ARRAY_SIZE(times); ++i) {
            if (times[i] > 0) {
                if (!first) {
                    ss << ' ';
                }
                ss << times[i] << names[i];
                first = false;
            }
        }
    } while (false);

    size_t written = buf.Size() - ss.Avail();
    os.Write(buf.Data(), written);
}

void Time(IOutputStream& l) {
    l << millisec();
}

void TimeHumanReadable(IOutputStream& l) {
    char timeStr[30];
    const time_t t = time(nullptr);

    l << ctime_r(&t, timeStr);
}
