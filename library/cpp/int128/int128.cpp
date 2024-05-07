#include "int128.h"

#include <tuple>

IOutputStream& operator<<(IOutputStream& out, const ui128& other) {
    // see http://stackoverflow.com/questions/4361441/c-print-a-biginteger-in-base-10
    // and http://stackoverflow.com/questions/8023414/how-to-convert-a-128-bit-integer-to-a-decimal-ascii-string-in-c
    int d[39] = {0};
    int i;
    int j;
    for (i = 63; i > -1; i--) {
        if ((other.High_ >> i) & 1)
            ++d[0];
        for (j = 0; j < 39; j++)
            d[j] *= 2;
        for (j = 0; j < 38; j++) {
            d[j + 1] += d[j] / 10;
            d[j] %= 10;
        }
    }
    for (i = 63; i > -1; i--) {
        if ((other.Low_ >> i) & 1)
            ++d[0];
        if (i > 0)
            for (j = 0; j < 39; j++)
                d[j] *= 2;
        for (j = 0; j < 38; j++) {
            d[j + 1] += d[j] / 10;
            d[j] %= 10;
        }
    }
    for (i = 38; i > 0; i--)
        if (d[i] > 0)
            break;
    for (; i > -1; i--)
        out << static_cast<char>('0' + d[i]);

    return out;
}

void TSerializer<ui128>::Save(IOutputStream* out, const ui128& Number) {
    ::Save(out, GetHigh(Number));
    ::Save(out, GetLow(Number));
}

void TSerializer<ui128>::Load(IInputStream* in, ui128& Number) {
    ui64 High;
    ui64 Low;
    ::Load(in, High);
    ::Load(in, Low);
    Number = ui128(High, Low);
}

IOutputStream& operator<<(IOutputStream& out, const i128& other)
{
    if (other >= 0) {
        out << ui128{other};
    } else {
        out << '-' << ui128{-other};
    }
    return out;
}
