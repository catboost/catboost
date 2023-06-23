#pragma once

#include "mem.h"
#include "output.h"

#include <util/datetime/base.h>
#include <util/generic/strbuf.h>
#include <util/generic/flags.h>
#include <util/memory/tempbuf.h>
#include <util/string/cast.h>

enum ENumberFormatFlag {
    HF_FULL = 0x01, /**< Output number with leading zeros. */
    HF_ADDX = 0x02, /**< Output '0x' or '0b' before hex/bin digits. */
};
Y_DECLARE_FLAGS(ENumberFormat, ENumberFormatFlag);
Y_DECLARE_OPERATORS_FOR_FLAGS(ENumberFormat);

enum ESizeFormat {
    SF_QUANTITY, /**< Base 1000, usual suffixes. 1100 gets turned into "1.1K". */
    SF_BYTES,    /**< Base 1024, byte suffix. 1100 gets turned into "1.07KiB". */
};

namespace NFormatPrivate {
    template <size_t Value>
    struct TLog2: std::integral_constant<size_t, TLog2<Value / 2>::value + 1> {};

    template <>
    struct TLog2<1>: std::integral_constant<size_t, 0> {};

    template <typename T>
    inline void StreamWrite(T& stream, const char* s, size_t size) {
        stream.write(s, size);
    }

    template <>
    inline void StreamWrite(IOutputStream& stream, const char* s, size_t size) {
        stream.Write(s, size);
    }

    template <>
    inline void StreamWrite(TStringStream& stream, const char* s, size_t size) {
        stream.Write(s, size);
    }

    template <typename T>
    static inline void WriteChars(T& os, char c, size_t count) {
        if (count == 0)
            return;
        TTempBuf buf(count);
        memset(buf.Data(), c, count);
        StreamWrite(os, buf.Data(), count);
    }

    template <typename T>
    struct TLeftPad {
        T Value;
        size_t Width;
        char Padc;

        inline TLeftPad(const T& value, size_t width, char padc)
            : Value(value)
            , Width(width)
            , Padc(padc)
        {
        }
    };

    template <typename T>
    IOutputStream& operator<<(IOutputStream& o, const TLeftPad<T>& lp) {
        TTempBuf buf;
        TMemoryOutput ss(buf.Data(), buf.Size());
        ss << lp.Value;
        size_t written = buf.Size() - ss.Avail();
        if (lp.Width > written) {
            WriteChars(o, lp.Padc, lp.Width - written);
        }
        o.Write(buf.Data(), written);
        return o;
    }

    template <typename T>
    struct TRightPad {
        T Value;
        size_t Width;
        char Padc;

        inline TRightPad(const T& value, size_t width, char padc)
            : Value(value)
            , Width(width)
            , Padc(padc)
        {
        }
    };

    template <typename T>
    IOutputStream& operator<<(IOutputStream& o, const TRightPad<T>& lp) {
        TTempBuf buf;
        TMemoryOutput ss(buf.Data(), buf.Size());
        ss << lp.Value;
        size_t written = buf.Size() - ss.Avail();
        o.Write(buf.Data(), written);
        if (lp.Width > written) {
            WriteChars(o, lp.Padc, lp.Width - written);
        }
        return o;
    }

    template <typename T, size_t Base>
    struct TBaseNumber {
        T Value;
        ENumberFormat Flags;

        template <typename OtherT>
        inline TBaseNumber(OtherT value, ENumberFormat flags)
            : Value(value)
            , Flags(flags)
        {
        }
    };

    template <typename T, size_t Base>
    using TUnsignedBaseNumber = TBaseNumber<std::make_unsigned_t<std::remove_cv_t<T>>, Base>;

    template <typename TStream, typename T, size_t Base>
    TStream& ToStreamImpl(TStream& stream, const TBaseNumber<T, Base>& value) {
        char buf[8 * sizeof(T) + 1]; /* Add 1 for sign. */
        TStringBuf str(buf, IntToString<Base>(value.Value, buf, sizeof(buf)));

        if (str[0] == '-') {
            stream << '-';
            str.Skip(1);
        }

        if (value.Flags & HF_ADDX) {
            if (Base == 16) {
                stream << TStringBuf("0x");
            } else if (Base == 2) {
                stream << TStringBuf("0b");
            }
        }

        if (value.Flags & HF_FULL) {
            WriteChars(stream, '0', (8 * sizeof(T) + TLog2<Base>::value - 1) / TLog2<Base>::value - str.size());
        }

        stream << str;
        return stream;
    }

    template <typename T, size_t Base>
    IOutputStream& operator<<(IOutputStream& stream, const TBaseNumber<T, Base>& value) {
        return ToStreamImpl(stream, value);
    }

    template <typename T, size_t Base>
    std::ostream& operator<<(std::ostream& stream, const TBaseNumber<T, Base>& value) {
        return ToStreamImpl(stream, value);
    }

    template <typename Char, size_t Base>
    struct TBaseText {
        TBasicStringBuf<Char> Text;

        inline TBaseText(const TBasicStringBuf<Char> text)
            : Text(text)
        {
        }
    };

    template <typename Char, size_t Base>
    IOutputStream& operator<<(IOutputStream& os, const TBaseText<Char, Base>& text) {
        for (size_t i = 0; i < text.Text.size(); ++i) {
            if (i != 0) {
                os << ' ';
            }
            os << TUnsignedBaseNumber<Char, Base>(text.Text[i], HF_FULL);
        }
        return os;
    }

    template <typename T>
    struct TFloatPrecision {
        using TdVal = std::remove_cv_t<T>;
        static_assert(std::is_floating_point<TdVal>::value, "expect std::is_floating_point<TdVal>::value");

        TdVal Value;
        EFloatToStringMode Mode;
        int NDigits;
    };

    template <typename T>
    IOutputStream& operator<<(IOutputStream& o, const TFloatPrecision<T>& prec) {
        char buf[512];
        size_t count = FloatToString(prec.Value, buf, sizeof(buf), prec.Mode, prec.NDigits);
        o << TStringBuf(buf, count);
        return o;
    }

    struct THumanReadableDuration {
        TDuration Value;

        constexpr THumanReadableDuration(const TDuration& value)
            : Value(value)
        {
        }
    };

    struct THumanReadableSize {
        double Value;
        ESizeFormat Format;
    };
}

/**
 * Output manipulator basically equivalent to `std::setw` and `std::setfill`
 * combined.
 *
 * When written into a `IOutputStream`, writes out padding characters first,
 * and then provided value.
 *
 * Example usage:
 * @code
 * stream << LeftPad(12345, 10, '0'); // Will output "0000012345"
 * @endcode
 *
 * @param value                         Value to output.
 * @param width                         Target total width.
 * @param padc                          Character to use for padding.
 * @see RightPad
 */
template <typename T>
static constexpr ::NFormatPrivate::TLeftPad<T> LeftPad(const T& value, const size_t width, const char padc = ' ') noexcept {
    return ::NFormatPrivate::TLeftPad<T>(value, width, padc);
}

template <typename T, int N>
static constexpr ::NFormatPrivate::TLeftPad<const T*> LeftPad(const T (&value)[N], const size_t width, const char padc = ' ') noexcept {
    return ::NFormatPrivate::TLeftPad<const T*>(value, width, padc);
}

/**
 * Output manipulator similar to `std::setw` and `std::setfill`.
 *
 * When written into a `IOutputStream`, writes provided value first, and then
 * the padding characters.
 *
 * Example usage:
 * @code
 * stream << RightPad("column1", 10, ' '); // Will output "column1   "
 * @endcode
 *
 * @param value                         Value to output.
 * @param width                         Target total width.
 * @param padc                          Character to use for padding.
 * @see LeftPad
 */
template <typename T>
static constexpr ::NFormatPrivate::TRightPad<T> RightPad(const T& value, const size_t width, const char padc = ' ') noexcept {
    return ::NFormatPrivate::TRightPad<T>(value, width, padc);
}

template <typename T, int N>
static constexpr ::NFormatPrivate::TRightPad<const T*> RightPad(const T (&value)[N], const size_t width, const char padc = ' ') noexcept {
    return ::NFormatPrivate::TRightPad<const T*>(value, width, padc);
}

/**
 * Output manipulator similar to `std::setbase(16)`.
 *
 * When written into a `IOutputStream`, writes out the provided value in
 * hexadecimal form. The value is treated as unsigned, even if its type is in
 * fact signed.
 *
 * Example usage:
 * @code
 * stream << Hex(-1);   // Will output "0xFFFFFFFF"
 * stream << Hex(1ull); // Will output "0x0000000000000001"
 * @endcode
 *
 * @param value                         Value to output.
 * @param flags                         Output flags.
 */
template <typename T>
static constexpr ::NFormatPrivate::TUnsignedBaseNumber<T, 16> Hex(const T& value, const ENumberFormat flags = HF_FULL | HF_ADDX) noexcept {
    return {value, flags};
}

/**
 * Output manipulator similar to `std::setbase(16)`.
 *
 * When written into a `IOutputStream`, writes out the provided value in
 * hexadecimal form.
 *
 * Example usage:
 * @code
 * stream << SHex(-1);   // Will output "-0x00000001"
 * stream << SHex(1ull); // Will output "0x0000000000000001"
 * @endcode
 *
 * @param value                         Value to output.
 * @param flags                         Output flags.
 */
template <typename T>
static constexpr ::NFormatPrivate::TBaseNumber<T, 16> SHex(const T& value, const ENumberFormat flags = HF_FULL | HF_ADDX) noexcept {
    return {value, flags};
}

/**
 * Output manipulator similar to `std::setbase(2)`.
 *
 * When written into a `IOutputStream`, writes out the provided value in
 * binary form. The value is treated as unsigned, even if its type is in
 * fact signed.
 *
 * Example usage:
 * @code
 * stream << Bin(-1);   // Will output "0b11111111111111111111111111111111"
 * stream << Bin(1);    // Will output "0b00000000000000000000000000000001"
 * @endcode
 *
 * @param value                         Value to output.
 * @param flags                         Output flags.
 */
template <typename T>
static constexpr ::NFormatPrivate::TUnsignedBaseNumber<T, 2> Bin(const T& value, const ENumberFormat flags = HF_FULL | HF_ADDX) noexcept {
    return {value, flags};
}

/**
 * Output manipulator similar to `std::setbase(2)`.
 *
 * When written into a `IOutputStream`, writes out the provided value in
 * binary form.
 *
 * Example usage:
 * @code
 * stream << SBin(-1);   // Will output "-0b00000000000000000000000000000001"
 * stream << SBin(1);    // Will output "0b00000000000000000000000000000001"
 * @endcode
 *
 * @param value                         Value to output.
 * @param flags                         Output flags.
 */
template <typename T>
static constexpr ::NFormatPrivate::TBaseNumber<T, 2> SBin(const T& value, const ENumberFormat flags = HF_FULL | HF_ADDX) noexcept {
    return {value, flags};
}

/**
 * Output manipulator for hexadecimal string output.
 *
 * When written into a `IOutputStream`, writes out the provided characters
 * in hexadecimal form divided by space character.
 *
 * Example usage:
 * @code
 * stream << HexText(TStringBuf("abcи"));  // Will output "61 62 63 D0 B8"
 * stream << HexText(TWtringBuf(u"abcи")); // Will output "0061 0062 0063 0438"
 * @endcode
 *
 * @param value                         String to output.
 */
template <typename TChar>
static inline ::NFormatPrivate::TBaseText<TChar, 16> HexText(const TBasicStringBuf<TChar> value) {
    return ::NFormatPrivate::TBaseText<TChar, 16>(value);
}

/**
 * Output manipulator for binary string output.
 *
 * When written into a `IOutputStream`, writes out the provided characters
 * in binary form divided by space character.
 *
 * Example usage:
 * @code
 * stream << BinText(TStringBuf("aaa"));  // Will output "01100001 01100001 01100001"
 * @endcode
 *
 * @param value                         String to output.
 */
template <typename TChar>
static inline ::NFormatPrivate::TBaseText<TChar, 2> BinText(const TBasicStringBuf<TChar> value) {
    return ::NFormatPrivate::TBaseText<TChar, 2>(value);
}

/**
 * Output manipulator for printing `TDuration` values.
 *
 * When written into a `IOutputStream`, writes out the provided `TDuration`
 * in auto-adjusted human-readable format.
 *
 * Example usage:
 * @code
 * stream << HumanReadable(TDuration::MicroSeconds(100));   // Will output "100us"
 * stream << HumanReadable(TDuration::Seconds(3672));       // Will output "1h 1m 12s"
 * @endcode
 *
 * @param value                         Value to output.
 */
static constexpr ::NFormatPrivate::THumanReadableDuration HumanReadable(const TDuration duration) noexcept {
    return ::NFormatPrivate::THumanReadableDuration(duration);
}

/**
 * Output manipulator for writing out human-readable number of elements / memory
 * amount in `ls -h` style.
 *
 * When written into a `IOutputStream`, writes out the provided unsigned integer
 * variable with small precision and a suffix (like 'K', 'M', 'G' for numbers, or
 * 'B', 'KiB', 'MiB', 'GiB' for bytes).
 *
 * For quantities, base 1000 is used. For bytes, base is 1024.
 *
 * Example usage:
 * @code
 * stream <<  HumanReadableSize(1024, SF_QUANTITY);                          // Will output    "1.02K"
 * stream <<  HumanReadableSize(1024, SF_BYTES);                             // Will output    "1KiB"
 * stream << "average usage " << HumanReadableSize(100 / 3., SF_BYTES);     // Will output    "average usage "33.3B""
 * @endcode
 *
 * @param value                         Value to output.
 * @param format                        Format to use.
 */
static constexpr ::NFormatPrivate::THumanReadableSize HumanReadableSize(const double size, ESizeFormat format) noexcept {
    return {size, format};
}

void Time(IOutputStream& l);
void TimeHumanReadable(IOutputStream& l);

/**
 * Output manipulator for adjusting precision of floating point values.
 *
 * When written into a `IOutputStream`, writes out the provided floating point
 * variable with given precision. The behavior depends on provided `mode`.
 *
 * Example usage:
 * @code
 * stream <<  Prec(1.2345678901234567, PREC_AUTO);   // Will output "1.2345678901234567"
 * @endcode
 *
 * @param value                         float or double to output.
 * @param mode                          Output mode.
 * @param ndigits                       Number of significant digits (in `PREC_NDIGITS` and `PREC_POINT_DIGITS` mode).
 * @see EFloatToStringMode
 */
template <typename T>
static constexpr ::NFormatPrivate::TFloatPrecision<T> Prec(const T& value, const EFloatToStringMode mode, const int ndigits = 0) noexcept {
    return {value, mode, ndigits};
}

/**
 * Output manipulator for adjusting precision of floating point values.
 *
 * When written into a `IOutputStream`, writes out the provided floating point
 * variable with given precision. The behavior is equivalent to `Prec(value, PREC_NDIGITS, ndigits)`.
 *
 * Example usage:
 * @code
 * stream <<  Prec(1.2345678901234567, 3);   // Will output "1.23"
 * @endcode
 *
 * @param value                         float or double to output.
 * @param ndigits                       Number of significant digits.
 */
template <typename T>
static constexpr ::NFormatPrivate::TFloatPrecision<T> Prec(const T& value, const int ndigits) noexcept {
    return {value, PREC_NDIGITS, ndigits};
}
