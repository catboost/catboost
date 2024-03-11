#include <util/system/defaults.h>

#if defined(_freebsd_) && !defined(__LONG_LONG_SUPPORTED)
    #define __LONG_LONG_SUPPORTED
#endif

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include <util/string/type.h>
#include <util/string/cast.h>
#include <util/string/escape.h>

#include <contrib/libs/double-conversion/double-conversion/double-conversion.h>

#include <util/generic/string.h>
#include <util/system/yassert.h>
#include <util/generic/yexception.h>
#include <util/generic/typetraits.h>
#include <util/generic/ylimits.h>
#include <util/generic/singleton.h>
#include <util/generic/utility.h>

using double_conversion::DoubleToStringConverter;
using double_conversion::StringBuilder;
using double_conversion::StringToDoubleConverter;

/*
 * ------------------------------ formatters ------------------------------
 */

namespace {

    // clang-format off
    constexpr int LetterToIntMap[] = {
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 0, 1,
        2, 3, 4, 5, 6, 7, 8, 9, 20, 20,
        20, 20, 20, 20, 20, 10, 11, 12, 13, 14,
        15, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 10, 11, 12,
        13, 14, 15,
    };
    // clang-format on

    template <class T>
    std::make_signed_t<T> NegatePositiveSigned(T value) noexcept {
        return value > 0 ? (-std::make_signed_t<T>(value - 1) - 1) : 0;
    }

    template <class T>
    struct TFltModifiers;

    template <class T, int base, class TChar>
    Y_NO_INLINE size_t FormatInt(T value, TChar* buf, size_t len) {
        return TIntStringBuf<T, base, TChar>::Convert(value, buf, len);
    }

    template <class T>
    inline size_t FormatFlt(T t, char* buf, size_t len) {
        const int ret = snprintf(buf, len, TFltModifiers<T>::ModifierWrite, t);

        Y_ENSURE(ret >= 0 && (size_t)ret <= len, TStringBuf("cannot format float"));

        return (size_t)ret;
    }

    enum EParseStatus {
        PS_OK = 0,
        PS_EMPTY_STRING,
        PS_PLUS_STRING,
        PS_MINUS_STRING,
        PS_BAD_SYMBOL,
        PS_OVERFLOW,
    };

    constexpr ui8 SAFE_LENS[4][17] = {
        {0, 0, 7, 5, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1},
        {0, 0, 15, 10, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3},
        {0, 0, 31, 20, 15, 13, 12, 11, 10, 10, 9, 9, 8, 8, 8, 8, 7},
        {0, 0, 63, 40, 31, 27, 24, 22, 21, 20, 19, 18, 17, 17, 16, 16, 15},
    };

    inline constexpr ui8 ConstLog2(ui8 x) noexcept {
        return x == 1 ? 0 : 1 + ConstLog2(x / 2);
    }

    template <unsigned BASE, class TChar, class T>
    inline std::enable_if_t<(BASE > 10), bool> CharToDigit(TChar c, T* digit) noexcept {
        unsigned uc = c;

        if (uc >= Y_ARRAY_SIZE(LetterToIntMap)) {
            return false;
        }

        *digit = LetterToIntMap[uc];

        return *digit < BASE;
    }

    template <unsigned BASE, class TChar, class T>
    inline std::enable_if_t<(BASE <= 10), bool> CharToDigit(TChar c, T* digit) noexcept {
        return (c >= '0') && ((*digit = (c - '0')) < BASE);
    }

    template <class T, unsigned base, class TChar>
    struct TBasicIntParser {
        static_assert(1 < base && base < 17, "Expect 1 < base && base < 17.");
        static_assert(std::is_unsigned<T>::value, "TBasicIntParser can only handle unsigned integers.");

        enum : unsigned {
            BASE_POW_2 = base * base,
        };

        static inline EParseStatus Parse(const TChar** ppos, const TChar* end, T max, T* target) noexcept {
            Y_ASSERT(*ppos != end); /* This check should be somewhere up the stack. */
            const size_t maxSafeLen = SAFE_LENS[ConstLog2(sizeof(T))][base];

            // can parse without overflow
            if (size_t(end - *ppos) <= maxSafeLen) {
                T result;

                if (ParseFast(*ppos, end, &result) && result <= max) {
                    *target = result;

                    return PS_OK;
                }
            }

            return ParseSlow(ppos, end, max, target);
        }

        static inline bool ParseFast(const TChar* pos, const TChar* end, T* target) noexcept {
            T result = T();
            T d1;
            T d2;

            // we have end > pos
            auto beforeEnd = end - 1;

            while (pos < beforeEnd && CharToDigit<base>(*pos, &d1) && CharToDigit<base>(*(pos + 1), &d2)) {
                result = result * BASE_POW_2 + d1 * base + d2;
                pos += 2;
            }

            while (pos != end && CharToDigit<base>(*pos, &d1)) {
                result = result * base + d1;
                ++pos;
            }

            *target = result;

            return pos == end;
        }

        static inline EParseStatus ParseSlow(const TChar** ppos, const TChar* end, T max, T* target) noexcept {
            T result = T();
            T preMulMax = max / base;
            const TChar* pos = *ppos;

            while (pos != end) {
                T digit;

                if (!CharToDigit<base>(*pos, &digit)) {
                    *ppos = pos;

                    return PS_BAD_SYMBOL;
                }

                if (result > preMulMax) {
                    return PS_OVERFLOW;
                }

                result *= base;

                if (result > max - digit) {
                    return PS_OVERFLOW;
                }

                result += digit;
                pos++;
            }

            *target = result;

            return PS_OK;
        }
    };

    template <class T>
    struct TBounds {
        T PositiveMax;
        T NegativeMax;
    };

    template <class T, unsigned base, class TChar>
    struct TIntParser {
        static_assert(1 < base && base < 17, "Expect 1 < base && base < 17.");
        static_assert(std::is_integral<T>::value, "T must be an integral type.");

        enum {
            IsSigned = std::is_signed<T>::value
        };

        using TUnsigned = std::make_unsigned_t<T>;

        static inline EParseStatus Parse(const TChar** ppos, const TChar* end, const TBounds<TUnsigned>& bounds, T* target) {
            const TChar* pos = *ppos;
            if (pos == end) {
                return PS_EMPTY_STRING;
            }

            bool negative = false;
            TUnsigned max;
            if (*pos == '+') {
                pos++;
                max = bounds.PositiveMax;

                if (pos == end) {
                    return PS_PLUS_STRING;
                }
            } else if (IsSigned && *pos == '-') {
                pos++;
                max = bounds.NegativeMax;
                negative = true;

                if (pos == end) {
                    return PS_MINUS_STRING;
                }
            } else {
                max = bounds.PositiveMax;
            }

            TUnsigned result;
            EParseStatus error = TBasicIntParser<TUnsigned, base, TChar>::Parse(&pos, end, max, &result);
            if (error != PS_OK) {
                *ppos = pos;
                return error;
            }

            if (IsSigned) {
                *target = negative ? NegatePositiveSigned(result) : static_cast<T>(result);
            } else {
                *target = result;
            }
            return PS_OK;
        }
    };

    template <class TChar>
    [[noreturn]] static Y_NO_INLINE void ThrowParseError(EParseStatus status, const TChar* data, size_t len, const TChar* pos) {
        Y_ASSERT(status != PS_OK);

        typedef TBasicString<TChar> TStringType;

        switch (status) {
            case PS_EMPTY_STRING:
                ythrow TFromStringException() << TStringBuf("Cannot parse empty string as number. ");
            case PS_PLUS_STRING:
                ythrow TFromStringException() << TStringBuf("Cannot parse string \"+\" as number. ");
            case PS_MINUS_STRING:
                ythrow TFromStringException() << TStringBuf("Cannot parse string \"-\" as number. ");
            case PS_BAD_SYMBOL:
                ythrow TFromStringException() << TStringBuf("Unexpected symbol \"") << EscapeC(*pos) << TStringBuf("\" at pos ") << (pos - data) << TStringBuf(" in string ") << TStringType(data, len).Quote() << TStringBuf(". ");
            case PS_OVERFLOW:
                ythrow TFromStringException() << TStringBuf("Integer overflow in string ") << TStringType(data, len).Quote() << TStringBuf(". ");
            default:
                ythrow yexception() << TStringBuf("Unknown error code in string converter. ");
        }
    }

    template <typename T, typename TUnsigned, int base, typename TChar>
    Y_NO_INLINE T ParseInt(const TChar* data, size_t len, const TBounds<TUnsigned>& bounds) {
        T result;
        const TChar* pos = data;
        EParseStatus status = TIntParser<T, base, TChar>::Parse(&pos, pos + len, bounds, &result);

        if (status == PS_OK) {
            return result;
        } else {
            ThrowParseError(status, data, len, pos);
        }
    }

    template <typename T, typename TUnsigned, int base, typename TChar>
    Y_NO_INLINE bool TryParseInt(const TChar* data, size_t len, const TBounds<TUnsigned>& bounds, T* result) {
        return TIntParser<T, base, TChar>::Parse(&data, data + len, bounds, result) == PS_OK;
    }

    template <class T>
    inline T ParseFlt(const char* data, size_t len) {
        /*
         * TODO
         */

        if (len > 256) {
            len = 256;
        }

        char* c = (char*)alloca(len + 1);
        memcpy(c, data, len);
        c[len] = 0;

        T ret;
        char ec;

        // try to read a value and an extra character in order to catch cases when
        // the string start with a valid float but is followed by unexpected characters
        if (sscanf(c, TFltModifiers<T>::ModifierReadAndChar, &ret, &ec) == 1) {
            return ret;
        }

        ythrow TFromStringException() << TStringBuf("cannot parse float(") << TStringBuf(data, len) << TStringBuf(")");
    }

#define DEF_FLT_MOD(type, modifierWrite, modifierRead)                    \
    template <>                                                           \
    struct TFltModifiers<type> {                                          \
        static const char* const ModifierWrite;                           \
        static const char* const ModifierReadAndChar;                     \
    };                                                                    \
                                                                          \
    const char* const TFltModifiers<type>::ModifierWrite = modifierWrite; \
    const char* const TFltModifiers<type>::ModifierReadAndChar = modifierRead "%c";

    DEF_FLT_MOD(long double, "%.10Lg", "%Lg")

#undef DEF_FLT_MOD

    /* The following constants are initialized in terms of <climits> constants to make
     * sure they go into binary as actual values and there is no associated
     * initialization code.
     * */
    constexpr TBounds<ui64> bSBounds = {static_cast<ui64>(SCHAR_MAX), static_cast<ui64>(UCHAR_MAX - SCHAR_MAX)};
    constexpr TBounds<ui64> bUBounds = {static_cast<ui64>(UCHAR_MAX), 0};
    constexpr TBounds<ui64> sSBounds = {static_cast<ui64>(SHRT_MAX), static_cast<ui64>(USHRT_MAX - SHRT_MAX)};
    constexpr TBounds<ui64> sUBounds = {static_cast<ui64>(USHRT_MAX), 0};
    constexpr TBounds<ui64> iSBounds = {static_cast<ui64>(INT_MAX), static_cast<ui64>(UINT_MAX - INT_MAX)};
    constexpr TBounds<ui64> iUBounds = {static_cast<ui64>(UINT_MAX), 0};
    constexpr TBounds<ui64> lSBounds = {static_cast<ui64>(LONG_MAX), static_cast<ui64>(ULONG_MAX - LONG_MAX)};
    constexpr TBounds<ui64> lUBounds = {static_cast<ui64>(ULONG_MAX), 0};
    constexpr TBounds<ui64> llSBounds = {static_cast<ui64>(LLONG_MAX), static_cast<ui64>(ULLONG_MAX - LLONG_MAX)};
    constexpr TBounds<ui64> llUBounds = {static_cast<ui64>(ULLONG_MAX), 0};
}

#define DEF_INT_SPEC_II(TYPE, ITYPE, BASE)                              \
    template <>                                                         \
    size_t IntToString<BASE, TYPE>(TYPE value, char* buf, size_t len) { \
        return FormatInt<ITYPE, BASE, char>(value, buf, len);           \
    }

#define DEF_INT_SPEC_I(TYPE, ITYPE)                                \
    template <>                                                    \
    size_t ToStringImpl<TYPE>(TYPE value, char* buf, size_t len) { \
        return FormatInt<ITYPE, 10, char>(value, buf, len);        \
    }                                                              \
    DEF_INT_SPEC_II(TYPE, ITYPE, 2)                                \
    DEF_INT_SPEC_II(TYPE, ITYPE, 8)                                \
    DEF_INT_SPEC_II(TYPE, ITYPE, 10)                               \
    DEF_INT_SPEC_II(TYPE, ITYPE, 16)

#define DEF_INT_SPEC(TYPE)           \
    DEF_INT_SPEC_I(signed TYPE, i64) \
    DEF_INT_SPEC_I(unsigned TYPE, ui64)

DEF_INT_SPEC(char)
DEF_INT_SPEC(short)
DEF_INT_SPEC(int)
DEF_INT_SPEC(long)
DEF_INT_SPEC(long long)

#ifdef __cpp_char8_t
template <>
size_t ToStringImpl<char8_t>(char8_t value, char* buf, size_t len) {
    return FormatInt<ui64, 10, char>(value, buf, len);
}
#endif

using TCharIType = std::conditional_t<std::is_signed<char>::value, i64, ui64>;
using TWCharIType = std::conditional_t<std::is_signed<wchar_t>::value, i64, ui64>;

DEF_INT_SPEC_I(char, TCharIType)
DEF_INT_SPEC_I(wchar_t, TWCharIType)
DEF_INT_SPEC_I(wchar16, ui64) // wchar16 is always unsigned
DEF_INT_SPEC_I(wchar32, ui64) // wchar32 is always unsigned

#undef DEF_INT_SPEC
#undef DEF_INT_SPEC_I
#undef DEF_INT_SPEC_II

#define DEF_FLT_SPEC(type)                                     \
    template <>                                                \
    size_t ToStringImpl<type>(type t, char* buf, size_t len) { \
        return FormatFlt<type>(t, buf, len);                   \
    }

DEF_FLT_SPEC(long double)

#undef DEF_FLT_SPEC

template <>
size_t ToStringImpl<bool>(bool t, char* buf, size_t len) {
    Y_ENSURE(len, TStringBuf("zero length"));
    *buf = t ? '1' : '0';
    return 1;
}

/*
 * ------------------------------ parsers ------------------------------
 */

template <>
bool TryFromStringImpl<bool>(const char* data, size_t len, bool& result) {
    if (len == 1) {
        if (data[0] == '0') {
            result = false;
            return true;
        } else if (data[0] == '1') {
            result = true;
            return true;
        }
    }
    TStringBuf buf(data, len);
    if (IsTrue(buf)) {
        result = true;
        return true;
    } else if (IsFalse(buf)) {
        result = false;
        return true;
    }
    return false;
}

template <>
bool FromStringImpl<bool>(const char* data, size_t len) {
    bool result;

    if (!TryFromStringImpl<bool>(data, len, result)) {
        ythrow TFromStringException() << TStringBuf("Cannot parse bool(") << TStringBuf(data, len) << TStringBuf("). ");
    }

    return result;
}

template <>
TString FromStringImpl<TString>(const char* data, size_t len) {
    return TString(data, len);
}

template <>
TStringBuf FromStringImpl<TStringBuf>(const char* data, size_t len) {
    return TStringBuf(data, len);
}

template <>
std::string FromStringImpl<std::string>(const char* data, size_t len) {
    return std::string(data, len);
}

template <>
std::filesystem::path FromStringImpl<std::filesystem::path>(const char* data, size_t len) {
    return std::filesystem::path(std::string(data, len));
}

template <>
TUtf16String FromStringImpl<TUtf16String>(const wchar16* data, size_t len) {
    return TUtf16String(data, len);
}

template <>
TWtringBuf FromStringImpl<TWtringBuf>(const wchar16* data, size_t len) {
    return TWtringBuf(data, len);
}

// Try-versions
template <>
bool TryFromStringImpl<TStringBuf>(const char* data, size_t len, TStringBuf& result) {
    result = {data, len};
    return true;
}

template <>
bool TryFromStringImpl<TString>(const char* data, size_t len, TString& result) {
    result = TString(data, len);
    return true;
}

template <>
bool TryFromStringImpl<std::string>(const char* data, size_t len, std::string& result) {
    result.assign(data, len);
    return true;
}

template <>
bool TryFromStringImpl<TWtringBuf>(const wchar16* data, size_t len, TWtringBuf& result) {
    result = {data, len};
    return true;
}

template <>
bool TryFromStringImpl<TUtf16String>(const wchar16* data, size_t len, TUtf16String& result) {
    result = TUtf16String(data, len);
    return true;
}

#define DEF_INT_SPEC_III(CHAR, TYPE, ITYPE, BOUNDS, BASE)                      \
    template <>                                                                \
    TYPE IntFromString<TYPE, BASE>(const CHAR* data, size_t len) {             \
        return ParseInt<ITYPE, ui64, BASE>(data, len, BOUNDS);                 \
    }                                                                          \
    template <>                                                                \
    bool TryIntFromString<BASE>(const CHAR* data, size_t len, TYPE& result) {  \
        ITYPE tmp;                                                             \
        bool status = TryParseInt<ITYPE, ui64, BASE>(data, len, BOUNDS, &tmp); \
        if (status) {                                                          \
            result = tmp;                                                      \
        }                                                                      \
        return status;                                                         \
    }

#define DEF_INT_SPEC_II(CHAR, TYPE, ITYPE, BOUNDS)                             \
    template <>                                                                \
    TYPE FromStringImpl<TYPE>(const CHAR* data, size_t len) {                  \
        return ParseInt<ITYPE, ui64, 10>(data, len, BOUNDS);                   \
    }                                                                          \
    template <>                                                                \
    bool TryFromStringImpl<TYPE>(const CHAR* data, size_t len, TYPE& result) { \
        ITYPE tmp;                                                             \
        bool status = TryParseInt<ITYPE, ui64, 10>(data, len, BOUNDS, &tmp);   \
        if (status) {                                                          \
            result = tmp;                                                      \
        }                                                                      \
        return status;                                                         \
    }                                                                          \
    DEF_INT_SPEC_III(CHAR, TYPE, ITYPE, BOUNDS, 2)                             \
    DEF_INT_SPEC_III(CHAR, TYPE, ITYPE, BOUNDS, 8)                             \
    DEF_INT_SPEC_III(CHAR, TYPE, ITYPE, BOUNDS, 10)                            \
    DEF_INT_SPEC_III(CHAR, TYPE, ITYPE, BOUNDS, 16)

#define DEF_INT_SPEC_I(TYPE, ITYPE, BOUNDS)    \
    DEF_INT_SPEC_II(char, TYPE, ITYPE, BOUNDS) \
    DEF_INT_SPEC_II(wchar16, TYPE, ITYPE, BOUNDS)

#define DEF_INT_SPEC(TYPE, ID)                    \
    DEF_INT_SPEC_I(signed TYPE, i64, ID##SBounds) \
    DEF_INT_SPEC_I(unsigned TYPE, ui64, ID##UBounds)

#define DEF_INT_SPEC_FIXED_WIDTH(TYPE, ID) \
    DEF_INT_SPEC_I(TYPE, i64, ID##SBounds) \
    DEF_INT_SPEC_I(u##TYPE, ui64, ID##UBounds)

DEF_INT_SPEC_FIXED_WIDTH(i8, b)
DEF_INT_SPEC(short, s)
DEF_INT_SPEC(int, i)
DEF_INT_SPEC(long, l)
DEF_INT_SPEC(long long, ll)

#undef DEF_INT_SPEC_FIXED_WIDTH
#undef DEF_INT_SPEC
#undef DEF_INT_SPEC_I
#undef DEF_INT_SPEC_II
#undef DEF_INT_SPEC_III

#define DEF_FLT_SPEC(type)                                    \
    template <>                                               \
    type FromStringImpl<type>(const char* data, size_t len) { \
        return ParseFlt<type>(data, len);                     \
    }

DEF_FLT_SPEC(long double)

#undef DEF_FLT_SPEC

// Using StrToD for float and double because it is faster than sscanf.
// Exception-free, specialized for float types
template <>
bool TryFromStringImpl<double>(const char* data, size_t len, double& result) {
    if (!len) {
        return false;
    }

    char* se = nullptr;
    double d = StrToD(data, data + len, &se);

    if (se != data + len) {
        return false;
    }
    result = d;
    return true;
}

template <>
bool TryFromStringImpl<float>(const char* data, size_t len, float& result) {
    double d;
    if (TryFromStringImpl<double>(data, len, d)) {
        result = static_cast<float>(d);
        return true;
    }
    return false;
}

template <>
bool TryFromStringImpl<long double>(const char* data, size_t len, long double& result) {
    double d;
    if (TryFromStringImpl<double>(data, len, d)) {
        result = static_cast<long double>(d);
        return true;
    }
    return false;
}

// Exception-throwing, specialized for float types
template <>
double FromStringImpl<double>(const char* data, size_t len) {
    double d = 0.0;
    if (!TryFromStringImpl(data, len, d)) {
        ythrow TFromStringException() << TStringBuf("cannot parse float(") << TStringBuf(data, len) << TStringBuf(")");
    }
    return d;
}

template <>
float FromStringImpl<float>(const char* data, size_t len) {
    return static_cast<float>(FromStringImpl<double>(data, len));
}

double StrToD(const char* b, const char* e, char** se) {
    struct TCvt: public StringToDoubleConverter {
        inline TCvt()
            : StringToDoubleConverter(ALLOW_TRAILING_JUNK | ALLOW_HEX | ALLOW_LEADING_SPACES, 0.0, NAN, nullptr, nullptr)
        {
        }
    };

    int out = 0;

    const auto res = SingletonWithPriority<TCvt, 0>()->StringToDouble(b, e - b, &out);

    if (se) {
        *se = (char*)(b + out);
    }

    return res;
}

double StrToD(const char* b, char** se) {
    return StrToD(b, b + strlen(b), se);
}

namespace {
    static inline DoubleToStringConverter& ToStringConverterNoPad() noexcept {
        struct TCvt: public DoubleToStringConverter {
            inline TCvt() noexcept
                : DoubleToStringConverter(EMIT_POSITIVE_EXPONENT_SIGN, "inf", "nan", 'e', -10, 21, 4, 0)
            {
            }
        };

        return *SingletonWithPriority<TCvt, 0>();
    }

    struct TBuilder {
        alignas(StringBuilder) char Store[sizeof(StringBuilder)];
        StringBuilder* SB;

        inline TBuilder(char* buf, size_t len) noexcept
            : SB(new (Store) StringBuilder(buf, len))
        {
        }
    };

    static inline size_t FixZeros(char* buf, size_t len) noexcept {
        auto end = buf + len;
        auto point = (char*)memchr(buf, '.', len);

        if (!point) {
            return len;
        }

        auto exp = (char*)memchr(point, 'e', end - point);

        if (!exp) {
            exp = end;
        }

        auto c = exp;

        c -= 1;

        while (point < c && *c == '0') {
            --c;
        }

        if (*c == '.') {
            --c;
        }

        memmove(c + 1, exp, end - exp);

        return c - buf + 1 + end - exp;
    }

    static inline size_t FixEnd(char* buf, size_t len) noexcept {
        if (len > 2) {
            auto sign = buf[len - 2];

            if (sign == '-' || sign == '+') {
                buf[len] = buf[len - 1];
                buf[len - 1] = '0';
                ++len;
            }
        }

        buf[len] = 0;

        return len;
    }

    static inline size_t DoDtoa(double d, char* buf, size_t len, int prec) noexcept {
        TBuilder sb(buf, len);

        Y_ABORT_UNLESS(ToStringConverterNoPad().ToPrecision(d, prec, sb.SB), "conversion failed");

        return FixEnd(buf, FixZeros(buf, sb.SB->position()));
    }
}

template <>
size_t ToStringImpl<double>(double d, char* buf, size_t len) {
    return DoDtoa(d, buf, len, 10);
}

template <>
size_t ToStringImpl<float>(float f, char* buf, size_t len) {
    return DoDtoa(f, buf, len, 6);
}

size_t FloatToString(float t, char* buf, size_t len, EFloatToStringMode mode, int ndigits) {
    if (mode == PREC_AUTO) {
        TBuilder sb(buf, len);

        Y_ABORT_UNLESS(ToStringConverterNoPad().ToShortestSingle(t, sb.SB), "conversion failed");

        return FixEnd(buf, sb.SB->position());
    }

    return FloatToString((double)t, buf, len, mode, ndigits);
}

size_t FloatToString(double t, char* buf, size_t len, EFloatToStringMode mode, int ndigits) {
    if (mode == PREC_NDIGITS) {
        auto minDigits = DoubleToStringConverter::kMinPrecisionDigits;
        auto maxDigits = DoubleToStringConverter::kMaxPrecisionDigits;

        return DoDtoa(t, buf, len, ClampVal(ndigits, minDigits, maxDigits));
    }

    TBuilder sb(buf, len);

    if (mode == PREC_AUTO) {
        Y_ABORT_UNLESS(ToStringConverterNoPad().ToShortest(t, sb.SB), "conversion failed");

        return FixEnd(buf, sb.SB->position());
    }

    if (!ToStringConverterNoPad().ToFixed(t, ndigits, sb.SB)) {
        return FloatToString(t, buf, len, PREC_AUTO);
    }

    if (mode == PREC_POINT_DIGITS_STRIP_ZEROES) {
        return FixZeros(buf, sb.SB->position());
    }

    return sb.SB->position();
}
