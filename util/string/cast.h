#pragma once

#include <util/system/defaults.h>
#include <util/stream/str.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/typetraits.h>
#include <util/generic/yexception.h>

/*
 * specialized for all arithmetic types
 */

template <class T>
size_t ToStringImpl(T t, char* buf, size_t len);

/**
 * Converts @c t to string writing not more than @c len bytes to output buffer @c buf.
 * No NULL terminator appended! Throws exception on buffer overflow.
 * @return number of bytes written
 */
template <class T>
inline size_t ToString(const T& t, char* buf, size_t len) {
    using TParam = typename TTypeTraits<T>::TFuncParam;

    return ToStringImpl<TParam>(t, buf, len);
}

/**
 * Floating point to string conversion mode, values are enforced by `dtoa_impl.cpp`.
 */
enum EFloatToStringMode {
    /** 0.1f -> "0.1", 0.12345678f -> "0.12345678", ignores ndigits. */
    PREC_AUTO = 0,

    /** "%g" mode, writes up to the given number of significant digits:
     *  0.1f -> "0.1", 0.12345678f -> "0.123457" for ndigits=6, 1.2e-06f -> "1.2e-06" */
    PREC_NDIGITS = 2,

    /** "%f" mode, writes the given number of digits after decimal point:
     *  0.1f -> "0.100000", 1.2e-06f -> "0.000001" for ndigits=6 */
    PREC_POINT_DIGITS = 3,

    /** same as PREC_POINT_DIGITS, but stripping trailing zeroes:
     * 0.1f for ndgigits=6 -> "0.1" */
    PREC_POINT_DIGITS_STRIP_ZEROES = 4
};

size_t FloatToString(float t, char* buf, size_t len, EFloatToStringMode mode = PREC_AUTO, int ndigits = 0);
size_t FloatToString(double t, char* buf, size_t len, EFloatToStringMode mode = PREC_AUTO, int ndigits = 0);

template <typename T>
inline TString FloatToString(const T& t, EFloatToStringMode mode = PREC_AUTO, int ndigits = 0) {
    char buf[512]; // Max<double>() with mode = PREC_POINT_DIGITS has 309 digits before the decimal point
    size_t count = FloatToString(t, buf, sizeof(buf), mode, ndigits);
    return TString(buf, count);
}

namespace NPrivate {
    template <class T, bool isSimple>
    struct TToString {
        static inline TString Cvt(const T& t) {
            char buf[512];

            return TString(buf, ToString<T>(t, buf, sizeof(buf)));
        }
    };

    template <class T>
    struct TToString<T, false> {
        static inline TString Cvt(const T& t) {
            TString s;
            TStringOutput o(s);
            o << t;
            return s;
        }
    };
}

/*
 * some clever implementations...
 */
template <class T>
inline TString ToString(const T& t) {
    using TR = std::remove_cv_t<T>;

    return ::NPrivate::TToString<TR, std::is_arithmetic<TR>::value>::Cvt((const TR&)t);
}

inline const TString& ToString(const TString& s) noexcept {
    return s;
}

inline const TString& ToString(TString& s) noexcept {
    return s;
}

inline TString ToString(const char* s) {
    return s;
}

inline TString ToString(char* s) {
    return s;
}

/*
 * Wrapper for wide strings.
 */
template <class T>
inline TUtf16String ToWtring(const T& t) {
    return TUtf16String::FromAscii(ToString(t));
}

inline const TUtf16String& ToWtring(const TUtf16String& w) {
    return w;
}

inline const TUtf16String& ToWtring(TUtf16String& w) {
    return w;
}

struct TFromStringException: public TBadCastException {
};

/*
 * specialized for:
 *  bool
 *  short
 *  unsigned short
 *  int
 *  unsigned int
 *  long
 *  unsigned long
 *  long long
 *  unsigned long long
 *  float
 *  double
 *  long double
 */
template <typename T, typename TChar>
T FromStringImpl(const TChar* data, size_t len);

template <typename T, typename TChar>
inline T FromString(const TChar* data, size_t len) {
    return ::FromStringImpl<T>(data, len);
}

template <typename T, typename TChar>
inline T FromString(const TChar* data) {
    return ::FromString<T>(data, TCharTraits<TChar>::GetLength(data));
}

template <class T>
inline T FromString(const TStringBuf& s) {
    return ::FromString<T>(s.data(), s.size());
}

template <class T>
inline T FromString(const TString& s) {
    return ::FromString<T>(s.data(), s.size());
}

template <>
inline TString FromString<TString>(const TString& s) {
    return s;
}

template <class T>
inline T FromString(const TWtringBuf& s) {
    return ::FromString<T, typename TWtringBuf::char_type>(s.data(), s.size());
}

template <class T>
inline T FromString(const TUtf16String& s) {
    return ::FromString<T, wchar16>(s.data(), s.size());
}

namespace NPrivate {
    template <typename TChar>
    class TFromString {
        const TChar* const Data;
        const size_t Len;

    public:
        inline TFromString(const TChar* data, size_t len)
            : Data(data)
            , Len(len)
        {
        }

        template <typename T>
        inline operator T() const {
            return FromString<T, TChar>(Data, Len);
        }
    };
}

template <typename TChar>
inline ::NPrivate::TFromString<TChar> FromString(const TChar* data, size_t len) {
    return ::NPrivate::TFromString<TChar>(data, len);
}

template <typename TChar>
inline ::NPrivate::TFromString<TChar> FromString(const TChar* data) {
    return ::NPrivate::TFromString<TChar>(data, TCharTraits<TChar>::GetLength(data));
}

template <typename T>
inline ::NPrivate::TFromString<typename T::TChar> FromString(const T& s) {
    return ::NPrivate::TFromString<typename T::TChar>(s.data(), s.size());
}

// Conversion exception free versions
template <typename T, typename TChar>
bool TryFromStringImpl(const TChar* data, size_t len, T& result);

/**
 * @param data Source string buffer pointer
 * @param len Source string length, in characters
 * @param result Place to store conversion result value.
 * If conversion error occurs, no value stored in @c result
 * @return @c true in case of successful conversion, @c false otherwise
 **/
template <typename T, typename TChar>
inline bool TryFromString(const TChar* data, size_t len, T& result) {
    return TryFromStringImpl<T>(data, len, result);
}

template <typename T, typename TChar>
inline bool TryFromString(const TChar* data, T& result) {
    return TryFromString<T>(data, TCharTraits<TChar>::GetLength(data), result);
}

template <class T, class TChar>
inline bool TryFromString(const TChar* data, const size_t len, T& result, const T& def) {
    if (TryFromString<T>(data, len, result)) {
        return true;
    }
    result = def;
    return false;
}

template <class T>
inline bool TryFromString(const TStringBuf& s, T& result) {
    return TryFromString<T>(s.data(), s.size(), result);
}

template <class T>
inline bool TryFromString(const TString& s, T& result) {
    return TryFromString<T>(s.data(), s.size(), result);
}

template <class T>
inline bool TryFromString(const TWtringBuf& s, T& result) {
    return TryFromString<T>(s.data(), s.size(), result);
}

template <class T>
inline bool TryFromString(const TUtf16String& s, T& result) {
    return TryFromString<T>(s.data(), s.size(), result);
}

template <class T, class TStringType>
inline bool TryFromStringWithDefault(const TStringType& s, T& result, const T& def) {
    return TryFromString<T>(s.data(), s.size(), result, def);
}

template <class T>
inline bool TryFromStringWithDefault(const char* s, T& result, const T& def) {
    return TryFromStringWithDefault<T>(TStringBuf(s), result, def);
}

template <class T, class TStringType>
inline bool TryFromStringWithDefault(const TStringType& s, T& result) {
    return TryFromStringWithDefault<T>(s, result, T());
}

// FromString methods with default value if data is invalid
template <class T, class TChar>
inline T FromString(const TChar* data, const size_t len, const T& def) {
    T result;
    TryFromString<T>(data, len, result, def);
    return result;
}

template <class T, class TStringType>
inline T FromStringWithDefault(const TStringType& s, const T& def) {
    return FromString<T>(s.data(), s.size(), def);
}

template <class T>
inline T FromStringWithDefault(const char* s, const T& def) {
    return FromStringWithDefault<T>(TStringBuf(s), def);
}

template <class T, class TStringType>
inline T FromStringWithDefault(const TStringType& s) {
    return FromStringWithDefault<T>(s, T());
}

double StrToD(const char* b, char** se);
double StrToD(const char* b, const char* e, char** se);

template <int base, class T>
size_t IntToString(T t, char* buf, size_t len);

template <int base, class T>
inline TString IntToString(T t) {
    static_assert(std::is_arithmetic<std::remove_cv_t<T>>::value, "expect std::is_arithmetic<std::remove_cv_t<T>>::value");

    char buf[256];

    return TString(buf, IntToString<base>(t, buf, sizeof(buf)));
}

template <int base, class TInt, class TChar>
bool TryIntFromString(const TChar* data, size_t len, TInt& result);

template <int base, class TInt, class TStringType>
inline bool TryIntFromString(const TStringType& s, TInt& result) {
    return TryIntFromString<base>(s.data(), s.size(), result);
}

template <class TInt, int base, class TChar>
TInt IntFromString(const TChar* str, size_t len);

template <class TInt, int base, class TChar>
inline TInt IntFromString(const TChar* str) {
    return IntFromString<TInt, base>(str, TCharTraits<TChar>::GetLength(str));
}

template <class TInt, int base, class TStringType>
inline TInt IntFromString(const TStringType& str) {
    return IntFromString<TInt, base>(str.data(), str.size());
}

static inline TString ToString(const TStringBuf str) {
    return TString(str);
}

static inline TUtf16String ToWtring(const TWtringBuf wtr) {
    return TUtf16String(wtr);
}

static inline TUtf32String ToUtf32String(const TUtf32StringBuf wtr) {
    return TUtf32String(wtr);
}
