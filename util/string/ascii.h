#pragma once

#include <util/system/defaults.h>
#include <util/system/compat.h>
#include <util/generic/string.h>

// ctype.h-like functions, locale-independent:
//      IsAscii{Upper,Lower,Digit,Alpha,Alnum,Space} and
//      AsciiTo{Upper,Lower}
//
// standard functions from <ctype.h> are locale dependent,
// and cause undefined behavior when called on chars outside [0..127] range

namespace NPrivate {
    enum ECharClass {
        CC_SPACE = 1,
        CC_UPPER = 2,
        CC_LOWER = 4,
        CC_DIGIT = 8,
        CC_ALPHA = 16,
        CC_ALNUM = 32,
        CC_ISHEX = 64,
        CC_PUNCT = 128,
    };

    extern const unsigned char ASCII_CLASS[256];
    extern const unsigned char ASCII_LOWER[256];

    template <class T>
    struct TDereference {
        using type = T;
    };

#ifndef TSTRING_IS_STD_STRING
    template <class String>
    struct TDereference<TBasicCharRef<String>> {
        using type = typename String::value_type;
    };
#endif

    template <class T>
    using TDereferenced = typename TDereference<T>::type;

    template <class T>
    bool RangeOk(T c) noexcept {
        static_assert(std::is_integral<T>::value, "Integral type character expected");

        if (sizeof(T) == 1) {
            return true;
        }

        return c >= static_cast<T>(0) && c <= static_cast<T>(127);
    }

#ifndef TSTRING_IS_STD_STRING
    template <class String>
    bool RangeOk(const TBasicCharRef<String>& c) {
        return RangeOk(static_cast<typename String::value_type>(c));
    }
#endif
}

constexpr bool IsAscii(const int c) noexcept {
    return !(c & ~0x7f);
}

inline bool IsAsciiSpace(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_SPACE;
}

inline bool IsAsciiUpper(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_UPPER;
}

inline bool IsAsciiLower(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_LOWER;
}

inline bool IsAsciiDigit(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_DIGIT;
}

inline bool IsAsciiAlpha(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_ALPHA;
}

inline bool IsAsciiAlnum(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_ALNUM;
}

inline bool IsAsciiHex(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_ISHEX;
}

inline bool IsAsciiPunct(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_PUNCT;
}

// some overloads

template <class T>
inline bool IsAsciiSpace(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiSpace(static_cast<unsigned char>(c));
}

template <class T>
inline bool IsAsciiUpper(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiUpper(static_cast<unsigned char>(c));
}

template <class T>
inline bool IsAsciiLower(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiLower(static_cast<unsigned char>(c));
}

template <class T>
inline bool IsAsciiDigit(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiDigit(static_cast<unsigned char>(c));
}

template <class T>
inline bool IsAsciiAlpha(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiAlpha(static_cast<unsigned char>(c));
}

template <class T>
inline bool IsAsciiAlnum(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiAlnum(static_cast<unsigned char>(c));
}

template <class T>
inline bool IsAsciiHex(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiHex(static_cast<unsigned char>(c));
}

template <class T>
inline bool IsAsciiPunct(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiPunct(static_cast<unsigned char>(c));
}

// some extra helpers
inline ui8 AsciiToLower(ui8 c) noexcept {
    return ::NPrivate::ASCII_LOWER[c];
}

inline char AsciiToLower(char c) noexcept {
    return (char)AsciiToLower((ui8)c);
}

template <class T>
inline ::NPrivate::TDereferenced<T> AsciiToLower(T c) noexcept {
    return (c >= 0 && c <= 127) ? (::NPrivate::TDereferenced<T>)AsciiToLower((ui8)c) : c;
}

template <class T>
inline ::NPrivate::TDereferenced<T> AsciiToUpper(T c) noexcept {
    return IsAsciiLower(c) ? (c + ('A' - 'a')) : c;
}

/**
 * ASCII case-insensitive string comparison (for proper UTF8 strings
 * case-insensitive comparison consider using @c library/cpp/charset).
 *
 * BUGS: Currently will NOT work properly with strings that contain
 * 0-terminator character inside. See IGNIETFERRO-1641 for details.
 *
 * @return                              true iff @c s1 ans @c s2 are case-insensitively equal.
 */
static inline bool AsciiEqualsIgnoreCase(const char* s1, const char* s2) noexcept {
    return ::stricmp(s1, s2) == 0;
}

/**
 * ASCII case-insensitive string comparison (for proper UTF8 strings
 * case-insensitive comparison consider using @c library/cpp/charset).
 *
 * BUGS: Currently will NOT work properly with strings that contain
 * 0-terminator character inside. See IGNIETFERRO-1641 for details.
 *
 * @return                              true iff @c s1 ans @c s2 are case-insensitively equal.
 */
static inline bool AsciiEqualsIgnoreCase(const TStringBuf s1, const TStringBuf s2) noexcept {
    if (s1.size() != s2.size()) {
        return false;
    }
    if (s1.empty()) {
        return true;
    }
    return ::strnicmp(s1.data(), s2.data(), s1.size()) == 0;
}

/**
 * ASCII case-insensitive string comparison (for proper UTF8 strings
 * case-insensitive comparison consider using @c library/cpp/charset).
 *
 * BUGS: Currently will NOT work properly with strings that contain
 * 0-terminator character inside. See IGNIETFERRO-1641 for details.
 *
 * @return                              0 if strings are equal, negative if @c s1 < @c s2
 *                                      and positive otherwise.
 *                                      (same value as @c stricmp does).
 */
static inline int AsciiCompareIgnoreCase(const char* s1, const char* s2) noexcept {
    return ::stricmp(s1, s2);
}

/**
 * ASCII case-insensitive string comparison (for proper UTF8 strings
 * case-insensitive comparison consider using @c library/cpp/charset).
 *
 * BUGS: Currently will NOT work properly with strings that contain
 * 0-terminator character inside. See IGNIETFERRO-1641 for details.
 *
 * @return
 * - zero if strings are equal
 * - negative if @c s1 < @c s2
 * - positive otherwise,
 * similar to stricmp.
 */
Y_PURE_FUNCTION int AsciiCompareIgnoreCase(const TStringBuf s1, const TStringBuf s2) noexcept;

/**
 * ASCII case-sensitive string comparison (for proper UTF8 strings
 * case-sensitive comparison consider using @c library/cpp/charset).
 *
 * BUGS: Currently will NOT work properly with strings that contain
 * 0-terminator character inside. See IGNIETFERRO-1641 for details.
 *
 * @return                              true iff @c s2 are case-sensitively prefix of @c s1.
 */
static inline bool AsciiHasPrefix(const TStringBuf s1, const TStringBuf s2) noexcept {
    return (s1.size() >= s2.size()) && memcmp(s1.data(), s2.data(), s2.size()) == 0;
}

/**
 * ASCII case-insensitive string comparison (for proper UTF8 strings
 * case-insensitive comparison consider using @c library/cpp/charset).
 *
 * @return                              true iff @c s2 are case-insensitively prefix of @c s1.
 */
static inline bool AsciiHasPrefixIgnoreCase(const TStringBuf s1, const TStringBuf s2) noexcept {
    return (s1.size() >= s2.size()) && ::strnicmp(s1.data(), s2.data(), s2.size()) == 0;
}

/**
 * ASCII case-insensitive string comparison (for proper UTF8 strings
 * case-insensitive comparison consider using @c library/cpp/charset).
 *
 * @return                              true iff @c s2 are case-insensitively suffix of @c s1.
 */
static inline bool AsciiHasSuffixIgnoreCase(const TStringBuf s1, const TStringBuf s2) noexcept {
    return (s1.size() >= s2.size()) && ::strnicmp((s1.data() + (s1.size() - s2.size())), s2.data(), s2.size()) == 0;
}
