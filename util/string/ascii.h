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

    // clang-format off
    inline constexpr unsigned char ASCII_CLASS[256] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x68, 0x68, 0x68, 0x68, 0x68, 0x68, 0x68, 0x68, 0x68, 0x68, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x72, 0x72, 0x72, 0x72, 0x72, 0x72, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32,
        0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x32, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x74, 0x74, 0x74, 0x74, 0x74, 0x74, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34,
        0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x80, 0x80, 0x80, 0x80, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    inline constexpr unsigned char ASCII_LOWER[256] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
        208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    };
    // clang-format on

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
    constexpr bool RangeOk(T c) noexcept {
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
} // namespace NPrivate

constexpr bool IsAscii(const int c) noexcept {
    return !(c & ~0x7f);
}

constexpr bool IsAsciiSpace(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_SPACE;
}

constexpr bool IsAsciiUpper(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_UPPER;
}

constexpr bool IsAsciiLower(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_LOWER;
}

constexpr bool IsAsciiDigit(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_DIGIT;
}

constexpr bool IsAsciiAlpha(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_ALPHA;
}

constexpr bool IsAsciiAlnum(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_ALNUM;
}

constexpr bool IsAsciiHex(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_ISHEX;
}

constexpr bool IsAsciiPunct(unsigned char c) {
    return ::NPrivate::ASCII_CLASS[c] & ::NPrivate::CC_PUNCT;
}

// some overloads

template <class T>
constexpr bool IsAsciiSpace(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiSpace(static_cast<unsigned char>(c));
}

template <class T>
constexpr bool IsAsciiUpper(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiUpper(static_cast<unsigned char>(c));
}

template <class T>
constexpr bool IsAsciiLower(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiLower(static_cast<unsigned char>(c));
}

template <class T>
constexpr bool IsAsciiDigit(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiDigit(static_cast<unsigned char>(c));
}

template <class T>
constexpr bool IsAsciiAlpha(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiAlpha(static_cast<unsigned char>(c));
}

template <class T>
constexpr bool IsAsciiAlnum(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiAlnum(static_cast<unsigned char>(c));
}

template <class T>
constexpr bool IsAsciiHex(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiHex(static_cast<unsigned char>(c));
}

template <class T>
constexpr bool IsAsciiPunct(T c) {
    return ::NPrivate::RangeOk(c) && IsAsciiPunct(static_cast<unsigned char>(c));
}

// some extra helpers
constexpr ui8 AsciiToLower(ui8 c) noexcept {
    return ::NPrivate::ASCII_LOWER[c];
}

constexpr char AsciiToLower(char c) noexcept {
    return (char)AsciiToLower((ui8)c);
}

template <class T>
constexpr ::NPrivate::TDereferenced<T> AsciiToLower(T c) noexcept {
    return (c >= 0 && c <= 127) ? (::NPrivate::TDereferenced<T>)AsciiToLower((ui8)c) : c;
}

template <class T>
constexpr ::NPrivate::TDereferenced<T> AsciiToUpper(T c) noexcept {
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
