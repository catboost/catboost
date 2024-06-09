#pragma once

#include <contrib/libs/libc_compat/string.h>

#include <util/string/ascii.h>

#include <string>

namespace NPrivate {
    template <typename TImpl>
    struct TCommonCaseInsensitiveCharTraits : private std::char_traits<char> {
        static bool eq(char c1, char c2) {
            return TImpl::ToCommonCase(c1) == TImpl::ToCommonCase(c2);
        }

        static bool lt(char c1, char c2) {
            return TImpl::ToCommonCase(c1) < TImpl::ToCommonCase(c2);
        }

        static const char* find(const char* s, std::size_t n, char a);

        using std::char_traits<char>::assign;
        using std::char_traits<char>::char_type;
        using std::char_traits<char>::copy;
        using std::char_traits<char>::length;
        using std::char_traits<char>::move;
    };
} // namespace NPrivate

struct TCaseInsensitiveCharTraits : public ::NPrivate::TCommonCaseInsensitiveCharTraits<TCaseInsensitiveCharTraits> {
    static int compare(const char* s1, const char* s2, std::size_t n);

private:
    friend ::NPrivate::TCommonCaseInsensitiveCharTraits<TCaseInsensitiveCharTraits>;

    // XXX return unsigned char. Current impl depends on char signedness, and if char is signed,
    // TCaseInsensitiveCharTraits::compare returns different result from std::char_traits<char>::compare for non-ascii strings.
    static char ToCommonCase(char ch) {
        return std::toupper((unsigned char)ch);
    }
};

struct TCaseInsensitiveAsciiCharTraits : public ::NPrivate::TCommonCaseInsensitiveCharTraits<TCaseInsensitiveCharTraits> {
    // WARN: does not work with null bytes (`compare("ab\0c", "ab\0d", 4)` returns 0).
    static int compare(const char* s1, const char* s2, std::size_t n) {
        return ::strncasecmp(s1, s2, n);
    }

private:
    friend ::NPrivate::TCommonCaseInsensitiveCharTraits<TCaseInsensitiveAsciiCharTraits>;

    static unsigned char ToCommonCase(char ch) {
        return AsciiToLower(ch);
    }
};
