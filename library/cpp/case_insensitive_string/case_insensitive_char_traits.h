#pragma once

#include <contrib/libs/libc_compat/string.h>

#include <string>

struct TCaseInsensitiveCharTraits : private std::char_traits<char> {
    static bool eq(char c1, char c2) {
        return to_upper(c1) == to_upper(c2);
    }

    static bool lt(char c1, char c2) {
        return to_upper(c1) < to_upper(c2);
    }

    static int compare(const char* s1, const char* s2, std::size_t n);

    static const char* find(const char* s, std::size_t n, char a);

    using std::char_traits<char>::assign;
    using std::char_traits<char>::char_type;
    using std::char_traits<char>::copy;
    using std::char_traits<char>::length;
    using std::char_traits<char>::move;

private:
    static char to_upper(char ch) {
        return std::toupper((unsigned char)ch);
    }
};
