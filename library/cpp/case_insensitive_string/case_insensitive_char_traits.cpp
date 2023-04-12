#include "case_insensitive_char_traits.h"
#include "case_insensitive_string.h"

#include <util/string/escape.h>

int TCaseInsensitiveCharTraits::compare(const char* s1, const char* s2, std::size_t n) {
    while (n-- != 0) {
        if (to_upper(*s1) < to_upper(*s2)) {
            return -1;
        }
        if (to_upper(*s1) > to_upper(*s2)) {
            return 1;
        }
        ++s1;
        ++s2;
    }
    return 0;
}

const char* TCaseInsensitiveCharTraits::find(const char* s, std::size_t n, char a) {
    auto const ua(to_upper(a));
    while (n-- != 0) {
        if (to_upper(*s) == ua)
            return s;
        s++;
    }
    return nullptr;
}

TCaseInsensitiveString EscapeC(const TCaseInsensitiveString& str) {
    const auto result = EscapeC(str.data(), str.size());
    return {result.data(), result.size()};
}

