#include "case_insensitive_char_traits.h"

#include <util/string/escape.h>

template <typename TImpl>
const char* ::NPrivate::TCommonCaseInsensitiveCharTraits<TImpl>::find(const char* s, std::size_t n, char a) {
    const auto ca(TImpl::ToCommonCase(a));
    while (n-- != 0) {
        if (TImpl::ToCommonCase(*s) == ca)
            return s;
        s++;
    }
    return nullptr;
}

int TCaseInsensitiveCharTraits::compare(const char* s1, const char* s2, std::size_t n) {
    while (n-- != 0) {
        auto c1 = ToCommonCase(*s1), c2 = ToCommonCase(*s2);
        if (c1 < c2) {
            return -1;
        }
        if (c1 > c2) {
            return 1;
        }
        ++s1;
        ++s2;
    }
    return 0;
}

template struct ::NPrivate::TCommonCaseInsensitiveCharTraits<TCaseInsensitiveCharTraits>;
template struct ::NPrivate::TCommonCaseInsensitiveCharTraits<TCaseInsensitiveAsciiCharTraits>;
