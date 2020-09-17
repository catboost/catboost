#include "chartraits.h"

#include <util/string/strspn.h>

Y_PURE_FUNCTION static inline const char* FindChr(const char* s, char c, size_t len) noexcept {
    const char* ret = TCharTraits<char>::Find(s, c, len);

    return ret ? ret : (s + len);
}

Y_PURE_FUNCTION static inline const char* FindTwo(const char* s, const char* c, size_t len) noexcept {
    const char* e = s + len;

    while (s != e && *s != c[0] && *s != c[1]) {
        ++s;
    }

    return s;
}

Y_PURE_FUNCTION const char* FastFindFirstOf(const char* s, size_t len, const char* set, size_t setlen) {
    switch (setlen) {
        case 0:
            return s + len;

        case 1:
            return FindChr(s, *set, len);

        case 2:
            return FindTwo(s, set, len);

        default:
            return TCompactStrSpn(set, set + setlen).FindFirstOf(s, s + len);
    }
}

Y_PURE_FUNCTION const char* FastFindFirstNotOf(const char* s, size_t len, const char* set, size_t setlen) {
    return TCompactStrSpn(set, set + setlen).FindFirstNotOf(s, s + len);
}
