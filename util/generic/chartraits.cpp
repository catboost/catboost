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

void TMutableCharTraits<wchar16>::Reverse(wchar16* start, size_t len) {
    if (!len) {
        return;
    }
    const wchar16* end = start + len;
    TArrayHolder<wchar16> temp_buffer(new wchar16[len]);
    wchar16* rbegin = temp_buffer.Get() + len;
    for (wchar16* p = start; p < end;) {
        const size_t symbol_size = W16SymbolSize(p, end);
        rbegin -= symbol_size;
        ::MemCopy(rbegin, p, symbol_size);
        p += symbol_size;
    }
    ::MemCopy(start, temp_buffer.Get(), len);
}
