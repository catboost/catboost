#pragma once

#include <util/system/defaults.h>

#include <algorithm>
#include <functional>
#include <utility>

template <class T>
inline bool IsZero(const T* begin, const T* end) {
    return std::find_if(begin, end, [](const T& other) { return other != T(0); }) == end;
}

template <size_t Size>
inline bool IsZero(const char* p) {
    size_t sizeInUI64 = Size / 8;
    const char* pEndUi64 = p + sizeInUI64 * 8;
    if (sizeInUI64 && !IsZero<ui64>((const ui64*)p, (const ui64*)pEndUi64)) {
        return false;
    }
    return IsZero(pEndUi64, p + Size);
}

#define IS_ZERO_INTSZ(INT)                           \
    template <>                                      \
    inline bool IsZero<sizeof(INT)>(const char* p) { \
        return (*(INT*)p) == INT(0);                 \
    }

IS_ZERO_INTSZ(ui8)
IS_ZERO_INTSZ(ui16)
IS_ZERO_INTSZ(ui32)
IS_ZERO_INTSZ(ui64)

#undef IS_ZERO_INTSZ

// If you want to use this to check all fields in a struct make sure it's w/o holes or #pragma pack(1)
template <class T>
bool IsZero(const T& t) {
    return IsZero<sizeof(T)>((const char*)&t);
}
