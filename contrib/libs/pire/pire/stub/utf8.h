#pragma once

#include <library/cpp/charset/codepage.h>
#include <util/charset/unidata.h>

inline wchar32 to_lower(wchar32 c) {
    return ToLower(c);
}
inline wchar32 to_upper(wchar32 c) {
    return ToUpper(c);
}

inline bool is_digit(wchar32 c) {
    return IsDigit(c);
}

inline bool is_upper(wchar32 c) {
    return IsUpper(c);
}
