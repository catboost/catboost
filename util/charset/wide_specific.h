#pragma once

#include <util/system/types.h>
#include <util/system/yassert.h>

inline constexpr bool IsW16SurrogateLead(wchar16 c) noexcept {
    return 0xD800 <= c && c <= 0xDBFF;
}

inline constexpr bool IsW16SurrogateTail(wchar16 c) noexcept {
    return 0xDC00 <= c && c <= 0xDFFF;
}

inline size_t W16SymbolSize(const wchar16* begin, const wchar16* end) {
    Y_ASSERT(begin < end);

    if ((begin + 1 != end) && IsW16SurrogateLead(*begin) && IsW16SurrogateTail(*(begin + 1))) {
        return 2;
    }

    return 1;
}
