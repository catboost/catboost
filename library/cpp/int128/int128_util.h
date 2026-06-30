#pragma once

#include <util/generic/bitops.h>
#include <limits>

namespace NPrivateInt128 {
    // will be moved to util/ later
    template <typename T>
    constexpr unsigned CountLeadingZeroBits(const T value) {
        if (value == 0) {
            return std::numeric_limits<std::make_unsigned_t<T>>::digits;
        }
        return std::numeric_limits<std::make_unsigned_t<T>>::digits - GetValueBitCount(value);
    }
}
