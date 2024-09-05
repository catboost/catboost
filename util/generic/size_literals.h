#pragma once

#include "yexception.h"
#include <util/system/types.h>
#include <limits>

// Unsigned literals

constexpr ui64 operator"" _B(unsigned long long value) noexcept {
    return value;
}

constexpr ui64 operator"" _KB(unsigned long long value) noexcept {
    return value * 1024;
}

constexpr ui64 operator"" _MB(unsigned long long value) noexcept {
    return value * 1024_KB;
}

constexpr ui64 operator"" _GB(unsigned long long value) noexcept {
    return value * 1024_MB;
}

constexpr ui64 operator"" _TB(unsigned long long value) noexcept {
    return value * 1024_GB;
}

constexpr ui64 operator"" _PB(unsigned long long value) noexcept {
    return value * 1024_TB;
}

constexpr ui64 operator"" _EB(unsigned long long value) noexcept {
    return value * 1024_PB;
}

// Signed literals

namespace NPrivate {
    constexpr i64 SignedCast(ui64 value) {
        return value <= static_cast<ui64>(std::numeric_limits<i64>::max())
                   ? static_cast<i64>(value)
                   : ythrow yexception() << "The resulting value " << value << " does not fit into the i64 type";
    }
} // namespace NPrivate

constexpr i64 operator"" _KBs(const unsigned long long value) noexcept {
    return ::NPrivate::SignedCast(value * 1024);
}

constexpr i64 operator"" _MBs(unsigned long long value) noexcept {
    return ::NPrivate::SignedCast(value * 1024_KB);
}

constexpr i64 operator"" _GBs(unsigned long long value) noexcept {
    return ::NPrivate::SignedCast(value * 1024_MB);
}

constexpr i64 operator"" _TBs(unsigned long long value) noexcept {
    return ::NPrivate::SignedCast(value * 1024_GB);
}

constexpr i64 operator"" _PBs(unsigned long long value) noexcept {
    return ::NPrivate::SignedCast(value * 1024_TB);
}

constexpr i64 operator"" _EBs(unsigned long long value) noexcept {
    return ::NPrivate::SignedCast(value * 1024_PB);
}
