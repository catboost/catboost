#pragma once

#include <limits>

#if defined(max) || defined(min)
    #error "stop defining 'min' and 'max' macros, evil people"
#endif

template <class T>
static constexpr T Max() noexcept {
    return std::numeric_limits<T>::max();
}

template <class T>
static constexpr T Min() noexcept {
    return std::numeric_limits<T>::min();
}

namespace NPrivate {
    struct TMax {
        template <class T>
        constexpr operator T() const {
            return Max<T>();
        }
    };

    struct TMin {
        template <class T>
        constexpr operator T() const {
            return Min<T>();
        }
    };
}

static constexpr ::NPrivate::TMax Max() noexcept {
    return {};
}

static constexpr ::NPrivate::TMin Min() noexcept {
    return {};
}

namespace NPrivate {
    template <unsigned long long N>
    static constexpr double MaxFloorValue() {
        return N;
    }
    template <unsigned long long N>
    static constexpr double MaxCeilValue() {
        return N;
    }
    template <>
    constexpr double MaxFloorValue<0x7FFF'FFFF'FFFF'FFFFull>() {
        return 9223372036854774784.0; // 0x7FFFFFFFFFFFFC00p0
    }
    template <>
    constexpr double MaxCeilValue<0x7FFF'FFFF'FFFF'FFFFull>() {
        return 9223372036854775808.0; // 0x8000000000000000p0
    }
    template <>
    constexpr double MaxFloorValue<0xFFFF'FFFF'FFFF'FFFFull>() {
        return 18446744073709549568.0; // 0xFFFFFFFFFFFFF800p0
    }
    template <>
    constexpr double MaxCeilValue<0xFFFF'FFFF'FFFF'FFFFull>() {
        return 18446744073709551616.0; // 0x10000000000000000p0
    }
}

// MaxFloor<T> is the greatest double within the range of T.
//
// 1. If Max<T> is an exact double, MaxFloor<T> = Max<T> = MaxCeil<T>.
//    In this case some doubles above MaxFloor<T> cast to T may round
//    to Max<T> depending on the rounding mode.
//
// 2. Otherwise Max<T> is between MaxFloor<T> and MaxCeil<T>, and
//    MaxFloor<T> is the largest double that does not overflow T.
template <class T>
static constexpr double MaxFloor() noexcept {
    return ::NPrivate::MaxFloorValue<Max<T>()>();
}

// MaxCeil<T> is the smallest double not lesser than Max<T>.
//
// 1. If Max<T> is an exact double, MaxCeil<T> = Max<T> = MaxFloor<T>.
//
// 2. Otherwise Max<T> is between MaxFloor<T> and MaxCeil<T>, and
//    MaxCeil<T> is the smallest double that overflows T.
template <class T>
static constexpr double MaxCeil() noexcept {
    return ::NPrivate::MaxCeilValue<Max<T>()>();
}
