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
