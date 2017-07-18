#pragma once

#include "typetraits.h"

#include <cstring>

template <class T>
static constexpr const T& Min(const T& l, const T& r) {
    return l < r ? l : r;
}

template <typename T, typename... Args>
static constexpr const T& Min(const T& a, const T& b, const Args&... args) {
    return Min(a, Min(b, args...));
}

template <class T>
static constexpr const T& Max(const T& l, const T& r) {
    return l > r ? l : r;
}

template <typename T, typename... Args>
static constexpr const T& Max(const T& a, const T& b, const Args&... args) {
    return Max(a, Max(b, args...));
}

// replace with http://en.cppreference.com/w/cpp/algorithm/clamp in c++17
template <class T>
constexpr T ClampVal(const T& val, const T& min, const T& max) {
    return val < min ? min : (val > max ? max : val);
}

template <typename T = double, typename... Args>
static T Mean(const Args&... other) noexcept {
    const auto numArgs = sizeof...(other);

    auto sum = T();
    for (const auto& v : {other...}) {
        sum += v;
    }

    return sum / numArgs;
}

template <class T>
static inline void Zero(T& t) noexcept {
    memset((void*)&t, 0, sizeof(t));
}

namespace NSwapCheck {
    Y_HAS_MEMBER(swap);
    Y_HAS_MEMBER(Swap);

    template <class T>
    static inline void DoSwapImpl(T& l, T& r) {
        T tmp(std::move(l));

        l = std::move(r);
        r = std::move(tmp);
    }

    template <bool val>
    struct TSmallSwapSelector {
        template <class T>
        static inline void Swap(T& l, T& r) {
            DoSwapImpl(l, r);
        }
    };

    template <>
    struct TSmallSwapSelector<true> {
        template <class T>
        static inline void Swap(T& l, T& r) {
            l.swap(r);
        }
    };

    template <bool val>
    struct TBigSwapSelector {
        template <class T>
        static inline void Swap(T& l, T& r) {
            TSmallSwapSelector<THasswap<T>::Result>::Swap(l, r);
        }
    };

    template <>
    struct TBigSwapSelector<true> {
        template <class T>
        static inline void Swap(T& l, T& r) {
            l.Swap(r);
        }
    };

    template <class T>
    class TSwapSelector: public TBigSwapSelector<THasSwap<T>::Result> {
    };
}

/*
 * DoSwap better than ::Swap in member functions...
 */
template <class T>
static inline void DoSwap(T& l, T& r) {
    NSwapCheck::TSwapSelector<T>::Swap(l, r);
}

template <bool b>
struct TNullTmpl {
    template <class T>
    operator T() const {
        return (T)0;
    }
};

using TNull = TNullTmpl<0>;
