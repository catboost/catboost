#pragma once

#include "typetraits.h"

#include <util/system/compiler.h>

#include <cstring>

template <class T>
static constexpr const T& Min(const T& l Y_LIFETIME_BOUND, const T& r Y_LIFETIME_BOUND) {
    return r < l ? r : l;
}

template <typename T, typename... Args>
static constexpr const T& Min(const T& a Y_LIFETIME_BOUND, const T& b Y_LIFETIME_BOUND, const Args&... args Y_LIFETIME_BOUND) {
    return Min(a, Min(b, args...));
}

template <class T>
static constexpr const T& Max(const T& l Y_LIFETIME_BOUND, const T& r Y_LIFETIME_BOUND) {
    return l < r ? r : l;
}

template <typename T, typename... Args>
static constexpr const T& Max(const T& a Y_LIFETIME_BOUND, const T& b Y_LIFETIME_BOUND, const Args&... args Y_LIFETIME_BOUND) {
    return Max(a, Max(b, args...));
}

// replace with http://en.cppreference.com/w/cpp/algorithm/clamp in c++17
template <class T>
constexpr const T& ClampVal(const T& val Y_LIFETIME_BOUND, const T& min Y_LIFETIME_BOUND, const T& max Y_LIFETIME_BOUND) {
    return val < min ? min : (max < val ? max : val);
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

/**
 * Securely zero memory (compiler does not optimize this out)
 *
 * @param pointer   void pointer to start of memory block to be zeroed
 * @param count     size of memory block to be zeroed (in bytes)
 */
void SecureZero(void* pointer, size_t count) noexcept;

/**
 * Securely zero memory of given object (compiler does not optimize this out)
 *
 * @param t     reference to object, which must be zeroed
 */
template <class T>
static inline void SecureZero(T& t) noexcept {
    SecureZero((void*)&t, sizeof(t));
}

namespace NSwapCheck {
    Y_HAS_MEMBER(swap);
    Y_HAS_MEMBER(Swap);

    template <class T, class = void>
    struct TSwapSelector {
        static inline void Swap(T& l, T& r) noexcept(std::is_nothrow_move_constructible<T>::value&&
                                                         std::is_nothrow_move_assignable<T>::value) {
            T tmp(std::move(l));
            l = std::move(r);
            r = std::move(tmp);
        }
    };

    template <class T>
    struct TSwapSelector<T, std::enable_if_t<THasSwap<T>::value>> {
        static inline void Swap(T& l, T& r) noexcept(noexcept(l.Swap(r))) {
            l.Swap(r);
        }
    };

    template <class T>
    struct TSwapSelector<T, std::enable_if_t<THasswap<T>::value && !THasSwap<T>::value>> {
        static inline void Swap(T& l, T& r) noexcept(noexcept(l.swap(r))) {
            l.swap(r);
        }
    };
}

/*
 * DoSwap better than ::Swap in member functions...
 */
template <class T>
static inline void DoSwap(T& l, T& r) noexcept(noexcept(NSwapCheck::TSwapSelector<T>::Swap(l, r))) {
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

/*
 * Class for zero-initialize padding bytes in derived classes
 */
template <typename TDerived>
class TZeroInit {
protected:
    TZeroInit() {
        // Actually, safe because this as TDerived is not initialized yet.
        Zero(*static_cast<TDerived*>(this));
    }
};

struct TIdentity {
    template <class T>
    constexpr decltype(auto) operator()(T&& x) const noexcept {
        return std::forward<T>(x);
    }
};
