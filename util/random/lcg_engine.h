#pragma once

#include <utility>
#include <util/generic/typetraits.h>

// common engine for lcg-based RNG's
// http://en.wikipedia.org/wiki/Linear_congruential_generator

namespace NPrivate {
    template <typename T>
    T LcgAdvance(T seed, T lcgBase, T lcgAddend, T delta) noexcept;
}

template <typename T, T A, T C>
struct TFastLcgIterator {
    static_assert(C % 2 == 1, "C must be odd");

    static constexpr T Iterate(T x) noexcept {
        return x * A + C;
    }

    static inline T IterateMultiple(T x, T delta) noexcept {
        return ::NPrivate::LcgAdvance(x, A, C, delta);
    }
};

template <typename T, T A>
struct TLcgIterator {
    inline TLcgIterator(T seq) noexcept
        : C((seq << 1u) | (T)1) // C must be odd
    {
    }

    inline T Iterate(T x) noexcept {
        return x * A + C;
    }

    inline T IterateMultiple(T x, T delta) noexcept {
        return ::NPrivate::LcgAdvance(x, A, C, delta);
    }

    const T C;
};

template <class TIterator, class TMixer>
struct TLcgRngBase: public TIterator, public TMixer {
    using TStateType = decltype(std::declval<TIterator>().Iterate(0));
    using TResultType = decltype(std::declval<TMixer>().Mix(TStateType()));

    template <typename... Args>
    inline TLcgRngBase(TStateType seed, Args&&... args)
        : TIterator(std::forward<Args>(args)...)
        , X(seed)
    {
    }

    inline TResultType GenRand() noexcept {
        return this->Mix(X = this->Iterate(X));
    }

    inline void Advance(TStateType delta) noexcept {
        X = this->IterateMultiple(X, delta);
    }

    TStateType X;
};
