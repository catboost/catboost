#pragma once

#include <util/system/defaults.h>
#include <util/system/yassert.h>

namespace NPrivate {
    constexpr double ToRandReal1(const ui32 x) noexcept {
        return x * (double)(1.0 / 4294967295.0);
    }

    constexpr double ToRandReal2(const ui32 x) noexcept {
        return x * (double)(1.0 / 4294967296.0);
    }

    constexpr double ToRandReal3(const ui32 x) noexcept {
        return ((double)x + 0.5) * (double)(1.0 / 4294967296.0);
    }

    constexpr double ToRandReal1(const ui64 x) noexcept {
        return (x >> 11) * (double)(1.0 / 9007199254740991.0);
    }

    constexpr double ToRandReal2(const ui64 x) noexcept {
        return (x >> 11) * (double)(1.0 / 9007199254740992.0);
    }

    constexpr double ToRandReal3(const ui64 x) noexcept {
        return ((x >> 12) + 0.5) * (double)(1.0 / 4503599627370496.0);
    }

    constexpr double ToRandReal4(const ui64 x) noexcept {
        return double(x * (double)(1.0 / 18446744073709551616.0L));
    }

    template <class T>
    static inline ui64 ToRand64(T&& rng, ui32 x) noexcept {
        return ((ui64)x) | (((ui64)rng.GenRand()) << 32);
    }

    template <class T>
    static constexpr ui64 ToRand64(T&&, ui64 x) noexcept {
        return x;
    }

    /*
     * return value in range [0, max) from any generator
     */
    template <class T, class TRandGen>
    static T GenUniform(T max, TRandGen&& gen) {
        Y_ABORT_UNLESS(max > 0, "Invalid random number range [0, 0)");

        const T randmax = gen.RandMax() - gen.RandMax() % max;
        T rand;

        while ((rand = gen.GenRand()) >= randmax) {
            /* no-op */
        }

        return rand % max;
    }
} // namespace NPrivate

template <class TRandType, class T>
struct TCommonRNG {
    using TResult = TRandType;
    using result_type = TRandType;

    inline T& Engine() noexcept {
        return static_cast<T&>(*this);
    }

    static constexpr TResult _Min = TResult(0);
    static constexpr TResult _Max = TResult(-1);

    static constexpr TResult RandMax() noexcept {
        return _Max;
    }

    static constexpr TResult RandMin() noexcept {
        return _Min;
    }

    /* generates uniformly distributed random number on [0, t) interval */
    inline TResult Uniform(TResult t) noexcept {
        return ::NPrivate::GenUniform(t, Engine());
    }

    /* generates uniformly distributed random number on [f, t) interval */
    inline TResult Uniform(TResult f, TResult t) noexcept {
        return f + Uniform(t - f);
    }

    /* generates 64-bit random number for current(may be 32 bit) rng */
    inline ui64 GenRand64() noexcept {
        return ::NPrivate::ToRand64(Engine(), Engine().GenRand());
    }

    /* generates a random number on [0, 1]-real-interval */
    inline double GenRandReal1() noexcept {
        return ::NPrivate::ToRandReal1(Engine().GenRand());
    }

    /* generates a random number on [0, 1)-real-interval */
    inline double GenRandReal2() noexcept {
        return ::NPrivate::ToRandReal2(Engine().GenRand());
    }

    /* generates a random number on (0, 1)-real-interval */
    inline double GenRandReal3() noexcept {
        return ::NPrivate::ToRandReal3(Engine().GenRand());
    }

    /* generates a random number on [0, 1) with 53-bit resolution */
    inline double GenRandReal4() noexcept {
        return ::NPrivate::ToRandReal4(Engine().GenRand64());
    }

    // compatibility stuff
    inline TResult operator()() noexcept {
        return Engine().GenRand();
    }

    static constexpr TResult max() noexcept {
        return T::RandMax();
    }

    static constexpr TResult min() noexcept {
        return T::RandMin();
    }
};
