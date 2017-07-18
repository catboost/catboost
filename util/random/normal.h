#pragma once

#include <cmath>

// sometimes we need stateless normal distribution...

/*
 * normal distribution with Box-Muller transform
 * http://www.design.caltech.edu/erik/Misc/Gaussian.html
 */
template <typename T, typename TRng>
static inline T StdNormalDistribution(TRng&& rng) noexcept {
    T x;
    T y;
    T r;

    do {
        x = (T)rng.GenRandReal1() * T(2) - T(1);
        y = (T)rng.GenRandReal1() * T(2) - T(1);
        r = x * x + y * y;
    } while (r > T(1) || r <= T(0));

    return x * std::sqrt(-T(2) * std::log(r) / r);
}

template <typename T, typename TRng>
static inline T NormalDistribution(TRng&& rng, T m, T d) noexcept {
    return StdNormalDistribution<T>(rng) * d + m;
}

// specialized for float, double, long double
template <class T>
T StdNormalRandom() noexcept;

template <class T>
static inline T NormalRandom(T m, T d) noexcept {
    return StdNormalRandom<T>() * d + m;
}
