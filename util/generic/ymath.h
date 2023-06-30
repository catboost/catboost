#pragma once

#include <util/system/yassert.h>
#include <util/system/defaults.h>

#include <cmath>
#include <cfloat>
#include <cstdlib>

#include "typetraits.h"
#include "utility.h"

constexpr double PI = M_PI;
constexpr double M_LOG2_10 = 3.32192809488736234787; // log2(10)
constexpr double M_LN2_INV = M_LOG2E;                // 1 / ln(2) == log2(e)

/**
 * \returns                             Absolute value of the provided argument.
 */
template <class T>
constexpr T Abs(T value) {
    return std::abs(value);
}

/**
 * @returns                             Base 2 logarithm of the provided double
 *                                      precision floating point value.
 */
inline double Log2(double value) {
    return log(value) * M_LN2_INV;
}

/**
 * @returns                             Base 2 logarithm of the provided
 *                                      floating point value.
 */
inline float Log2(float value) {
    return logf(value) * static_cast<float>(M_LN2_INV);
}

/**
 * @returns                             Base 2 logarithm of the provided integral value.
 */
template <class T>
inline std::enable_if_t<std::is_integral<T>::value, double>
Log2(T value) {
    return Log2(static_cast<double>(value));
}

/** Returns 2^x */
double Exp2(double);
float Exp2f(float);

template <class T>
static constexpr T Sqr(const T t) noexcept {
    return t * t;
}

inline double Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline bool IsFinite(double f) {
#if defined(isfinite)
    return isfinite(f);
#elif defined(_win_)
    return _finite(f) != 0;
#elif defined(_darwin_)
    return isfinite(f);
#elif defined(__GNUC__)
    return __builtin_isfinite(f);
#elif defined(_STLP_VENDOR_STD)
    return _STLP_VENDOR_STD::isfinite(f);
#else
    return std::isfinite(f);
#endif
}

static inline bool IsNan(double f) {
#if defined(_win_)
    return _isnan(f) != 0;
#else
    return std::isnan(f);
#endif
}

inline bool IsValidFloat(double f) {
    return IsFinite(f) && !IsNan(f);
}

#ifdef _MSC_VER
double Erf(double x);
#else
inline double Erf(double x) {
    return erf(x);
}
#endif

/**
 * @returns                             Natural logarithm of the absolute value
 *                                      of the gamma function of provided argument.
 */
inline double LogGamma(double x) noexcept {
#if defined(_glibc_)
    int sign;

    (void)sign;

    return lgamma_r(x, &sign);
#elif defined(__GNUC__)
    return __builtin_lgamma(x);
#elif defined(_unix_)
    return lgamma(x);
#else
    extern double LogGammaImpl(double);
    return LogGammaImpl(x);
#endif
}

/**
 * @returns                             x^n for integer n, n >= 0.
 */
template <class T, class Int>
T Power(T x, Int n) {
    static_assert(std::is_integral<Int>::value, "only integer powers are supported");
    Y_ASSERT(n >= 0);
    if (n == 0) {
        return T(1);
    }
    while ((n & 1) == 0) {
        x = x * x;
        n >>= 1;
    }
    T result = x;
    n >>= 1;
    while (n > 0) {
        x = x * x;
        if (n & 1) {
            result = result * x;
        }
        n >>= 1;
    }
    return result;
}

/**
 * Compares two floating point values and returns true if they are considered equal.
 * The two numbers are compared in a relative way, where the exactness is stronger
 * the smaller the numbers are.
 *
 * Note that comparing values where either one is 0.0 will not work.
 * The solution to this is to compare against values greater than or equal to 1.0.
 *
 * @code
 * // Instead of comparing with 0.0
 * FuzzyEquals(0.0, 1.0e-200); // This will return false
 * // Compare adding 1 to both values will fix the problem
 * FuzzyEquals(1 + 0.0, 1 + 1.0e-200); // This will return true
 * @endcode
 */
inline bool FuzzyEquals(double p1, double p2, double eps = 1.0e-13) {
    return (Abs(p1 - p2) <= eps * Min(Abs(p1), Abs(p2)));
}

/**
 * @see FuzzyEquals(double, double, double)
 */
inline bool FuzzyEquals(float p1, float p2, float eps = 1.0e-6) {
    return (Abs(p1 - p2) <= eps * Min(Abs(p1), Abs(p2)));
}

namespace NUtilMathPrivate {
    template <bool IsSigned>
    struct TCeilDivImpl {};

    template <>
    struct TCeilDivImpl<true> {
        template <class T>
        static inline constexpr T Do(T x, T y) noexcept {
            return x / y + (((x < 0) ^ (y > 0)) && (x % y));
        }
    };

    template <>
    struct TCeilDivImpl<false> {
        template <class T>
        static inline constexpr T Do(T x, T y) noexcept {
            auto quot = x / y;
            return (x % y) ? (quot + 1) : quot;
        }
    };
}

/**
 * @returns Equivalent to ceil((double) x / (double) y) but using only integer arithmetic operations
 */
template <class T>
inline T
#if !defined(__NVCC__)
    constexpr
#endif
    CeilDiv(T x, T y) noexcept {
    static_assert(std::is_integral<T>::value, "Integral type required.");
    Y_ASSERT(y != 0);
    return ::NUtilMathPrivate::TCeilDivImpl<std::is_signed<T>::value>::Do(x, y);
}
