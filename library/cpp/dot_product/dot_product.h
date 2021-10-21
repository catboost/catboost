#pragma once

#include <util/system/types.h>
#include <util/system/compiler.h>

#include <numeric>

/**
 * Dot product (Inner product or scalar product) implementation using SSE when possible.
 */
namespace NDotProductImpl {
    extern i32 (*DotProductI8Impl)(const i8* lhs, const i8* rhs, size_t length) noexcept;
    extern ui32 (*DotProductUi8Impl)(const ui8* lhs, const ui8* rhs, size_t length) noexcept;
    extern i64 (*DotProductI32Impl)(const i32* lhs, const i32* rhs, size_t length) noexcept;
    extern float (*DotProductFloatImpl)(const float* lhs, const float* rhs, size_t length) noexcept;
    extern double (*DotProductDoubleImpl)(const double* lhs, const double* rhs, size_t length) noexcept;
}

Y_PURE_FUNCTION
inline i32 DotProduct(const i8* lhs, const i8* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductI8Impl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline ui32 DotProduct(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductUi8Impl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline i64 DotProduct(const i32* lhs, const i32* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductI32Impl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline float DotProduct(const float* lhs, const float* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductFloatImpl(lhs, rhs, length);
}

Y_PURE_FUNCTION
inline double DotProduct(const double* lhs, const double* rhs, size_t length) noexcept {
    return NDotProductImpl::DotProductDoubleImpl(lhs, rhs, length);
}

/**
 * Dot product to itself
 */
Y_PURE_FUNCTION
float L2NormSquared(const float* v, size_t length) noexcept;

// TODO(yazevnul): make `L2NormSquared` for double, this should be faster than `DotProduct`
// where `lhs == rhs` because it will save N load instructions.

template <typename T>
struct TTriWayDotProduct {
    T LL = 1;
    T LR = 0;
    T RR = 1;
};

enum class ETriWayDotProductComputeMask: unsigned {
    // basic
    LL = 0b100,
    LR = 0b010,
    RR = 0b001,

    // useful combinations
    All = 0b111,
    Left = 0b110, // skip computation of R·R
    Right = 0b011, // skip computation of L·L
};

Y_PURE_FUNCTION
TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, size_t length, unsigned mask) noexcept;

/**
 * For two vectors L and R computes 3 dot-products: L·L, L·R, R·R
 */
Y_PURE_FUNCTION
static inline TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, size_t length, ETriWayDotProductComputeMask mask = ETriWayDotProductComputeMask::All) noexcept {
    return TriWayDotProduct(lhs, rhs, length, static_cast<unsigned>(mask));
}

namespace NDotProduct {
    // Simpler wrapper allowing to use this functions as template argument.
    template <typename T>
    struct TDotProduct {
        using TResult = decltype(DotProduct(static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0));
        Y_PURE_FUNCTION
        inline TResult operator()(const T* l, const T* r, size_t length) const {
            return DotProduct(l, r, length);
        }
    };

    void DisableAvx2();
}

