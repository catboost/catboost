#pragma once

#include <util/system/types.h>
#include <util/system/compiler.h>

/**
 * Dot product (Inner product or scalar product) implementation using SSE when possible.
 */
Y_PURE_FUNCTION
i32 DotProduct(const i8* lhs, const i8* rhs, ui32 length) noexcept;

Y_PURE_FUNCTION
ui32 DotProduct(const ui8* lhs, const ui8* rhs, ui32 length) noexcept;

Y_PURE_FUNCTION
i64 DotProduct(const i32* lhs, const i32* rhs, ui32 length) noexcept;

Y_PURE_FUNCTION
float DotProduct(const float* lhs, const float* rhs, ui32 length) noexcept;

Y_PURE_FUNCTION
double DotProduct(const double* lhs, const double* rhs, ui32 length) noexcept;

/**
 * Dot product to itself
 */
Y_PURE_FUNCTION
float L2NormSquared(const float* v, ui32 length) noexcept;

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
TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, ui32 length, unsigned mask) noexcept;

/**
 * For two vectors L and R computes 3 dot-products: L·L, L·R, R·R
 */
Y_PURE_FUNCTION
static inline TTriWayDotProduct<float> TriWayDotProduct(const float* lhs, const float* rhs, ui32 length, ETriWayDotProductComputeMask mask = ETriWayDotProductComputeMask::All) noexcept {
    return TriWayDotProduct(lhs, rhs, length, static_cast<unsigned>(mask));
}

/**
 * Dot product implementation without SSE optimizations.
 */
Y_PURE_FUNCTION
ui32 DotProductSlow(const ui8* lhs, const ui8* rhs, ui32 length) noexcept;

Y_PURE_FUNCTION
i32 DotProductSlow(const i8* lhs, const i8* rhs, ui32 length) noexcept;

Y_PURE_FUNCTION
i64 DotProductSlow(const i32* lhs, const i32* rhs, ui32 length) noexcept;

Y_PURE_FUNCTION
float DotProductSlow(const float* lhs, const float* rhs, ui32 length) noexcept;

Y_PURE_FUNCTION
double DotProductSlow(const double* lhs, const double* rhs, ui32 length) noexcept;

namespace NDotProduct {
    // Simpler wrapper allowing to use this functions as template argument.
    template <typename T>
    struct TDotProduct {
        using TResult = decltype(DotProduct(static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0));
        Y_PURE_FUNCTION
        inline TResult operator()(const T* l, const T* r, ui32 length) const {
            return DotProduct(l, r, length);
        }
    };
}
