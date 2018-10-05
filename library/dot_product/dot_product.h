#pragma once

#include <util/system/types.h>
#include <util/system/compiler.h>

/**
 * Dot product (Inner product or scalar product) implementation using SSE when possible.
 */
Y_PURE_FUNCTION
i32 DotProduct(const i8* lhs, const i8* rhs, ui32 length) noexcept;

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

/**
 * Dot product implementation without SSE optimizations.
 */
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
