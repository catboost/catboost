#pragma once

#include <util/system/types.h>

/**
 * Dot product (Inner product or scalar product) implementation using SSE when possible.
 */
i32 DotProduct(const i8* lhs, const i8* rhs, int length) noexcept;
i64 DotProduct(const i32* lhs, const i32* rhs, int length) noexcept;
float DotProduct(const float* lhs, const float* rhs, int length) noexcept;
double DotProduct(const double* lhs, const double* rhs, int length) noexcept;

/**
 * Dot product to itself
 */
float L2NormSquared(const float* v, int length) noexcept;

/**
 * Dot product implementation without SSE optimizations.
 */
i32 DotProductSlow(const i8* lhs, const i8* rhs, int length) noexcept;
i64 DotProductSlow(const i32* lhs, const i32* rhs, int length) noexcept;
float DotProductSlow(const float* lhs, const float* rhs, int length) noexcept;
double DotProductSlow(const double* lhs, const double* rhs, int length) noexcept;

namespace NDotProduct {
    // Simpler wrapper allowing to use this functions as template argument.
    template <typename T>
    struct TDotProduct {
        using TResult = decltype(DotProduct(static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0));
        inline TResult operator()(const T* l, const T* r, int length) const {
            return DotProduct(l, r, length);
        }
    };
}
