#pragma once

#include <util/system/compiler.h>
#include <util/system/types.h>

#include <numeric>

/**
 * Dot product implementation without SSE optimizations.
 */
Y_PURE_FUNCTION
inline ui32 DotProductSimple(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    return std::inner_product(lhs, lhs + length, rhs, static_cast<ui32>(0u),
                              [](ui32 x1, ui16 x2) {return x1 + x2;},
                              [](ui16 x1, ui8 x2) {return x1 * x2;});
}

Y_PURE_FUNCTION
inline i32 DotProductSimple(const i8* lhs, const i8* rhs, size_t length) noexcept {
    return std::inner_product(lhs, lhs + length, rhs, static_cast<i32>(0),
                              [](i32 x1, i16 x2) {return x1 + x2;},
                              [](i16 x1, i8 x2) {return x1 * x2;});
}

Y_PURE_FUNCTION
inline i64 DotProductSimple(const i32* lhs, const i32* rhs, size_t length) noexcept {
    return std::inner_product(lhs, lhs + length, rhs, static_cast<i64>(0),
                              [](i64 x1, i64 x2) {return x1 + x2;},
                              [](i64 x1, i32 x2) {return x1 * x2;});
}

Y_PURE_FUNCTION
float DotProductSimple(const float* lhs, const float* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
double DotProductSimple(const double* lhs, const double* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
ui32 DotProductUI4Simple(const ui8* lhs, const ui8* rhs, size_t lengtInBytes) noexcept;

