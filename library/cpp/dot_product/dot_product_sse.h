#pragma once

#include <util/system/types.h>
#include <util/system/compiler.h>

Y_PURE_FUNCTION
i32 DotProductSse(const i8* lhs, const i8* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
ui32 DotProductSse(const ui8* lhs, const ui8* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
i64 DotProductSse(const i32* lhs, const i32* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
float DotProductSse(const float* lhs, const float* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
double DotProductSse(const double* lhs, const double* rhs, size_t length) noexcept;
