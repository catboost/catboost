#pragma once

#include <util/system/types.h>
#include <util/system/compiler.h>

Y_PURE_FUNCTION
i32 DotProductAvx2(const i8* lhs, const i8* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
ui32 DotProductAvx2(const ui8* lhs, const ui8* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
i64 DotProductAvx2(const i32* lhs, const i32* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
float DotProductAvx2(const float* lhs, const float* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
double DotProductAvx2(const double* lhs, const double* rhs, size_t length) noexcept;
