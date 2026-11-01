#pragma once

#include <util/system/types.h>
#include <util/system/compiler.h>

#include <stddef.h>

Y_PURE_FUNCTION
i32 DotProductVnni(const i8* lhs, const i8* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
i32 DotProduct64Vnni(const i8* lhs, const i8* rhs) noexcept;

Y_PURE_FUNCTION
i32 DotProduct128Vnni(const i8* lhs, const i8* rhs) noexcept;

Y_PURE_FUNCTION
i32 DotProduct256Vnni(const i8* lhs, const i8* rhs) noexcept;

Y_PURE_FUNCTION
i32 DotProduct512Vnni(const i8* lhs, const i8* rhs) noexcept;
