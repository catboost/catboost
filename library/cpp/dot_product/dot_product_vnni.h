#pragma once

#include <util/system/types.h>
#include <util/system/compiler.h>

#include <stddef.h>

Y_PURE_FUNCTION
i32 DotProductVnni(const i8* lhs, const i8* rhs, size_t length) noexcept;
