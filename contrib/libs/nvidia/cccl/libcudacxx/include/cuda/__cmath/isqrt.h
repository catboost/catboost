//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_ISQRT_H
#define _CUDA___CMATH_ISQRT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/integral.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Returns the square root of the given non-negative integer rounded down
//! @param __v The input number
//! @pre \p __v must be an integer type
//! @pre \p __v must be non-negative
//! @return The square root of \p __v rounded down
//! @warning If \p __v is negative, the behavior is undefined
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr _Tp isqrt(_Tp __v) noexcept
{
  if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
  {
    _CCCL_ASSERT(__v >= _Tp{0}, "cuda::isqrt requires non-negative input");
  }

  if (__v <= 1)
  {
    return __v;
  }

  using _Up = _CUDA_VSTD::make_unsigned_t<_Tp>;

  _Up __uv = static_cast<_Up>(__v);
  _Up __ret{};
  _Up __bit = static_cast<_Up>(_Up{1} << ((_CUDA_VSTD::bit_width(__uv) - 1) & (~1)));

  while (__bit != 0)
  {
    if (__uv >= __ret + __bit)
    {
      __uv -= __ret + __bit;
      __ret = (__ret >> 1) + __bit;
    }
    else
    {
      __ret >>= 1;
    }
    __bit >>= 2;
  }
  return static_cast<_Tp>(__ret);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_ISQRT_H
