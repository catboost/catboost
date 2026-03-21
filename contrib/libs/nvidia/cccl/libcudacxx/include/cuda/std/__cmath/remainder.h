// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_REMAINDER_H
#define _LIBCUDACXX___CMATH_REMAINDER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/promote.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <math.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// remainder

#if _CCCL_CHECK_BUILTIN(builtin_remainder) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_REMAINDERF(...) __builtin_remainderf(__VA_ARGS__)
#  define _CCCL_BUILTIN_REMAINDER(...)  __builtin_remainder(__VA_ARGS__)
#  define _CCCL_BUILTIN_REMAINDERL(...) __builtin_remainderl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_remainder)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "remainder"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_REMAINDERF
#  undef _CCCL_BUILTIN_REMAINDER
#  undef _CCCL_BUILTIN_REMAINDERFL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float remainder(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_REMAINDERF)
  return _CCCL_BUILTIN_REMAINDERF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_REMAINDERF ^^^ / vvv !_CCCL_BUILTIN_REMAINDERF vvv
  return ::remainderf(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_REMAINDERF ^^^
}

[[nodiscard]] _CCCL_API inline float remainderf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_REMAINDERF)
  return _CCCL_BUILTIN_REMAINDERF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_REMAINDERF ^^^ / vvv !_CCCL_BUILTIN_REMAINDERF vvv
  return ::remainderf(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_REMAINDERF ^^^
}

[[nodiscard]] _CCCL_API inline double remainder(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_REMAINDER)
  return _CCCL_BUILTIN_REMAINDER(__x, __y);
#else // ^^^ _CCCL_BUILTIN_REMAINDER ^^^ / vvv !_CCCL_BUILTIN_REMAINDER vvv
  return ::remainder(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_REMAINDER ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double remainder(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_REMAINDERL)
  return _CCCL_BUILTIN_REMAINDERL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_REMAINDERL ^^^ / vvv !_CCCL_BUILTIN_REMAINDERL vvv
  return ::remainderl(__x, __y);
#  endif // ^^^ !_CCCL_BUILTIN_REMAINDERL ^^^
}

[[nodiscard]] _CCCL_API inline long double remainderl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_REMAINDERL)
  return _CCCL_BUILTIN_REMAINDERL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_REMAINDERL ^^^ / vvv !_CCCL_BUILTIN_REMAINDERL vvv
  return ::remainderl(__x, __y);
#  endif // ^^^ !_CCCL_BUILTIN_REMAINDERL ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half remainder(__half __x, __half __y) noexcept
{
  return __float2half(_CUDA_VSTD::remainder(__half2float(__x), __half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 remainder(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::remainder(__bfloat162float(__x), __bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<is_arithmetic_v<_A1> && is_arithmetic_v<_A2>, int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2> remainder(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(is_same_v<_A1, __result_type> && is_same_v<_A2, __result_type>), "");
  return _CUDA_VSTD::remainder((__result_type) __x, (__result_type) __y);
}

// remquo

#if _CCCL_CHECK_BUILTIN(builtin_remquo) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_REMQUOF(...) __builtin_remquof(__VA_ARGS__)
#  define _CCCL_BUILTIN_REMQUO(...)  __builtin_remquo(__VA_ARGS__)
#  define _CCCL_BUILTIN_REMQUOL(...) __builtin_remquol(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_remquo)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "remquo"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_REMQUOF
#  undef _CCCL_BUILTIN_REMQUO
#  undef _CCCL_BUILTIN_REMQUOL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float remquo(float __x, float __y, int* __quotient) noexcept
{
#if defined(_CCCL_BUILTIN_REMQUOF)
  return _CCCL_BUILTIN_REMQUOF(__x, __y, __quotient);
#else // ^^^ _CCCL_BUILTIN_REMQUOF ^^^ / vvv !_CCCL_BUILTIN_REMQUOF vvv
  return ::remquof(__x, __y, __quotient);
#endif // ^^^ !_CCCL_BUILTIN_REMQUOF ^^^
}

[[nodiscard]] _CCCL_API inline float remquof(float __x, float __y, int* __quotient) noexcept
{
#if defined(_CCCL_BUILTIN_REMQUOF)
  return _CCCL_BUILTIN_REMQUOF(__x, __y, __quotient);
#else // ^^^ _CCCL_BUILTIN_REMQUOF ^^^ / vvv !_CCCL_BUILTIN_REMQUOF vvv
  return ::remquof(__x, __y, __quotient);
#endif // ^^^ !_CCCL_BUILTIN_REMQUOF ^^^
}

[[nodiscard]] _CCCL_API inline double remquo(double __x, double __y, int* __quotient) noexcept
{
#if defined(_CCCL_BUILTIN_REMQUO)
  return _CCCL_BUILTIN_REMQUO(__x, __y, __quotient);
#else // ^^^ _CCCL_BUILTIN_REMQUO ^^^ / vvv !_CCCL_BUILTIN_REMQUO vvv
  return ::remquo(__x, __y, __quotient);
#endif // ^^^ !_CCCL_BUILTIN_REMQUO ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double remquo(long double __x, long double __y, int* __quotient) noexcept
{
#  if defined(_CCCL_BUILTIN_REMQUOL)
  return _CCCL_BUILTIN_REMQUOL(__x, __y, __quotient);
#  else // ^^^ _CCCL_BUILTIN_REMQUOL ^^^ / vvv !_CCCL_BUILTIN_REMQUOL vvv
  return ::remquol(__x, __y, __quotient);
#  endif // ^^^ !_CCCL_BUILTIN_REMQUOL ^^^
}

[[nodiscard]] _CCCL_API inline long double remquol(long double __x, long double __y, int* __quotient) noexcept
{
#  if defined(_CCCL_BUILTIN_REMQUOL)
  return _CCCL_BUILTIN_REMQUOL(__x, __y, __quotient);
#  else // ^^^ _CCCL_BUILTIN_REMQUOL ^^^ / vvv !_CCCL_BUILTIN_REMQUOL vvv
  return ::remquol(__x, __y, __quotient);
#  endif // ^^^ !_CCCL_BUILTIN_REMQUOL ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half remquo(__half __x, __half __y, int* __quotient) noexcept
{
  return __float2half(_CUDA_VSTD::remquo(__half2float(__x), __half2float(__y), __quotient));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 remquo(__nv_bfloat16 __x, __nv_bfloat16 __y, int* __quotient) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::remquo(__bfloat162float(__x), __bfloat162float(__y), __quotient));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<is_arithmetic_v<_A1> && is_arithmetic_v<_A2>, int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2> remquo(_A1 __x, _A2 __y, int* __quotient) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(is_same_v<_A1, __result_type> && is_same_v<_A2, __result_type>), "");
  return _CUDA_VSTD::remquo((__result_type) __x, (__result_type) __y, __quotient);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_MODULO_H
