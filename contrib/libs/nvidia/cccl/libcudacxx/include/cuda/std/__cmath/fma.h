//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_FMA_H
#define _LIBCUDACXX___CMATH_FMA_H

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

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// fma

#if _CCCL_CHECK_BUILTIN(builtin_fma) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FMAF(...) __builtin_fmaf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMA(...)  __builtin_fma(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMAL(...) __builtin_fmal(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fmax)

[[nodiscard]] _CCCL_API inline float fma(float __x, float __y, float __z) noexcept
{
#if defined(_CCCL_BUILTIN_FMAF)
  return _CCCL_BUILTIN_FMAF(__x, __y, __z);
#else // ^^^ _CCCL_BUILTIN_FMAF ^^^ / vvv !_CCCL_BUILTIN_FMAF vvv
  return ::fmaf(__x, __y, __z);
#endif // ^^^ !_CCCL_BUILTIN_FMAF ^^^
}

[[nodiscard]] _CCCL_API inline float fmaf(float __x, float __y, float __z) noexcept
{
#if defined(_CCCL_BUILTIN_FMAF)
  return _CCCL_BUILTIN_FMAF(__x, __y, __z);
#else // ^^^ _CCCL_BUILTIN_FMAF ^^^ / vvv !_CCCL_BUILTIN_FMAF vvv
  return ::fmaf(__x, __y, __z);
#endif // ^^^ !_CCCL_BUILTIN_FMAF ^^^
}

[[nodiscard]] _CCCL_API inline double fma(double __x, double __y, double __z) noexcept
{
#if defined(_CCCL_BUILTIN_FMA)
  return _CCCL_BUILTIN_FMA(__x, __y, __z);
#else // ^^^ _CCCL_BUILTIN_FMA ^^^ / vvv !_CCCL_BUILTIN_FMA vvv
  return ::fma(__x, __y, __z);
#endif // ^^^ !_CCCL_BUILTIN_FMA ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double fma(long double __x, long double __y, long double __z) noexcept
{
#  if defined(_CCCL_BUILTIN_FMAL)
  return _CCCL_BUILTIN_FMAL(__x, __y, __z);
#  else // ^^^ _CCCL_BUILTIN_FMAL ^^^ / vvv !_CCCL_BUILTIN_FMAL vvv
  return ::fmal(__x, __y, __z);
#  endif // ^^^ !_CCCL_BUILTIN_FMAL ^^^
}

[[nodiscard]] _CCCL_API inline long double fmal(long double __x, long double __y, long double __z) noexcept
{
#  if defined(_CCCL_BUILTIN_FMAL)
  return _CCCL_BUILTIN_FMAL(__x, __y, __z);
#  else // ^^^ _CCCL_BUILTIN_FMAL ^^^ / vvv !_CCCL_BUILTIN_FMAL vvv
  return ::fmal(__x, __y, __z);
#  endif // ^^^ !_CCCL_BUILTIN_FMAL ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half fma(__half __x, __half __y, __half __z) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_53,
    (return ::__hfma(__x, __y, __z);),
    (return ::__float2half(_CUDA_VSTD::fma(::__half2float(__x), ::__half2float(__y), ::__half2float(__z)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 fma(__nv_bfloat16 __x, __nv_bfloat16 __y, __nv_bfloat16 __z) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_80,
    (return ::__hfma(__x, __y, __z);),
    (return ::__float2bfloat16(
              _CUDA_VSTD::fma(::__bfloat162float(__x), ::__bfloat162float(__y), ::__bfloat162float(__z)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1,
          class _A2,
          class _A3,
          enable_if_t<is_arithmetic_v<_A1> && is_arithmetic_v<_A2> && is_arithmetic_v<_A3>, int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2, _A3> fma(_A1 __x, _A2 __y, _A3 __z) noexcept
{
  using __result_type = __promote_t<_A1, _A2, _A3>;
  static_assert(!(is_same_v<_A1, __result_type> && is_same_v<_A2, __result_type> && is_same_v<_A3, __result_type>) );
  return _CUDA_VSTD::fma((__result_type) __x, (__result_type) __y, (__result_type) __z);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_FMA_H
