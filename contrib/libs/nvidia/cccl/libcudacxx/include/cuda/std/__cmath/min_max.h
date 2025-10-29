// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_MIN_MAX_H
#define _LIBCUDACXX___CMATH_MIN_MAX_H

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
#include <cuda/std/__type_traits/promote.h>

#include <nv/target>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// fmax

#if _CCCL_CHECK_BUILTIN(builtin_fmax) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FMAXF(...) __builtin_fmaxf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMAX(...)  __builtin_fmax(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMAXL(...) __builtin_fmaxl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fmax)

[[nodiscard]] _CCCL_API inline float fmax(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_FMAX)
  return _CCCL_BUILTIN_FMAXF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FMAX ^^^ / vvv !_CCCL_BUILTIN_FMAX vvv
  return ::fmaxf(__x, __y);
#endif // !_CCCL_BUILTIN_FMAX
}

[[nodiscard]] _CCCL_API inline float fmaxf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_FMAX)
  return _CCCL_BUILTIN_FMAXF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FMAX ^^^ / vvv !_CCCL_BUILTIN_FMAX vvv
  return ::fmaxf(__x, __y);
#endif // !_CCCL_BUILTIN_FMAX
}

[[nodiscard]] _CCCL_API inline double fmax(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_FMAX)
  return _CCCL_BUILTIN_FMAX(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FMAX ^^^ / vvv !_CCCL_BUILTIN_FMAX vvv
  return ::fmax(__x, __y);
#endif // !_CCCL_BUILTIN_FMAX
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double fmax(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_FMAX)
  return _CCCL_BUILTIN_FMAXL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_FMAX ^^^ / vvv !_CCCL_BUILTIN_FMAX vvv
  return ::fmaxl(__x, __y);
#  endif // !_CCCL_BUILTIN_FMAX
}
[[nodiscard]] _CCCL_API inline long double fmaxl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_FMAX)
  return _CCCL_BUILTIN_FMAXL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_FMAX ^^^ / vvv !_CCCL_BUILTIN_FMAX vvv
  return ::fmaxl(__x, __y);
#  endif // !_CCCL_BUILTIN_FMAX
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half fmax(__half __x, __half __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return ::__hmax(__x, __y);),
                    (return __float2half(_CUDA_VSTD::fmaxf(__half2float(__x), __half2float(__y)));))
}
template <class _A1, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<float, _A1> fmax(__half __x, _A1 __y) noexcept
{
  return _CUDA_VSTD::fmaxf(__half2float(__x), __y);
}

template <class _A1, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, float> fmax(_A1 __x, __half __y) noexcept
{
  return _CUDA_VSTD::fmaxf(__x, __half2float(__y));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 fmax(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return ::__hmax(__x, __y);),
                    (return __float2bfloat16(_CUDA_VSTD::fmaxf(__bfloat162float(__x), __bfloat162float(__y)));))
}
template <class _A1, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<float, _A1> fmax(__nv_bfloat16 __x, _A1 __y) noexcept
{
  return _CUDA_VSTD::fmaxf(__bfloat162float(__x), __y);
}

template <class _A1, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, float> fmax(_A1 __x, __nv_bfloat16 __y) noexcept
{
  return _CUDA_VSTD::fmaxf(__x, __bfloat162float(__y));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2> fmax(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)), "");
  return _CUDA_VSTD::fmax((__result_type) __x, (__result_type) __y);
}

// fmin

#if _CCCL_CHECK_BUILTIN(builtin_fmin) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FMINF(...) __builtin_fminf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMIN(...)  __builtin_fmin(__VA_ARGS__)
#  define _CCCL_BUILTIN_FMINL(...) __builtin_fminl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fmin)

[[nodiscard]] _CCCL_API inline float fmin(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_FMIN)
  return _CCCL_BUILTIN_FMINF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FMIN ^^^ / vvv !_CCCL_BUILTIN_FMIN vvv
  return ::fminf(__x, __y);
#endif // !_CCCL_BUILTIN_FMIN
}

[[nodiscard]] _CCCL_API inline float fminf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_FMIN)
  return _CCCL_BUILTIN_FMINF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FMIN ^^^ / vvv !_CCCL_BUILTIN_FMIN vvv
  return ::fminf(__x, __y);
#endif // !_CCCL_BUILTIN_FMIN
}

[[nodiscard]] _CCCL_API inline double fmin(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_FMIN)
  return _CCCL_BUILTIN_FMIN(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FMIN ^^^ / vvv !_CCCL_BUILTIN_FMIN vvv
  return ::fmin(__x, __y);
#endif // !_CCCL_BUILTIN_FMIN
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double fmin(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_FMIN)
  return _CCCL_BUILTIN_FMINL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_FMIN ^^^ / vvv !_CCCL_BUILTIN_FMIN vvv
  return ::fminl(__x, __y);
#  endif // !_CCCL_BUILTIN_FMIN
}
[[nodiscard]] _CCCL_API inline long double fminl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_FMIN)
  return _CCCL_BUILTIN_FMINL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_FMIN ^^^ / vvv !_CCCL_BUILTIN_FMIN vvv
  return ::fminl(__x, __y);
#  endif // !_CCCL_BUILTIN_FMIN
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half fmin(__half __x, __half __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return ::__hmin(__x, __y);),
                    (return __float2half(_CUDA_VSTD::fminf(__half2float(__x), __half2float(__y)));))
}
template <class _A1, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<float, _A1> fmin(__half __x, _A1 __y) noexcept
{
  return _CUDA_VSTD::fminf(__half2float(__x), __y);
}

template <class _A1, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, float> fmin(_A1 __x, __half __y) noexcept
{
  return _CUDA_VSTD::fminf(__x, __half2float(__y));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 fmin(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return ::__hmin(__x, __y);),
                    (return __float2bfloat16(_CUDA_VSTD::fminf(__bfloat162float(__x), __bfloat162float(__y)));))
}
template <class _A1, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<float, _A1> fmin(__nv_bfloat16 __x, _A1 __y) noexcept
{
  return _CUDA_VSTD::fminf(__bfloat162float(__x), __y);
}

template <class _A1, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, float> fmin(_A1 __x, __nv_bfloat16 __y) noexcept
{
  return _CUDA_VSTD::fminf(__x, __bfloat162float(__y));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2> fmin(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)), "");
  return _CUDA_VSTD::fmin((__result_type) __x, (__result_type) __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_MIN_MAX_H
