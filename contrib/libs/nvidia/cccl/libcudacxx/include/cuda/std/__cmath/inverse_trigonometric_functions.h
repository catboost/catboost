// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_INVERSE_TRIGONOMETRIC_FUNCTIONS_H
#define _LIBCUDACXX___CMATH_INVERSE_TRIGONOMETRIC_FUNCTIONS_H

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
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/promote.h>

#include <nv/target>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// acos

#if _CCCL_CHECK_BUILTIN(builtin_acos) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ACOSF(...) __builtin_acosf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ACOS(...)  __builtin_acos(__VA_ARGS__)
#  define _CCCL_BUILTIN_ACOSL(...) __builtin_acosl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_acos)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ACOSF
#  undef _CCCL_BUILTIN_ACOS
#  undef _CCCL_BUILTIN_ACOSL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float acos(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSF)
  return _CCCL_BUILTIN_ACOSF(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSF ^^^ / vvv !_CCCL_BUILTIN_ACOSF vvv
  return ::acosf(__x);
#endif // !_CCCL_BUILTIN_ACOSF
}

[[nodiscard]] _CCCL_API inline float acosf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSF)
  return _CCCL_BUILTIN_ACOSF(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSF ^^^ / vvv !_CCCL_BUILTIN_ACOSF vvv
  return ::acosf(__x);
#endif // !_CCCL_BUILTIN_ACOSF
}

[[nodiscard]] _CCCL_API inline double acos(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOS)
  return _CCCL_BUILTIN_ACOS(__x);
#else // ^^^ _CCCL_BUILTIN_ACOS ^^^ / vvv !_CCCL_BUILTIN_ACOS vvv
  return ::acos(__x);
#endif // !_CCCL_BUILTIN_ACOS
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double acos(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ACOSL)
  return _CCCL_BUILTIN_ACOSL(__x);
#  else // ^^^ _CCCL_BUILTIN_ACOSL ^^^ / vvv !_CCCL_BUILTIN_ACOSL vvv
  return ::acosl(__x);
#  endif // !_CCCL_BUILTIN_ACOSL
}

[[nodiscard]] _CCCL_API inline long double acosl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ACOSL)
  return _CCCL_BUILTIN_ACOSL(__x);
#  else // ^^^ _CCCL_BUILTIN_ACOSL ^^^ / vvv !_CCCL_BUILTIN_ACOSL vvv
  return ::acosl(__x);
#  endif // !_CCCL_BUILTIN_ACOSL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half acos(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::acosf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 acos(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::acosf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double acos(_Integer __x) noexcept
{
  return _CUDA_VSTD::acos((double) __x);
}

// asin

#if _CCCL_CHECK_BUILTIN(builtin_asin) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ASINF(...) __builtin_asinf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ASIN(...)  __builtin_asin(__VA_ARGS__)
#  define _CCCL_BUILTIN_ASINL(...) __builtin_asinl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_asin)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ASINF
#  undef _CCCL_BUILTIN_ASIN
#  undef _CCCL_BUILTIN_ASINL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float asin(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINF)
  return _CCCL_BUILTIN_ASINF(__x);
#else // ^^^ _CCCL_BUILTIN_ASINF ^^^ / vvv !_CCCL_BUILTIN_ASINF vvv
  return ::asinf(__x);
#endif // !_CCCL_BUILTIN_ASINF
}

[[nodiscard]] _CCCL_API inline float asinf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINF)
  return _CCCL_BUILTIN_ASINF(__x);
#else // ^^^ _CCCL_BUILTIN_ASINF ^^^ / vvv !_CCCL_BUILTIN_ASINF vvv
  return ::asinf(__x);
#endif // !_CCCL_BUILTIN_ASINF
}

[[nodiscard]] _CCCL_API inline double asin(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASIN)
  return _CCCL_BUILTIN_ASIN(__x);
#else // ^^^ _CCCL_BUILTIN_ASIN ^^^ / vvv !_CCCL_BUILTIN_ASIN vvv
  return ::asin(__x);
#endif // !_CCCL_BUILTIN_ASIN
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double asin(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ASINL)
  return _CCCL_BUILTIN_ASINL(__x);
#  else // ^^^ _CCCL_BUILTIN_ASINL ^^^ / vvv !_CCCL_BUILTIN_ASINL vvv
  return ::asinl(__x);
#  endif // !_CCCL_BUILTIN_ASINL
}

[[nodiscard]] _CCCL_API inline long double asinl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ASINL)
  return _CCCL_BUILTIN_ASINL(__x);
#  else // ^^^ _CCCL_BUILTIN_ASINL ^^^ / vvv !_CCCL_BUILTIN_ASINL vvv
  return ::asinl(__x);
#  endif // !_CCCL_BUILTIN_ASINL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half asin(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::asinf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 asin(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::asinf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double asin(_Integer __x) noexcept
{
  return _CUDA_VSTD::asin((double) __x);
}

// atan

#if _CCCL_CHECK_BUILTIN(builtin_atan) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ATANF(...) __builtin_atanf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATAN(...)  __builtin_atan(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATANL(...) __builtin_atanl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_atan)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ATANF
#  undef _CCCL_BUILTIN_ATAN
#  undef _CCCL_BUILTIN_ATANL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float atan(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANF)
  return _CCCL_BUILTIN_ATANF(__x);
#else // ^^^ _CCCL_BUILTIN_ATANF ^^^ / vvv !_CCCL_BUILTIN_ATANF vvv
  return ::atanf(__x);
#endif // !_CCCL_BUILTIN_ATANF
}

[[nodiscard]] _CCCL_API inline float atanf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANF)
  return _CCCL_BUILTIN_ATANF(__x);
#else // ^^^ _CCCL_BUILTIN_ATANF ^^^ / vvv !_CCCL_BUILTIN_ATANF vvv
  return ::atanf(__x);
#endif // !_CCCL_BUILTIN_ATANF
}

[[nodiscard]] _CCCL_API inline double atan(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATAN)
  return _CCCL_BUILTIN_ATAN(__x);
#else // ^^^ _CCCL_BUILTIN_ATAN ^^^ / vvv !_CCCL_BUILTIN_ATAN vvv
  return ::atan(__x);
#endif // !_CCCL_BUILTIN_ATAN
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double atan(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ATANL)
  return _CCCL_BUILTIN_ATANL(__x);
#  else // ^^^ _CCCL_BUILTIN_ATANL ^^^ / vvv !_CCCL_BUILTIN_ATANL vvv
  return ::atanl(__x);
#  endif // !_CCCL_BUILTIN_ATANL
}

[[nodiscard]] _CCCL_API inline long double atanl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ATANL)
  return _CCCL_BUILTIN_ATANL(__x);
#  else // ^^^ _CCCL_BUILTIN_ATANL ^^^ / vvv !_CCCL_BUILTIN_ATANL vvv
  return ::atanl(__x);
#  endif // !_CCCL_BUILTIN_ATANL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half atan(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::atanf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 atan(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::atanf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double atan(_Integer __x) noexcept
{
  return _CUDA_VSTD::atan((double) __x);
}

// atan2

#if _CCCL_CHECK_BUILTIN(builtin_atan2) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ATAN2F(...) __builtin_atan2f(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATAN2(...)  __builtin_atan2(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATAN2L(...) __builtin_atan2l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_atan2)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ATAN2F
#  undef _CCCL_BUILTIN_ATAN2
#  undef _CCCL_BUILTIN_ATAN2L
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float atan2(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_ATAN2F)
  return _CCCL_BUILTIN_ATAN2F(__x, __y);
#else // ^^^ _CCCL_BUILTIN_ATAN2F ^^^ // vvv !_CCCL_BUILTIN_ATAN2F vvv
  return ::atan2f(__x, __y);
#endif // !_CCCL_BUILTIN_ATAN2F
}

[[nodiscard]] _CCCL_API inline float atan2f(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_ATAN2F)
  return _CCCL_BUILTIN_ATAN2F(__x, __y);
#else // ^^^ _CCCL_BUILTIN_ATAN2F ^^^ // vvv !_CCCL_BUILTIN_ATAN2F vvv
  return ::atan2f(__x, __y);
#endif // !_CCCL_BUILTIN_ATAN2F
}

[[nodiscard]] _CCCL_API inline double atan2(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_ATAN2)
  return _CCCL_BUILTIN_ATAN2(__x, __y);
#else // ^^^ _CCCL_BUILTIN_ATAN2 ^^^ // vvv !_CCCL_BUILTIN_ATAN2 vvv
  return ::atan2(__x, __y);
#endif // !_CCCL_BUILTIN_ATAN2
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double atan2(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ATAN2L)
  return _CCCL_BUILTIN_ATAN2L(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ATAN2L ^^^ // vvv !_CCCL_BUILTIN_ATAN2L vvv
  return ::atan2l(__x, __y);
#  endif // !_CCCL_BUILTIN_ATAN2L
}

[[nodiscard]] _CCCL_API inline long double atan2l(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ATAN2L)
  return _CCCL_BUILTIN_ATAN2L(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ATAN2L ^^^ // vvv !_CCCL_BUILTIN_ATAN2L vvv
  return ::atan2l(__x, __y);
#  endif // !_CCCL_BUILTIN_ATAN2L
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half atan2(__half __x, __half __y) noexcept
{
  return __float2half(_CUDA_VSTD::atan2f(__half2float(__x), __half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 atan2(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::atan2f(__bfloat162float(__x), __bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2> atan2(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)), "");
  return _CUDA_VSTD::atan2((__result_type) __x, (__result_type) __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_INVERSE_TRIGONOMETRIC_FUNCTIONS_H
