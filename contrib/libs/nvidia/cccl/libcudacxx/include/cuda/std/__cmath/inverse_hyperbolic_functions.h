// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_INVERSE_HYPERBOLIC_FUNCTIONS_H
#define _LIBCUDACXX___CMATH_INVERSE_HYPERBOLIC_FUNCTIONS_H

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
#include <cuda/std/__type_traits/is_integral.h>

#include <nv/target>

#if !_CCCL_COMPILER(NVRTC)
#  include <math.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// acosh

#if _CCCL_CHECK_BUILTIN(builtin_acosh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ACOSHF(...) __builtin_acoshf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ACOSH(...)  __builtin_acosh(__VA_ARGS__)
#  define _CCCL_BUILTIN_ACOSHL(...) __builtin_acoshl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_acosh)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ACOSHF
#  undef _CCCL_BUILTIN_ACOSH
#  undef _CCCL_BUILTIN_ACOSHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float acosh(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSHF)
  return _CCCL_BUILTIN_ACOSHF(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSHF ^^^ / vvv !_CCCL_BUILTIN_ACOSHF vvv
  return ::acoshf(__x);
#endif // !_CCCL_BUILTIN_ACOSHF
}

[[nodiscard]] _CCCL_API inline float acoshf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSHF)
  return _CCCL_BUILTIN_ACOSHF(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSHF ^^^ / vvv !_CCCL_BUILTIN_ACOSHF vvv
  return ::acoshf(__x);
#endif // !_CCCL_BUILTIN_ACOSHF
}

[[nodiscard]] _CCCL_API inline double acosh(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ACOSH)
  return _CCCL_BUILTIN_ACOSH(__x);
#else // ^^^ _CCCL_BUILTIN_ACOSH ^^^ / vvv !_CCCL_BUILTIN_ACOSH vvv
  return ::acosh(__x);
#endif // !_CCCL_BUILTIN_ACOSH
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double acosh(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ACOSHL)
  return _CCCL_BUILTIN_ACOSHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ACOSHL ^^^ / vvv !_CCCL_BUILTIN_ACOSHL vvv
  return ::acoshl(__x);
#  endif // !_CCCL_BUILTIN_ACOSHL
}

[[nodiscard]] _CCCL_API inline long double acoshl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ACOSHL)
  return _CCCL_BUILTIN_ACOSHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ACOSHL ^^^ / vvv !_CCCL_BUILTIN_ACOSHL vvv
  return ::acoshl(__x);
#  endif // !_CCCL_BUILTIN_ACOSHL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half acosh(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::acoshf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 acosh(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::acoshf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double acosh(_Integer __x) noexcept
{
  return _CUDA_VSTD::acosh((double) __x);
}

// asinh

#if _CCCL_CHECK_BUILTIN(builtin_asinh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ASINHF(...) __builtin_asinhf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ASINH(...)  __builtin_asinh(__VA_ARGS__)
#  define _CCCL_BUILTIN_ASINHL(...) __builtin_asinhl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_asin)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ASINHF
#  undef _CCCL_BUILTIN_ASINH
#  undef _CCCL_BUILTIN_ASINHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float asinh(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINHF)
  return _CCCL_BUILTIN_ASINHF(__x);
#else // ^^^ _CCCL_BUILTIN_ASINHF ^^^ / vvv !_CCCL_BUILTIN_ASINHF vvv
  return ::asinhf(__x);
#endif // !_CCCL_BUILTIN_ASINHF
}

[[nodiscard]] _CCCL_API inline float asinhf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINHF)
  return _CCCL_BUILTIN_ASINHF(__x);
#else // ^^^ _CCCL_BUILTIN_ASINHF ^^^ / vvv !_CCCL_BUILTIN_ASINHF vvv
  return ::asinhf(__x);
#endif // !_CCCL_BUILTIN_ASINHF
}

[[nodiscard]] _CCCL_API inline double asinh(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ASINH)
  return _CCCL_BUILTIN_ASINH(__x);
#else // ^^^ _CCCL_BUILTIN_ASINH ^^^ / vvv !_CCCL_BUILTIN_ASINH vvv
  return ::asinh(__x);
#endif // !_CCCL_BUILTIN_ASINH
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double asinh(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ASINHL)
  return _CCCL_BUILTIN_ASINHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ASINHL ^^^ / vvv !_CCCL_BUILTIN_ASINHL vvv
  return ::asinhl(__x);
#  endif // !_CCCL_BUILTIN_ASINHL
}

[[nodiscard]] _CCCL_API inline long double asinhl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ASINHL)
  return _CCCL_BUILTIN_ASINHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ASINHL ^^^ / vvv !_CCCL_BUILTIN_ASINHL vvv
  return ::asinhl(__x);
#  endif // !_CCCL_BUILTIN_ASINHL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half asinh(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::asinhf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 asinh(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::asinhf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double asinh(_Integer __x) noexcept
{
  return _CUDA_VSTD::asinh((double) __x);
}

// atanh

#if _CCCL_CHECK_BUILTIN(builtin_atanh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ATANHF(...) __builtin_atanhf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATANH(...)  __builtin_atanh(__VA_ARGS__)
#  define _CCCL_BUILTIN_ATANHL(...) __builtin_atanhl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_atanh)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_ATANHF
#  undef _CCCL_BUILTIN_ATANH
#  undef _CCCL_BUILTIN_ATANHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float atanh(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANHF)
  return _CCCL_BUILTIN_ATANHF(__x);
#else // ^^^ _CCCL_BUILTIN_ATANHF ^^^ / vvv !_CCCL_BUILTIN_ATANHF vvv
  return ::atanhf(__x);
#endif // !_CCCL_BUILTIN_ATANHF
}

[[nodiscard]] _CCCL_API inline float atanhf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANHF)
  return _CCCL_BUILTIN_ATANHF(__x);
#else // ^^^ _CCCL_BUILTIN_ATANHF ^^^ / vvv !_CCCL_BUILTIN_ATANHF vvv
  return ::atanhf(__x);
#endif // !_CCCL_BUILTIN_ATANHF
}

[[nodiscard]] _CCCL_API inline double atanh(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ATANH)
  return _CCCL_BUILTIN_ATANH(__x);
#else // ^^^ _CCCL_BUILTIN_ATANH ^^^ / vvv !_CCCL_BUILTIN_ATANH vvv
  return ::atanh(__x);
#endif // !_CCCL_BUILTIN_ATANH
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double atanh(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ATANHL)
  return _CCCL_BUILTIN_ATANHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ATANHL ^^^ / vvv !_CCCL_BUILTIN_ATANHL vvv
  return ::atanhl(__x);
#  endif // !_CCCL_BUILTIN_ATANHL
}

[[nodiscard]] _CCCL_API inline long double atanhl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ATANHL)
  return _CCCL_BUILTIN_ATANHL(__x);
#  else // ^^^ _CCCL_BUILTIN_ATANHL ^^^ / vvv !_CCCL_BUILTIN_ATANHL vvv
  return ::atanhl(__x);
#  endif // !_CCCL_BUILTIN_ATANHL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half atanh(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::atanhf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 atanh(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::atanhf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double atanh(_Integer __x) noexcept
{
  return _CUDA_VSTD::atanh((double) __x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_INVERSE_HYPERBOLIC_FUNCTIONS_H
