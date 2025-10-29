// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_HYPERBOLIC_FUNCTIONS_H
#define _LIBCUDACXX___CMATH_HYPERBOLIC_FUNCTIONS_H

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

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// cosh

#if _CCCL_CHECK_BUILTIN(builtin_cosh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_COSHF(...) __builtin_coshf(__VA_ARGS__)
#  define _CCCL_BUILTIN_COSH(...)  __builtin_cosh(__VA_ARGS__)
#  define _CCCL_BUILTIN_COSHL(...) __builtin_coshl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_cosh)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_COSHF
#  undef _CCCL_BUILTIN_COSH
#  undef _CCCL_BUILTIN_COSHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float cosh(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_COSHF)
  return _CCCL_BUILTIN_COSHF(__x);
#else // ^^^ _CCCL_BUILTIN_COSHF ^^^ / vvv !_CCCL_BUILTIN_COSHF vvv
  return ::coshf(__x);
#endif // !_CCCL_BUILTIN_COSHF
}

[[nodiscard]] _CCCL_API inline float coshf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_COSHF)
  return _CCCL_BUILTIN_COSHF(__x);
#else // ^^^ _CCCL_BUILTIN_COSHF ^^^ / vvv !_CCCL_BUILTIN_COSHF vvv
  return ::coshf(__x);
#endif // !_CCCL_BUILTIN_COSHF
}

[[nodiscard]] _CCCL_API inline double cosh(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_COSH)
  return _CCCL_BUILTIN_COSH(__x);
#else // ^^^ _CCCL_BUILTIN_COSH ^^^ / vvv !_CCCL_BUILTIN_COSH vvv
  return ::cosh(__x);
#endif // !_CCCL_BUILTIN_COSH
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double cosh(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_COSHL)
  return _CCCL_BUILTIN_COSHL(__x);
#  else // ^^^ _CCCL_BUILTIN_COSHL ^^^ / vvv !_CCCL_BUILTIN_COSHL vvv
  return ::coshl(__x);
#  endif // !_CCCL_BUILTIN_COSHL
}

[[nodiscard]] _CCCL_API inline long double coshl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_COSHL)
  return _CCCL_BUILTIN_COSHL(__x);
#  else // ^^^ _CCCL_BUILTIN_COSHL ^^^ / vvv !_CCCL_BUILTIN_COSHL vvv
  return ::coshl(__x);
#  endif // !_CCCL_BUILTIN_COSHL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half cosh(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::coshf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 cosh(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::coshf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double cosh(_Integer __x) noexcept
{
  return _CUDA_VSTD::cosh((double) __x);
}

// sinh

#if _CCCL_CHECK_BUILTIN(builtin_sinh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SINHF(...) __builtin_sinhf(__VA_ARGS__)
#  define _CCCL_BUILTIN_SINH(...)  __builtin_sinh(__VA_ARGS__)
#  define _CCCL_BUILTIN_SINHL(...) __builtin_sinhl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_sin)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_SINHF
#  undef _CCCL_BUILTIN_SINH
#  undef _CCCL_BUILTIN_SINHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float sinh(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SINHF)
  return _CCCL_BUILTIN_SINHF(__x);
#else // ^^^ _CCCL_BUILTIN_SINHF ^^^ / vvv !_CCCL_BUILTIN_SINHF vvv
  return ::sinhf(__x);
#endif // !_CCCL_BUILTIN_SINHF
}

[[nodiscard]] _CCCL_API inline float sinhf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SINHF)
  return _CCCL_BUILTIN_SINHF(__x);
#else // ^^^ _CCCL_BUILTIN_SINHF ^^^ / vvv !_CCCL_BUILTIN_SINHF vvv
  return ::sinhf(__x);
#endif // !_CCCL_BUILTIN_SINHF
}

[[nodiscard]] _CCCL_API inline double sinh(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_SINH)
  return _CCCL_BUILTIN_SINH(__x);
#else // ^^^ _CCCL_BUILTIN_SINH ^^^ / vvv !_CCCL_BUILTIN_SINH vvv
  return ::sinh(__x);
#endif // !_CCCL_BUILTIN_SINH
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double sinh(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SINHL)
  return _CCCL_BUILTIN_SINHL(__x);
#  else // ^^^ _CCCL_BUILTIN_SINHL ^^^ / vvv !_CCCL_BUILTIN_SINHL vvv
  return ::sinhl(__x);
#  endif // !_CCCL_BUILTIN_SINHL
}

[[nodiscard]] _CCCL_API inline long double sinhl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SINHL)
  return _CCCL_BUILTIN_SINHL(__x);
#  else // ^^^ _CCCL_BUILTIN_SINHL ^^^ / vvv !_CCCL_BUILTIN_SINHL vvv
  return ::sinhl(__x);
#  endif // !_CCCL_BUILTIN_SINHL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half sinh(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::sinhf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 sinh(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::sinhf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double sinh(_Integer __x) noexcept
{
  return _CUDA_VSTD::sinh((double) __x);
}

// tanh

#if _CCCL_CHECK_BUILTIN(builtin_tanh) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_TANHF(...) __builtin_tanhf(__VA_ARGS__)
#  define _CCCL_BUILTIN_TANH(...)  __builtin_tanh(__VA_ARGS__)
#  define _CCCL_BUILTIN_TANHL(...) __builtin_tanhl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_tan)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_TANHF
#  undef _CCCL_BUILTIN_TANH
#  undef _CCCL_BUILTIN_TANHL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float tanh(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TANHF)
  return _CCCL_BUILTIN_TANHF(__x);
#else // ^^^ _CCCL_BUILTIN_TANHF ^^^ / vvv !_CCCL_BUILTIN_TANHF vvv
  return ::tanhf(__x);
#endif // !_CCCL_BUILTIN_TANHF
}

[[nodiscard]] _CCCL_API inline float tanhf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TANHF)
  return _CCCL_BUILTIN_TANHF(__x);
#else // ^^^ _CCCL_BUILTIN_TANHF ^^^ / vvv !_CCCL_BUILTIN_TANHF vvv
  return ::tanhf(__x);
#endif // !_CCCL_BUILTIN_TANHF
}

[[nodiscard]] _CCCL_API inline double tanh(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_TANH)
  return _CCCL_BUILTIN_TANH(__x);
#else // ^^^ _CCCL_BUILTIN_TANH ^^^ / vvv !_CCCL_BUILTIN_TANH vvv
  return ::tanh(__x);
#endif // !_CCCL_BUILTIN_TANH
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double tanh(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TANHL)
  return _CCCL_BUILTIN_TANHL(__x);
#  else // ^^^ _CCCL_BUILTIN_TANHL ^^^ / vvv !_CCCL_BUILTIN_TANHL vvv
  return ::tanhl(__x);
#  endif // !_CCCL_BUILTIN_TANHL
}

[[nodiscard]] _CCCL_API inline long double tanhl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TANHL)
  return _CCCL_BUILTIN_TANHL(__x);
#  else // ^^^ _CCCL_BUILTIN_TANHL ^^^ / vvv !_CCCL_BUILTIN_TANHL vvv
  return ::tanhl(__x);
#  endif // !_CCCL_BUILTIN_TANHL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half tanh(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::tanhf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 tanh(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::tanhf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double tanh(_Integer __x) noexcept
{
  return _CUDA_VSTD::tanh((double) __x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_HYPERBOLIC_FUNCTIONS_H
