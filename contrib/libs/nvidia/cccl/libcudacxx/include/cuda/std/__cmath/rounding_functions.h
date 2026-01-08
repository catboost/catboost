// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ROUNDING_FUNCTIONS_H
#define _LIBCUDACXX___CMATH_ROUNDING_FUNCTIONS_H

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
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/promote.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <math.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// ceil

#if _CCCL_CHECK_BUILTIN(builtin_ceil) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_CEILF(...) __builtin_ceilf(__VA_ARGS__)
#  define _CCCL_BUILTIN_CEIL(...)  __builtin_ceil(__VA_ARGS__)
#  define _CCCL_BUILTIN_CEILL(...) __builtin_ceill(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ceil)

[[nodiscard]] _CCCL_API inline float ceil(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_CEILF)
  return _CCCL_BUILTIN_CEILF(__x);
#else // ^^^ _CCCL_BUILTIN_CEILF ^^^ // vvv !_CCCL_BUILTIN_CEILF vvv
  return ::ceilf(__x);
#endif // !_CCCL_BUILTIN_CEILF
}

[[nodiscard]] _CCCL_API inline float ceilf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_CEILF)
  return _CCCL_BUILTIN_CEILF(__x);
#else // ^^^ _CCCL_BUILTIN_CEILF ^^^ // vvv !_CCCL_BUILTIN_CEILF vvv
  return ::ceilf(__x);
#endif // !_CCCL_BUILTIN_CEILF
}

[[nodiscard]] _CCCL_API inline double ceil(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_CEIL)
  return _CCCL_BUILTIN_CEIL(__x);
#else // ^^^ _CCCL_BUILTIN_CEIL ^^^ // vvv !_CCCL_BUILTIN_CEIL vvv
  return ::ceil(__x);
#endif // !_CCCL_BUILTIN_CEIL
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double ceil(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_CEILL)
  return _CCCL_BUILTIN_CEILL(__x);
#  else // ^^^ _CCCL_BUILTIN_CEILL ^^^ // vvv !_CCCL_BUILTIN_CEILL vvv
  return ::ceill(__x);
#  endif // !_CCCL_BUILTIN_CEILL
}
[[nodiscard]] _CCCL_API inline long double ceill(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_CEILL)
  return _CCCL_BUILTIN_CEILL(__x);
#  else // ^^^ _CCCL_BUILTIN_CEILL ^^^ // vvv !_CCCL_BUILTIN_CEILL vvv
  return ::ceill(__x);
#  endif // !_CCCL_BUILTIN_CEILL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half ceil(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hceil(__x);), (return __float2half(_CUDA_VSTD::ceil(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 ceil(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hceil(__x);), (return __float2bfloat16(_CUDA_VSTD::ceil(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double ceil(_Integer __x) noexcept
{
  return _CUDA_VSTD::ceil((double) __x);
}

// floor

#if _CCCL_CHECK_BUILTIN(builtin_floor) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FLOORF(...) __builtin_floorf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FLOOR(...)  __builtin_floor(__VA_ARGS__)
#  define _CCCL_BUILTIN_FLOORL(...) __builtin_floorl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_floor)

[[nodiscard]] _CCCL_API inline float floor(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_FLOORF)
  return _CCCL_BUILTIN_FLOORF(__x);
#else // ^^^ _CCCL_BUILTIN_FLOORF ^^^ // vvv !_CCCL_BUILTIN_FLOORF vvv
  return ::floorf(__x);
#endif // !_CCCL_BUILTIN_FLOORF
}

[[nodiscard]] _CCCL_API inline float floorf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_FLOORF)
  return _CCCL_BUILTIN_FLOORF(__x);
#else // ^^^ _CCCL_BUILTIN_FLOORF ^^^ // vvv !_CCCL_BUILTIN_FLOORF vvv
  return ::floorf(__x);
#endif // !_CCCL_BUILTIN_FLOORF
}

[[nodiscard]] _CCCL_API inline double floor(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_FLOOR)
  return _CCCL_BUILTIN_FLOOR(__x);
#else // ^^^ _CCCL_BUILTIN_FLOOR ^^^ // vvv !_CCCL_BUILTIN_FLOOR vvv
  return ::floor(__x);
#endif // !_CCCL_BUILTIN_FLOOR
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double floor(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_FLOORL)
  return _CCCL_BUILTIN_FLOORL(__x);
#  else // ^^^ _CCCL_BUILTIN_FLOORL ^^^ // vvv !_CCCL_BUILTIN_FLOORL vvv
  return ::floorl(__x);
#  endif // !_CCCL_BUILTIN_FLOORL
}

[[nodiscard]] _CCCL_API inline long double floorl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_FLOORL)
  return _CCCL_BUILTIN_FLOORL(__x);
#  else // ^^^ _CCCL_BUILTIN_FLOORL ^^^ // vvv !_CCCL_BUILTIN_FLOORL vvv
  return ::floorl(__x);
#  endif // !_CCCL_BUILTIN_FLOORL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half floor(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hfloor(__x);), (return __float2half(_CUDA_VSTD::floor(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 floor(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hfloor(__x);), (return __float2bfloat16(_CUDA_VSTD::floor(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double floor(_Integer __x) noexcept
{
  return _CUDA_VSTD::floor((double) __x);
}

// llrint

#if _CCCL_CHECK_BUILTIN(builtin_llrint) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LLRINTF(...) __builtin_llrintf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LLRINT(...)  __builtin_llrint(__VA_ARGS__)
#  define _CCCL_BUILTIN_LLRINTL(...) __builtin_llrintl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_llrint)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "llrint"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LLRINTF
#  undef _CCCL_BUILTIN_LLRINT
#  undef _CCCL_BUILTIN_LLRINTL
#endif // _CCCL_CUDA_COMPILER(CLANG)

_CCCL_API inline long long llrint(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLRINTF)
  return _CCCL_BUILTIN_LLRINTF(__x);
#else // ^^^ _CCCL_BUILTIN_LLRINTF ^^^ // vvv !_CCCL_BUILTIN_LLRINTF vvv
  return ::llrintf(__x);
#endif // !_CCCL_BUILTIN_LLRINTF
}

_CCCL_API inline long long llrintf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLRINTF)
  return _CCCL_BUILTIN_LLRINTF(__x);
#else // ^^^ _CCCL_BUILTIN_LLRINTF ^^^ // vvv !_CCCL_BUILTIN_LLRINTF vvv
  return ::llrintf(__x);
#endif // !_CCCL_BUILTIN_LLRINTF
}

_CCCL_API inline long long llrint(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLRINT)
  return _CCCL_BUILTIN_LLRINT(__x);
#else // ^^^ _CCCL_BUILTIN_LLRINT ^^^ // vvv !_CCCL_BUILTIN_LLRINT vvv
  return ::llrint(__x);
#endif // !_CCCL_BUILTIN_LLRINT
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_API inline long long llrint(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LLRINTL)
  return _CCCL_BUILTIN_LLRINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_LLRINTL ^^^ // vvv !_CCCL_BUILTIN_LLRINTL vvv
  return ::llrintl(__x);
#  endif // !_CCCL_BUILTIN_LLRINTL
}

_CCCL_API inline long long llrintl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LLRINTL)
  return _CCCL_BUILTIN_LLRINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_LLRINTL ^^^ // vvv !_CCCL_BUILTIN_LLRINTL vvv
  return ::llrintl(__x);
#  endif // !_CCCL_BUILTIN_LLRINTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline long long llrint(__half __x) noexcept
{
  return _CUDA_VSTD::llrintf(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline long long llrint(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::llrintf(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_API inline long long llrint(_Integer __x) noexcept
{
  return _CUDA_VSTD::llrint((double) __x);
}

// llround

#if _CCCL_CHECK_BUILTIN(builtin_llround) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LLROUNDF(...) __builtin_llroundf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LLROUND(...)  __builtin_llround(__VA_ARGS__)
#  define _CCCL_BUILTIN_LLROUNDL(...) __builtin_llroundl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_llround)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "llround"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LLROUNDF
#  undef _CCCL_BUILTIN_LLROUND
#  undef _CCCL_BUILTIN_LLROUNDL
#endif // _CCCL_CUDA_COMPILER(CLANG)

_CCCL_API inline long long llround(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLROUNDF)
  return _CCCL_BUILTIN_LLROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_LLROUNDF ^^^ // vvv !_CCCL_BUILTIN_LLROUNDF vvv
  return ::llroundf(__x);
#endif // !_CCCL_BUILTIN_LLROUNDF
}

_CCCL_API inline long long llroundf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLROUNDF)
  return _CCCL_BUILTIN_LLROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_LLROUNDF ^^^ // vvv !_CCCL_BUILTIN_LLROUNDF vvv
  return ::llroundf(__x);
#endif // !_CCCL_BUILTIN_LLROUNDF
}

_CCCL_API inline long long llround(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LLROUND)
  return _CCCL_BUILTIN_LLROUND(__x);
#else // ^^^ _CCCL_BUILTIN_LLROUND ^^^ // vvv !_CCCL_BUILTIN_LLROUND vvv
  return ::llround(__x);
#endif // !_CCCL_BUILTIN_LLROUND
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_API inline long long llround(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LLROUNDL)
  return _CCCL_BUILTIN_LLROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_LLROUNDL ^^^ // vvv !_CCCL_BUILTIN_LLROUNDL vvv
  return ::llroundl(__x);
#  endif // !_CCCL_BUILTIN_LLROUNDL
}

_CCCL_API inline long long llroundl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LLROUNDL)
  return _CCCL_BUILTIN_LLROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_LLROUNDL ^^^ // vvv !_CCCL_BUILTIN_LLROUNDL vvv
  return ::llroundl(__x);
#  endif // !_CCCL_BUILTIN_LLROUNDL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline long long llround(__half __x) noexcept
{
  return _CUDA_VSTD::llroundf(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline long long llround(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::llroundf(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_API inline long long llround(_Integer __x) noexcept
{
  return _CUDA_VSTD::llround((double) __x);
}

// lrint

#if _CCCL_CHECK_BUILTIN(builtin_lrint) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LRINTF(...) __builtin_lrintf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LRINT(...)  __builtin_lrint(__VA_ARGS__)
#  define _CCCL_BUILTIN_LRINTL(...) __builtin_lrintl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_lrint)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "lrint"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LRINTF
#  undef _CCCL_BUILTIN_LRINT
#  undef _CCCL_BUILTIN_LRINTL
#endif // _CCCL_CUDA_COMPILER(CLANG)

_CCCL_API inline long lrint(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LRINTF)
  return _CCCL_BUILTIN_LRINTF(__x);
#else // ^^^ _CCCL_BUILTIN_LRINTF ^^^ // vvv !_CCCL_BUILTIN_LRINTF vvv
  return ::lrintf(__x);
#endif // !_CCCL_BUILTIN_LRINTF
}

_CCCL_API inline long lrintf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LRINTF)
  return _CCCL_BUILTIN_LRINTF(__x);
#else // ^^^ _CCCL_BUILTIN_LRINTF ^^^ // vvv !_CCCL_BUILTIN_LRINTF vvv
  return ::lrintf(__x);
#endif // !_CCCL_BUILTIN_LRINTF
}

_CCCL_API inline long lrint(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LRINT)
  return _CCCL_BUILTIN_LRINT(__x);
#else // ^^^ _CCCL_BUILTIN_LRINT ^^^ // vvv !_CCCL_BUILTIN_LRINT vvv
  return ::lrint(__x);
#endif // !_CCCL_BUILTIN_LRINT
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_API inline long lrint(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LRINTL)
  return _CCCL_BUILTIN_LRINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_LRINTL ^^^ // vvv !_CCCL_BUILTIN_LRINTL vvv
  return ::lrintl(__x);
#  endif // !_CCCL_BUILTIN_LRINTL
}

_CCCL_API inline long lrintl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LRINTL)
  return _CCCL_BUILTIN_LRINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_LRINTL ^^^ // vvv !_CCCL_BUILTIN_LRINTL vvv
  return ::lrintl(__x);
#  endif // !_CCCL_BUILTIN_LRINTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline long lrint(__half __x) noexcept
{
  return _CUDA_VSTD::lrintf(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline long lrint(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::lrintf(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_API inline long lrint(_Integer __x) noexcept
{
  return _CUDA_VSTD::lrint((double) __x);
}

// lround

#if _CCCL_CHECK_BUILTIN(builtin_lround) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LROUNDF(...) __builtin_lroundf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LROUND(...)  __builtin_lround(__VA_ARGS__)
#  define _CCCL_BUILTIN_LROUNDL(...) __builtin_lroundl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_lround)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "lround"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LROUNDF
#  undef _CCCL_BUILTIN_LROUND
#  undef _CCCL_BUILTIN_LROUNDL
#endif // _CCCL_CUDA_COMPILER(CLANG)

_CCCL_API inline long lround(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LROUNDF)
  return _CCCL_BUILTIN_LROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_LROUNDF ^^^ // vvv !_CCCL_BUILTIN_LROUNDF vvv
  return ::lroundf(__x);
#endif // !_CCCL_BUILTIN_LROUNDF
}

_CCCL_API inline long lroundf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LROUNDF)
  return _CCCL_BUILTIN_LROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_LROUNDF ^^^ // vvv !_CCCL_BUILTIN_LROUNDF vvv
  return ::lroundf(__x);
#endif // !_CCCL_BUILTIN_LROUNDF
}

_CCCL_API inline long lround(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LROUND)
  return _CCCL_BUILTIN_LROUND(__x);
#else // ^^^ _CCCL_BUILTIN_LROUND ^^^ // vvv !_CCCL_BUILTIN_LROUND vvv
  return ::lround(__x);
#endif // !_CCCL_BUILTIN_LROUND
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_API inline long lround(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LROUNDL)
  return _CCCL_BUILTIN_LROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_LROUNDL ^^^ // vvv !_CCCL_BUILTIN_LROUNDL vvv
  return ::lroundl(__x);
#  endif // !_CCCL_BUILTIN_LROUNDL
}

_CCCL_API inline long lroundl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LROUNDL)
  return _CCCL_BUILTIN_LROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_LROUNDL ^^^ // vvv !_CCCL_BUILTIN_LROUNDL vvv
  return ::lroundl(__x);
#  endif // !_CCCL_BUILTIN_LROUNDL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline long lround(__half __x) noexcept
{
  return _CUDA_VSTD::lroundf(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline long lround(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::lroundf(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_API inline long lround(_Integer __x) noexcept
{
  return _CUDA_VSTD::lround((double) __x);
}

// nearbyint

#if _CCCL_CHECK_BUILTIN(builtin_nearbyint) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_NEARBYINTF(...) __builtin_nearbyintf(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEARBYINT(...)  __builtin_nearbyint(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEARBYINTL(...) __builtin_nearbyintl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nearbyint)

[[nodiscard]] _CCCL_API inline float nearbyint(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_NEARBYINTF)
  return _CCCL_BUILTIN_NEARBYINTF(__x);
#else // ^^^ _CCCL_BUILTIN_NEARBYINTF ^^^ // vvv !_CCCL_BUILTIN_NEARBYINTF vvv
  return ::nearbyintf(__x);
#endif // !_CCCL_BUILTIN_NEARBYINTF
}

[[nodiscard]] _CCCL_API inline float nearbyintf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_NEARBYINTF)
  return _CCCL_BUILTIN_NEARBYINTF(__x);
#else // ^^^ _CCCL_BUILTIN_NEARBYINTF ^^^ // vvv !_CCCL_BUILTIN_NEARBYINTF vvv
  return ::nearbyintf(__x);
#endif // !_CCCL_BUILTIN_NEARBYINTF
}

[[nodiscard]] _CCCL_API inline double nearbyint(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_NEARBYINT)
  return _CCCL_BUILTIN_NEARBYINT(__x);
#else // ^^^ _CCCL_BUILTIN_NEARBYINT ^^^ // vvv !_CCCL_BUILTIN_NEARBYINT vvv
  return ::nearbyint(__x);
#endif // !_CCCL_BUILTIN_NEARBYINT
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double nearbyint(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_NEARBYINTL)
  return _CCCL_BUILTIN_NEARBYINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_NEARBYINTL ^^^ // vvv !_CCCL_BUILTIN_NEARBYINTL vvv
  return ::nearbyintl(__x);
#  endif // !_CCCL_BUILTIN_NEARBYINTL
}

[[nodiscard]] _CCCL_API inline long double nearbyintl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_NEARBYINTL)
  return _CCCL_BUILTIN_NEARBYINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_NEARBYINTL ^^^ // vvv !_CCCL_BUILTIN_NEARBYINTL vvv
  return ::nearbyintl(__x);
#  endif // !_CCCL_BUILTIN_NEARBYINTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half nearbyint(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::nearbyintf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 nearbyint(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::nearbyintf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double nearbyint(_Integer __x) noexcept
{
  return _CUDA_VSTD::nearbyint((double) __x);
}

// nextafter

#if _CCCL_CHECK_BUILTIN(builtin_nextafter) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_NEXTAFTERF(...) __builtin_nextafterf(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEXTAFTER(...)  __builtin_nextafter(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEXTAFTERL(...) __builtin_nextafterl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nextafter)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "nextafter"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_NEXTAFTERF
#  undef _CCCL_BUILTIN_NEXTAFTER
#  undef _CCCL_BUILTIN_NEXTAFTERL
#endif // _CCCL_CUDA_COMPILER(CLANG)

_CCCL_API inline float nextafter(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_NEXTAFTERF)
  return _CCCL_BUILTIN_NEXTAFTERF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_NEXTAFTERF ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTERF vvv
  return ::nextafterf(__x, __y);
#endif // !_CCCL_BUILTIN_NEXTAFTERF
}

_CCCL_API inline float nextafterf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_NEXTAFTERF)
  return _CCCL_BUILTIN_NEXTAFTERF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_NEXTAFTERF ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTERF vvv
  return ::nextafterf(__x, __y);
#endif // !_CCCL_BUILTIN_NEXTAFTERF
}

_CCCL_API inline double nextafter(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_NEXTAFTER)
  return _CCCL_BUILTIN_NEXTAFTER(__x, __y);
#else // ^^^ _CCCL_BUILTIN_NEXTAFTER ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTER vvv
  return ::nextafter(__x, __y);
#endif // !_CCCL_BUILTIN_NEXTAFTER
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_API inline long double nextafter(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTAFTERL)
  return _CCCL_BUILTIN_NEXTAFTERL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTAFTERL ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTERL vvv
  return ::nextafterl(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTAFTERL
}

_CCCL_API inline long double nextafterl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTAFTERL)
  return _CCCL_BUILTIN_NEXTAFTERL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTAFTERL ^^^ // vvv !_CCCL_BUILTIN_NEXTAFTERL vvv
  return ::nextafterl(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTAFTERL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half nextafter(__half __x, __half __y) noexcept
{
  return __float2half(_CUDA_VSTD::nextafterf(__half2float(__x), __half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 nextafter(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::nextafterf(__bfloat162float(__x), __bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2), int> = 0>
_CCCL_API inline __promote_t<_A1, _A2> nextafter(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)), "");
  return _CUDA_VSTD::nextafter(static_cast<__result_type>(__x), static_cast<__result_type>(__y));
}

// nexttoward

#if _CCCL_CHECK_BUILTIN(builtin_nexttoward) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_NEXTTOWARDF(...) __builtin_nexttowardf(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEXTTOWARD(...)  __builtin_nexttoward(__VA_ARGS__)
#  define _CCCL_BUILTIN_NEXTTOWARDL(...) __builtin_nexttowardl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_nexttoward)

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_API inline float nexttoward(float __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARDF)
  return _CCCL_BUILTIN_NEXTTOWARDF(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARDF ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARDF vvv
  return ::nexttowardf(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARDF
}

_CCCL_API inline float nexttowardf(float __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARDF)
  return _CCCL_BUILTIN_NEXTTOWARDF(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARDF ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARDF vvv
  return ::nexttowardf(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARDF
}

_CCCL_API inline double nexttoward(double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARD)
  return _CCCL_BUILTIN_NEXTTOWARD(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARD ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARD vvv
  return ::nexttoward(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARD
}

_CCCL_API inline long double nexttoward(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARDL)
  return _CCCL_BUILTIN_NEXTTOWARDL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARDL ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARDL vvv
  return ::nexttowardl(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARDL
}

_CCCL_API inline long double nexttowardl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_NEXTTOWARDL)
  return _CCCL_BUILTIN_NEXTTOWARDL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_NEXTTOWARDL ^^^ // vvv !_CCCL_BUILTIN_NEXTTOWARDL vvv
  return ::nexttowardl(__x, __y);
#  endif // !_CCCL_BUILTIN_NEXTTOWARDL
}

#  if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half nexttoward(__half __x, long double __y) noexcept
{
  return __float2half(_CUDA_VSTD::nexttowardf(__half2float(__x), __y));
}
#  endif // _LIBCUDACXX_HAS_NVFP16()

#  if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 nexttoward(__nv_bfloat16 __x, long double __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::nexttowardf(__bfloat162float(__x), __y));
}
#  endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_API inline double nexttoward(_Integer __x, long double __y) noexcept
{
  return _CUDA_VSTD::nexttoward((double) __x, __y);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

// rint

#if _CCCL_CHECK_BUILTIN(builtin_rint) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_RINTF(...) __builtin_rintf(__VA_ARGS__)
#  define _CCCL_BUILTIN_RINT(...)  __builtin_rint(__VA_ARGS__)
#  define _CCCL_BUILTIN_RINTL(...) __builtin_rintl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_rint)

[[nodiscard]] _CCCL_API inline float rint(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_RINTF)
  return _CCCL_BUILTIN_RINTF(__x);
#else // ^^^ _CCCL_BUILTIN_RINTF ^^^ // vvv !_CCCL_BUILTIN_RINTF vvv
  return ::rintf(__x);
#endif // !_CCCL_BUILTIN_RINTF
}

[[nodiscard]] _CCCL_API inline float rintf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_RINTF)
  return _CCCL_BUILTIN_RINTF(__x);
#else // ^^^ _CCCL_BUILTIN_RINTF ^^^ // vvv !_CCCL_BUILTIN_RINTF vvv
  return ::rintf(__x);
#endif // !_CCCL_BUILTIN_RINTF
}

[[nodiscard]] _CCCL_API inline double rint(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_RINT)
  return _CCCL_BUILTIN_RINT(__x);
#else // ^^^ _CCCL_BUILTIN_RINT ^^^ // vvv !_CCCL_BUILTIN_RINT vvv
  return ::rint(__x);
#endif // !_CCCL_BUILTIN_RINT
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double rint(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_RINTL)
  return _CCCL_BUILTIN_RINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_RINTL ^^^ // vvv !_CCCL_BUILTIN_RINTL vvv
  return ::rintl(__x);
#  endif // !_CCCL_BUILTIN_RINTL
}

[[nodiscard]] _CCCL_API inline long double rintl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_RINTL)
  return _CCCL_BUILTIN_RINTL(__x);
#  else // ^^^ _CCCL_BUILTIN_RINTL ^^^ // vvv !_CCCL_BUILTIN_RINTL vvv
  return ::rintl(__x);
#  endif // !_CCCL_BUILTIN_RINTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half rint(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hrint(__x);), (return __float2half(_CUDA_VSTD::rint(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 rint(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hrint(__x);), (return __float2bfloat16(_CUDA_VSTD::rint(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double rint(_Integer __x) noexcept
{
  return _CUDA_VSTD::rint((double) __x);
}

// round

#if _CCCL_CHECK_BUILTIN(builtin_round) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ROUNDF(...) __builtin_roundf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ROUND(...)  __builtin_round(__VA_ARGS__)
#  define _CCCL_BUILTIN_ROUNDL(...) __builtin_roundl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_round)

[[nodiscard]] _CCCL_API inline float round(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ROUNDF)
  return _CCCL_BUILTIN_ROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_ROUNDF ^^^ // vvv !_CCCL_BUILTIN_ROUNDF vvv
  return ::roundf(__x);
#endif // !_CCCL_BUILTIN_ROUNDF
}

[[nodiscard]] _CCCL_API inline float roundf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ROUNDF)
  return _CCCL_BUILTIN_ROUNDF(__x);
#else // ^^^ _CCCL_BUILTIN_ROUNDF ^^^ // vvv !_CCCL_BUILTIN_ROUNDF vvv
  return ::roundf(__x);
#endif // !_CCCL_BUILTIN_ROUNDF
}

[[nodiscard]] _CCCL_API inline double round(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ROUND)
  return _CCCL_BUILTIN_ROUND(__x);
#else // ^^^ _CCCL_BUILTIN_ROUND ^^^ // vvv !_CCCL_BUILTIN_ROUND vvv
  return ::round(__x);
#endif // !_CCCL_BUILTIN_ROUND
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double round(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ROUNDL)
  return _CCCL_BUILTIN_ROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_ROUNDL ^^^ // vvv !_CCCL_BUILTIN_ROUNDL vvv
  return ::roundl(__x);
#  endif // !_CCCL_BUILTIN_ROUNDL
}

[[nodiscard]] _CCCL_API inline long double roundl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ROUNDL)
  return _CCCL_BUILTIN_ROUNDL(__x);
#  else // ^^^ _CCCL_BUILTIN_ROUNDL ^^^ // vvv !_CCCL_BUILTIN_ROUNDL vvv
  return ::roundl(__x);
#  endif // !_CCCL_BUILTIN_ROUNDL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half round(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::roundf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 round(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::roundf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double round(_Integer __x) noexcept
{
  return _CUDA_VSTD::round((double) __x);
}

// trunc

#if _CCCL_CHECK_BUILTIN(builtin_trunc) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_TRUNCF(...) __builtin_truncf(__VA_ARGS__)
#  define _CCCL_BUILTIN_TRUNC(...)  __builtin_trunc(__VA_ARGS__)
#  define _CCCL_BUILTIN_TRUNCL(...) __builtin_truncl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_trunc)

[[nodiscard]] _CCCL_API inline float trunc(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TRUNCF)
  return _CCCL_BUILTIN_TRUNCF(__x);
#else // ^^^ _CCCL_BUILTIN_TRUNCF ^^^ // vvv !_CCCL_BUILTIN_TRUNCF vvv
  return ::truncf(__x);
#endif // !_CCCL_BUILTIN_TRUNCF
}

[[nodiscard]] _CCCL_API inline float truncf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TRUNCF)
  return _CCCL_BUILTIN_TRUNCF(__x);
#else // ^^^ _CCCL_BUILTIN_TRUNCF ^^^ // vvv !_CCCL_BUILTIN_TRUNCF vvv
  return ::truncf(__x);
#endif // !_CCCL_BUILTIN_TRUNCF
}

[[nodiscard]] _CCCL_API inline double trunc(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_TRUNC)
  return _CCCL_BUILTIN_TRUNC(__x);
#else // ^^^ _CCCL_BUILTIN_TRUNC ^^^ // vvv !_CCCL_BUILTIN_TRUNC vvv
  return ::trunc(__x);
#endif // !_CCCL_BUILTIN_TRUNC
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double trunc(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TRUNCL)
  return _CCCL_BUILTIN_TRUNCL(__x);
#  else // ^^^ _CCCL_BUILTIN_TRUNCL ^^^ // vvv !_CCCL_BUILTIN_TRUNCL vvv
  return ::truncl(__x);
#  endif // !_CCCL_BUILTIN_TRUNCL
}

[[nodiscard]] _CCCL_API inline long double truncl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TRUNCL)
  return _CCCL_BUILTIN_TRUNCL(__x);
#  else // ^^^ _CCCL_BUILTIN_TRUNCL ^^^ // vvv !_CCCL_BUILTIN_TRUNCL vvv
  return ::truncl(__x);
#  endif // !_CCCL_BUILTIN_TRUNCL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half trunc(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::htrunc(__x);), (return __float2half(_CUDA_VSTD::trunc(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 trunc(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::htrunc(__x);), (return __float2bfloat16(_CUDA_VSTD::trunc(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double trunc(_Integer __x) noexcept
{
  return _CUDA_VSTD::trunc((double) __x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_ROUNDING_FUNCTIONS_H
