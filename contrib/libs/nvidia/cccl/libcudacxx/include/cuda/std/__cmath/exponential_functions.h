// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MATH_EXPONENTIAL_FUNCTIONS_H
#define _LIBCUDACXX___MATH_EXPONENTIAL_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/promote.h>
#include <cuda/std/cstdint>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// exp

#if _CCCL_CHECK_BUILTIN(builtin_exp) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_EXPF(...) __builtin_expf(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXP(...)  __builtin_exp(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXPL(...) __builtin_expl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_exp)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "expf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_EXPF
#  undef _CCCL_BUILTIN_EXP
#  undef _CCCL_BUILTIN_EXPL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float exp(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPF)
  return _CCCL_BUILTIN_EXPF(__x);
#else // ^^^ _CCCL_BUILTIN_EXPF ^^^ // vvv !_CCCL_BUILTIN_EXPF vvv
  return ::expf(__x);
#endif // !_CCCL_BUILTIN_EXPF
}

[[nodiscard]] _CCCL_API inline float expf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPF)
  return _CCCL_BUILTIN_EXPF(__x);
#else // ^^^ _CCCL_BUILTIN_EXPF ^^^ // vvv !_CCCL_BUILTIN_EXPF vvv
  return ::expf(__x);
#endif // !_CCCL_BUILTIN_EXPF
}

[[nodiscard]] _CCCL_API inline double exp(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXP)
  return _CCCL_BUILTIN_EXP(__x);
#else // ^^^ _CCCL_BUILTIN_EXP ^^^ // vvv !_CCCL_BUILTIN_EXP vvv
  return ::exp(__x);
#endif // !_CCCL_BUILTIN_EXP
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double exp(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXPL)
  return _CCCL_BUILTIN_EXPL(__x);
#  else // ^^^ _CCCL_BUILTIN_EXPL ^^^ // vvv !_CCCL_BUILTIN_EXPL vvv
  return ::expl(__x);
#  endif // !_CCCL_BUILTIN_EXPL
}

[[nodiscard]] _CCCL_API inline long double expl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXPL)
  return _CCCL_BUILTIN_EXPL(__x);
#  else // ^^^ _CCCL_BUILTIN_EXPL ^^^ // vvv !_CCCL_BUILTIN_EXPL vvv
  return ::expl(__x);
#  endif // !_CCCL_BUILTIN_EXPL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half exp(__half __x) noexcept
{
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (return ::hexp(__x);), ({
                        float __xf            = __half2float(__x);
                        __xf                  = ::expf(__xf);
                        __half_raw __ret_repr = ::__float2half_rn(__xf);

                        uint16_t __repr = _CUDA_VSTD::__fp_get_storage(__x);
                        switch (__repr)
                        {
                          case 8057:
                          case 9679:
                            __ret_repr.x -= 1;
                            break;

                          default:;
                        }

                        return __ret_repr;
                      }))
  }
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 exp(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hexp(__x);), (return __float2bfloat16(_CUDA_VSTD::expf(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double exp(_Integer __x) noexcept
{
  return _CUDA_VSTD::exp((double) __x);
}

// frexp

#if _CCCL_CHECK_BUILTIN(builtin_frexp) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FREXPF(...) __builtin_frexpf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FREXP(...)  __builtin_frexp(__VA_ARGS__)
#  define _CCCL_BUILTIN_FREXPL(...) __builtin_frexpl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_frexp)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "frexp"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_FREXPF
#  undef _CCCL_BUILTIN_FREXP
#  undef _CCCL_BUILTIN_FREXPL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float frexp(float __x, int* __e) noexcept
{
#if defined(_CCCL_BUILTIN_FREXPF)
  return _CCCL_BUILTIN_FREXPF(__x, __e);
#else // ^^^ _CCCL_BUILTIN_FREXPF ^^^ // vvv !_CCCL_BUILTIN_FREXPF vvv
  return ::frexpf(__x, __e);
#endif // !_CCCL_BUILTIN_FREXPF
}

[[nodiscard]] _CCCL_API inline float frexpf(float __x, int* __e) noexcept
{
#if defined(_CCCL_BUILTIN_FREXPF)
  return _CCCL_BUILTIN_FREXPF(__x, __e);
#else // ^^^ _CCCL_BUILTIN_FREXPF ^^^ // vvv !_CCCL_BUILTIN_FREXPF vvv
  return ::frexpf(__x, __e);
#endif // !_CCCL_BUILTIN_FREXPF
}

[[nodiscard]] _CCCL_API inline double frexp(double __x, int* __e) noexcept
{
#if defined(_CCCL_BUILTIN_FREXP)
  return _CCCL_BUILTIN_FREXP(__x, __e);
#else // ^^^ _CCCL_BUILTIN_FREXP ^^^ // vvv !_CCCL_BUILTIN_FREXP vvv
  return ::frexp(__x, __e);
#endif // !_CCCL_BUILTIN_FREXP
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double frexp(long double __x, int* __e) noexcept
{
#  if defined(_CCCL_BUILTIN_FREXPL)
  return _CCCL_BUILTIN_FREXPL(__x, __e);
#  else // ^^^ _CCCL_BUILTIN_FREXPL ^^^ // vvv !_CCCL_BUILTIN_FREXPL vvv
  return ::frexpl(__x, __e);
#  endif // !_CCCL_BUILTIN_FREXPL
}

[[nodiscard]] _CCCL_API inline long double frexpl(long double __x, int* __e) noexcept
{
#  if defined(_CCCL_BUILTIN_FREXPL)
  return _CCCL_BUILTIN_FREXPL(__x, __e);
#  else // ^^^ _CCCL_BUILTIN_FREXPL ^^^ // vvv !_CCCL_BUILTIN_FREXPL vvv
  return ::frexpl(__x, __e);
#  endif // !_CCCL_BUILTIN_FREXPL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half frexp(__half __x, int* __e) noexcept
{
  return __float2half(_CUDA_VSTD::frexpf(__half2float(__x), __e));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 frexp(__nv_bfloat16 __x, int* __e) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::frexpf(__bfloat162float(__x), __e));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double frexp(_Integer __x, int* __e) noexcept
{
  return _CUDA_VSTD::frexp((double) __x, __e);
}

// ldexp

#if _CCCL_CHECK_BUILTIN(builtin_ldexp) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LDEXPF(...) __builtin_ldexpf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LDEXP(...)  __builtin_ldexp(__VA_ARGS__)
#  define _CCCL_BUILTIN_LDEXPL(...) __builtin_ldexpl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ldexp)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "ldexp"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LDEXPF
#  undef _CCCL_BUILTIN_LDEXP
#  undef _CCCL_BUILTIN_LDEXPL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float ldexp(float __x, int __e) noexcept
{
#if defined(_CCCL_BUILTIN_LDEXPF)
  return _CCCL_BUILTIN_LDEXPF(__x, __e);
#else // ^^^ _CCCL_BUILTIN_LDEXPF ^^^ // vvv !_CCCL_BUILTIN_LDEXPF vvv
  return ::ldexpf(__x, __e);
#endif // !_CCCL_BUILTIN_LDEXPF
}

[[nodiscard]] _CCCL_API inline float ldexpf(float __x, int __e) noexcept
{
#if defined(_CCCL_BUILTIN_LDEXPF)
  return _CCCL_BUILTIN_LDEXPF(__x, __e);
#else // ^^^ _CCCL_BUILTIN_LDEXPF ^^^ // vvv !_CCCL_BUILTIN_LDEXPF vvv
  return ::ldexpf(__x, __e);
#endif // !_CCCL_BUILTIN_LDEXPF
}

[[nodiscard]] _CCCL_API inline double ldexp(double __x, int __e) noexcept
{
#if defined(_CCCL_BUILTIN_LDEXP)
  return _CCCL_BUILTIN_LDEXP(__x, __e);
#else // ^^^ _CCCL_BUILTIN_LDEXP ^^^ // vvv !_CCCL_BUILTIN_LDEXP vvv
  return ::ldexp(__x, __e);
#endif // !_CCCL_BUILTIN_LDEXP
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double ldexp(long double __x, int __e) noexcept
{
#  if defined(_CCCL_BUILTIN_LDEXPL)
  return _CCCL_BUILTIN_LDEXPL(__x, __e);
#  else // ^^^ _CCCL_BUILTIN_LDEXPL ^^^ // vvv !_CCCL_BUILTIN_LDEXPL vvv
  return ::ldexpl(__x, __e);
#  endif // !_CCCL_BUILTIN_LDEXPL
}

[[nodiscard]] _CCCL_API inline long double ldexpl(long double __x, int __e) noexcept
{
#  if defined(_CCCL_BUILTIN_LDEXPL)
  return _CCCL_BUILTIN_LDEXPL(__x, __e);
#  else // ^^^ _CCCL_BUILTIN_LDEXPL ^^^ // vvv !_CCCL_BUILTIN_LDEXPL vvv
  return ::ldexpl(__x, __e);
#  endif // !_CCCL_BUILTIN_LDEXPL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half ldexp(__half __x, int __e) noexcept
{
  return __float2half(_CUDA_VSTD::ldexpf(__half2float(__x), __e));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 ldexp(__nv_bfloat16 __x, int __e) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::ldexpf(__bfloat162float(__x), __e));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double ldexp(_Integer __x, int __e) noexcept
{
  return _CUDA_VSTD::ldexp((double) __x, __e);
}

// exp2

#if _CCCL_CHECK_BUILTIN(builtin_exp2) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_EXP2F(...) __builtin_exp2f(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXP2(...)  __builtin_exp2(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXP2L(...) __builtin_exp2l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_exp2)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "exp2"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_EXP2F
#  undef _CCCL_BUILTIN_EXP2
#  undef _CCCL_BUILTIN_EXP2L
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float exp2(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXP2F)
  return _CCCL_BUILTIN_EXP2F(__x);
#else // ^^^ _CCCL_BUILTIN_EXP2F ^^^ // vvv !_CCCL_BUILTIN_EXP2F vvv
  return ::exp2f(__x);
#endif // !_CCCL_BUILTIN_EXP2F
}

[[nodiscard]] _CCCL_API inline float exp2f(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXP2F)
  return _CCCL_BUILTIN_EXP2F(__x);
#else // ^^^ _CCCL_BUILTIN_EXP2F ^^^ // vvv !_CCCL_BUILTIN_EXP2F vvv
  return ::exp2f(__x);
#endif // !_CCCL_BUILTIN_EXP2F
}

[[nodiscard]] _CCCL_API inline double exp2(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXP2)
  return _CCCL_BUILTIN_EXP2(__x);
#else // ^^^ _CCCL_BUILTIN_EXP2 ^^^ // vvv !_CCCL_BUILTIN_EXP2 vvv
  return ::exp2(__x);
#endif // !_CCCL_BUILTIN_EXP2
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double exp2(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXP2L)
  return _CCCL_BUILTIN_EXP2L(__x);
#  else // ^^^ _CCCL_BUILTIN_EXP2L ^^^ // vvv !_CCCL_BUILTIN_EXP2L vvv
  return ::exp2l(__x);
#  endif // !_CCCL_BUILTIN_EXP2L
}

[[nodiscard]] _CCCL_API inline long double exp2l(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXP2L)
  return _CCCL_BUILTIN_EXP2L(__x);
#  else // ^^^ _CCCL_BUILTIN_EXP2L ^^^ // vvv !_CCCL_BUILTIN_EXP2L vvv
  return ::exp2l(__x);
#  endif // !_CCCL_BUILTIN_EXP2L
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half exp2(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hexp2(__x);), (return __float2half(_CUDA_VSTD::exp2f(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 exp2(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hexp2(__x);), (return __float2bfloat16(_CUDA_VSTD::exp2f(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double exp2(_Integer __x) noexcept
{
  return _CUDA_VSTD::exp2((double) __x);
}

// expm1

#if _CCCL_CHECK_BUILTIN(builtin_expm1) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_EXPM1F(...) __builtin_expm1f(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXPM1(...)  __builtin_expm1(__VA_ARGS__)
#  define _CCCL_BUILTIN_EXPM1L(...) __builtin_expm1l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_expm1)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "expm1"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_EXPM1F
#  undef _CCCL_BUILTIN_EXPM1
#  undef _CCCL_BUILTIN_EXPM1L
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float expm1(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPM1F)
  return _CCCL_BUILTIN_EXPM1F(__x);
#else // ^^^ _CCCL_BUILTIN_EXPM1F ^^^ // vvv !_CCCL_BUILTIN_EXPM1F vvv
  return ::expm1f(__x);
#endif // !_CCCL_BUILTIN_EXPM1F
}

[[nodiscard]] _CCCL_API inline float expm1f(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPM1F)
  return _CCCL_BUILTIN_EXPM1F(__x);
#else // ^^^ _CCCL_BUILTIN_EXPM1F ^^^ // vvv !_CCCL_BUILTIN_EXPM1F vvv
  return ::expm1f(__x);
#endif // !_CCCL_BUILTIN_EXPM1F
}

[[nodiscard]] _CCCL_API inline double expm1(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPM1)
  return _CCCL_BUILTIN_EXPM1(__x);
#else // ^^^ _CCCL_BUILTIN_EXPM1 ^^^ // vvv !_CCCL_BUILTIN_EXPM1 vvv
  return ::expm1(__x);
#endif // !_CCCL_BUILTIN_EXPM1
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double expm1(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXPM1L)
  return _CCCL_BUILTIN_EXPM1L(__x);
#  else // ^^^ _CCCL_BUILTIN_EXPM1L ^^^ // vvv !_CCCL_BUILTIN_EXPM1L vvv
  return ::expm1l(__x);
#  endif // !_CCCL_BUILTIN_EXPM1L
}

[[nodiscard]] _CCCL_API inline long double expm1l(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXPM1L)
  return _CCCL_BUILTIN_EXPM1L(__x);
#  else // ^^^ _CCCL_BUILTIN_EXPM1L ^^^ // vvv !_CCCL_BUILTIN_EXPM1L vvv
  return ::expm1l(__x);
#  endif // !_CCCL_BUILTIN_EXPM1L
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half expm1(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::expm1f(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 expm1(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::expm1f(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double expm1(_Integer __x) noexcept
{
  return _CUDA_VSTD::expm1((double) __x);
}

// scalbln

#if _CCCL_CHECK_BUILTIN(builtin_scalbln) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SCALBLNF(...) __builtin_scalblnf(__VA_ARGS__)
#  define _CCCL_BUILTIN_SCALBLN(...)  __builtin_scalbln(__VA_ARGS__)
#  define _CCCL_BUILTIN_SCALBLNL(...) __builtin_scalblnl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_scalbln)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "scalblnf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_SCALBLNF
#  undef _CCCL_BUILTIN_SCALBLN
#  undef _CCCL_BUILTIN_SCALBLNL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float scalbln(float __x, long __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBLNF)
  return _CCCL_BUILTIN_SCALBLNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBLNF ^^^ // vvv !_CCCL_BUILTIN_SCALBLNF vvv
  return ::scalblnf(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBLNF
}

[[nodiscard]] _CCCL_API inline float scalblnf(float __x, long __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBLNF)
  return _CCCL_BUILTIN_SCALBLNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBLNF ^^^ // vvv !_CCCL_BUILTIN_SCALBLNF vvv
  return ::scalblnf(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBLNF
}

[[nodiscard]] _CCCL_API inline double scalbln(double __x, long __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBLN)
  return _CCCL_BUILTIN_SCALBLN(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBLN ^^^ // vvv !_CCCL_BUILTIN_SCALBLN vvv
  return ::scalbln(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBLN
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double scalbln(long double __x, long __y) noexcept
{
#  if defined(_CCCL_BUILTIN_SCALBLNL)
  return _CCCL_BUILTIN_SCALBLNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_SCALBLNL ^^^ // vvv !_CCCL_BUILTIN_SCALBLNL vvv
  return ::scalblnl(__x, __y);
#  endif // !_CCCL_BUILTIN_SCALBLNL
}

[[nodiscard]] _CCCL_API inline long double scalblnl(long double __x, long __y) noexcept
{
#  if defined(_CCCL_BUILTIN_SCALBLNL)
  return _CCCL_BUILTIN_SCALBLNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_SCALBLNL ^^^ // vvv !_CCCL_BUILTIN_SCALBLNL vvv
  return ::scalblnl(__x, __y);
#  endif // !_CCCL_BUILTIN_SCALBLNL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half scalbln(__half __x, long __y) noexcept
{
  return __float2half(_CUDA_VSTD::scalblnf(__half2float(__x), __y));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 scalbln(__nv_bfloat16 __x, long __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::scalblnf(__bfloat162float(__x), __y));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double scalbln(_Integer __x, long __y) noexcept
{
  return _CUDA_VSTD::scalbln((double) __x, __y);
}

// scalbn

#if _CCCL_CHECK_BUILTIN(builtin_scalbn) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SCALBNF(...) __builtin_scalbnf(__VA_ARGS__)
#  define _CCCL_BUILTIN_SCALBN(...)  __builtin_scalbn(__VA_ARGS__)
#  define _CCCL_BUILTIN_SCALBNL(...) __builtin_scalbnl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_scalbn)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "scalbnf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_SCALBNF
#  undef _CCCL_BUILTIN_SCALBN
#  undef _CCCL_BUILTIN_SCALBNL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float scalbn(float __x, int __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBNF)
  return _CCCL_BUILTIN_SCALBNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBNF ^^^ // vvv !_CCCL_BUILTIN_SCALBNF vvv
  return ::scalbnf(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBNF
}

[[nodiscard]] _CCCL_API inline float scalbnf(float __x, int __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBNF)
  return _CCCL_BUILTIN_SCALBNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBNF ^^^ // vvv !_CCCL_BUILTIN_SCALBNF vvv
  return ::scalbnf(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBNF
}

[[nodiscard]] _CCCL_API inline double scalbn(double __x, int __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBN)
  return _CCCL_BUILTIN_SCALBN(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBN ^^^ // vvv !_CCCL_BUILTIN_SCALBN vvv
  return ::scalbn(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBN
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double scalbn(long double __x, int __y) noexcept
{
#  if defined(_CCCL_BUILTIN_SCALBNL)
  return _CCCL_BUILTIN_SCALBNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_SCALBNL ^^^ // vvv !_CCCL_BUILTIN_SCALBNL vvv
  return ::scalbnl(__x, __y);
#  endif // !_CCCL_BUILTIN_SCALBNL
}

[[nodiscard]] _CCCL_API inline long double scalbnl(long double __x, int __y) noexcept
{
#  if defined(_CCCL_BUILTIN_SCALBNL)
  return _CCCL_BUILTIN_SCALBNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_SCALBNL ^^^ // vvv !_CCCL_BUILTIN_SCALBNL vvv
  return ::scalbnl(__x, __y);
#  endif // !_CCCL_BUILTIN_SCALBNL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half scalbn(__half __x, int __y) noexcept
{
  return __float2half(_CUDA_VSTD::scalbnf(__half2float(__x), __y));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 scalbn(__nv_bfloat16 __x, int __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::scalbnf(__bfloat162float(__x), __y));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
[[nodiscard]] _CCCL_API inline double scalbn(_Integer __x, int __y) noexcept
{
  return _CUDA_VSTD::scalbn((double) __x, __y);
}

// pow

#if _CCCL_CHECK_BUILTIN(builtin_pow) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_POWF(...) __builtin_powf(__VA_ARGS__)
#  define _CCCL_BUILTIN_POW(...)  __builtin_pow(__VA_ARGS__)
#  define _CCCL_BUILTIN_POWL(...) __builtin_powl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_pow)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "pow"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_POWF
#  undef _CCCL_BUILTIN_POW
#  undef _CCCL_BUILTIN_POWL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float pow(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_POWF)
  return _CCCL_BUILTIN_POWF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_POWF ^^^ // vvv !_CCCL_BUILTIN_POWF vvv
  return ::powf(__x, __y);
#endif // !_CCCL_BUILTIN_POWF
}

[[nodiscard]] _CCCL_API inline float powf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_POWF)
  return _CCCL_BUILTIN_POWF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_POWF ^^^ // vvv !_CCCL_BUILTIN_POWF vvv
  return ::powf(__x, __y);
#endif // !_CCCL_BUILTIN_POWF
}

[[nodiscard]] _CCCL_API inline double pow(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_POW)
  return _CCCL_BUILTIN_POW(__x, __y);
#else // ^^^ _CCCL_BUILTIN_POW ^^^ // vvv !_CCCL_BUILTIN_POW vvv
  return ::pow(__x, __y);
#endif // !_CCCL_BUILTIN_POW
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double pow(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_POWL)
  return _CCCL_BUILTIN_POWL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_POWL ^^^ // vvv !_CCCL_BUILTIN_POWL vvv
  return ::powl(__x, __y);
#  endif // !_CCCL_BUILTIN_POWL
}

[[nodiscard]] _CCCL_API inline long double powl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_POWL)
  return _CCCL_BUILTIN_POWL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_POWL ^^^ // vvv !_CCCL_BUILTIN_POWL vvv
  return ::powl(__x, __y);
#  endif // !_CCCL_BUILTIN_POWL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half pow(__half __x, __half __y) noexcept
{
  return __float2half(_CUDA_VSTD::powf(__half2float(__x), __half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 pow(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::powf(__bfloat162float(__x), __bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2> pow(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)), "");
  return _CUDA_VSTD::pow((__result_type) __x, (__result_type) __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MATH_EXPONENTIAL_FUNCTIONS_H
