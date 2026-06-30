//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_COPYSIGN_H
#define _LIBCUDACXX___CMATH_COPYSIGN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_CHECK_BUILTIN(builtin_copysign) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_COPYSIGNF(...) __builtin_copysignf(__VA_ARGS__)
#  define _CCCL_BUILTIN_COPYSIGN(...)  __builtin_copysign(__VA_ARGS__)
#  define _CCCL_BUILTIN_COPYSIGNL(...) __builtin_copysignl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_copysign)

[[nodiscard]] _CCCL_API inline float copysign(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_COPYSIGNF)
  return _CCCL_BUILTIN_COPYSIGNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_COPYSIGNF ^^^ / vvv !_CCCL_BUILTIN_COPYSIGNF vvv
  return ::copysignf(__x, __y);
#endif // !_CCCL_BUILTIN_COPYSIGNF
}

[[nodiscard]] _CCCL_API inline float copysignf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_COPYSIGNF)
  return _CCCL_BUILTIN_COPYSIGNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_COPYSIGNF ^^^ / vvv !_CCCL_BUILTIN_COPYSIGNF vvv
  return ::copysignf(__x, __y);
#endif // !_CCCL_BUILTIN_COPYSIGNF
}

[[nodiscard]] _CCCL_API inline double copysign(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_COPYSIGN)
  return _CCCL_BUILTIN_COPYSIGN(__x, __y);
#else // ^^^ _CCCL_BUILTIN_COPYSIGN ^^^ / vvv !_CCCL_BUILTIN_COPYSIGN vvv
  return ::copysign(__x, __y);
#endif // !_CCCL_BUILTIN_COPYSIGN
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double copysign(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_COPYSIGNL)
  return _CCCL_BUILTIN_COPYSIGNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_COPYSIGNL ^^^ / vvv !_CCCL_BUILTIN_COPYSIGNL vvv
  return ::copysignl(__x, __y);
#  endif // !_CCCL_BUILTIN_COPYSIGNL
}

[[nodiscard]] _CCCL_API inline long double copysignl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_COPYSIGNL)
  return _CCCL_BUILTIN_COPYSIGNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_COPYSIGNL ^^^ / vvv !_CCCL_BUILTIN_COPYSIGNL vvv
  return ::copysignl(__x, __y);
#  endif // !_CCCL_BUILTIN_COPYSIGNL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __copysign_impl(_Tp __x, [[maybe_unused]] _Tp __y) noexcept
{
  if constexpr (numeric_limits<_Tp>::is_signed)
  {
    const auto __val = (_CUDA_VSTD::__fp_get_storage(__x) & __fp_exp_mant_mask_of_v<_Tp>)
                     | (_CUDA_VSTD::__fp_get_storage(__y) & __fp_sign_mask_of_v<_Tp>);
    return _CUDA_VSTD::__fp_from_storage<_Tp>(static_cast<__fp_storage_of_t<_Tp>>(__val));
  }
  else
  {
    return __x;
  }
}

#if _CCCL_HAS_NVFP16()
[[nodiscard]] _CCCL_API constexpr __half copysign(__half __x, __half __y) noexcept
{
  return _CUDA_VSTD::__copysign_impl(__x, __y);
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
[[nodiscard]] _CCCL_API constexpr __nv_bfloat16 copysign(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return _CUDA_VSTD::__copysign_impl(__x, __y);
}
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
[[nodiscard]] _CCCL_API constexpr __nv_fp8_e4m3 copysign(__nv_fp8_e4m3 __x, __nv_fp8_e4m3 __y) noexcept
{
  return _CUDA_VSTD::__copysign_impl(__x, __y);
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
[[nodiscard]] _CCCL_API constexpr __nv_fp8_e5m2 copysign(__nv_fp8_e5m2 __x, __nv_fp8_e5m2 __y) noexcept
{
  return _CUDA_VSTD::__copysign_impl(__x, __y);
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
[[nodiscard]] _CCCL_API constexpr __nv_fp8_e8m0 copysign(__nv_fp8_e8m0 __x, __nv_fp8_e8m0 __y) noexcept
{
  return _CUDA_VSTD::__copysign_impl(__x, __y);
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
[[nodiscard]] _CCCL_API constexpr __nv_fp6_e2m3 copysign(__nv_fp6_e2m3 __x, __nv_fp6_e2m3 __y) noexcept
{
  return _CUDA_VSTD::__copysign_impl(__x, __y);
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
[[nodiscard]] _CCCL_API constexpr __nv_fp6_e3m2 copysign(__nv_fp6_e3m2 __x, __nv_fp6_e3m2 __y) noexcept
{
  return _CUDA_VSTD::__copysign_impl(__x, __y);
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
[[nodiscard]] _CCCL_API constexpr __nv_fp4_e2m1 copysign(__nv_fp4_e2m1 __x, __nv_fp4_e2m1 __y) noexcept
{
  return _CUDA_VSTD::__copysign_impl(__x, __y);
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
[[nodiscard]] _CCCL_API constexpr double copysign(_Tp __x, [[maybe_unused]] _Tp __y) noexcept
{
  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    const auto __x_dbl = static_cast<double>(__x);
    if (__y < 0)
    {
      return (__x < 0) ? __x_dbl : -__x_dbl;
    }
    else
    {
      return (__x < 0) ? -__x_dbl : __x_dbl;
    }
  }
  else
  {
    return static_cast<double>(__x);
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_COPYSIGN_H
