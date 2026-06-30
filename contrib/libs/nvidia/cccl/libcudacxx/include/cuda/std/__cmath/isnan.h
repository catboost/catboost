//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ISNAN_H
#define _LIBCUDACXX___CMATH_ISNAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integral.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <math.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_CHECK_BUILTIN(builtin_isnan) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISNAN(...) __builtin_isnan(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isnan)

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool __isnan_impl(_Tp __x) noexcept
{
  static_assert(_CCCL_TRAIT(is_floating_point, _Tp), "Only standard floating-point types are supported");
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return ::isnan(__x);
  }
  return __x != __x;
}

[[nodiscard]] _CCCL_API constexpr bool isnan(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#else // ^^^ _CCCL_BUILTIN_ISNAN ^^^ / vvv !_CCCL_BUILTIN_ISNAN vvv
  return _CUDA_VSTD::__isnan_impl(__x);
#endif // ^^^ !_CCCL_BUILTIN_ISNAN ^^^
}

[[nodiscard]] _CCCL_API constexpr bool isnan(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#else // ^^^ _CCCL_BUILTIN_ISNAN ^^^ / vvv !_CCCL_BUILTIN_ISNAN vvv
  return _CUDA_VSTD::__isnan_impl(__x);
#endif // ^^^ !_CCCL_BUILTIN_ISNAN ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API constexpr bool isnan(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#  else // ^^^ _CCCL_BUILTIN_ISNAN ^^^ / vvv !_CCCL_BUILTIN_ISNAN vvv
  return _CUDA_VSTD::__isnan_impl(__x);
#  endif // ^^^ !_CCCL_BUILTIN_ISNAN ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _CCCL_HAS_NVFP16()
[[nodiscard]] _CCCL_API constexpr bool isnan(__half __x) noexcept
{
#  if _LIBCUDACXX_HAS_NVFP16()
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return ::__hisnan(__x);
  }
#  endif // _LIBCUDACXX_HAS_NVFP16()

  const auto __storage = _CUDA_VSTD::__fp_get_storage(__x);
  return ((__storage & __fp_exp_mask_of_v<__half>) == __fp_exp_mask_of_v<__half>)
      && (__storage & __fp_mant_mask_of_v<__half>);
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
[[nodiscard]] _CCCL_API constexpr bool isnan(__nv_bfloat16 __x) noexcept
{
#  if _LIBCUDACXX_HAS_NVFP16()
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    return ::__hisnan(__x);
  }
#  endif // _LIBCUDACXX_HAS_NVFP16()

  const auto __storage = _CUDA_VSTD::__fp_get_storage(__x);
  return ((__storage & __fp_exp_mask_of_v<__nv_bfloat16>) == __fp_exp_mask_of_v<__nv_bfloat16>)
      && (__storage & __fp_mant_mask_of_v<__nv_bfloat16>);
}
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
[[nodiscard]] _CCCL_API constexpr bool isnan(__nv_fp8_e4m3 __x) noexcept
{
  return (__x.__x & __fp_exp_mant_mask_of_v<__nv_fp8_e4m3>) == __fp_exp_mant_mask_of_v<__nv_fp8_e4m3>;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
[[nodiscard]] _CCCL_API constexpr bool isnan(__nv_fp8_e5m2 __x) noexcept
{
  return ((__x.__x & __fp_exp_mask_of_v<__nv_fp8_e5m2>) == __fp_exp_mask_of_v<__nv_fp8_e5m2>)
      && (__x.__x & __fp_mant_mask_of_v<__nv_fp8_e5m2>);
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
[[nodiscard]] _CCCL_API constexpr bool isnan(__nv_fp8_e8m0 __x) noexcept
{
  return (__x.__x & __fp_exp_mask_of_v<__nv_fp8_e8m0>) == __fp_exp_mask_of_v<__nv_fp8_e8m0>;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
[[nodiscard]] _CCCL_API constexpr bool isnan(__nv_fp6_e2m3) noexcept
{
  return false;
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
[[nodiscard]] _CCCL_API constexpr bool isnan(__nv_fp6_e3m2) noexcept
{
  return false;
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
[[nodiscard]] _CCCL_API constexpr bool isnan(__nv_fp4_e2m1) noexcept
{
  return false;
}
#endif // _CCCL_HAS_NVFP4_E2M1()

#if _CCCL_HAS_FLOAT128()
[[nodiscard]] _CCCL_API constexpr bool isnan(__float128 __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#  else // ^^^ _CCCL_BUILTIN_ISNAN ^^^ / vvv !_CCCL_BUILTIN_ISNAN vvv
  return _CUDA_VSTD::__isnan_impl(__x);
#  endif // ^^^ !_CCCL_BUILTIN_ISNAN ^^^
}
#endif // _CCCL_HAS_FLOAT128()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
[[nodiscard]] _CCCL_API constexpr bool isnan(_Tp) noexcept
{
  return false;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_ISNAN_H
