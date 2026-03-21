//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_FORMAT_H
#define _LIBCUDACXX___FLOATING_POINT_FORMAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__fwd/fp.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cfloat>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class __fp_format
{
  __binary16, // IEEE 754 binary16
  __binary32, // IEEE 754 binary32
  __binary64, // IEEE 754 binary64
  __binary128, // IEEE 754 binary128
  __bfloat16, // Google's 16-bit brain float
  __fp80_x86, // x86 80-bit extended precision
  __fp8_nv_e4m3, // NVIDIA's __nv_fp8_e4m3
  __fp8_nv_e5m2, // NVIDIA's __nv_fp8_e5m2
  __fp8_nv_e8m0, // NVIDIA's __nv_fp8_e8m0
  __fp6_nv_e2m3, // NVIDIA's __nv_fp6_e2m3
  __fp6_nv_e3m2, // NVIDIA's __nv_fp6_e3m2
  __fp4_nv_e2m1, // NVIDIA's __nv_fp4_e2m1

  __invalid,
};

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __fp_format __fp_format_of_v_impl() noexcept
{
  if constexpr (_CCCL_TRAIT(is_same, _Tp, float))
  {
    return __fp_format::__binary32;
  }
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, double))
  {
    return __fp_format::__binary64;
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, long double))
  {
    if (LDBL_MIN_EXP == -1021 && LDBL_MAX_EXP == 1024 && LDBL_MANT_DIG == 53)
    {
      return __fp_format::__binary64;
    }
    else if (LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && LDBL_MANT_DIG == 64 && sizeof(long double) == 16)
    {
      return __fp_format::__fp80_x86;
    }
    else if (LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && LDBL_MANT_DIG == 113)
    {
      return __fp_format::__binary128;
    }
    else // Unsupported long double format
    {
      return __fp_format::__invalid;
    }
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __half))
  {
    return __fp_format::__binary16;
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_bfloat16))
  {
    return __fp_format::__bfloat16;
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e4m3))
  {
    return __fp_format::__fp8_nv_e4m3;
  }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e5m2))
  {
    return __fp_format::__fp8_nv_e5m2;
  }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp8_e8m0))
  {
    return __fp_format::__fp8_nv_e8m0;
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e2m3))
  {
    return __fp_format::__fp6_nv_e2m3;
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp6_e3m2))
  {
    return __fp_format::__fp6_nv_e3m2;
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __nv_fp4_e2m1))
  {
    return __fp_format::__fp4_nv_e2m1;
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  else if constexpr (_CCCL_TRAIT(is_same, _Tp, __float128))
  {
    return __fp_format::__binary128;
  }
#endif // _CCCL_HAS_FLOAT128()
  else
  {
    return __fp_format::__invalid;
  }
}

template <class _Tp>
inline constexpr __fp_format __fp_format_of_v = _CUDA_VSTD::__fp_format_of_v_impl<_Tp>();

template <class _Tp>
inline constexpr __fp_format __fp_format_of_v<const _Tp> = __fp_format_of_v<_Tp>;

template <class _Tp>
inline constexpr __fp_format __fp_format_of_v<volatile _Tp> = __fp_format_of_v<_Tp>;

template <class _Tp>
inline constexpr __fp_format __fp_format_of_v<const volatile _Tp> = __fp_format_of_v<_Tp>;

template <__fp_format _Fmt>
inline constexpr __fp_format __fp_format_of_v<__cccl_fp<_Fmt>> = _Fmt;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_FORMAT_H
