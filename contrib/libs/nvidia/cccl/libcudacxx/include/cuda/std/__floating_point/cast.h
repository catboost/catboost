//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_CAST_H
#define _LIBCUDACXX___FLOATING_POINT_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _To, class _From>
[[nodiscard]] _CCCL_API inline _To __fp_cast(_From __v) noexcept
{
  if constexpr (_CCCL_TRAIT(is_same, _From, float))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return __v;
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return static_cast<double>(__v);
    }
#if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return static_cast<long double>(__v);
    }
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return ::__float2half(__v);
    }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return ::__float2bfloat16(__v);
    }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp8_e4m3>(::__nv_cvt_float_to_fp8(__v, __NV_NOSAT, __NV_E4M3));
    }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp8_e5m2>(::__nv_cvt_float_to_fp8(__v, __NV_NOSAT, __NV_E5M2));
    }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp8_e8m0>(::__nv_cvt_float_to_e8m0(__v, __NV_NOSAT, ::cudaRoundZero));
    }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp6_e2m3>(::__nv_cvt_float_to_fp6(__v, __NV_E2M3, ::cudaRoundNearest));
    }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp6_e3m2>(::__nv_cvt_float_to_fp6(__v, __NV_E3M2, ::cudaRoundNearest));
    }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp4_e2m1>(::__nv_cvt_float_to_fp4(__v, __NV_E2M1, ::cudaRoundNearest));
    }
#endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
  else if constexpr (_CCCL_TRAIT(is_same, _From, double))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return static_cast<float>(__v);
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return __v;
    }
#if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return static_cast<long double>(__v);
    }
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return ::__double2half(__v);
    }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return ::__double2bfloat16(__v);
    }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp8_e4m3>(::__nv_cvt_double_to_fp8(__v, __NV_NOSAT, __NV_E4M3));
    }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp8_e5m2>(::__nv_cvt_double_to_fp8(__v, __NV_NOSAT, __NV_E5M2));
    }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp8_e8m0>(::__nv_cvt_double_to_e8m0(__v, __NV_NOSAT, ::cudaRoundZero));
    }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp6_e2m3>(
        ::__nv_cvt_double_to_fp6(__v, __NV_E2M3, ::cudaRoundNearest));
    }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp6_e3m2>(
        ::__nv_cvt_double_to_fp6(__v, __NV_E3M2, ::cudaRoundNearest));
    }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp4_e2m1>(
        ::__nv_cvt_double_to_fp4(__v, __NV_E2M1, ::cudaRoundNearest));
    }
#endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (_CCCL_TRAIT(is_same, _From, long double))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return static_cast<float>(__v);
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return static_cast<double>(__v);
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return __v;
    }
#  if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return _CUDA_VSTD::__fp_cast<__half>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return _CUDA_VSTD::__fp_cast<__nv_bfloat16>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e4m3>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e5m2>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e8m0>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e2m3>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e3m2>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp4_e2m1>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  else if constexpr (_CCCL_TRAIT(is_same, _From, __half))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return ::__half2float(__v);
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return _CUDA_VSTD::__fp_cast<double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return _CUDA_VSTD::__fp_cast<long double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return __v;
    }
#  if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return _CUDA_VSTD::__fp_cast<__nv_bfloat16>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp8_e4m3>(::__nv_cvt_halfraw_to_fp8(__v, __NV_NOSAT, __NV_E4M3));
    }
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp8_e5m2>(::__nv_cvt_halfraw_to_fp8(__v, __NV_NOSAT, __NV_E5M2));
    }
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e8m0>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp6_e2m3>(::__nv_cvt_halfraw_to_fp6(__v, __NV_E2M3, cudaRoundNearest));
    }
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp6_e3m2>(::__nv_cvt_halfraw_to_fp6(__v, __NV_E3M2, cudaRoundNearest));
    }
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp4_e2m1>(::__nv_cvt_halfraw_to_fp4(__v, __NV_E2M1, cudaRoundNearest));
    }
#  endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (_CCCL_TRAIT(is_same, _From, __nv_bfloat16))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return ::__bfloat162float(__v);
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return _CUDA_VSTD::__fp_cast<double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return _CUDA_VSTD::__fp_cast<long double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return _CUDA_VSTD::__fp_cast<__half>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return __v;
    }
#  if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp8_e4m3>(::__nv_cvt_bfloat16raw_to_fp8(__v, __NV_NOSAT, __NV_E4M3));
    }
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_from_storage<__nv_fp8_e5m2>(::__nv_cvt_bfloat16raw_to_fp8(__v, __NV_NOSAT, __NV_E5M2));
    }
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp8_e8m0>(
        ::__nv_cvt_bfloat16raw_to_e8m0(__v, __NV_NOSAT, ::cudaRoundZero));
    }
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp6_e2m3>(
        ::__nv_cvt_bfloat16raw_to_fp6(__v, __NV_E2M3, ::cudaRoundNearest));
    }
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp6_e3m2>(
        ::__nv_cvt_bfloat16raw_to_fp6(__v, __NV_E3M2, ::cudaRoundNearest));
    }
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return ::cuda::std::__fp_from_storage<__nv_fp4_e2m1>(
        ::__nv_cvt_bfloat16raw_to_fp4(__v, __NV_E2M1, ::cudaRoundNearest));
    }
#  endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (_CCCL_TRAIT(is_same, _From, __nv_fp8_e4m3))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return _CUDA_VSTD::__fp_cast<float>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return _CUDA_VSTD::__fp_cast<double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return _CUDA_VSTD::__fp_cast<long double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return ::__nv_cvt_fp8_to_halfraw(__v.__x, __NV_E4M3);
    }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return _CUDA_VSTD::__fp_cast<__nv_bfloat16>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return __v;
    }
#  if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e5m2>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e8m0>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e2m3>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e3m2>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp4_e2m1>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (_CCCL_TRAIT(is_same, _From, __nv_fp8_e5m2))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return _CUDA_VSTD::__fp_cast<float>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return _CUDA_VSTD::__fp_cast<double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return _CUDA_VSTD::__fp_cast<long double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return ::__nv_cvt_fp8_to_halfraw(__v.__x, __NV_E5M2);
    }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return _CUDA_VSTD::__fp_cast<__nv_bfloat16>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e4m3>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return __v;
    }
#  if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e8m0>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e2m3>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e3m2>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp4_e2m1>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (_CCCL_TRAIT(is_same, _From, __nv_fp8_e8m0))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return _CUDA_VSTD::__fp_cast<float>(_CUDA_VSTD::__fp_cast<__nv_bfloat16>(__v));
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return _CUDA_VSTD::__fp_cast<double>(_CUDA_VSTD::__fp_cast<__nv_bfloat16>(__v));
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return _CUDA_VSTD::__fp_cast<long double>(_CUDA_VSTD::__fp_cast<__nv_bfloat16>(__v));
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return _CUDA_VSTD::__fp_cast<__half>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return ::__nv_cvt_e8m0_to_bf16raw(__v.__x);
    }
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e4m3>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e5m2>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return __v;
    }
#  if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e2m3>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e3m2>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp4_e2m1>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (_CCCL_TRAIT(is_same, _From, __nv_fp6_e2m3))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return _CUDA_VSTD::__fp_cast<float>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return _CUDA_VSTD::__fp_cast<double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return _CUDA_VSTD::__fp_cast<long double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return ::__nv_cvt_fp6_to_halfraw(__v.__x, __NV_E2M3);
    }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return _CUDA_VSTD::__fp_cast<__nv_bfloat16>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e4m3>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e5m2>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e8m0>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return __v;
    }
#  if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e3m2>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp4_e2m1>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (_CCCL_TRAIT(is_same, _From, __nv_fp6_e3m2))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return _CUDA_VSTD::__fp_cast<float>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return _CUDA_VSTD::__fp_cast<double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return _CUDA_VSTD::__fp_cast<long double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return ::__nv_cvt_fp6_to_halfraw(__v.__x, __NV_E3M2);
    }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return _CUDA_VSTD::__fp_cast<__nv_bfloat16>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e4m3>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e5m2>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e8m0>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e2m3>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return __v;
    }
#  if _CCCL_HAS_NVFP4_E2M1()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp4_e2m1>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP4_E2M1()
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (_CCCL_TRAIT(is_same, _From, __nv_fp4_e2m1))
  {
    if constexpr (_CCCL_TRAIT(is_same, _To, float))
    {
      return _CUDA_VSTD::__fp_cast<float>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
    else if constexpr (_CCCL_TRAIT(is_same, _To, double))
    {
      return _CUDA_VSTD::__fp_cast<double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (_CCCL_TRAIT(is_same, _To, long double))
    {
      return _CUDA_VSTD::__fp_cast<long double>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __half))
    {
      return ::__nv_cvt_fp4_to_halfraw(__v.__x, __NV_E2M1);
    }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_bfloat16))
    {
      return _CUDA_VSTD::__fp_cast<__nv_bfloat16>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e4m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e4m3>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e5m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e5m2>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp8_e8m0))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp8_e8m0>(_CUDA_VSTD::__fp_cast<float>(__v));
    }
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e2m3))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e2m3>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp6_e3m2))
    {
      return _CUDA_VSTD::__fp_cast<__nv_fp6_e3m2>(_CUDA_VSTD::__fp_cast<__half>(__v));
    }
#  endif // _CCCL_HAS_NVFP6_E3M2()
    else if constexpr (_CCCL_TRAIT(is_same, _To, __nv_fp4_e2m1))
    {
      return __v;
    }
    else
    {
      static_assert(__always_false_v<_To>, "Unsupported floating point format");
    }
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
  else
  {
    static_assert(__always_false_v<_From>, "Unsupported floating point format");
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_CAST_H
