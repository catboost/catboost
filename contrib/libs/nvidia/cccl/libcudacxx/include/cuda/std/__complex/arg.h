//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_ARG_H
#define _LIBCUDACXX___COMPLEX_ARG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/inverse_trigonometric_functions.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/is_integral.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// arg

template <class _Tp>
[[nodiscard]] _CCCL_API inline _Tp arg(const complex<_Tp>& __c)
{
  return _CUDA_VSTD::atan2(__c.imag(), __c.real());
}

[[nodiscard]] _CCCL_API inline float arg(float __re)
{
  return _CUDA_VSTD::atan2f(0.F, __re);
}

[[nodiscard]] _CCCL_API inline double arg(double __re)
{
  return _CUDA_VSTD::atan2(0.0, __re);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double arg(long double __re)
{
  return _CUDA_VSTD::atan2l(0.L, __re);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 arg(__nv_bfloat16 __re)
{
  return _CUDA_VSTD::atan2(::__int2bfloat16_rn(0), __re);
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half arg(__half __re)
{
  return _CUDA_VSTD::atan2(::__int2half_rn(0), __re);
}
#endif // _LIBCUDACXX_HAS_NVFP16()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_integral_v<_Tp>)
[[nodiscard]] _CCCL_API inline double arg(_Tp __re)
{
  // integrals need to be promoted to double
  return _CUDA_VSTD::arg(static_cast<double>(__re));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_MATH_H
