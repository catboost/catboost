//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_INVERSE_TRIGONOMETRIC_FUNCTIONS_H
#define _LIBCUDACXX___COMPLEX_INVERSE_TRIGONOMETRIC_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/exponential_functions.h>
#include <cuda/std/__complex/inverse_hyperbolic_functions.h>
#include <cuda/std/__complex/logarithms.h>
#include <cuda/std/__complex/nvbf16.h>
#include <cuda/std/__complex/nvfp16.h>
#include <cuda/std/__complex/roots.h>
#include <cuda/std/limits>
#include <cuda/std/numbers>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// asin

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> asin(const complex<_Tp>& __x)
{
  complex<_Tp> __z = _CUDA_VSTD::asinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> acos(const complex<_Tp>& __x)
{
  constexpr _Tp __pi = __numbers<_Tp>::__pi();
  if (_CUDA_VSTD::isinf(__x.real()))
  {
    if (_CUDA_VSTD::isnan(__x.imag()))
    {
      return complex<_Tp>(__x.imag(), __x.real());
    }
    if (_CUDA_VSTD::isinf(__x.imag()))
    {
      if (__x.real() < _Tp(0))
      {
        return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
      }
      return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
    }
    if (__x.real() < _Tp(0))
    {
      return complex<_Tp>(__pi, _CUDA_VSTD::signbit(__x.imag()) ? -__x.real() : __x.real());
    }
    return complex<_Tp>(_Tp(0), _CUDA_VSTD::signbit(__x.imag()) ? __x.real() : -__x.real());
  }
  if (_CUDA_VSTD::isnan(__x.real()))
  {
    if (_CUDA_VSTD::isinf(__x.imag()))
    {
      return complex<_Tp>(__x.real(), -__x.imag());
    }
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (_CUDA_VSTD::isinf(__x.imag()))
  {
    return complex<_Tp>(__pi / _Tp(2), -__x.imag());
  }
  if (__x.real() == _Tp(0) && (__x.imag() == _Tp(0) || _CUDA_VSTD::isnan(__x.imag())))
  {
    return complex<_Tp>(__pi / _Tp(2), -__x.imag());
  }
  complex<_Tp> __z = _CUDA_VSTD::log(__x + _CUDA_VSTD::sqrt(_CUDA_VSTD::__sqr(__x) - _Tp(1)));
  if (_CUDA_VSTD::signbit(__x.imag()))
  {
    return complex<_Tp>(_CUDA_VSTD::abs(__z.imag()), _CUDA_VSTD::abs(__z.real()));
  }
  return complex<_Tp>(_CUDA_VSTD::abs(__z.imag()), -_CUDA_VSTD::abs(__z.real()));
}

// We have performance issues with some trigonometric functions with extended floating point types
#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_API inline complex<__nv_bfloat16> acos(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{_CUDA_VSTD::acos(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_API inline complex<__half> acos(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::acos(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

// atan

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> atan(const complex<_Tp>& __x)
{
  complex<_Tp> __z = _CUDA_VSTD::atanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_INVERSE_TRIGONOMETRIC_FUNCTIONS_H
