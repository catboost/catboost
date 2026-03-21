//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_MATH_H
#define _LIBCUDACXX___COMPLEX_MATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/hypot.h>
#include <cuda/std/__cmath/inverse_trigonometric_functions.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/vector_support.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// abs

template <class _Tp>
[[nodiscard]] _CCCL_API inline _Tp abs(const complex<_Tp>& __c)
{
  return _CUDA_VSTD::hypot(__c.real(), __c.imag());
}

// norm

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp norm(const complex<_Tp>& __c)
{
  if (_CUDA_VSTD::isinf(__c.real()))
  {
    return _CUDA_VSTD::abs(__c.real());
  }
  if (_CUDA_VSTD::isinf(__c.imag()))
  {
    return _CUDA_VSTD::abs(__c.imag());
  }
  return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __cccl_complex_value_type<_Tp> norm(_Tp __re)
{
  return static_cast<__cccl_complex_value_type<_Tp>>(__re) * __re;
}

// conj

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> conj(const complex<_Tp>& __c)
{
  return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __cccl_complex_complex_type<_Tp> conj(_Tp __re)
{
  return __cccl_complex_complex_type<_Tp>(__re);
}

// proj

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> proj(const complex<_Tp>& __c)
{
  complex<_Tp> __r = __c;
  if (_CUDA_VSTD::isinf(__c.real()) || _CUDA_VSTD::isinf(__c.imag()))
  {
    __r = complex<_Tp>(numeric_limits<_Tp>::infinity(), _CUDA_VSTD::copysign(_Tp(0), __c.imag()));
  }
  return __r;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((_CCCL_TRAIT(is_floating_point, _Tp) || _CCCL_TRAIT(__is_extended_floating_point, _Tp)))
[[nodiscard]] _CCCL_API inline __cccl_complex_complex_type<_Tp> proj(_Tp __re)
{
  if (_CUDA_VSTD::isinf(__re))
  {
    __re = _CUDA_VSTD::abs(__re);
  }
  return complex<_Tp>(__re);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
[[nodiscard]] _CCCL_API inline __cccl_complex_complex_type<_Tp> proj(_Tp __re)
{
  return __cccl_complex_complex_type<_Tp>(__re);
}

// polar

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> polar(const _Tp& __rho, const _Tp& __theta = _Tp())
{
  if (_CUDA_VSTD::isnan(__rho) || _CUDA_VSTD::signbit(__rho))
  {
    return complex<_Tp>(numeric_limits<_Tp>::quiet_NaN(), numeric_limits<_Tp>::quiet_NaN());
  }
  if (_CUDA_VSTD::isnan(__theta))
  {
    if (_CUDA_VSTD::isinf(__rho))
    {
      return complex<_Tp>(__rho, __theta);
    }
    return complex<_Tp>(__theta, __theta);
  }
  if (_CUDA_VSTD::isinf(__theta))
  {
    if (_CUDA_VSTD::isinf(__rho))
    {
      return complex<_Tp>(__rho, numeric_limits<_Tp>::quiet_NaN());
    }
    return complex<_Tp>(numeric_limits<_Tp>::quiet_NaN(), numeric_limits<_Tp>::quiet_NaN());
  }
  _Tp __x = __rho * _CUDA_VSTD::cos(__theta);
  if (_CUDA_VSTD::isnan(__x))
  {
    __x = 0;
  }
  _Tp __y = __rho * _CUDA_VSTD::sin(__theta);
  if (_CUDA_VSTD::isnan(__y))
  {
    __y = 0;
  }
  return complex<_Tp>(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_MATH_H
