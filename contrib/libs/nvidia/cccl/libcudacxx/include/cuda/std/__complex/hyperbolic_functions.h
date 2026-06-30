//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_HYPERBOLIC_FUNCTIONS_H
#define _LIBCUDACXX___COMPLEX_HYPERBOLIC_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/hyperbolic_functions.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// sinh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> sinh(const complex<_Tp>& __x)
{
  if (_CUDA_VSTD::isinf(__x.real()) && !_CUDA_VSTD::isfinite(__x.imag()))
  {
    return complex<_Tp>(__x.real(), numeric_limits<_Tp>::quiet_NaN());
  }
  if (__x.real() == _Tp(0) && !_CUDA_VSTD::isfinite(__x.imag()))
  {
    return complex<_Tp>(__x.real(), numeric_limits<_Tp>::quiet_NaN());
  }
  if (__x.imag() == _Tp(0) && !_CUDA_VSTD::isfinite(__x.real()))
  {
    return __x;
  }
  return complex<_Tp>(_CUDA_VSTD::sinh(__x.real()) * _CUDA_VSTD::cos(__x.imag()),
                      _CUDA_VSTD::cosh(__x.real()) * _CUDA_VSTD::sin(__x.imag()));
}

// cosh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> cosh(const complex<_Tp>& __x)
{
  if (_CUDA_VSTD::isinf(__x.real()) && !_CUDA_VSTD::isfinite(__x.imag()))
  {
    return complex<_Tp>(_CUDA_VSTD::abs(__x.real()), numeric_limits<_Tp>::quiet_NaN());
  }
  if (__x.real() == _Tp(0) && !_CUDA_VSTD::isfinite(__x.imag()))
  {
    return complex<_Tp>(numeric_limits<_Tp>::quiet_NaN(), __x.real());
  }
  if (__x.real() == _Tp(0) && __x.imag() == _Tp(0))
  {
    return complex<_Tp>(_Tp(1), __x.imag());
  }
  if (__x.imag() == _Tp(0) && !_CUDA_VSTD::isfinite(__x.real()))
  {
    return complex<_Tp>(_CUDA_VSTD::abs(__x.real()), __x.imag());
  }
  return complex<_Tp>(_CUDA_VSTD::cosh(__x.real()) * _CUDA_VSTD::cos(__x.imag()),
                      _CUDA_VSTD::sinh(__x.real()) * _CUDA_VSTD::sin(__x.imag()));
}

// tanh

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> tanh(const complex<_Tp>& __x)
{
  if (_CUDA_VSTD::isinf(__x.real()))
  {
    if (!_CUDA_VSTD::isfinite(__x.imag()))
    {
      return complex<_Tp>(_CUDA_VSTD::copysign(_Tp(1), __x.real()), _Tp(0));
    }
    return complex<_Tp>(_CUDA_VSTD::copysign(_Tp(1), __x.real()),
                        _CUDA_VSTD::copysign(_Tp(0), _CUDA_VSTD::sin(_Tp(2) * __x.imag())));
  }
  if (_CUDA_VSTD::isnan(__x.real()) && __x.imag() == _Tp(0))
  {
    return __x;
  }
  _Tp __2r(_Tp(2) * __x.real());
  _Tp __2i(_Tp(2) * __x.imag());
  _Tp __d(_CUDA_VSTD::cosh(__2r) + _CUDA_VSTD::cos(__2i));
  _Tp __2rsh(_CUDA_VSTD::sinh(__2r));
  if (_CUDA_VSTD::isinf(__2rsh) && _CUDA_VSTD::isinf(__d))
  {
    return complex<_Tp>(__2rsh > _Tp(0) ? _Tp(1) : _Tp(-1), __2i > _Tp(0) ? _Tp(0) : _Tp(-0.));
  }
  return complex<_Tp>(__2rsh / __d, _CUDA_VSTD::sin(__2i) / __d);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_HYPERBOLIC_FUNCTIONS_H
