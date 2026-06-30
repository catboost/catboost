//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_ROOTS_H
#define _LIBCUDACXX___COMPLEX_ROOTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__complex/arg.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/math.h>
#include <cuda/std/__complex/roots.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// sqrt

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> sqrt(const complex<_Tp>& __x)
{
  if (_CUDA_VSTD::isinf(__x.imag()))
  {
    return complex<_Tp>(numeric_limits<_Tp>::infinity(), __x.imag());
  }
  if (_CUDA_VSTD::isinf(__x.real()))
  {
    if (__x.real() > _Tp(0))
    {
      return complex<_Tp>(__x.real(),
                          _CUDA_VSTD::isnan(__x.imag()) ? __x.imag() : _CUDA_VSTD::copysign(_Tp(0), __x.imag()));
    }
    return complex<_Tp>(_CUDA_VSTD::isnan(__x.imag()) ? __x.imag() : _Tp(0),
                        _CUDA_VSTD::copysign(__x.real(), __x.imag()));
  }
  return _CUDA_VSTD::polar(_CUDA_VSTD::sqrt(_CUDA_VSTD::abs(__x)), _CUDA_VSTD::arg(__x) / _Tp(2));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_ROOTS_H
