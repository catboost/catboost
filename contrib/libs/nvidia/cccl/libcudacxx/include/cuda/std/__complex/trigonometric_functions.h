//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_TRIGONOMETRIC_FUNCTIONS_H
#define _LIBCUDACXX___COMPLEX_TRIGONOMETRIC_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/hyperbolic_functions.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// sin

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> sin(const complex<_Tp>& __x)
{
  complex<_Tp> __z = _CUDA_VSTD::sinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> cos(const complex<_Tp>& __x)
{
  return _CUDA_VSTD::cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> tan(const complex<_Tp>& __x)
{
  complex<_Tp> __z = _CUDA_VSTD::tanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_TRIGONOMETRIC_FUNCTIONS_H
