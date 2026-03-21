//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_LOGARITHMS_H
#define _LIBCUDACXX___COMPLEX_LOGARITHMS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__complex/arg.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/logarithms.h>
#include <cuda/std/__complex/math.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// log

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> log(const complex<_Tp>& __x)
{
  return complex<_Tp>(_CUDA_VSTD::log(_CUDA_VSTD::abs(__x)), _CUDA_VSTD::arg(__x));
}

// log10

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> log10(const complex<_Tp>& __x)
{
  return _CUDA_VSTD::log(__x) / _CUDA_VSTD::log(_Tp(10));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_LOGARITHMS_H
