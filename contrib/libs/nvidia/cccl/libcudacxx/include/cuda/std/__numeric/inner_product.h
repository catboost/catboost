// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NUMERIC_INNER_PRODUCT_H
#define _LIBCUDACXX___NUMERIC_INNER_PRODUCT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator1, class _InputIterator2, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp
inner_product(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _Tp __init)
{
  for (; __first1 != __last1; ++__first1, (void) ++__first2)
  {
    __init = _CUDA_VSTD::move(__init) + *__first1 * *__first2;
  }
  return __init;
}

template <class _InputIterator1, class _InputIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
[[nodiscard]] _CCCL_API constexpr _Tp inner_product(
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _Tp __init,
  _BinaryOperation1 __binary_op1,
  _BinaryOperation2 __binary_op2)
{
  for (; __first1 != __last1; ++__first1, (void) ++__first2)
  {
    __init = __binary_op1(_CUDA_VSTD::move(__init), __binary_op2(*__first1, *__first2));
  }
  return __init;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___NUMERIC_INNER_PRODUCT_H
