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

#ifndef _LIBCUDACXX___NUMERIC_REDUCE_H
#define _LIBCUDACXX___NUMERIC_REDUCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Tp, class _BinaryOp>
[[nodiscard]] _CCCL_API constexpr _Tp reduce(_InputIterator __first, _InputIterator __last, _Tp __init, _BinaryOp __b)
{
  for (; __first != __last; ++__first)
  {
    __init = __b(_CUDA_VSTD::move(__init), *__first);
  }
  return __init;
}

template <class _InputIterator, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp reduce(_InputIterator __first, _InputIterator __last, _Tp __init)
{
  return _CUDA_VSTD::reduce(__first, __last, __init, _CUDA_VSTD::plus<>());
}

template <class _InputIterator>
[[nodiscard]] _CCCL_API constexpr typename iterator_traits<_InputIterator>::value_type
reduce(_InputIterator __first, _InputIterator __last)
{
  return _CUDA_VSTD::reduce(__first, __last, typename iterator_traits<_InputIterator>::value_type{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___NUMERIC_REDUCE_H
