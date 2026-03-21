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

#ifndef _LIBCUDACXX___NUMERIC_ACCUMULATE_H
#define _LIBCUDACXX___NUMERIC_ACCUMULATE_H

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

template <class _InputIterator, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp accumulate(_InputIterator __first, _InputIterator __last, _Tp __init)
{
  for (; __first != __last; ++__first)
  {
    __init = _CUDA_VSTD::move(__init) + *__first;
  }
  return __init;
}

template <class _InputIterator, class _Tp, class _BinaryOperation>
[[nodiscard]] _CCCL_API constexpr _Tp
accumulate(_InputIterator __first, _InputIterator __last, _Tp __init, _BinaryOperation __binary_op)
{
  for (; __first != __last; ++__first)
  {
    __init = __binary_op(_CUDA_VSTD::move(__init), *__first);
  }
  return __init;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___NUMERIC_ACCUMULATE_H
