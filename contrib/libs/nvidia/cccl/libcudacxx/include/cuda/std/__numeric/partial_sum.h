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

#ifndef _LIBCUDACXX___NUMERIC_PARTIAL_SUM_H
#define _LIBCUDACXX___NUMERIC_PARTIAL_SUM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _OutputIterator>
_CCCL_API constexpr _OutputIterator partial_sum(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  if (__first != __last)
  {
    typename iterator_traits<_InputIterator>::value_type __t(*__first);
    *__result = __t;
    for (++__first, (void) ++__result; __first != __last; ++__first, (void) ++__result)
    {
      __t       = _CUDA_VSTD::move(__t) + *__first;
      *__result = __t;
    }
  }
  return __result;
}

template <class _InputIterator, class _OutputIterator, class _BinaryOperation>
_CCCL_API constexpr _OutputIterator
partial_sum(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _BinaryOperation __binary_op)
{
  if (__first != __last)
  {
    typename iterator_traits<_InputIterator>::value_type __t(*__first);
    *__result = __t;
    for (++__first, (void) ++__result; __first != __last; ++__first, (void) ++__result)
    {
      __t       = __binary_op(_CUDA_VSTD::move(__t), *__first);
      *__result = __t;
    }
  }
  return __result;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___NUMERIC_PARTIAL_SUM_H
