//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_FIND_FIRST_OF_H
#define _LIBCUDACXX___ALGORITHM_FIND_FIRST_OF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator1 __find_first_of_ce(
  _ForwardIterator1 __first1,
  _ForwardIterator1 __last1,
  _ForwardIterator2 __first2,
  _ForwardIterator2 __last2,
  _BinaryPredicate __pred)
{
  for (; __first1 != __last1; ++__first1)
  {
    for (_ForwardIterator2 __j = __first2; __j != __last2; ++__j)
    {
      if (__pred(*__first1, *__j))
      {
        return __first1;
      }
    }
  }
  return __last1;
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator1 find_first_of(
  _ForwardIterator1 __first1,
  _ForwardIterator1 __last1,
  _ForwardIterator2 __first2,
  _ForwardIterator2 __last2,
  _BinaryPredicate __pred)
{
  return _CUDA_VSTD::__find_first_of_ce(__first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator1 find_first_of(
  _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
  return _CUDA_VSTD::__find_first_of_ce(__first1, __last1, __first2, __last2, __equal_to{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_FIND_FIRST_OF_H
