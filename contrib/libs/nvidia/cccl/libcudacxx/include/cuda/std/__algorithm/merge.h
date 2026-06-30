//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MERGE_H
#define _LIBCUDACXX___ALGORITHM_MERGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/comp_ref_type.h>
#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__iterator/iterator_traits.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Compare, class _InputIterator1, class _InputIterator2, class _OutputIterator>
_CCCL_API constexpr _OutputIterator __merge(
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _InputIterator2 __last2,
  _OutputIterator __result,
  _Compare __comp)
{
  for (; __first1 != __last1; ++__result)
  {
    if (__first2 == __last2)
    {
      return _CUDA_VSTD::copy(__first1, __last1, __result);
    }
    if (__comp(*__first2, *__first1))
    {
      *__result = *__first2;
      ++__first2;
    }
    else
    {
      *__result = *__first1;
      ++__first1;
    }
  }
  return _CUDA_VSTD::copy(__first2, __last2, __result);
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare>
_CCCL_API constexpr _OutputIterator
merge(_InputIterator1 __first1,
      _InputIterator1 __last1,
      _InputIterator2 __first2,
      _InputIterator2 __last2,
      _OutputIterator __result,
      _Compare __comp)
{
  return _CUDA_VSTD::__merge<__comp_ref_type<_Compare>>(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator>
_CCCL_API constexpr _OutputIterator
merge(_InputIterator1 __first1,
      _InputIterator1 __last1,
      _InputIterator2 __first2,
      _InputIterator2 __last2,
      _OutputIterator __result)
{
  return _CUDA_VSTD::merge(__first1, __last1, __first2, __last2, __result, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_MERGE_H
