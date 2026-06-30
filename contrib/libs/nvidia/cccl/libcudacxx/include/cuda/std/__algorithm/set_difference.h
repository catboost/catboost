//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SET_DIFFERENCE_H
#define _LIBCUDACXX___ALGORITHM_SET_DIFFERENCE_H

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
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Comp, class _InIter1, class _Sent1, class _InIter2, class _Sent2, class _OutIter>
_CCCL_API constexpr pair<remove_cvref_t<_InIter1>, remove_cvref_t<_OutIter>> __set_difference(
  _InIter1&& __first1, _Sent1&& __last1, _InIter2&& __first2, _Sent2&& __last2, _OutIter&& __result, _Comp&& __comp)
{
  while (__first1 != __last1 && __first2 != __last2)
  {
    if (__comp(*__first1, *__first2))
    {
      *__result = *__first1;
      ++__first1;
      ++__result;
    }
    else if (__comp(*__first2, *__first1))
    {
      ++__first2;
    }
    else
    {
      ++__first1;
      ++__first2;
    }
  }
  return _CUDA_VSTD::__copy<_AlgPolicy>(
    _CUDA_VSTD::move(__first1), _CUDA_VSTD::move(__last1), _CUDA_VSTD::move(__result));
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare>
_CCCL_API constexpr _OutputIterator set_difference(
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _InputIterator2 __last2,
  _OutputIterator __result,
  _Compare __comp)
{
  return _CUDA_VSTD::__set_difference<_ClassicAlgPolicy, __comp_ref_type<_Compare>>(
           __first1, __last1, __first2, __last2, __result, __comp)
    .second;
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator>
_CCCL_API constexpr _OutputIterator set_difference(
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _InputIterator2 __last2,
  _OutputIterator __result)
{
  return _CUDA_VSTD::__set_difference<_ClassicAlgPolicy>(__first1, __last1, __first2, __last2, __result, __less{})
    .second;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_SET_DIFFERENCE_H
