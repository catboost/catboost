//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_IS_PERMUTATION_H
#define _LIBCUDACXX___ALGORITHM_IS_PERMUTATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr bool is_permutation(
  _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _BinaryPredicate __pred)
{
  //  shorten sequences as much as possible by lopping of any equal prefix
  for (; __first1 != __last1; ++__first1, (void) ++__first2)
  {
    if (!__pred(*__first1, *__first2))
    {
      break;
    }
  }
  if (__first1 == __last1)
  {
    return true;
  }

  //  __first1 != __last1 && *__first1 != *__first2
  using _Diff1 = __iter_diff_t<_ForwardIterator1>;
  _Diff1 __l1  = _CUDA_VSTD::distance(__first1, __last1);
  if (__l1 == _Diff1(1))
  {
    return false;
  }
  _ForwardIterator2 __last2 = _CUDA_VSTD::next(__first2, __l1);
  // For each element in [f1, l1) see if there are the same number of
  //    equal elements in [f2, l2)
  for (_ForwardIterator1 __i = __first1; __i != __last1; ++__i)
  {
    //  Have we already counted the number of *__i in [f1, l1)?
    _ForwardIterator1 __match = __first1;
    for (; __match != __i; ++__match)
    {
      if (__pred(*__match, *__i))
      {
        break;
      }
    }
    if (__match == __i)
    {
      // Count number of *__i in [f2, l2)
      _Diff1 __c2 = 0;
      for (_ForwardIterator2 __j = __first2; __j != __last2; ++__j)
      {
        if (__pred(*__i, *__j))
        {
          ++__c2;
        }
      }
      if (__c2 == 0)
      {
        return false;
      }
      // Count number of *__i in [__i, l1) (we can start with 1)
      _Diff1 __c1 = 1;
      for (_ForwardIterator1 __j = _CUDA_VSTD::next(__i); __j != __last1; ++__j)
      {
        if (__pred(*__i, *__j))
        {
          ++__c1;
        }
      }
      if (__c1 != __c2)
      {
        return false;
      }
    }
  }
  return true;
}

template <class _ForwardIterator1, class _ForwardIterator2>
[[nodiscard]] _CCCL_API constexpr bool
is_permutation(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2)
{
  return _CUDA_VSTD::is_permutation(__first1, __last1, __first2, __equal_to{});
}

_CCCL_EXEC_CHECK_DISABLE
template <class _BinaryPredicate, class _ForwardIterator1, class _ForwardIterator2>
[[nodiscard]] _CCCL_API constexpr bool __is_permutation(
  _ForwardIterator1 __first1,
  _ForwardIterator1 __last1,
  _ForwardIterator2 __first2,
  _ForwardIterator2 __last2,
  _BinaryPredicate __pred,
  forward_iterator_tag,
  forward_iterator_tag)
{
  //  shorten sequences as much as possible by lopping of any equal prefix
  for (; __first1 != __last1 && __first2 != __last2; ++__first1, (void) ++__first2)
  {
    if (!__pred(*__first1, *__first2))
    {
      break;
    }
  }
  if (__first1 == __last1)
  {
    return __first2 == __last2;
  }
  else if (__first2 == __last2)
  {
    return false;
  }

  using _Diff1 = __iter_diff_t<_ForwardIterator1>;
  _Diff1 __l1  = _CUDA_VSTD::distance(__first1, __last1);

  using _Diff2 = __iter_diff_t<_ForwardIterator2>;
  _Diff2 __l2  = _CUDA_VSTD::distance(__first2, __last2);
  if (__l1 != __l2)
  {
    return false;
  }

  // For each element in [f1, l1) see if there are the same number of
  //    equal elements in [f2, l2)
  for (_ForwardIterator1 __i = __first1; __i != __last1; ++__i)
  {
    //  Have we already counted the number of *__i in [f1, l1)?
    _ForwardIterator1 __match = __first1;
    for (; __match != __i; ++__match)
    {
      if (__pred(*__match, *__i))
      {
        break;
      }
    }
    if (__match == __i)
    {
      // Count number of *__i in [f2, l2)
      _Diff1 __c2 = 0;
      for (_ForwardIterator2 __j = __first2; __j != __last2; ++__j)
      {
        if (__pred(*__i, *__j))
        {
          ++__c2;
        }
      }
      if (__c2 == 0)
      {
        return false;
      }
      // Count number of *__i in [__i, l1) (we can start with 1)
      _Diff1 __c1 = 1;
      for (_ForwardIterator1 __j = _CUDA_VSTD::next(__i); __j != __last1; ++__j)
      {
        if (__pred(*__i, *__j))
        {
          ++__c1;
        }
      }
      if (__c1 != __c2)
      {
        return false;
      }
    }
  }
  return true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _BinaryPredicate, class _RandomAccessIterator1, class _RandomAccessIterator2>
[[nodiscard]] _CCCL_API constexpr bool __is_permutation(
  _RandomAccessIterator1 __first1,
  _RandomAccessIterator2 __last1,
  _RandomAccessIterator1 __first2,
  _RandomAccessIterator2 __last2,
  _BinaryPredicate __pred,
  random_access_iterator_tag,
  random_access_iterator_tag)
{
  if (__last1 - __first1 != __last2 - __first2)
  {
    return false;
  }
  return _CUDA_VSTD::
    is_permutation<_RandomAccessIterator1, _RandomAccessIterator2, add_lvalue_reference_t<_BinaryPredicate>>(
      __first1, __last1, __first2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr bool is_permutation(
  _ForwardIterator1 __first1,
  _ForwardIterator1 __last1,
  _ForwardIterator2 __first2,
  _ForwardIterator2 __last2,
  _BinaryPredicate __pred)
{
  return _CUDA_VSTD::__is_permutation<add_lvalue_reference_t<_BinaryPredicate>>(
    __first1,
    __last1,
    __first2,
    __last2,
    __pred,
    __iterator_category_type<_ForwardIterator1>{},
    __iterator_category_type<_ForwardIterator2>{});
}

template <class _ForwardIterator1, class _ForwardIterator2>
[[nodiscard]] _CCCL_API constexpr bool is_permutation(
  _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
  return _CUDA_VSTD::__is_permutation(
    __first1,
    __last1,
    __first2,
    __last2,
    __equal_to{},
    __iterator_category_type<_ForwardIterator1>{},
    __iterator_category_type<_ForwardIterator2>{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_IS_PERMUTATION_H
