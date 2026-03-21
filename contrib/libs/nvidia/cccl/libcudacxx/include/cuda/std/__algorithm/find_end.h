//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_FIND_END_H
#define _LIBCUDACXX___ALGORITHM_FIND_END_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _BinaryPredicate, class _ForwardIterator1, class _ForwardIterator2>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator1 __find_end(
  _ForwardIterator1 __first1,
  _ForwardIterator1 __last1,
  _ForwardIterator2 __first2,
  _ForwardIterator2 __last2,
  _BinaryPredicate __pred,
  forward_iterator_tag,
  forward_iterator_tag)
{
  // modeled after search algorithm
  _ForwardIterator1 __r = __last1; // __last1 is the "default" answer
  if (__first2 == __last2)
  {
    return __r;
  }
  while (true)
  {
    while (true)
    {
      if (__first1 == __last1) // if source exhausted return last correct answer
      {
        return __r; //    (or __last1 if never found)
      }
      if (__pred(*__first1, *__first2))
      {
        break;
      }
      ++__first1;
    }
    // *__first1 matches *__first2, now match elements after here
    _ForwardIterator1 __m1 = __first1;
    _ForwardIterator2 __m2 = __first2;
    while (true)
    {
      if (++__m2 == __last2)
      { // Pattern exhausted, record answer and search for another one
        __r = __first1;
        ++__first1;
        break;
      }
      if (++__m1 == __last1) // Source exhausted, return last answer
      {
        return __r;
      }
      if (!__pred(*__m1, *__m2)) // mismatch, restart with a new __first
      {
        ++__first1;
        break;
      } // else there is a match, check next elements
    }
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _BinaryPredicate, class _BidirectionalIterator1, class _BidirectionalIterator2>
[[nodiscard]] _CCCL_API constexpr _BidirectionalIterator1 __find_end(
  _BidirectionalIterator1 __first1,
  _BidirectionalIterator1 __last1,
  _BidirectionalIterator2 __first2,
  _BidirectionalIterator2 __last2,
  _BinaryPredicate __pred,
  bidirectional_iterator_tag,
  bidirectional_iterator_tag)
{
  // modeled after search algorithm (in reverse)
  if (__first2 == __last2)
  {
    return __last1; // Everything matches an empty sequence
  }
  _BidirectionalIterator1 __l1 = __last1;
  _BidirectionalIterator2 __l2 = __last2;
  --__l2;
  while (true)
  {
    // Find last element in sequence 1 that matches *(__last2-1), with a minimum of loop checks
    while (true)
    {
      if (__first1 == __l1) // return __last1 if no element matches *__first2
      {
        return __last1;
      }
      if (__pred(*--__l1, *__l2))
      {
        break;
      }
    }
    // *__l1 matches *__l2, now match elements before here
    _BidirectionalIterator1 __m1 = __l1;
    _BidirectionalIterator2 __m2 = __l2;
    while (true)
    {
      if (__m2 == __first2) // If pattern exhausted, __m1 is the answer (works for 1 element pattern)
      {
        return __m1;
      }
      if (__m1 == __first1) // Otherwise if source exhausted, pattern not found
      {
        return __last1;
      }
      if (!__pred(*--__m1, *--__m2)) // if there is a mismatch, restart with a new __l1
      {
        break;
      } // else there is a match, check next elements
    }
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _BinaryPredicate, class _RandomAccessIterator1, class _RandomAccessIterator2>
[[nodiscard]] _CCCL_API constexpr _RandomAccessIterator1 __find_end(
  _RandomAccessIterator1 __first1,
  _RandomAccessIterator1 __last1,
  _RandomAccessIterator2 __first2,
  _RandomAccessIterator2 __last2,
  _BinaryPredicate __pred,
  random_access_iterator_tag,
  random_access_iterator_tag)
{
  // Take advantage of knowing source and pattern lengths.  Stop short when source is smaller than pattern
  __iter_diff_t<_RandomAccessIterator2> __len2 = __last2 - __first2;
  if (__len2 == 0)
  {
    return __last1;
  }
  __iter_diff_t<_RandomAccessIterator1> __len1 = __last1 - __first1;
  if (__len1 < __len2)
  {
    return __last1;
  }
  const _RandomAccessIterator1 __s = __first1 + (__len2 - 1); // End of pattern match can't go before here
  _RandomAccessIterator1 __l1      = __last1;
  _RandomAccessIterator2 __l2      = __last2;
  --__l2;
  while (true)
  {
    while (true)
    {
      if (__s == __l1)
      {
        return __last1;
      }
      if (__pred(*--__l1, *__l2))
      {
        break;
      }
    }
    _RandomAccessIterator1 __m1 = __l1;
    _RandomAccessIterator2 __m2 = __l2;
    while (true)
    {
      if (__m2 == __first2)
      {
        return __m1;
      }
      // no need to check range on __m1 because __s guarantees we have enough source
      if (!__pred(*--__m1, *--__m2))
      {
        break;
      }
    }
  }
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator1 find_end(
  _ForwardIterator1 __first1,
  _ForwardIterator1 __last1,
  _ForwardIterator2 __first2,
  _ForwardIterator2 __last2,
  _BinaryPredicate __pred)
{
  return _CUDA_VSTD::__find_end<add_lvalue_reference_t<_BinaryPredicate>>(
    __first1,
    __last1,
    __first2,
    __last2,
    __pred,
    typename iterator_traits<_ForwardIterator1>::iterator_category{},
    typename iterator_traits<_ForwardIterator2>::iterator_category{});
}

template <class _ForwardIterator1, class _ForwardIterator2>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator1
find_end(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
  return _CUDA_VSTD::find_end(__first1, __last1, __first2, __last2, __equal_to{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_FIND_END_H
