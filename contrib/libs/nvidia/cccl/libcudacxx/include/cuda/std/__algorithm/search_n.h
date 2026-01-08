//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SEARCH_N_H
#define _LIBCUDACXX___ALGORITHM_SEARCH_N_H

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
#include <cuda/std/__utility/convert_to_integral.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _BinaryPredicate, class _ForwardIterator, class _Size, class _Tp>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator __search_n(
  _ForwardIterator __first,
  _ForwardIterator __last,
  _Size __count,
  const _Tp& __value_,
  _BinaryPredicate __pred,
  forward_iterator_tag)
{
  if (__count <= 0)
  {
    return __first;
  }
  while (true)
  {
    // Find first element in sequence that matches __value_, with a minimum of loop checks
    while (true)
    {
      if (__first == __last) // return __last if no element matches __value_
      {
        return __last;
      }
      if (__pred(*__first, __value_))
      {
        break;
      }
      ++__first;
    }
    // *__first matches __value_, now match elements after here
    _ForwardIterator __m = __first;
    _Size __c(0);
    while (true)
    {
      if (++__c == __count) // If pattern exhausted, __first is the answer (works for 1 element pattern)
      {
        return __first;
      }
      if (++__m == __last) // Otherwise if source exhausted, pattern not found
      {
        return __last;
      }
      if (!__pred(*__m, __value_)) // if there is a mismatch, restart with a new __first
      {
        __first = __m;
        ++__first;
        break;
      } // else there is a match, check next elements
    }
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _BinaryPredicate, class _RandomAccessIterator, class _Size, class _Tp>
[[nodiscard]] _CCCL_API constexpr _RandomAccessIterator __search_n(
  _RandomAccessIterator __first,
  _RandomAccessIterator __last,
  _Size __count,
  const _Tp& __value_,
  _BinaryPredicate __pred,
  random_access_iterator_tag)
{
  if (__count <= 0)
  {
    return __first;
  }
  _Size __len = static_cast<_Size>(__last - __first);
  if (__len < __count)
  {
    return __last;
  }
  const _RandomAccessIterator __s = __last - (__count - 1); // Start of pattern match can't go beyond here
  while (true)
  {
    // Find first element in sequence that matches __value_, with a minimum of loop checks
    while (true)
    {
      if (__first >= __s) // return __last if no element matches __value_
      {
        return __last;
      }
      if (__pred(*__first, __value_))
      {
        break;
      }
      ++__first;
    }
    // *__first matches __value_, now match elements after here
    _RandomAccessIterator __m = __first;
    _Size __c(0);
    while (true)
    {
      if (++__c == __count) // If pattern exhausted, __first is the answer (works for 1 element pattern)
      {
        return __first;
      }
      ++__m; // no need to check range on __m because __s guarantees we have enough source
      if (!__pred(*__m, __value_)) // if there is a mismatch, restart with a new __first
      {
        __first = __m;
        ++__first;
        break;
      } // else there is a match, check next elements
    }
  }
}

template <class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
search_n(_ForwardIterator __first, _ForwardIterator __last, _Size __count, const _Tp& __value_, _BinaryPredicate __pred)
{
  return _CUDA_VSTD::__search_n<add_lvalue_reference_t<_BinaryPredicate>>(
    __first,
    __last,
    __convert_to_integral(__count),
    __value_,
    __pred,
    typename iterator_traits<_ForwardIterator>::iterator_category());
}

template <class _ForwardIterator, class _Size, class _Tp>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
search_n(_ForwardIterator __first, _ForwardIterator __last, _Size __count, const _Tp& __value_)
{
  return _CUDA_VSTD::search_n(__first, __last, __convert_to_integral(__count), __value_, __equal_to{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_SEARCH_N_H
