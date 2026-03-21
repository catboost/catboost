//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SHIFT_RIGHT_H
#define _LIBCUDACXX___ALGORITHM_SHIFT_RIGHT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/move.h>
#include <cuda/std/__algorithm/move_backward.h>
#include <cuda/std/__algorithm/swap_ranges.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __shift_right(
  _ForwardIterator __first,
  _ForwardIterator __last,
  typename iterator_traits<_ForwardIterator>::difference_type __n,
  random_access_iterator_tag)
{
  decltype(__n) __d = __last - __first;
  if (__n >= __d)
  {
    return __last;
  }
  _ForwardIterator __m = __first + (__d - __n);
  return _CUDA_VSTD::move_backward(__first, __m, __last);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __shift_right(
  _ForwardIterator __first,
  _ForwardIterator __last,
  typename iterator_traits<_ForwardIterator>::difference_type __n,
  bidirectional_iterator_tag)
{
  _ForwardIterator __m = __last;
  for (; __n > 0; --__n)
  {
    if (__m == __first)
    {
      return __last;
    }
    --__m;
  }
  return _CUDA_VSTD::move_backward(__first, __m, __last);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __shift_right(
  _ForwardIterator __first,
  _ForwardIterator __last,
  typename iterator_traits<_ForwardIterator>::difference_type __n,
  forward_iterator_tag)
{
  _ForwardIterator __ret = __first;
  for (; __n > 0; --__n)
  {
    if (__ret == __last)
    {
      return __last;
    }
    ++__ret;
  }

  // We have an __n-element scratch space from __first to __ret.
  // Slide an __n-element window [__trail, __lead) from left to right.
  // We're essentially doing swap_ranges(__first, __ret, __trail, __lead)
  // over and over; but once __lead reaches __last we needn't bother
  // to save the values of elements [__trail, __last).

  auto __trail = __first;
  auto __lead  = __ret;
  while (__trail != __ret)
  {
    if (__lead == __last)
    {
      _CUDA_VSTD::move(__first, __trail, __ret);
      return __ret;
    }
    ++__trail;
    ++__lead;
  }

  _ForwardIterator __mid = __first;
  while (true)
  {
    if (__lead == __last)
    {
      __trail = _CUDA_VSTD::move(__mid, __ret, __trail);
      _CUDA_VSTD::move(__first, __mid, __trail);
      return __ret;
    }
    swap(*__mid, *__trail);
    ++__mid;
    ++__trail;
    ++__lead;
    if (__mid == __ret)
    {
      __mid = __first;
    }
  }
}

template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator shift_right(
  _ForwardIterator __first, _ForwardIterator __last, typename iterator_traits<_ForwardIterator>::difference_type __n)
{
  if (__n == 0)
  {
    return __first;
  }

  using _IterCategory = typename iterator_traits<_ForwardIterator>::iterator_category;
  return __shift_right(__first, __last, __n, _IterCategory());
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_SHIFT_RIGHT_H
