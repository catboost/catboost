//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_IS_HEAP_UNTIL_H
#define _LIBCUDACXX___ALGORITHM_IS_HEAP_UNTIL_H

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
#include <cuda/std/__iterator/iterator_traits.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Compare, class _RandomAccessIterator>
_CCCL_API constexpr _RandomAccessIterator
__is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare&& __comp)
{
  using difference_type      = typename iterator_traits<_RandomAccessIterator>::difference_type;
  difference_type __len      = __last - __first;
  difference_type __p        = 0;
  difference_type __c        = 1;
  _RandomAccessIterator __pp = __first;
  while (__c < __len)
  {
    _RandomAccessIterator __cp = __first + __c;
    if (__comp(*__pp, *__cp))
    {
      return __cp;
    }
    ++__c;
    ++__cp;
    if (__c == __len)
    {
      return __last;
    }
    if (__comp(*__pp, *__cp))
    {
      return __cp;
    }
    ++__p;
    ++__pp;
    __c = 2 * __p + 1;
  }
  return __last;
}

template <class _RandomAccessIterator, class _Compare>
[[nodiscard]] _CCCL_API constexpr _RandomAccessIterator
is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
  return _CUDA_VSTD::__is_heap_until(__first, __last, static_cast<__comp_ref_type<_Compare>>(__comp));
}

template <class _RandomAccessIterator>
[[nodiscard]] _CCCL_API constexpr _RandomAccessIterator
is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
  return _CUDA_VSTD::__is_heap_until(__first, __last, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_IS_HEAP_UNTIL_H
