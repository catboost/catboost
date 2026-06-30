//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MAKE_HEAP_H
#define _LIBCUDACXX___ALGORITHM_MAKE_HEAP_H

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
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__algorithm/sift_down.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_CCCL_API constexpr void __make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare&& __comp)
{
  __comp_ref_type<_Compare> __comp_ref = __comp;

  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;
  difference_type __n   = __last - __first;
  if (__n > 1)
  {
    // start from the first parent, there is no need to consider children
    for (difference_type __start = (__n - 2) / 2; __start >= 0; --__start)
    {
      _CUDA_VSTD::__sift_down<_AlgPolicy>(__first, __comp_ref, __n, __first + __start);
    }
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _RandomAccessIterator, class _Compare>
_CCCL_API constexpr void make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
  _CUDA_VSTD::__make_heap<_ClassicAlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __comp);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _RandomAccessIterator>
_CCCL_API constexpr void make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
  _CUDA_VSTD::make_heap(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_MAKE_HEAP_H
