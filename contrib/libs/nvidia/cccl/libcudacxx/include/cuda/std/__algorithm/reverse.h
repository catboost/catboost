//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_REVERSE_H
#define _LIBCUDACXX___ALGORITHM_REVERSE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/iter_swap.h>
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BidirectionalIterator>
_CCCL_API constexpr void
__reverse_impl(_BidirectionalIterator __first, _BidirectionalIterator __last, bidirectional_iterator_tag)
{
  while (__first != __last)
  {
    if (__first == --__last)
    {
      break;
    }
    _IterOps<_AlgPolicy>::iter_swap(__first, __last);
    ++__first;
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _RandomAccessIterator>
_CCCL_API constexpr void
__reverse_impl(_RandomAccessIterator __first, _RandomAccessIterator __last, random_access_iterator_tag)
{
  if (__first != __last)
  {
    for (; __first < --__last; ++__first)
    {
      _IterOps<_AlgPolicy>::iter_swap(__first, __last);
    }
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BidirectionalIterator, class _Sentinel>
_CCCL_API constexpr void __reverse(_BidirectionalIterator __first, _Sentinel __last)
{
  using _IterCategory = typename _IterOps<_AlgPolicy>::template __iterator_category<_BidirectionalIterator>;
  _CUDA_VSTD::__reverse_impl<_AlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), _IterCategory());
}

_CCCL_EXEC_CHECK_DISABLE
template <class _BidirectionalIterator>
_CCCL_API constexpr void reverse(_BidirectionalIterator __first, _BidirectionalIterator __last)
{
  _CUDA_VSTD::__reverse<_ClassicAlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_REVERSE_H
