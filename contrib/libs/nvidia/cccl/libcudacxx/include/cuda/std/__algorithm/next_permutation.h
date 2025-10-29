//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_NEXT_PERMUTATION_H
#define _LIBCUDACXX___ALGORITHM_NEXT_PERMUTATION_H

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
#include <cuda/std/__algorithm/reverse.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Compare, class _BidirectionalIterator, class _Sentinel>
_CCCL_API constexpr pair<_BidirectionalIterator, bool>
__next_permutation(_BidirectionalIterator __first, _Sentinel __last, _Compare&& __comp)
{
  using _Result = pair<_BidirectionalIterator, bool>;

  _BidirectionalIterator __last_iter = _IterOps<_AlgPolicy>::next(__first, __last);
  _BidirectionalIterator __i         = __last_iter;
  if (__first == __last || __first == --__i)
  {
    return _Result(_CUDA_VSTD::move(__last_iter), false);
  }

  while (true)
  {
    _BidirectionalIterator __ip1 = __i;
    if (__comp(*--__i, *__ip1))
    {
      _BidirectionalIterator __j = __last_iter;
      while (!__comp(*__i, *--__j))
        ;
      _IterOps<_AlgPolicy>::iter_swap(__i, __j);
      _CUDA_VSTD::__reverse<_AlgPolicy>(__ip1, __last_iter);
      return _Result(_CUDA_VSTD::move(__last_iter), true);
    }
    if (__i == __first)
    {
      _CUDA_VSTD::__reverse<_AlgPolicy>(__first, __last_iter);
      return _Result(_CUDA_VSTD::move(__last_iter), false);
    }
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _BidirectionalIterator, class _Compare>
_CCCL_API constexpr bool next_permutation(_BidirectionalIterator __first, _BidirectionalIterator __last, _Compare __comp)
{
  return _CUDA_VSTD::__next_permutation<_ClassicAlgPolicy>(
           _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), static_cast<__comp_ref_type<_Compare>>(__comp))
    .second;
}

template <class _BidirectionalIterator>
_CCCL_API constexpr bool next_permutation(_BidirectionalIterator __first, _BidirectionalIterator __last)
{
  return _CUDA_VSTD::next_permutation(__first, __last, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_NEXT_PERMUTATION_H
