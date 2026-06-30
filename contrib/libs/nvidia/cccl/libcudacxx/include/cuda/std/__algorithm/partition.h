//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PARTITION_H
#define _LIBCUDACXX___ALGORITHM_PARTITION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Predicate, class _AlgPolicy, class _ForwardIterator, class _Sentinel>
_CCCL_API constexpr pair<_ForwardIterator, _ForwardIterator>
__partition_impl(_ForwardIterator __first, _Sentinel __last, _Predicate __pred, forward_iterator_tag)
{
  while (true)
  {
    if (__first == __last)
    {
      return _CUDA_VSTD::make_pair(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__first));
    }
    if (!__pred(*__first))
    {
      break;
    }
    ++__first;
  }

  _ForwardIterator __p = __first;
  while (++__p != __last)
  {
    if (__pred(*__p))
    {
      _IterOps<_AlgPolicy>::iter_swap(__first, __p);
      ++__first;
    }
  }
  return _CUDA_VSTD::make_pair(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__p));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Predicate, class _AlgPolicy, class _BidirectionalIterator, class _Sentinel>
_CCCL_API constexpr pair<_BidirectionalIterator, _BidirectionalIterator>
__partition_impl(_BidirectionalIterator __first, _Sentinel __sentinel, _Predicate __pred, bidirectional_iterator_tag)
{
  _BidirectionalIterator __original_last = _IterOps<_AlgPolicy>::next(__first, __sentinel);
  _BidirectionalIterator __last          = __original_last;

  while (true)
  {
    while (true)
    {
      if (__first == __last)
      {
        return _CUDA_VSTD::make_pair(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__original_last));
      }
      if (!__pred(*__first))
      {
        break;
      }
      ++__first;
    }
    do
    {
      if (__first == --__last)
      {
        return _CUDA_VSTD::make_pair(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__original_last));
      }
    } while (!__pred(*__last));
    _IterOps<_AlgPolicy>::iter_swap(__first, __last);
    ++__first;
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _ForwardIterator, class _Sentinel, class _Predicate, class _IterCategory>
_CCCL_API constexpr pair<_ForwardIterator, _ForwardIterator>
__partition(_ForwardIterator __first, _Sentinel __last, _Predicate&& __pred, _IterCategory __iter_category)
{
  return _CUDA_VSTD::__partition_impl<remove_cvref_t<_Predicate>&, _AlgPolicy>(
    _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __pred, __iter_category);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator, class _Predicate>
_CCCL_API constexpr _ForwardIterator partition(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
  using _IterCategory = typename iterator_traits<_ForwardIterator>::iterator_category;
  auto __result       = _CUDA_VSTD::__partition<_ClassicAlgPolicy>(
    _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __pred, _IterCategory());
  return __result.first;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_PARTITION_H
