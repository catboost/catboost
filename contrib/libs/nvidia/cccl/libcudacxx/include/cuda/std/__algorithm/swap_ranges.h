// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SWAP_RANGES_H
#define _LIBCUDACXX___ALGORITHM_SWAP_RANGES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// 2+2 iterators: the shorter size will be used.
_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _ForwardIterator1, class _Sentinel1, class _ForwardIterator2, class _Sentinel2>
_CCCL_API constexpr pair<_ForwardIterator1, _ForwardIterator2>
__swap_ranges(_ForwardIterator1 __first1, _Sentinel1 __last1, _ForwardIterator2 __first2, _Sentinel2 __last2)
{
  while (__first1 != __last1 && __first2 != __last2)
  {
    _IterOps<_AlgPolicy>::iter_swap(__first1, __first2);
    ++__first1;
    ++__first2;
  }

  return pair<_ForwardIterator1, _ForwardIterator2>(_CUDA_VSTD::move(__first1), _CUDA_VSTD::move(__first2));
}

// 2+1 iterators: size2 >= size1.
_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _ForwardIterator1, class _Sentinel1, class _ForwardIterator2>
_CCCL_API constexpr pair<_ForwardIterator1, _ForwardIterator2>
__swap_ranges(_ForwardIterator1 __first1, _Sentinel1 __last1, _ForwardIterator2 __first2)
{
  while (__first1 != __last1)
  {
    _IterOps<_AlgPolicy>::iter_swap(__first1, __first2);
    ++__first1;
    ++__first2;
  }

  return pair<_ForwardIterator1, _ForwardIterator2>(_CUDA_VSTD::move(__first1), _CUDA_VSTD::move(__first2));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator1, class _ForwardIterator2>
_CCCL_API constexpr _ForwardIterator2
swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2)
{
  return _CUDA_VSTD::__swap_ranges<_ClassicAlgPolicy>(
           _CUDA_VSTD::move(__first1), _CUDA_VSTD::move(__last1), _CUDA_VSTD::move(__first2))
    .second;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_SWAP_RANGES_H
