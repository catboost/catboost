//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MOVE_BACKWARD_H
#define _LIBCUDACXX___ALGORITHM_MOVE_BACKWARD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__algorithm/unwrap_iter.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BidirectionalIterator, class _OutputIterator>
_CCCL_API constexpr pair<_BidirectionalIterator, _OutputIterator>
__move_backward(_BidirectionalIterator __first, _BidirectionalIterator __last, _OutputIterator __result)
{
  while (__first != __last)
  {
    *--__result = _IterOps<_AlgPolicy>::__iter_move(--__last);
  }
  return {__first, __result};
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy,
          class _Tp,
          class _Up,
          enable_if_t<_CCCL_TRAIT(is_same, remove_const_t<_Tp>, _Up), int> = 0,
          enable_if_t<_CCCL_TRAIT(is_trivially_copyable, _Up), int>        = 0>
_CCCL_API constexpr pair<_Tp*, _Up*> __move_backward(_Tp* __first, _Tp* __last, _Up* __result)
{
  const ptrdiff_t __n = __last - __first;
  if (__n > 0)
  {
    if (_CUDA_VSTD::__dispatch_memmove(__result - __n, __first, __n))
    {
      return {__last, __result - __n};
    }
    for (ptrdiff_t __i = 1; __i <= __n; ++__i)
    {
      *(__result - __i) = _IterOps<_AlgPolicy>::__iter_move((__last - __i));
    }
  }
  return {__last, __result - __n};
}

template <class _BidirectionalIterator1, class _BidirectionalIterator2>
_CCCL_API constexpr _BidirectionalIterator2
move_backward(_BidirectionalIterator1 __first, _BidirectionalIterator1 __last, _BidirectionalIterator2 __result)
{
  return _CUDA_VSTD::__move_backward<_ClassicAlgPolicy>(
           _CUDA_VSTD::__unwrap_iter(__first), _CUDA_VSTD::__unwrap_iter(__last), _CUDA_VSTD::__unwrap_iter(__result))
    .second;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_MOVE_BACKWARD_H
