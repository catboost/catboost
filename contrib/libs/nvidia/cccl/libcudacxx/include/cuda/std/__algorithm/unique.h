//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_UNIQUE_H
#define _LIBCUDACXX___ALGORITHM_UNIQUE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/adjacent_find.h>
#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Iter, class _Sent, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::pair<_Iter, _Iter>
__unique(_Iter __first, _Sent __last, _BinaryPredicate&& __pred)
{
  __first = _CUDA_VSTD::adjacent_find(__first, __last, __pred);
  if (__first != __last)
  {
    // ...  a  a  ?  ...
    //      f     i
    _Iter __i = __first;
    for (++__i; ++__i != __last;)
    {
      if (!__pred(*__first, *__i))
      {
        *++__first = _IterOps<_AlgPolicy>::__iter_move(__i);
      }
    }
    ++__first;
    return _CUDA_VSTD::pair<_Iter, _Iter>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__i));
  }
  return _CUDA_VSTD::pair<_Iter, _Iter>(__first, __first);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
unique(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred)
{
  return _CUDA_VSTD::__unique<_ClassicAlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __pred).first;
}

template <class _ForwardIterator>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator unique(_ForwardIterator __first, _ForwardIterator __last)
{
  return _CUDA_VSTD::unique(__first, __last, __equal_to{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_UNIQUE_H
