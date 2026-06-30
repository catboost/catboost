//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_UPPER_BOUND_H
#define _LIBCUDACXX___ALGORITHM_UPPER_BOUND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/half_positive.h>
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Compare, class _Iter, class _Sent, class _Tp, class _Proj>
_CCCL_API constexpr _Iter
__upper_bound(_Iter __first, _Sent __last, const _Tp& __value, _Compare&& __comp, _Proj&& __proj)
{
  auto __len = _IterOps<_AlgPolicy>::distance(__first, __last);
  while (__len != 0)
  {
    auto __half_len = _CUDA_VSTD::__half_positive(__len);
    auto __mid      = _IterOps<_AlgPolicy>::next(__first, __half_len);
    if (_CUDA_VSTD::__invoke(__comp, __value, _CUDA_VSTD::__invoke(__proj, *__mid)))
    {
      __len = __half_len;
    }
    else
    {
      __first = ++__mid;
      __len -= __half_len + 1;
    }
  }
  return __first;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator, class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
upper_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, _Compare __comp)
{
  static_assert(is_copy_constructible<_ForwardIterator>::value, "Iterator has to be copy constructible");
  return _CUDA_VSTD::__upper_bound<_ClassicAlgPolicy>(
    _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __value, _CUDA_VSTD::move(__comp), _CUDA_VSTD::identity());
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator, class _Tp>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
upper_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
  return _CUDA_VSTD::upper_bound(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __value, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_UPPER_BOUND_H
