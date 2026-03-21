//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PUSH_HEAP_H
#define _LIBCUDACXX___ALGORITHM_PUSH_HEAP_H

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
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_CCCL_API constexpr void
__sift_up(_RandomAccessIterator __first,
          _RandomAccessIterator __last,
          _Compare&& __comp,
          typename iterator_traits<_RandomAccessIterator>::difference_type __len)
{
  using value_type = typename iterator_traits<_RandomAccessIterator>::value_type;

  if (__len > 1)
  {
    __len                       = (__len - 2) / 2;
    _RandomAccessIterator __ptr = __first + __len;

    if (__comp(*__ptr, *--__last))
    {
      value_type __t(_IterOps<_AlgPolicy>::__iter_move(__last));
      do
      {
        *__last = _IterOps<_AlgPolicy>::__iter_move(__ptr);
        __last  = __ptr;
        if (__len == 0)
        {
          break;
        }
        __len = (__len - 1) / 2;
        __ptr = __first + __len;
      } while (__comp(*__ptr, __t));

      *__last = _CUDA_VSTD::move(__t);
    }
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _RandomAccessIterator, class _Compare>
_CCCL_API constexpr void __push_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare& __comp)
{
  typename iterator_traits<_RandomAccessIterator>::difference_type __len = __last - __first;
  _CUDA_VSTD::__sift_up<_AlgPolicy, __comp_ref_type<_Compare>>(
    _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __comp, __len);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _RandomAccessIterator, class _Compare>
_CCCL_API constexpr void push_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
  static_assert(_CUDA_VSTD::is_copy_constructible<_RandomAccessIterator>::value,
                "Iterators must be copy constructible.");
  static_assert(_CUDA_VSTD::is_copy_assignable<_RandomAccessIterator>::value, "Iterators must be copy assignable.");

  _CUDA_VSTD::__push_heap<_ClassicAlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __comp);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _RandomAccessIterator>
_CCCL_API constexpr void push_heap(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
  _CUDA_VSTD::push_heap(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_PUSH_HEAP_H
