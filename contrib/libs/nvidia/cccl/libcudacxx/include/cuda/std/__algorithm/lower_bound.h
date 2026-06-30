//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_LOWER_BOUND_H
#define _LIBCUDACXX___ALGORITHM_LOWER_BOUND_H

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
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Iter, class _Sent, class _Type, class _Proj, class _Comp>
_CCCL_API constexpr _Iter __lower_bound(_Iter __first, _Sent __last, const _Type& __value, _Comp& __comp, _Proj& __proj)
{
  auto __len = _IterOps<_AlgPolicy>::distance(__first, __last);

  while (__len != 0)
  {
    auto __l2 = _CUDA_VSTD::__half_positive(__len);
    _Iter __m = __first;
    _IterOps<_AlgPolicy>::advance(__m, __l2);
    if (_CUDA_VSTD::__invoke(__comp, _CUDA_VSTD::__invoke(__proj, *__m), __value))
    {
      __first = ++__m;
      __len -= __l2 + 1;
    }
    else
    {
      __len = __l2;
    }
  }
  return __first;
}

template <class _ForwardIterator, class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, _Compare __comp)
{
  static_assert(__is_callable<_Compare, decltype(*__first), const _Tp&>::value, "The comparator has to be callable");
  auto __proj = _CUDA_VSTD::identity();
  return _CUDA_VSTD::__lower_bound<_ClassicAlgPolicy>(__first, __last, __value, __comp, __proj);
}

template <class _ForwardIterator, class _Tp>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
  return _CUDA_VSTD::lower_bound(__first, __last, __value, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_LOWER_BOUND_H
