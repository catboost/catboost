//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_INCLUDES_H
#define _LIBCUDACXX___ALGORITHM_INCLUDES_H

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
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Iter1, class _Sent1, class _Iter2, class _Sent2, class _Comp, class _Proj1, class _Proj2>
_CCCL_API constexpr bool __includes(
  _Iter1 __first1, _Sent1 __last1, _Iter2 __first2, _Sent2 __last2, _Comp&& __comp, _Proj1&& __proj1, _Proj2&& __proj2)
{
  for (; __first2 != __last2; ++__first1)
  {
    if (__first1 == __last1
        || _CUDA_VSTD::__invoke(
          __comp, _CUDA_VSTD::__invoke(__proj2, *__first2), _CUDA_VSTD::__invoke(__proj1, *__first1)))
    {
      return false;
    }
    if (!_CUDA_VSTD::__invoke(
          __comp, _CUDA_VSTD::__invoke(__proj1, *__first1), _CUDA_VSTD::__invoke(__proj2, *__first2)))
    {
      ++__first2;
    }
  }
  return true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator1, class _InputIterator2, class _Compare>
[[nodiscard]] _CCCL_API constexpr bool includes(
  _InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2, _Compare __comp)
{
  static_assert(__is_callable<_Compare, decltype(*__first1), decltype(*__first2)>::value,
                "Comparator has to be callable");

  return _CUDA_VSTD::__includes(
    _CUDA_VSTD::move(__first1),
    _CUDA_VSTD::move(__last1),
    _CUDA_VSTD::move(__first2),
    _CUDA_VSTD::move(__last2),
    static_cast<__comp_ref_type<_Compare>>(__comp),
    identity(),
    identity());
}

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator1, class _InputIterator2>
[[nodiscard]] _CCCL_API constexpr bool
includes(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2)
{
  return _CUDA_VSTD::includes(
    _CUDA_VSTD::move(__first1),
    _CUDA_VSTD::move(__last1),
    _CUDA_VSTD::move(__first2),
    _CUDA_VSTD::move(__last2),
    __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_INCLUDES_H
