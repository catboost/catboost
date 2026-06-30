//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_LEXICOGRAPHICAL_COMPARE_H
#define _LIBCUDACXX___ALGORITHM_LEXICOGRAPHICAL_COMPARE_H

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
#include <cuda/std/__iterator/iterator_traits.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Compare, class _InputIterator1, class _InputIterator2>
[[nodiscard]] _CCCL_API constexpr bool __lexicographical_compare(
  _InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2, _Compare __comp)
{
  for (; __first2 != __last2; ++__first1, (void) ++__first2)
  {
    if (__first1 == __last1 || __comp(*__first1, *__first2))
    {
      return true;
    }
    if (__comp(*__first2, *__first1))
    {
      return false;
    }
  }
  return false;
}

template <class _InputIterator1, class _InputIterator2, class _Compare>
[[nodiscard]] _CCCL_API constexpr bool lexicographical_compare(
  _InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2, _Compare __comp)
{
  return __lexicographical_compare<__comp_ref_type<_Compare>>(__first1, __last1, __first2, __last2, __comp);
}

template <class _InputIterator1, class _InputIterator2>
[[nodiscard]] _CCCL_API constexpr bool lexicographical_compare(
  _InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2)
{
  return _CUDA_VSTD::lexicographical_compare(__first1, __last1, __first2, __last2, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_LEXICOGRAPHICAL_COMPARE_H
