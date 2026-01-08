//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_ITER_SWAP_H
#define _LIBCUDACXX___ALGORITHM_ITER_SWAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

//! Intentionally not an algorithm to avoid breaking types that pull in `::std::iter_swap` via ADL
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__iter_swap)
// "Poison pill" overload to intentionally create ambiguity with the unconstrained
// `std::iter_swap` function.
template <class _ForwardIterator1, class _ForwardIterator2>
void iter_swap(_ForwardIterator1, _ForwardIterator2) = delete;

template <class _ForwardIterator1, class _ForwardIterator2>
_CCCL_CONCEPT __unqualified_iter_swap =
  _CCCL_REQUIRES_EXPR((_ForwardIterator1, _ForwardIterator2), _ForwardIterator1&& __a, _ForwardIterator2&& __b)(
    iter_swap(_CUDA_VSTD::forward<_ForwardIterator1>(__a), _CUDA_VSTD::forward<_ForwardIterator2>(__b)));

template <class _ForwardIterator1, class _ForwardIterator2>
_CCCL_CONCEPT __readable_swappable =
  _CCCL_REQUIRES_EXPR((_ForwardIterator1, _ForwardIterator2), _ForwardIterator1 __a, _ForwardIterator2 __b)(
    requires(!__unqualified_iter_swap<_ForwardIterator1, _ForwardIterator2>), swap(*__a, *__b));

struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _ForwardIterator1, class _ForwardIterator2)
  _CCCL_REQUIRES(__unqualified_iter_swap<_ForwardIterator1, _ForwardIterator2>)
  _CCCL_API constexpr void operator()(_ForwardIterator1&& __a, _ForwardIterator2&& __b) const
    noexcept(noexcept(iter_swap(_CUDA_VSTD::declval<_ForwardIterator1>(), _CUDA_VSTD::declval<_ForwardIterator2>())))
  {
    (void) iter_swap(_CUDA_VSTD::forward<_ForwardIterator1>(__a), _CUDA_VSTD::forward<_ForwardIterator2>(__b));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _ForwardIterator1, class _ForwardIterator2)
  _CCCL_REQUIRES(__readable_swappable<_ForwardIterator1, _ForwardIterator2>)
  _CCCL_API constexpr void operator()(_ForwardIterator1&& __a, _ForwardIterator2&& __b) const
    noexcept(noexcept(swap(*_CUDA_VSTD::declval<_ForwardIterator1>(), *_CUDA_VSTD::declval<_ForwardIterator2>())))
  {
    swap(*__a, *__b);
  }
};

_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
// This is a global constant to avoid breaking types that pull in `::std::iter_swap` via ADL
_CCCL_GLOBAL_CONSTANT auto iter_swap = __iter_swap::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_ITER_SWAP_H
