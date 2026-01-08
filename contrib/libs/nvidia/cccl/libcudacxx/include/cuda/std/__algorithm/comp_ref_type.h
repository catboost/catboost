//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_COMP_REF_TYPE_H
#define _LIBCUDACXX___ALGORITHM_COMP_REF_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Compare>
struct __debug_less
{
  _Compare& __comp_;
  _CCCL_API constexpr __debug_less(_Compare& __c)
      : __comp_(__c)
  {}

  template <class _Tp, class _Up>
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __x, const _Up& __y)
  {
    bool __r = __comp_(__x, __y);
    if (__r)
    {
      __do_compare_assert(0, __y, __x);
    }
    return __r;
  }

  template <class _Tp, class _Up>
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp& __x, _Up& __y)
  {
    bool __r = __comp_(__x, __y);
    if (__r)
    {
      __do_compare_assert(0, __y, __x);
    }
    return __r;
  }

  template <class _LHS, class _RHS>
  _CCCL_API constexpr decltype((void) declval<_Compare&>()(declval<_LHS&>(), declval<_RHS&>()))
  __do_compare_assert(int, [[maybe_unused]] _LHS& __l, [[maybe_unused]] _RHS& __r)
  {
    _CCCL_ASSERT(!__comp_(__l, __r), "Comparator does not induce a strict weak ordering");
  }

  template <class _LHS, class _RHS>
  _CCCL_API constexpr void __do_compare_assert(long, _LHS&, _RHS&)
  {}
};

// Pass the comparator by lvalue reference. Or in debug mode, using a
// debugging wrapper that stores a reference.
#ifdef _CCCL_ENABLE_DEBUG_MODE
template <class _Comp>
using __comp_ref_type = __debug_less<_Comp>;
#else // ^^^ _LIBCUDACXX_ENABLE_DEBUG_MODE ^^^ / vvv !_LIBCUDACXX_ENABLE_DEBUG_MODE vvv
template <class _Comp>
using __comp_ref_type = _Comp&;
#endif // !_LIBCUDACXX_ENABLE_DEBUG_MODE

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_COMP_REF_TYPE_H
