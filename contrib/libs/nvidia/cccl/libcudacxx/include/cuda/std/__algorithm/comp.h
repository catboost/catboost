//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_COMP_H
#define _LIBCUDACXX___ALGORITHM_COMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/__type_traits/predicate_traits.h>
#endif // _LIBCUDACXX_HAS_STRING

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __equal_to
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _T1& __lhs, const _T2& __rhs) const
    noexcept(noexcept(__lhs == __rhs))
  {
    return __lhs == __rhs;
  }
};

#if defined(_LIBCUDACXX_HAS_STRING)
template <class _Lhs, class _Rhs>
struct __is_trivial_equality_predicate<__equal_to, _Lhs, _Rhs> : true_type
{};
#endif // _LIBCUDACXX_HAS_STRING

struct __less
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class _Up>
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tp& __lhs, const _Up& __rhs) const
    noexcept(noexcept(__lhs < __rhs))
  {
    return __lhs < __rhs;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_COMP_H
