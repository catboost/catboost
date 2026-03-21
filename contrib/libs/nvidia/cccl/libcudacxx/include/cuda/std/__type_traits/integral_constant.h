//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_INTEGRAL_CONSTANT_H
#define _LIBCUDACXX___TYPE_TRAITS_INTEGRAL_CONSTANT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, _Tp __v>
struct _CCCL_TYPE_VISIBILITY_DEFAULT integral_constant
{
  static constexpr const _Tp value = __v;
  using value_type                 = _Tp;
  using type                       = integral_constant;
  _CCCL_API constexpr operator value_type() const noexcept
  {
    return value;
  }
  _CCCL_API constexpr value_type operator()() const noexcept
  {
    return value;
  }
};

template <class _Tp, _Tp __v>
constexpr const _Tp integral_constant<_Tp, __v>::value;

using true_type  = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <bool _Val>
using _BoolConstant _LIBCUDACXX_DEPRECATED _CCCL_NODEBUG_ALIAS = integral_constant<bool, _Val>;

template <bool __b>
using bool_constant = integral_constant<bool, __b>;

// deprecated [Since 2.7.0]
#define _LIBCUDACXX_BOOL_CONSTANT(__b) bool_constant<(__b)>

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_INTEGRAL_CONSTANT_H
