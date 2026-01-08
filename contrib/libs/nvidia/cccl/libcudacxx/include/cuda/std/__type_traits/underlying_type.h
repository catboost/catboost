//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_UNDERLYING_TYPE_H
#define _LIBCUDACXX___TYPE_TRAITS_UNDERLYING_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_enum.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_UNDERLYING_TYPE) && !defined(_LIBCUDACXX_USE_UNDERLYING_TYPE_FALLBACK)

template <class _Tp, bool = is_enum_v<_Tp>>
struct __cccl_underlying_type_impl
{
  using type = _CCCL_BUILTIN_UNDERLYING_TYPE(_Tp);
};

template <class _Tp>
struct __cccl_underlying_type_impl<_Tp, false>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT underlying_type : __cccl_underlying_type_impl<_Tp>
{};

template <class _Tp>
using underlying_type_t _CCCL_NODEBUG_ALIAS = typename underlying_type<_Tp>::type;

#else // ^^^ _CCCL_BUILTIN_UNDERLYING_TYPE ^^^ / vvv !_CCCL_BUILTIN_UNDERLYING_TYPE vvv

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT underlying_type
{
  static_assert(__always_false_v<_Tp>,
                "The underyling_type trait requires compiler "
                "support. Either no such support exists or "
                "libcu++ does not know how to use it.");
};

#endif // !_CCCL_BUILTIN_UNDERLYING_TYPE

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_UNDERLYING_TYPE_H
