//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REMOVE_VOLATILE_H
#define _LIBCUDACXX___TYPE_TRAITS_REMOVE_VOLATILE_H

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

#if defined(_CCCL_BUILTIN_REMOVE_VOLATILE) && !defined(_LIBCUDACXX_USE_REMOVE_VOLATILE_FALLBACK)
template <class _Tp>
struct remove_volatile
{
  using type _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_REMOVE_VOLATILE(_Tp);
};

template <class _Tp>
using remove_volatile_t _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_REMOVE_VOLATILE(_Tp);

#else
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_volatile
{
  using type = _Tp;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_volatile<volatile _Tp>
{
  using type = _Tp;
};

template <class _Tp>
using remove_volatile_t _CCCL_NODEBUG_ALIAS = typename remove_volatile<_Tp>::type;

#endif // defined(_CCCL_BUILTIN_REMOVE_VOLATILE) && !defined(_LIBCUDACXX_USE_REMOVE_VOLATILE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_REMOVE_VOLATILE_H
