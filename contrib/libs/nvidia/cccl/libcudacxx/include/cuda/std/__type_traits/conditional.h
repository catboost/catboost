//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_CONDITIONAL_H
#define _LIBCUDACXX___TYPE_TRAITS_CONDITIONAL_H

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

template <bool>
struct _IfImpl;

template <>
struct _IfImpl<true>
{
  template <class _IfRes, class _ElseRes>
  using _Select _CCCL_NODEBUG_ALIAS = _IfRes;
};

template <>
struct _IfImpl<false>
{
  template <class _IfRes, class _ElseRes>
  using _Select _CCCL_NODEBUG_ALIAS = _ElseRes;
};

template <bool _Cond, class _IfRes, class _ElseRes>
using _If _CCCL_NODEBUG_ALIAS = typename _IfImpl<_Cond>::template _Select<_IfRes, _ElseRes>;

template <bool _Bp, class _If, class _Then>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional
{
  using type = _If;
};
template <class _If, class _Then>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional<false, _If, _Then>
{
  using type = _Then;
};

template <bool _Bp, class _If, class _Then>
using conditional_t _CCCL_NODEBUG_ALIAS = typename conditional<_Bp, _If, _Then>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_CONDITIONAL_H
