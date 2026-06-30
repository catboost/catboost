//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SAME_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SAME_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_SAME) && !defined(_LIBCUDACXX_USE_IS_SAME_FALLBACK)

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_same : bool_constant<_CCCL_BUILTIN_IS_SAME(_Tp, _Up)>
{};

template <class _Tp, class _Up>
inline constexpr bool is_same_v = _CCCL_BUILTIN_IS_SAME(_Tp, _Up);

// _IsSame<T,U> has the same effect as is_same<T,U> but instantiates fewer types:
// is_same<A,B> and is_same<C,D> are guaranteed to be different types, but
// _IsSame<A,B> and _IsSame<C,D> are the same type (namely, false_type).
// Neither GCC nor Clang can mangle the __is_same builtin, so _IsSame
// mustn't be directly used anywhere that contributes to name-mangling
// (such as in a dependent return type).

template <class _Tp, class _Up>
using _IsSame = bool_constant<_CCCL_BUILTIN_IS_SAME(_Tp, _Up)>;

template <class _Tp, class _Up>
using _IsNotSame = bool_constant<!_CCCL_BUILTIN_IS_SAME(_Tp, _Up)>;

#else

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_same : public false_type
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_same<_Tp, _Tp> : public true_type
{};

template <class _Tp, class _Up>
inline constexpr bool is_same_v = false;
template <class _Tp>
inline constexpr bool is_same_v<_Tp, _Tp> = true;

// _IsSame<T,U> has the same effect as is_same<T,U> but instantiates fewer types:
// is_same<A,B> and is_same<C,D> are guaranteed to be different types, but
// _IsSame<A,B> and _IsSame<C,D> are the same type (namely, false_type).
// Neither GCC nor Clang can mangle the __is_same builtin, so _IsSame
// mustn't be directly used anywhere that contributes to name-mangling
// (such as in a dependent return type).

template <class _Tp, class _Up>
using _IsSame = bool_constant<_CCCL_TRAIT(is_same, _Tp, _Up)>;

template <class _Tp, class _Up>
using _IsNotSame = bool_constant<!_CCCL_TRAIT(is_same, _Tp, _Up)>;

#endif // defined(_CCCL_BUILTIN_IS_SAME) && !defined(_LIBCUDACXX_USE_IS_SAME_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SAME_H
