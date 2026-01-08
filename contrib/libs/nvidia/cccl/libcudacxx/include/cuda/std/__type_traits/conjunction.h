//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_CONJUNCTION_H
#define _LIBCUDACXX___TYPE_TRAITS_CONJUNCTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class...>
using __expand_to_true = true_type;

template <class... _Pred>
_CCCL_HOST_DEVICE __expand_to_true<enable_if_t<_Pred::value>...> __and_helper(int);

template <class...>
_CCCL_HOST_DEVICE false_type __and_helper(...);

// _And always performs lazy evaluation of its arguments.
//
// However, `_And<_Pred...>` itself will evaluate its result immediately (without having to
// be instantiated) since it is an alias, unlike `conjunction<_Pred...>`, which is a struct.
// If you want to defer the evaluation of `_And<_Pred...>` itself, use `_Lazy<_And, _Pred...>`.
template <class... _Pred>
using _And _CCCL_NODEBUG_ALIAS = decltype(__and_helper<_Pred...>(0));

template <class...>
struct conjunction : true_type
{};

template <class _Arg>
struct conjunction<_Arg> : _Arg
{};

template <class _Arg, class... _Args>
struct conjunction<_Arg, _Args...> : _If<!bool(_Arg::value), _Arg, conjunction<_Args...>>
{};

template <class... _Args>
inline constexpr bool conjunction_v = conjunction<_Args...>::value;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_CONJUNCTION_H
