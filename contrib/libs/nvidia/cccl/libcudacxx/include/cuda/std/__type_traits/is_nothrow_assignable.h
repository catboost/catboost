//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_ASSIGNABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_ASSIGNABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_NOTHROW_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_ASSIGNABLE_FALLBACK)

template <class _Tp, class _Arg>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_nothrow_assignable : public integral_constant<bool, _CCCL_BUILTIN_IS_NOTHROW_ASSIGNABLE(_Tp, _Arg)>
{};

template <class _Tp, class _Arg>
inline constexpr bool is_nothrow_assignable_v = _CCCL_BUILTIN_IS_NOTHROW_ASSIGNABLE(_Tp, _Arg);

#else // ^^^ _CCCL_BUILTIN_IS_NOTHROW_ASSIGNABLE ^^^ / vvv !_CCCL_BUILTIN_IS_NOTHROW_ASSIGNABLE vvv

template <bool, class _Tp, class _Arg>
struct __cccl_is_nothrow_assignable;

template <class _Tp, class _Arg>
struct __cccl_is_nothrow_assignable<false, _Tp, _Arg> : public false_type
{};

template <class _Tp, class _Arg>
struct __cccl_is_nothrow_assignable<true, _Tp, _Arg>
    : public integral_constant<bool, noexcept(_CUDA_VSTD::declval<_Tp>() = _CUDA_VSTD::declval<_Arg>())>
{};

template <class _Tp, class _Arg>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_nothrow_assignable : public __cccl_is_nothrow_assignable<is_assignable<_Tp, _Arg>::value, _Tp, _Arg>
{};

template <class _Tp, class _Arg>
inline constexpr bool is_nothrow_assignable_v = is_nothrow_assignable<_Tp, _Arg>::value;

#endif // !_CCCL_BUILTIN_IS_NOTHROW_ASSIGNABLE

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_ASSIGNABLE_H
