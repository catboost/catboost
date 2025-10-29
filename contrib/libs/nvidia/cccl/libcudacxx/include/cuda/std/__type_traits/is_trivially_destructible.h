//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_DESTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_DESTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_destructible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__type_traits/remove_all_extents.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_TRIVIALLY_DESTRUCTIBLE_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_trivially_destructible : public integral_constant<bool, _CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE(_Tp)>
{};

#elif defined(_CCCL_BUILTIN_HAS_TRIVIAL_DESTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_TRIVIAL_DESTRUCTOR_FALLBACK)

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_trivially_destructible
    : public integral_constant<bool, is_destructible<_Tp>::value && _CCCL_BUILTIN_HAS_TRIVIAL_DESTRUCTOR(_Tp)>
{};
_CCCL_SUPPRESS_DEPRECATED_POP

#else

template <class _Tp>
struct __cccl_trivial_destructor : public integral_constant<bool, is_scalar<_Tp>::value || is_reference<_Tp>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_trivially_destructible : public __cccl_trivial_destructor<remove_all_extents_t<_Tp>>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_trivially_destructible<_Tp[]> : public false_type
{};

#endif // defined(_CCCL_BUILTIN_HAS_TRIVIAL_DESTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_TRIVIAL_DESTRUCTOR_FALLBACK)

template <class _Tp>
inline constexpr bool is_trivially_destructible_v = is_trivially_destructible<_Tp>::value;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_DESTRUCTIBLE_H
