//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_CONSTRUCTIBLE_FALLBACK)

template <class _Tp, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_nothrow_constructible : public integral_constant<bool, _CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE(_Tp, _Args...)>
{};

template <class _Tp, class... _Args>
inline constexpr bool is_nothrow_constructible_v = _CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE(_Tp, _Args...);

#else

template <bool, bool, class _Tp, class... _Args>
struct __cccl_is_nothrow_constructible;

template <class _Tp, class... _Args>
struct __cccl_is_nothrow_constructible</*is constructible*/ true, /*is reference*/ false, _Tp, _Args...>
    : public integral_constant<bool, noexcept(_Tp(_CUDA_VSTD::declval<_Args>()...))>
{};

template <class _Tp>
_CCCL_API inline void __implicit_conversion_to(_Tp) noexcept
{}

template <class _Tp, class _Arg>
struct __cccl_is_nothrow_constructible</*is constructible*/ true, /*is reference*/ true, _Tp, _Arg>
    : public integral_constant<bool, noexcept(__implicit_conversion_to<_Tp>(_CUDA_VSTD::declval<_Arg>()))>
{};

template <class _Tp, bool _IsReference, class... _Args>
struct __cccl_is_nothrow_constructible</*is constructible*/ false, _IsReference, _Tp, _Args...> : public false_type
{};

template <class _Tp, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_constructible
    : __cccl_is_nothrow_constructible<is_constructible<_Tp, _Args...>::value, is_reference<_Tp>::value, _Tp, _Args...>
{};

template <class _Tp, size_t _Ns>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_constructible<_Tp[_Ns]>
    : __cccl_is_nothrow_constructible<is_constructible<_Tp>::value, is_reference<_Tp>::value, _Tp>
{};

template <class _Tp, class... _Args>
inline constexpr bool is_nothrow_constructible_v = is_nothrow_constructible<_Tp, _Args...>::value;

#endif // defined(_CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_CONSTRUCTIBLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H
