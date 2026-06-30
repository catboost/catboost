//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H

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
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// is_nothrow_destructible

#if defined(_CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_DESTRUCTIBLE_FALLBACK)

template <class _Tp>
struct is_nothrow_destructible : public integral_constant<bool, _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_nothrow_destructible_v = _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE(_Tp);

#else // ^^^ _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE ^^^ / vvv !_CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE vvv

template <class _Tp, bool = is_destructible<_Tp>::value>
struct __cccl_is_nothrow_destructible : false_type
{};

template <class _Tp>
struct __cccl_is_nothrow_destructible<_Tp, true>
    : public integral_constant<bool, noexcept(_CUDA_VSTD::declval<_Tp>().~_Tp())>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible : public __cccl_is_nothrow_destructible<_Tp>
{};

template <class _Tp, size_t _Ns>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp[_Ns]> : public is_nothrow_destructible<_Tp>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp&> : public true_type
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp&&> : public true_type
{};

template <class _Tp>
inline constexpr bool is_nothrow_destructible_v = is_nothrow_destructible<_Tp>::value;

#endif // !_CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H
