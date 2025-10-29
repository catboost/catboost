//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/is_trivially_default_constructible.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_TRIVIAL) && !defined(_LIBCUDACXX_USE_IS_TRIVIAL_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_trivial : public integral_constant<bool, _CCCL_BUILTIN_IS_TRIVIAL(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_trivial_v = _CCCL_BUILTIN_IS_TRIVIAL(_Tp);

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_trivial
    : public integral_constant<bool, is_trivially_copyable<_Tp>::value && is_trivially_default_constructible<_Tp>::value>
{};

template <class _Tp>
inline constexpr bool is_trivial_v = is_trivial<_Tp>::value;

#endif // defined(_CCCL_BUILTIN_IS_TRIVIAL) && !defined(_LIBCUDACXX_USE_IS_TRIVIAL_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIAL_H
