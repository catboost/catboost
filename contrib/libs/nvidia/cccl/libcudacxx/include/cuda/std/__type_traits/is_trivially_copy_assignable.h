//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_COPY_ASSIGNABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_COPY_ASSIGNABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/is_trivially_assignable.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_TRIVIALLY_ASSIGNABLE_FALLBACK)

template <class _Tp>
struct is_trivially_copy_assignable
    : public integral_constant<bool,
                               _CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE(
                                 add_lvalue_reference_t<_Tp>, add_lvalue_reference_t<typename add_const<_Tp>::type>)>
{};

template <class _Tp>
inline constexpr bool is_trivially_copy_assignable_v = _CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE(
  add_lvalue_reference_t<_Tp>, add_lvalue_reference_t<typename add_const<_Tp>::type>);

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_trivially_copy_assignable
    : public is_trivially_assignable<add_lvalue_reference_t<_Tp>, add_lvalue_reference_t<typename add_const<_Tp>::type>>
{};

template <class _Tp>
inline constexpr bool is_trivially_copy_assignable_v = is_trivially_copy_assignable<_Tp>::value;

#endif // defined(_CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_TRIVIALLY_ASSIGNABLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_COPY_ASSIGNABLE_H
