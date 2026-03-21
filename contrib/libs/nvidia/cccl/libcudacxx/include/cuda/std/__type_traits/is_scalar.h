//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SCALAR_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SCALAR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_member_pointer.h>
#include <cuda/std/__type_traits/is_null_pointer.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_SCALAR) && !defined(_LIBCUDACXX_USE_IS_SCALAR_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_scalar : public bool_constant<_CCCL_BUILTIN_IS_SCALAR(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_scalar_v = _CCCL_BUILTIN_IS_SCALAR(_Tp);

#else

template <class _Tp>
inline constexpr bool is_scalar_v =
  is_arithmetic_v<_Tp> || is_member_pointer_v<_Tp> || is_pointer_v<_Tp> || is_null_pointer_v<_Tp> || is_enum_v<_Tp>;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_scalar : public bool_constant<is_scalar_v<_Tp>>
{};

#endif // defined(_CCCL_BUILTIN_IS_SCALAR) && !defined(_LIBCUDACXX_USE_IS_SCALAR_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SCALAR_H
