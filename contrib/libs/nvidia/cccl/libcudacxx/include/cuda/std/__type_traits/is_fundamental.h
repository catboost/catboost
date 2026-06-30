//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_FUNDAMENTAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_FUNDAMENTAL_H

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
#include <cuda/std/__type_traits/is_null_pointer.h>
#include <cuda/std/__type_traits/is_void.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_FUNDAMENTAL) && !defined(_LIBCUDACXX_USE_IS_FUNDAMENTAL_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_fundamental : public bool_constant<_CCCL_BUILTIN_IS_FUNDAMENTAL(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_fundamental_v = _CCCL_BUILTIN_IS_FUNDAMENTAL(_Tp);

#else // ^^^ _CCCL_BUILTIN_IS_FUNDAMENTAL ^^^ / vvv !_CCCL_BUILTIN_IS_FUNDAMENTAL vvv

template <class _Tp>
inline constexpr bool is_fundamental_v = is_void_v<_Tp> || is_null_pointer_v<_Tp> || is_arithmetic_v<_Tp>;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_fundamental : public bool_constant<is_fundamental_v<_Tp>>
{};

#endif // !_CCCL_BUILTIN_IS_FUNDAMENTAL

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_FUNDAMENTAL_H
