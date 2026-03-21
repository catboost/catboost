//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_MEMBER_POINTER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_MEMBER_POINTER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_member_function_pointer.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_MEMBER_POINTER) && !defined(_LIBCUDACXX_USE_IS_MEMBER_POINTER_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_member_pointer : public integral_constant<bool, _CCCL_BUILTIN_IS_MEMBER_POINTER(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_member_pointer_v = _CCCL_BUILTIN_IS_MEMBER_POINTER(_Tp);

#else // ^^^ _CCCL_BUILTIN_IS_MEMBER_POINTER ^^^ / vvv !_CCCL_BUILTIN_IS_MEMBER_POINTER vvv

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_member_pointer : public integral_constant<bool, __cccl_is_member_pointer<remove_cv_t<_Tp>>::__is_member>
{};

template <class _Tp>
inline constexpr bool is_member_pointer_v = is_member_pointer<_Tp>::value;

#endif // !_CCCL_BUILTIN_IS_MEMBER_POINTER

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_MEMBER_POINTER_H
