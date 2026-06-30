//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_POINTER) && !defined(_LIBCUDACXX_USE_IS_POINTER_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_pointer : public bool_constant<_CCCL_BUILTIN_IS_POINTER(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_pointer_v = _CCCL_BUILTIN_IS_POINTER(_Tp);

#else

template <class _Tp>
inline constexpr bool __cccl_is_pointer_helper_v = false;

template <class _Tp>
inline constexpr bool __cccl_is_pointer_helper_v<_Tp*> = true;

template <class _Tp>
inline constexpr bool is_pointer_v = __cccl_is_pointer_helper_v<remove_cv_t<_Tp>>;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_pointer : public bool_constant<is_pointer_v<_Tp>>
{};

#endif // defined(_CCCL_BUILTIN_IS_POINTER) && !defined(_LIBCUDACXX_USE_IS_POINTER_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_H
