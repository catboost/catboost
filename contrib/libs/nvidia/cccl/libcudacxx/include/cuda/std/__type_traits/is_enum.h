//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_ENUM_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_ENUM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_member_pointer.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_union.h>
#include <cuda/std/__type_traits/is_void.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_ENUM) && !defined(_LIBCUDACXX_USE_IS_ENUM_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_enum : public integral_constant<bool, _CCCL_BUILTIN_IS_ENUM(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_enum_v = _CCCL_BUILTIN_IS_ENUM(_Tp);

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_enum
    : public integral_constant<
        bool,
        !is_void<_Tp>::value && !is_integral<_Tp>::value && !is_floating_point<_Tp>::value && !is_array<_Tp>::value
          && !is_pointer<_Tp>::value && !is_reference<_Tp>::value && !is_member_pointer<_Tp>::value
          && !is_union<_Tp>::value && !is_class<_Tp>::value && !is_function<_Tp>::value>
{};

template <class _Tp>
inline constexpr bool is_enum_v = is_enum<_Tp>::value;

#endif // defined(_CCCL_BUILTIN_IS_ENUM) && !defined(_LIBCUDACXX_USE_IS_ENUM_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_ENUM_H
