//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_HAS_VIRTUAL_DESTRUCTOR_H
#define _LIBCUDACXX___TYPE_TRAITS_HAS_VIRTUAL_DESTRUCTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_HAS_VIRTUAL_DESTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_VIRTUAL_DESTRUCTOR_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
has_virtual_destructor : public integral_constant<bool, _CCCL_BUILTIN_HAS_VIRTUAL_DESTRUCTOR(_Tp)>
{};

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT has_virtual_destructor : public false_type
{};

#endif // defined(_CCCL_BUILTIN_HAS_VIRTUAL_DESTRUCTOR) && !defined(_LIBCUDACXX_USE_HAS_VIRTUAL_DESTRUCTOR_FALLBACK)

template <class _Tp>
inline constexpr bool has_virtual_destructor_v = has_virtual_destructor<_Tp>::value;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_HAS_VIRTUAL_DESTRUCTOR_H
