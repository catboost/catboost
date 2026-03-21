//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_CLASS_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_CLASS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_union.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __two
{
  char __lx[2];
};

#if defined(_CCCL_BUILTIN_IS_CLASS) && !defined(_LIBCUDACXX_USE_IS_CLASS_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_class : public integral_constant<bool, _CCCL_BUILTIN_IS_CLASS(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_class_v = _CCCL_BUILTIN_IS_CLASS(_Tp);

#else

namespace __is_class_imp
{
template <class _Tp>
_CCCL_HOST_DEVICE char __test(int _Tp::*);
template <class _Tp>
_CCCL_HOST_DEVICE __two __test(...);
} // namespace __is_class_imp

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_class : public integral_constant<bool, sizeof(__is_class_imp::__test<_Tp>(0)) == 1 && !is_union<_Tp>::value>
{};

template <class _Tp>
inline constexpr bool is_class_v = is_class<_Tp>::value;

#endif // defined(_CCCL_BUILTIN_IS_CLASS) && !defined(_LIBCUDACXX_USE_IS_CLASS_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_CLASS_H
