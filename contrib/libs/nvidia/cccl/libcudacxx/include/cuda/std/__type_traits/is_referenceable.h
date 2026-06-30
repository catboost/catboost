//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCEABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCEABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_REFERENCEABLE) && !defined(_LIBCUDACXX_USE_IS_REFERENCEABLE_FALLBACK)

template <class _Tp>
struct __cccl_is_referenceable : public integral_constant<bool, _CCCL_BUILTIN_IS_REFERENCEABLE(_Tp)>
{};

#else
struct __cccl_is_referenceable_impl
{
  template <class _Tp>
  _CCCL_HOST_DEVICE static _Tp& __test(int);
  template <class _Tp>
  _CCCL_HOST_DEVICE static false_type __test(...);
};

template <class _Tp>
struct __cccl_is_referenceable
    : integral_constant<bool, _IsNotSame<decltype(__cccl_is_referenceable_impl::__test<_Tp>(0)), false_type>::value>
{};
#endif // defined(_CCCL_BUILTIN_IS_REFERENCEABLE) && !defined(_LIBCUDACXX_USE_IS_REFERENCEABLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCEABLE_H
