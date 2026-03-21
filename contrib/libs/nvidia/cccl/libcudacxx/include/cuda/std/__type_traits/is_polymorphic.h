//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_POLYMORPHIC_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_POLYMORPHIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_class.h> // __two
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_POLYMORPHIC) && !defined(_LIBCUDACXX_USE_IS_POLYMORPHIC_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_polymorphic : public integral_constant<bool, _CCCL_BUILTIN_IS_POLYMORPHIC(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_polymorphic_v = _CCCL_BUILTIN_IS_POLYMORPHIC(_Tp);

#else

template <typename _Tp>
_CCCL_HOST_DEVICE char& __is_polymorphic_impl(
  enable_if_t<sizeof((_Tp*) dynamic_cast<const volatile void*>(_CUDA_VSTD::declval<_Tp*>())) != 0, int>);
template <typename _Tp>
_CCCL_HOST_DEVICE __two& __is_polymorphic_impl(...);

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_polymorphic : public integral_constant<bool, sizeof(__is_polymorphic_impl<_Tp>(0)) == 1>
{};

template <class _Tp>
inline constexpr bool is_polymorphic_v = is_polymorphic<_Tp>::value;

#endif // defined(_CCCL_BUILTIN_IS_POLYMORPHIC) && !defined(_LIBCUDACXX_USE_IS_POLYMORPHIC_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_POLYMORPHIC_H
