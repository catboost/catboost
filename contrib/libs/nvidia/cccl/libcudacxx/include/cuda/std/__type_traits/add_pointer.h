//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_ADD_POINTER_H
#define _LIBCUDACXX___TYPE_TRAITS_ADD_POINTER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_referenceable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_ADD_POINTER) && !defined(_LIBCUDACXX_USE_ADD_POINTER_FALLBACK)

template <class _Tp>
using add_pointer_t _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_ADD_POINTER(_Tp);

#else // ^^^ _CCCL_BUILTIN_ADD_POINTER ^^^ / vvv !_CCCL_BUILTIN_ADD_POINTER vvv
template <class _Tp, bool = __cccl_is_referenceable<_Tp>::value || is_void<_Tp>::value>
struct __add_pointer_impl
{
  using type _CCCL_NODEBUG_ALIAS = remove_reference_t<_Tp>*;
};
template <class _Tp>
struct __add_pointer_impl<_Tp, false>
{
  using type _CCCL_NODEBUG_ALIAS = _Tp;
};

template <class _Tp>
using add_pointer_t _CCCL_NODEBUG_ALIAS = typename __add_pointer_impl<_Tp>::type;

#endif // !_CCCL_BUILTIN_ADD_POINTER

template <class _Tp>
struct add_pointer
{
  using type _CCCL_NODEBUG_ALIAS = add_pointer_t<_Tp>;
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_ADD_POINTER_H
