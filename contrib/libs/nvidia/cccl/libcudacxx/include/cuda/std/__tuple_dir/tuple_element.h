//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_TUPLE_ELEMENT_H
#define _LIBCUDACXX___TUPLE_TUPLE_ELEMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/add_cv.h>
#include <cuda/std/__type_traits/add_volatile.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element;

template <size_t _Ip, class _Tp>
using tuple_element_t _CCCL_NODEBUG_ALIAS = typename tuple_element<_Ip, _Tp>::type;

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = volatile tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const volatile tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class... _Types>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, __tuple_types<_Types...>>
{
  static_assert(_Ip < sizeof...(_Types), "tuple_element index out of range");
  using type _CCCL_NODEBUG_ALIAS = __type_index_c<_Ip, _Types...>;
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TUPLE_TUPLE_ELEMENT_H
