//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_TUPLE_SIZE_H
#define _LIBCUDACXX___TUPLE_TUPLE_SIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size;

template <class _Tp, class...>
using __enable_if_tuple_size_imp = _Tp;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__enable_if_tuple_size_imp<const _Tp,
                                      enable_if_t<!is_volatile<_Tp>::value>,
                                      integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : public integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__enable_if_tuple_size_imp<volatile _Tp,
                                      enable_if_t<!is_const<_Tp>::value>,
                                      integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : public integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__enable_if_tuple_size_imp<const volatile _Tp, integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : public integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size<tuple<_Tp...>> : public integral_constant<size_t, sizeof...(_Tp)>
{};

template <class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__tuple_types<_Tp...>> : public integral_constant<size_t, sizeof...(_Tp)>
{};

template <class _Tp>
inline constexpr size_t tuple_size_v = tuple_size<_Tp>::value;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TUPLE_TUPLE_SIZE_H
