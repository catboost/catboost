//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_IN_PLACE_H
#define _LIBCUDACXX___UTILITY_IN_PLACE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_t() = default;
};
_CCCL_GLOBAL_CONSTANT in_place_t in_place{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_type_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_type_t() = default;
};
template <class _Tp>
inline constexpr in_place_type_t<_Tp> in_place_type{};

template <size_t _Idx>
struct _CCCL_TYPE_VISIBILITY_DEFAULT in_place_index_t
{
  _CCCL_HIDE_FROM_ABI explicit in_place_index_t() = default;
};
template <size_t _Idx>
inline constexpr in_place_index_t<_Idx> in_place_index{};

template <class _Tp>
struct __is_inplace_type_imp : false_type
{};
template <class _Tp>
struct __is_inplace_type_imp<in_place_type_t<_Tp>> : true_type
{};

template <class _Tp>
using __is_inplace_type = __is_inplace_type_imp<remove_cvref_t<_Tp>>;

template <class _Tp>
struct __is_inplace_index_imp : false_type
{};
template <size_t _Idx>
struct __is_inplace_index_imp<in_place_index_t<_Idx>> : true_type
{};

template <class _Tp>
using __is_inplace_index = __is_inplace_index_imp<remove_cvref_t<_Tp>>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_IN_PLACE_H
