//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_CAN_EXTRACT_KEY_H
#define _LIBCUDACXX___TYPE_TRAITS_CAN_EXTRACT_KEY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_const_ref.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// These traits are used in __tree and __hash_table
struct __extract_key_fail_tag
{};
struct __extract_key_self_tag
{};
struct __extract_key_first_tag
{};

template <class _ValTy, class _Key, class _RawValTy = __remove_const_ref_t<_ValTy>>
struct __can_extract_key
    : conditional_t<_IsSame<_RawValTy, _Key>::value, __extract_key_self_tag, __extract_key_fail_tag>
{};

template <class _Pair, class _Key, class _First, class _Second>
struct __can_extract_key<_Pair, _Key, pair<_First, _Second>>
    : conditional_t<_IsSame<remove_const_t<_First>, _Key>::value, __extract_key_first_tag, __extract_key_fail_tag>
{};

// __can_extract_map_key uses true_type/false_type instead of the tags.
// It returns true if _Key != _ContainerValueTy (the container is a map not a set)
// and _ValTy == _Key.
template <class _ValTy, class _Key, class _ContainerValueTy, class _RawValTy = __remove_const_ref_t<_ValTy>>
struct __can_extract_map_key : integral_constant<bool, _IsSame<_RawValTy, _Key>::value>
{};

// This specialization returns __extract_key_fail_tag for non-map containers
// because _Key == _ContainerValueTy
template <class _ValTy, class _Key, class _RawValTy>
struct __can_extract_map_key<_ValTy, _Key, _Key, _RawValTy> : false_type
{};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_CAN_EXTRACT_KEY_H
