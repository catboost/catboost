// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_CONCEPTS_H
#define _LIBCUDACXX___MDSPAN_CONCEPTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __mdspan_detail
{

// [mdspan.layout.stride.expo]/3
template <class>
struct __is_extents : false_type
{};

template <class _Tp>
inline constexpr bool __is_extents_v = __is_extents<_Tp>::value;

// [mdspan.layout.general]/2
template <class _Layout, class _Mapping>
inline constexpr bool __is_mapping_of =
  _CCCL_TRAIT(is_same, typename _Layout::template mapping<typename _Mapping::extents_type>, _Mapping);

// [mdspan.layout.reqmts]/1
template <class _Mapping>
_CCCL_CONCEPT __layout_mapping_req_type = _CCCL_REQUIRES_EXPR((_Mapping))(
  requires(copyable<_Mapping>),
  requires(equality_comparable<_Mapping>),
  requires(_CCCL_TRAIT(is_nothrow_move_constructible, _Mapping)),
  requires(_CCCL_TRAIT(is_move_assignable, _Mapping)),
  requires(_CCCL_TRAIT(is_nothrow_swappable, _Mapping)));

// [mdspan.layout.reqmts]/2-4
template <class _Mapping>
_CCCL_CONCEPT __layout_mapping_req_types = _CCCL_REQUIRES_EXPR((_Mapping))(
  requires(__is_extents_v<typename _Mapping::extents_type>),
  requires(same_as<typename _Mapping::index_type, typename _Mapping::extents_type::index_type>),
  requires(same_as<typename _Mapping::rank_type, typename _Mapping::extents_type::rank_type>),
  requires(__is_mapping_of<typename _Mapping::layout_type, _Mapping>));

template <class _Mapping>
_CCCL_CONCEPT __layout_mapping_req_members = _CCCL_REQUIRES_EXPR((_Mapping), const _Mapping& __map)(
  _Same_as(const typename _Mapping::extents_type&) __map.extents(),
  _Same_as(typename _Mapping::index_type) __map.required_span_size(),
  _Same_as(bool) __map.is_unique(),
  _Same_as(bool) __map.is_exhaustive(),
  _Same_as(bool) __map.is_strided());

template <class _Mapping>
_CCCL_CONCEPT __layout_mapping_req = _CCCL_REQUIRES_EXPR((_Mapping))(
  requires(__layout_mapping_req_type<_Mapping>),
  requires(__layout_mapping_req_types<_Mapping>),
  requires(__layout_mapping_req_members<_Mapping>));

// [mdspan.layout.stride.expo]/4
// NOTE: integral_constant<bool, _Mapping::is_always_strided()>::value only checks that this is a constant expression
template <class _Mapping>
_CCCL_CONCEPT __layout_mapping_alike = _CCCL_REQUIRES_EXPR((_Mapping))(
  requires(__is_mapping_of<typename _Mapping::layout_type, _Mapping>),
  requires(__is_extents_v<typename _Mapping::extents_type>),
  requires(same_as<bool, decltype(_Mapping::is_always_strided())>),
  requires(same_as<bool, decltype(_Mapping::is_always_exhaustive())>),
  requires(same_as<bool, decltype(_Mapping::is_always_unique())>),
  (integral_constant<bool, _Mapping::is_always_strided()>::value),
  (integral_constant<bool, _Mapping::is_always_exhaustive()>::value),
  (integral_constant<bool, _Mapping::is_always_unique()>::value));

template <class _IndexType, class... _Indices>
_CCCL_CONCEPT __all_convertible_to_index_type =
  (_CCCL_TRAIT(is_convertible, _Indices, _IndexType) && ... && true)
  && (_CCCL_TRAIT(is_nothrow_constructible, _IndexType, _Indices) && ... && true);

template <class _Extent, size_t _Size>
static constexpr bool __matches_dynamic_rank = (_Size == _Extent::rank_dynamic());

template <class _Extent, size_t _Size>
static constexpr bool __matches_static_rank = (_Size == _Extent::rank()) && (_Size != _Extent::rank_dynamic());

} // namespace __mdspan_detail

template <class _Tp, class _IndexType>
_CCCL_CONCEPT __index_pair_like = _CCCL_REQUIRES_EXPR((_Tp, _IndexType))(
  requires(__pair_like<_Tp>),
  requires(convertible_to<tuple_element_t<0, _Tp>, _IndexType>),
  requires(convertible_to<tuple_element_t<1, _Tp>, _IndexType>));

// [mdspan.submdspan.strided.slice]/3

template <class _Tp>
_CCCL_CONCEPT __index_like =
  _CCCL_TRAIT(is_signed, _Tp) || _CCCL_TRAIT(is_unsigned, _Tp) || __integral_constant_like<_Tp>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_CONCEPTS_H
