//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_TOTALLY_ORDERED_H
#define _LIBCUDACXX___CONCEPTS_TOTALLY_ORDERED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/boolean_testable.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__type_traits/common_reference.h>
#include <cuda/std/__type_traits/make_const_lvalue_ref.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.totallyordered]

template <class _Tp, class _Up>
concept __partially_ordered_with = requires(__make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u) {
  { __t < __u } -> __boolean_testable;
  { __t > __u } -> __boolean_testable;
  { __t <= __u } -> __boolean_testable;
  { __t >= __u } -> __boolean_testable;
  { __u < __t } -> __boolean_testable;
  { __u > __t } -> __boolean_testable;
  { __u <= __t } -> __boolean_testable;
  { __u >= __t } -> __boolean_testable;
};

template <class _Tp>
concept totally_ordered = equality_comparable<_Tp> && __partially_ordered_with<_Tp, _Tp>;

template <class _Tp, class _Up>
concept totally_ordered_with =
  totally_ordered<_Tp> && totally_ordered<_Up> && equality_comparable_with<_Tp, _Up>
  && totally_ordered<common_reference_t<__make_const_lvalue_ref<_Tp>, __make_const_lvalue_ref<_Up>>>
  && __partially_ordered_with<_Tp, _Up>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(
  __partially_ordered_with_,
  requires(__make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u)(
    requires(__boolean_testable<decltype(__t < __u)>),
    requires(__boolean_testable<decltype(__t > __u)>),
    requires(__boolean_testable<decltype(__t <= __u)>),
    requires(__boolean_testable<decltype(__t >= __u)>),
    requires(__boolean_testable<decltype(__u < __t)>),
    requires(__boolean_testable<decltype(__u > __t)>),
    requires(__boolean_testable<decltype(__u <= __t)>),
    requires(__boolean_testable<decltype(__u >= __t)>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT __partially_ordered_with = _CCCL_FRAGMENT(__partially_ordered_with_, _Tp, _Up);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__totally_ordered_,
                       requires()(requires(equality_comparable<_Tp>), requires(__partially_ordered_with<_Tp, _Tp>)));

template <class _Tp>
_CCCL_CONCEPT totally_ordered = _CCCL_FRAGMENT(__totally_ordered_, _Tp);

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(
  __totally_ordered_with_,
  requires()(requires(totally_ordered<_Tp>),
             requires(totally_ordered<_Up>),
             requires(equality_comparable_with<_Tp, _Up>),
             requires(totally_ordered<common_reference_t<__make_const_lvalue_ref<_Tp>, __make_const_lvalue_ref<_Up>>>),
             requires(__partially_ordered_with<_Tp, _Up>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT totally_ordered_with = _CCCL_FRAGMENT(__totally_ordered_with_, _Tp, _Up);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_TOTALLY_ORDERED_H
