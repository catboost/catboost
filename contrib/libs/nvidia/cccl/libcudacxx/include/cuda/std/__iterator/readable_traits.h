// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_READABLE_TRAITS_H
#define _LIBCUDACXX___ITERATOR_READABLE_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__fwd/iterator_traits.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_primary_template.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_extent.h>
#include <cuda/std/__type_traits/void_t.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

// [readable.traits]
template <class>
struct __cond_value_type
{};

template <class _Tp>
  requires is_object_v<_Tp>
struct __cond_value_type<_Tp>
{
  using value_type = remove_cv_t<_Tp>;
};

template <class _Tp>
concept __has_member_value_type = requires { typename _Tp::value_type; };

template <class _Tp>
concept __has_member_element_type = requires { typename _Tp::element_type; };

template <class>
struct indirectly_readable_traits
{};

template <class _Ip>
  requires is_array_v<_Ip>
struct indirectly_readable_traits<_Ip>
{
  using value_type = remove_cv_t<remove_extent_t<_Ip>>;
};

template <class _Ip>
struct indirectly_readable_traits<const _Ip> : indirectly_readable_traits<_Ip>
{};

template <class _Tp>
struct indirectly_readable_traits<_Tp*> : __cond_value_type<_Tp>
{};

template <__has_member_value_type _Tp>
struct indirectly_readable_traits<_Tp> : __cond_value_type<typename _Tp::value_type>
{};

template <__has_member_element_type _Tp>
struct indirectly_readable_traits<_Tp> : __cond_value_type<typename _Tp::element_type>
{};

template <__has_member_value_type _Tp>
  requires __has_member_element_type<_Tp>
struct indirectly_readable_traits<_Tp>
{};

template <__has_member_value_type _Tp>
  requires __has_member_element_type<_Tp>
        && same_as<remove_cv_t<typename _Tp::element_type>, remove_cv_t<typename _Tp::value_type>>
struct indirectly_readable_traits<_Tp> : __cond_value_type<typename _Tp::value_type>
{};

// Let `RI` be `remove_cvref_t<I>`. The type `iter_value_t<I>` denotes
// `indirectly_readable_traits<RI>::value_type` if `iterator_traits<RI>` names a specialization
// generated from the primary template, and `iterator_traits<RI>::value_type` otherwise.
template <class _Ip>
using iter_value_t =
  typename __select_traits<remove_cvref_t<_Ip>, indirectly_readable_traits<remove_cvref_t<_Ip>>>::value_type;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

// [readable.traits]
template <class, class = void>
struct __cond_value_type
{};

template <class _Tp>
struct __cond_value_type<_Tp, enable_if_t<_CCCL_TRAIT(is_object, _Tp)>>
{
  using value_type = remove_cv_t<_Tp>;
};

template <class _Tp, class = void>
inline constexpr bool __has_member_value_type = false;

template <class _Tp>
inline constexpr bool __has_member_value_type<_Tp, void_t<typename _Tp::value_type>> = true;

template <class _Tp, class = void>
inline constexpr bool __has_member_element_type = false;

template <class _Tp>
inline constexpr bool __has_member_element_type<_Tp, void_t<typename _Tp::element_type>> = true;

template <class, class = void>
struct indirectly_readable_traits
{};

template <class _Ip>
struct indirectly_readable_traits<_Ip, enable_if_t<!_CCCL_TRAIT(is_const, _Ip) && _CCCL_TRAIT(is_array, _Ip)>>
{
  using value_type = remove_cv_t<remove_extent_t<_Ip>>;
};

template <class _Ip>
struct indirectly_readable_traits<const _Ip> : indirectly_readable_traits<_Ip>
{};

template <class _Tp>
struct indirectly_readable_traits<_Tp*> : __cond_value_type<_Tp>
{};

template <class _Tp>
struct indirectly_readable_traits<
  _Tp,
  enable_if_t<!_CCCL_TRAIT(is_const, _Tp) && __has_member_value_type<_Tp> && !__has_member_element_type<_Tp>>>
    : __cond_value_type<typename _Tp::value_type>
{};

template <class _Tp>
struct indirectly_readable_traits<
  _Tp,
  enable_if_t<!_CCCL_TRAIT(is_const, _Tp) && !__has_member_value_type<_Tp> && __has_member_element_type<_Tp>>>
    : __cond_value_type<typename _Tp::element_type>
{};

template <class _Tp>
struct indirectly_readable_traits<
  _Tp,
  enable_if_t<!_CCCL_TRAIT(is_const, _Tp) && __has_member_value_type<_Tp> && __has_member_element_type<_Tp>
              && same_as<remove_cv_t<typename _Tp::element_type>, remove_cv_t<typename _Tp::value_type>>>>
    : __cond_value_type<typename _Tp::value_type>
{};

// Let `RI` be `remove_cvref_t<I>`. The type `iter_value_t<I>` denotes
// `indirectly_readable_traits<RI>::value_type` if `iterator_traits<RI>` names a specialization
// generated from the primary template, and `iterator_traits<RI>::value_type` otherwise.
template <class _Ip>
using iter_value_t =
  typename __select_traits<remove_cvref_t<_Ip>, indirectly_readable_traits<remove_cvref_t<_Ip>>>::value_type;

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_READABLE_TRAITS_H
