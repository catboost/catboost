//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_DESTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_DESTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_all_extents.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_DESTRUCTIBLE_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_destructible : public integral_constant<bool, _CCCL_BUILTIN_IS_DESTRUCTIBLE(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_destructible_v = _CCCL_BUILTIN_IS_DESTRUCTIBLE(_Tp);

#else // ^^^ _CCCL_BUILTIN_IS_DESTRUCTIBLE ^^^ / vvv !_CCCL_BUILTIN_IS_DESTRUCTIBLE vvv

//  if it's a reference, return true
//  if it's a function, return false
//  if it's   void,     return false
//  if it's an array of unknown bound, return false
//  Otherwise, return "declval<_Up&>().~_Up()" is well-formed
//    where _Up is remove_all_extents<_Tp>::type

template <class>
struct __is_destructible_apply
{
  using type = int;
};

template <typename _Tp>
struct __is_destructor_wellformed
{
  template <typename _Tp1>
  _CCCL_API inline static true_type
    __test(typename __is_destructible_apply<decltype(_CUDA_VSTD::declval<_Tp1&>().~_Tp1())>::type);

  template <typename _Tp1>
  _CCCL_API inline static false_type __test(...);

  static const bool value = decltype(__test<_Tp>(12))::value;
};

template <class _Tp, bool>
struct __destructible_imp;

template <class _Tp>
struct __destructible_imp<_Tp, false>
    : public integral_constant<bool, __is_destructor_wellformed<remove_all_extents_t<_Tp>>::value>
{};

template <class _Tp>
struct __destructible_imp<_Tp, true> : public true_type
{};

template <class _Tp, bool>
struct __destructible_false;

template <class _Tp>
struct __destructible_false<_Tp, false> : public __destructible_imp<_Tp, is_reference<_Tp>::value>
{};

template <class _Tp>
struct __destructible_false<_Tp, true> : public false_type
{};

template <class _Tp>
struct is_destructible : public __destructible_false<_Tp, is_function<_Tp>::value>
{};

template <class _Tp>
struct is_destructible<_Tp[]> : public false_type
{};

template <>
struct is_destructible<void> : public false_type
{};

template <class _Tp>
inline constexpr bool is_destructible_v = is_destructible<_Tp>::value;

#endif // !_CCCL_BUILTIN_IS_DESTRUCTIBLE

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_DESTRUCTIBLE_H
