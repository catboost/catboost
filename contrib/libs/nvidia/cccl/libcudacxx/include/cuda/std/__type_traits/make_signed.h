//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_MAKE_SIGNED_H
#define _LIBCUDACXX___TYPE_TRAITS_MAKE_SIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/nat.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_MAKE_SIGNED) && !defined(_LIBCUDACXX_USE_MAKE_SIGNED_FALLBACK)

template <class _Tp>
using make_signed_t _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_MAKE_SIGNED(_Tp);

#else
using __signed_types =
  __type_list<signed char,
              signed short,
              signed int,
              signed long,
              signed long long
#  if _CCCL_HAS_INT128()
              ,
              __int128_t
#  endif // _CCCL_HAS_INT128()
              >;

template <class _Tp, bool = is_integral<_Tp>::value || is_enum<_Tp>::value>
struct __make_signed_impl
{};

template <class _Tp>
struct __make_signed_impl<_Tp, true>
{
  struct __size_greater_equal_fn
  {
    template <class _Up>
    using __call = bool_constant<(sizeof(_Tp) <= sizeof(_Up))>;
  };

  using type = __type_front<__type_find_if<__signed_types, __size_greater_equal_fn>>;
};

template <>
struct __make_signed_impl<bool, true>
{};
template <>
struct __make_signed_impl<signed short, true>
{
  using type = short;
};
template <>
struct __make_signed_impl<unsigned short, true>
{
  using type = short;
};
template <>
struct __make_signed_impl<signed int, true>
{
  using type = int;
};
template <>
struct __make_signed_impl<unsigned int, true>
{
  using type = int;
};
template <>
struct __make_signed_impl<signed long, true>
{
  using type = long;
};
template <>
struct __make_signed_impl<unsigned long, true>
{
  using type = long;
};
template <>
struct __make_signed_impl<signed long long, true>
{
  using type = long long;
};
template <>
struct __make_signed_impl<unsigned long long, true>
{
  using type = long long;
};
#  if _CCCL_HAS_INT128()
template <>
struct __make_signed_impl<__int128_t, true>
{
  using type = __int128_t;
};
template <>
struct __make_signed_impl<__uint128_t, true>
{
  using type = __int128_t;
};
#  endif // _CCCL_HAS_INT128()

template <class _Tp>
using make_signed_t _CCCL_NODEBUG_ALIAS = __copy_cvref_t<_Tp, typename __make_signed_impl<remove_cv_t<_Tp>>::type>;

#endif // defined(_CCCL_BUILTIN_MAKE_SIGNED) && !defined(_LIBCUDACXX_USE_MAKE_SIGNED_FALLBACK)

template <class _Tp>
struct make_signed
{
  using type _CCCL_NODEBUG_ALIAS = make_signed_t<_Tp>;
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_MAKE_SIGNED_H
