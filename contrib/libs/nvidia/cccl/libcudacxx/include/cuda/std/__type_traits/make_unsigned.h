//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_MAKE_UNSIGNED_H
#define _LIBCUDACXX___TYPE_TRAITS_MAKE_UNSIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/nat.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_MAKE_UNSIGNED) && !defined(_LIBCUDACXX_USE_MAKE_UNSIGNED_FALLBACK)

template <class _Tp>
using make_unsigned_t _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_MAKE_UNSIGNED(_Tp);

#else
using __unsigned_types =
  __type_list<unsigned char,
              unsigned short,
              unsigned int,
              unsigned long,
              unsigned long long
#  if _CCCL_HAS_INT128()
              ,
              __uint128_t
#  endif // _CCCL_HAS_INT128()
              >;

template <class _Tp, bool = is_integral<_Tp>::value || is_enum<_Tp>::value>
struct __make_unsigned_impl
{};

template <class _Tp>
struct __make_unsigned_impl<_Tp, true>
{
  struct __size_greater_equal_fn
  {
    template <class _Up>
    using __call = bool_constant<(sizeof(_Tp) <= sizeof(_Up))>;
  };

  using type = __type_front<__type_find_if<__unsigned_types, __size_greater_equal_fn>>;
};

template <>
struct __make_unsigned_impl<bool, true>
{};
template <>
struct __make_unsigned_impl<signed short, true>
{
  using type = unsigned short;
};
template <>
struct __make_unsigned_impl<unsigned short, true>
{
  using type = unsigned short;
};
template <>
struct __make_unsigned_impl<signed int, true>
{
  using type = unsigned int;
};
template <>
struct __make_unsigned_impl<unsigned int, true>
{
  using type = unsigned int;
};
template <>
struct __make_unsigned_impl<signed long, true>
{
  using type = unsigned long;
};
template <>
struct __make_unsigned_impl<unsigned long, true>
{
  using type = unsigned long;
};
template <>
struct __make_unsigned_impl<signed long long, true>
{
  using type = unsigned long long;
};
template <>
struct __make_unsigned_impl<unsigned long long, true>
{
  using type = unsigned long long;
};
#  if _CCCL_HAS_INT128()
template <>
struct __make_unsigned_impl<__int128_t, true>
{
  using type = __uint128_t;
};
template <>
struct __make_unsigned_impl<__uint128_t, true>
{
  using type = __uint128_t;
};
#  endif // _CCCL_HAS_INT128()

template <class _Tp>
using make_unsigned_t _CCCL_NODEBUG_ALIAS = __copy_cvref_t<_Tp, typename __make_unsigned_impl<remove_cv_t<_Tp>>::type>;

#endif // defined(_CCCL_BUILTIN_MAKE_UNSIGNED) && !defined(_LIBCUDACXX_USE_MAKE_UNSIGNED_FALLBACK)

template <class _Tp>
struct make_unsigned
{
  using type _CCCL_NODEBUG_ALIAS = make_unsigned_t<_Tp>;
};

template <class _Tp>
_CCCL_API constexpr make_unsigned_t<_Tp> __to_unsigned_like(_Tp __x) noexcept
{
  return static_cast<make_unsigned_t<_Tp>>(__x);
}

template <class _Tp, class _Up>
using __copy_unsigned_t _CCCL_NODEBUG_ALIAS = conditional_t<_CCCL_TRAIT(is_unsigned, _Tp), make_unsigned_t<_Up>, _Up>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_MAKE_UNSIGNED_H
