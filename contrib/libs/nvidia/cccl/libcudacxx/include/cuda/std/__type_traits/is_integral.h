//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_INTEGRAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_INTEGRAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_INTEGRAL) && !defined(_LIBCUDACXX_USE_IS_INTEGRAL_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_integral : public bool_constant<_CCCL_BUILTIN_IS_INTEGRAL(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_integral_v = _CCCL_BUILTIN_IS_INTEGRAL(_Tp);

#else // ^^^ _CCCL_BUILTIN_IS_INTEGRAL ^^^ / vvv !_CCCL_BUILTIN_IS_INTEGRAL vvv

template <class _Tp>
inline constexpr bool __cccl_is_integral_helper_v = false;

template <>
inline constexpr bool __cccl_is_integral_helper_v<bool> = true;

// char types

template <>
inline constexpr bool __cccl_is_integral_helper_v<char> = true;

#  if _CCCL_HAS_CHAR8_T()
template <>
inline constexpr bool __cccl_is_integral_helper_v<char8_t> = true;
#  endif // _CCCL_HAS_CHAR8_T()

template <>
inline constexpr bool __cccl_is_integral_helper_v<char16_t> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<char32_t> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<wchar_t> = true;

// signed integer types

template <>
inline constexpr bool __cccl_is_integral_helper_v<signed char> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<short> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<int> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<long> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<long long> = true;

#  if _CCCL_HAS_INT128()
template <>
inline constexpr bool __cccl_is_integral_helper_v<__int128_t> = true;
#  endif // _CCCL_HAS_INT128()

// unsigned integer types

template <>
inline constexpr bool __cccl_is_integral_helper_v<unsigned char> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<unsigned short> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<unsigned int> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<unsigned long> = true;

template <>
inline constexpr bool __cccl_is_integral_helper_v<unsigned long long> = true;

#  if _CCCL_HAS_INT128()
template <>
inline constexpr bool __cccl_is_integral_helper_v<__uint128_t> = true;
#  endif // _CCCL_HAS_INT128()

template <class _Tp>
inline constexpr bool is_integral_v = __cccl_is_integral_helper_v<remove_cv_t<_Tp>>;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_integral : public bool_constant<is_integral_v<_Tp>>
{};

#endif // !_CCCL_BUILTIN_IS_INTEGRAL

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_INTEGRAL_H
