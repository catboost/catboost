//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
inline constexpr bool __cccl_is_unsigned_integer_v = false;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned char> = true;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned short> = true;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned int> = true;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned long> = true;

template <>
inline constexpr bool __cccl_is_unsigned_integer_v<unsigned long long> = true;

#if _CCCL_HAS_INT128()
template <>
inline constexpr bool __cccl_is_unsigned_integer_v<__uint128_t> = true;
#endif // _CCCL_HAS_INT128()

template <class _Tp>
inline constexpr bool __cccl_is_cv_unsigned_integer_v = __cccl_is_unsigned_integer_v<remove_cv_t<_Tp>>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H
