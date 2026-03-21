//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4197) //  top-level volatile in cast is ignored

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_UNSIGNED) && !defined(_LIBCUDACXX_USE_IS_UNSIGNED_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_unsigned : public bool_constant<_CCCL_BUILTIN_IS_UNSIGNED(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_unsigned_v = _CCCL_BUILTIN_IS_UNSIGNED(_Tp);

#else

template <class _Tp, bool = is_integral_v<_Tp>>
inline constexpr bool __cccl_is_unsigned_helper_v = false;

template <class _Tp>
inline constexpr bool __cccl_is_unsigned_helper_v<_Tp, true> = _Tp(0) < _Tp(-1);

template <class _Tp>
inline constexpr bool is_unsigned_v = is_arithmetic_v<_Tp> && __cccl_is_unsigned_helper_v<_Tp>;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_unsigned : public bool_constant<is_unsigned_v<_Tp>>
{};

#endif // defined(_CCCL_BUILTIN_IS_UNSIGNED) && !defined(_LIBCUDACXX_USE_IS_UNSIGNED_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_H
