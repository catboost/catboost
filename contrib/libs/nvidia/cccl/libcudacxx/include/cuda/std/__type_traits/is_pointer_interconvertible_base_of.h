//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_INTERCONVERTIBLE_BASE_OF_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_INTERCONVERTIBLE_BASE_OF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_class.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_BASE_OF)

template <class _Tp, class _Up>
inline constexpr bool is_pointer_interconvertible_base_of_v =
  _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_BASE_OF(_Tp, _Up);

#  if _CCCL_COMPILER(CLANG)
// clang's builtin evaluates is_pointer_interconvertible_base_of_v<T, T> to be false which is not right
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<_Tp, _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<_Tp, const _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<_Tp, volatile _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<_Tp, const volatile _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<const _Tp, _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<const _Tp, const _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<const _Tp, volatile _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<const _Tp, const volatile _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<volatile _Tp, _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<volatile _Tp, const _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<volatile _Tp, volatile _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<volatile _Tp, const volatile _Tp> =
  _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<const volatile _Tp, _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<const volatile _Tp, const _Tp> = _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<const volatile _Tp, volatile _Tp> =
  _CCCL_TRAIT(is_class, _Tp);
template <class _Tp>
inline constexpr bool is_pointer_interconvertible_base_of_v<const volatile _Tp, const volatile _Tp> =
  _CCCL_TRAIT(is_class, _Tp);
#  endif // _CCCL_COMPILER(CLANG)

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_pointer_interconvertible_base_of : bool_constant<is_pointer_interconvertible_base_of_v<_Tp, _Up>>
{};

#endif // _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_BASE_OF

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_INTERCONVERTIBLE_BASE_OF_H
