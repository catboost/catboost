//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_ARRAY_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_ARRAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// TODO: Clang incorrectly reports that __is_array is true for T[0].
//       Re-enable the branch once https://llvm.org/PR54705 is fixed.
#if defined(_CCCL_BUILTIN_IS_ARRAY) && !defined(_LIBCUDACXX_USE_IS_ARRAY_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_array : public bool_constant<_CCCL_BUILTIN_IS_ARRAY(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_array_v = _CCCL_BUILTIN_IS_ARRAY(_Tp);

#else

template <class _Tp>
inline constexpr bool is_array_v = false;

template <class _Tp>
inline constexpr bool is_array_v<_Tp[]> = true;

template <class _Tp, size_t _Np>
inline constexpr bool is_array_v<_Tp[_Np]> = true;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_array : public bool_constant<is_array_v<_Tp>>
{};

#endif // defined(_CCCL_BUILTIN_IS_ARRAY) && !defined(_LIBCUDACXX_USE_IS_ARRAY_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_ARRAY_H
