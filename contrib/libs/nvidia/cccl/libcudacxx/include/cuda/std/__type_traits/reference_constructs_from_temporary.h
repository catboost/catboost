//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REFERENCE_CONSTRUCTS_FROM_TEMPORARY_H
#define _LIBCUDACXX___TYPE_TRAITS_REFERENCE_CONSTRUCTS_FROM_TEMPORARY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
template <class _Tp, class _Up>
struct reference_constructs_from_temporary
    : integral_constant<bool, _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY(_Tp, _Up)>
{};

template <class _Tp, class _Up>
inline constexpr bool reference_constructs_from_temporary_v =
  _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY(_Tp, _Up);

#else

template <class _Tp, class _Up>
struct reference_constructs_from_temporary : integral_constant<bool, false>
{
  static_assert(__always_false_v<_Tp>, "The compiler does not support __reference_constructs_from_temporary");
};

template <class _Tp, class _Up>
inline constexpr bool reference_constructs_from_temporary_v = reference_constructs_from_temporary<_Tp, _Up>::value;

#endif // !_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_REFERENCE_CONSTRUCTS_FROM_TEMPORARY_H
