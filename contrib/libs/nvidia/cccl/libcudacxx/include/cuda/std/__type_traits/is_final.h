//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_FINAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_FINAL_H

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

#if defined(_CCCL_BUILTIN_IS_FINAL)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_final : public bool_constant<_CCCL_BUILTIN_IS_FINAL(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_final_v = _CCCL_BUILTIN_IS_FINAL(_Tp);

#else // ^^^ _CCCL_BUILTIN_IS_FINAL ^^^ / vvv !_CCCL_BUILTIN_IS_FINAL vvv

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_final : public false_type
{
  static_assert(__always_false_v<_Tp>, "is_final requires compiler support");
};

template <class _Tp>
inline constexpr bool is_final_v = is_final<_Tp>::value;

#endif // !_CCCL_BUILTIN_IS_FINAL

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_FINAL_H
