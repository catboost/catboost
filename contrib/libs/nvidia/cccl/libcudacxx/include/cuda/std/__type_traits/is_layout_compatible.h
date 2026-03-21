//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_LAYOUT_COMPATIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_LAYOUT_COMPATIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_LAYOUT_COMPATIBLE)

template <class _Tp, class _Up>
inline constexpr bool is_layout_compatible_v = _CCCL_BUILTIN_IS_LAYOUT_COMPATIBLE(_Tp, _Up);

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_layout_compatible : bool_constant<is_layout_compatible_v<_Tp, _Up>>
{};

#endif // _CCCL_BUILTIN_IS_LAYOUT_COMPATIBLE

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_LAYOUT_COMPATIBLE_H
