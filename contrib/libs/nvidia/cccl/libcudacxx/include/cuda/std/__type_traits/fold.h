//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_FOLD_H
#define _LIBCUDACXX___TYPE_TRAITS_FOLD_H

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

// Use fold expressions when possible to implement __fold_and[_v] and
// __fold_or[_v].
template <bool... _Preds>
inline constexpr bool __fold_and_v = (_Preds && ...);

template <bool... _Preds>
inline constexpr bool __fold_or_v = (_Preds || ...);

template <bool... _Preds>
using __fold_and = bool_constant<bool((_Preds && ...))>; // cast to bool to avoid error from gcc < 8

template <bool... _Preds>
using __fold_or = bool_constant<bool((_Preds || ...))>; // cast to bool to avoid error from gcc < 8

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_FOLD_H
