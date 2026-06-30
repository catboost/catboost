//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA__TYPE_TRAITS_IS_FLOATING_POINT_H
#define __CUDA__TYPE_TRAITS_IS_FLOATING_POINT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! Tells whether a type is a floating point type, including extended floating point types.
//! Users are allowed to specialize this template for their own types.
template <class _Tp>
inline constexpr bool is_floating_point_v =
  _CUDA_VSTD::is_floating_point_v<_CUDA_VSTD::remove_cv_t<_Tp>>
  || _CUDA_VSTD::__is_extended_floating_point_v<_CUDA_VSTD::remove_cv_t<_Tp>>;

// we define the trait as alias, so users cannot specialize it (they should specialize the variable template instead)
template <class _Tp>
using is_floating_point = _CUDA_VSTD::bool_constant<is_floating_point_v<_Tp>>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA__TYPE_TRAITS_IS_FLOATING_POINT_H
