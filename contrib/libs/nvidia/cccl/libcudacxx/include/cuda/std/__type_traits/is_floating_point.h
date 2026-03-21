//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_FLOATING_POINT_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_FLOATING_POINT_H

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

template <class _Tp>
inline constexpr bool __cccl_is_floating_point_helper_v = false;

template <>
inline constexpr bool __cccl_is_floating_point_helper_v<float> = true;

template <>
inline constexpr bool __cccl_is_floating_point_helper_v<double> = true;

template <>
inline constexpr bool __cccl_is_floating_point_helper_v<long double> = true;

template <class _Tp>
inline constexpr bool is_floating_point_v = __cccl_is_floating_point_helper_v<remove_cv_t<_Tp>>;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_floating_point : public bool_constant<is_floating_point_v<_Tp>>
{};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_FLOATING_POINT_H
