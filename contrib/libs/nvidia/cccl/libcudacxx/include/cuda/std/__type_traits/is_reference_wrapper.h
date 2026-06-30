//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/reference_wrapper.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
inline constexpr bool __cccl_is_reference_wrapper_v = false;

template <class _Tp>
inline constexpr bool __cccl_is_reference_wrapper_v<reference_wrapper<_Tp>> = true;

template <class _Tp>
inline constexpr bool __cccl_is_reference_wrapper_v<const reference_wrapper<_Tp>> = true;

template <class _Tp>
inline constexpr bool __cccl_is_reference_wrapper_v<volatile reference_wrapper<_Tp>> = true;

template <class _Tp>
inline constexpr bool __cccl_is_reference_wrapper_v<const volatile reference_wrapper<_Tp>> = true;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_ENABLE_IF_H
