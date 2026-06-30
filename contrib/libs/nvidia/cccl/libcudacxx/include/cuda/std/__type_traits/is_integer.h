//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_INTEGER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_INTEGER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_signed_integer.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __cccl_is_integer is a trait that tests whether a type is an integral type intended for arithmetic.
// In contrast to is_integral, __cccl_is_integer excludes bool and character types.
// See https://eel.is/c++draft/basic.fundamental#1
template <class _Tp>
inline constexpr bool __cccl_is_integer_v = __cccl_is_signed_integer_v<_Tp> || __cccl_is_unsigned_integer_v<_Tp>;

template <class _Tp>
inline constexpr bool __cccl_is_cv_integer_v =
  __cccl_is_cv_signed_integer_v<_Tp> || __cccl_is_cv_unsigned_integer_v<_Tp>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_INTEGER_H
