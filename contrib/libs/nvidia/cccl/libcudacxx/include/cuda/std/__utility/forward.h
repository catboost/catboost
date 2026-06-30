// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_FORWARD_H
#define _LIBCUDACXX___UTILITY_FORWARD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_BUILTIN_STD_FORWARD()

// The compiler treats ::std::forward_like as a builtin function so it does not need to be
// instantiated and will be compiled away even at -O0.
using ::std::forward;

#else // ^^^ _CCCL_HAS_BUILTIN_STD_FORWARD() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_FORWARD() vvv

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr _Tp&& forward(remove_reference_t<_Tp>& __t) noexcept
{
  return static_cast<_Tp&&>(__t);
}

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr _Tp&& forward(remove_reference_t<_Tp>&& __t) noexcept
{
  static_assert(!is_lvalue_reference<_Tp>::value, "cannot forward an rvalue as an lvalue");
  return static_cast<_Tp&&>(__t);
}

#endif // _CCCL_HAS_BUILTIN_STD_FORWARD()

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_FORWARD_H
