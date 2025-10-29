//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_AS_CONST_H
#define _LIBCUDACXX___UTILITY_AS_CONST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/add_const.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_BUILTIN_STD_AS_CONST()

// The compiler treats ::std::as_const as a builtin function so it does not need to be
// instantiated and will be compiled away even at -O0.
using ::std::as_const;

#else // ^^^ _CCCL_HAS_BUILTIN_STD_AS_CONST() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_AS_CONST() vvv

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr add_const_t<_Tp>& as_const(_Tp& __t) noexcept
{
  return __t;
}

template <class _Tp>
void as_const(const _Tp&&) = delete;

#endif // _CCCL_HAS_BUILTIN_STD_AS_CONST()

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_AS_CONST_H
