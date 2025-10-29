//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_CONSTANT_EVALUATED_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_CONSTANT_EVALUATED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
_CCCL_API constexpr bool is_constant_evaluated() noexcept
{
  return _CCCL_BUILTIN_IS_CONSTANT_EVALUATED();
}
_CCCL_API constexpr bool __cccl_default_is_constant_evaluated() noexcept
{
  return _CCCL_BUILTIN_IS_CONSTANT_EVALUATED();
}
#else // ^^^ _CCCL_BUILTIN_IS_CONSTANT_EVALUATED ^^^ / vvv !_CCCL_BUILTIN_IS_CONSTANT_EVALUATED vvv
_CCCL_API constexpr bool is_constant_evaluated() noexcept
{
  return false;
}
_CCCL_API constexpr bool __cccl_default_is_constant_evaluated() noexcept
{
  return true;
}
#endif // !_CCCL_BUILTIN_IS_CONSTANT_EVALUATED

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_CONSTANT_EVALUATED_H
