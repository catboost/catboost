//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_CORRESPONDING_MEMBER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_CORRESPONDING_MEMBER_H

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

#if defined(_CCCL_BUILTIN_IS_CORRESPONDING_MEMBER)

template <class _S1, class _S2, class _M1, class _M2>
[[nodiscard]] _CCCL_API constexpr bool is_corresponding_member(_M1 _S1::* __m1_ptr, _M2 _S2::* __m2_ptr) noexcept
{
  return _CCCL_BUILTIN_IS_CORRESPONDING_MEMBER(_S1, _S2, __m1_ptr, __m2_ptr);
}

#endif // _CCCL_BUILTIN_IS_CORRESPONDING_MEMBER

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_CORRESPONDING_MEMBER_H
