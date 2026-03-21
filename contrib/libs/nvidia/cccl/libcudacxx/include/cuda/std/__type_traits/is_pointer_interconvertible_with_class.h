//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS_H

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

#if defined(_CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS)

template <class _Sp, class _Mp>
[[nodiscard]] _CCCL_API constexpr bool is_pointer_interconvertible_with_class(_Mp _Sp::* __m_ptr) noexcept
{
  return _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS(_Sp, __m_ptr);
}

#endif // _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS_H
