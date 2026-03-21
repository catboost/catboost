//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_IGNORE_H
#define _LIBCUDACXX___TUPLE_IGNORE_H

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

struct __ignore_t
{
  __ignore_t() = default;

  template <class _Tp, class... _Ts>
  _CCCL_API constexpr __ignore_t(const _Tp&, const _Ts&...) noexcept
  {}

  template <class _Tp>
  _CCCL_API constexpr const __ignore_t& operator=(const _Tp&) const noexcept
  {
    return *this;
  }
};

namespace
{
_CCCL_GLOBAL_CONSTANT __ignore_t ignore = __ignore_t{};
} // namespace

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TUPLE_IGNORE_H
