// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDLIB_ABS_H
#define _LIBCUDACXX___CSTDLIB_ABS_H

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

[[nodiscard]] _CCCL_API constexpr int abs(int __val) noexcept
{
  return (__val < 0) ? -__val : __val;
}

[[nodiscard]] _CCCL_API constexpr long labs(long __val) noexcept
{
  return (__val < 0l) ? -__val : __val;
}

[[nodiscard]] _CCCL_API constexpr long abs(long __val) noexcept
{
  return _CUDA_VSTD::labs(__val);
}

[[nodiscard]] _CCCL_API constexpr long long llabs(long long __val) noexcept
{
  return (__val < 0ll) ? -__val : __val;
}

[[nodiscard]] _CCCL_API constexpr long long abs(long long __val) noexcept
{
  return _CUDA_VSTD::llabs(__val);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CSTDLIB_ABS_H
