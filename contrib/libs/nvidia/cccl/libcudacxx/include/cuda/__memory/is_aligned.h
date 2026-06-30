//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_IS_ALIGNED_H
#define _CUDA___MEMORY_IS_ALIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

[[nodiscard]] _CCCL_API inline bool is_aligned(const void* __ptr, _CUDA_VSTD::size_t __alignment) noexcept
{
  _CCCL_ASSERT(::cuda::is_power_of_two(__alignment), "alignment must be a power of two");
  return (reinterpret_cast<_CUDA_VSTD::uintptr_t>(__ptr) & (__alignment - 1)) == 0;
}

[[nodiscard]] _CCCL_API inline bool is_aligned(const volatile void* __ptr, _CUDA_VSTD::size_t __alignment) noexcept
{
  return ::cuda::is_aligned(const_cast<const void*>(__ptr), __alignment);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_IS_ALIGN_H
