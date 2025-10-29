// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDLIB_ALIGNED_ALLOC_H
#define _LIBCUDACXX___CSTDLIB_ALIGNED_ALLOC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__cstdlib/malloc.h>
#include <cuda/std/cstring>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION()
extern "C" _CCCL_DEVICE void* __cuda_syscall_aligned_malloc(size_t, size_t);
#endif // _CCCL_CUDA_COMPILATION()

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if !_CCCL_COMPILER(NVRTC)
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST void*
__aligned_alloc_host([[maybe_unused]] size_t __nbytes, [[maybe_unused]] size_t __align) noexcept
{
#  if _CCCL_OS(WINDOWS)
  _CCCL_ASSERT(false, "Use of aligned_alloc in host code is not supported on WIndows");
  return nullptr;
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
  return ::aligned_alloc(__align, __nbytes);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^
}
#endif // !_CCCL_COMPILER(NVRTC)

[[nodiscard]] _CCCL_API inline void* aligned_alloc(size_t __nbytes, size_t __align) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return _CUDA_VSTD::__aligned_alloc_host(__nbytes, __align);),
                    (return ::__cuda_syscall_aligned_malloc(__nbytes, __align);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CSTDLIB_ALIGNED_ALLOC_H
