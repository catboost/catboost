// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDLIB_MALLOC_H
#define _LIBCUDACXX___CSTDLIB_MALLOC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstring>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

using ::free;
using ::malloc;

#if _CCCL_CUDA_COMPILATION()
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void* __calloc_device(size_t __n, size_t __size) noexcept
{
  void* __ptr{};

  const size_t __nbytes = __n * __size;

  if (::__umul64hi(__n, __size) == 0)
  {
    __ptr = _CUDA_VSTD::malloc(__nbytes);
    if (__ptr != nullptr)
    {
      _CUDA_VSTD::memset(__ptr, 0, __nbytes);
    }
  }

  return __ptr;
}
#endif // _CCCL_CUDA_COMPILATION()

[[nodiscard]] _CCCL_API inline void* calloc(size_t __n, size_t __size) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (return ::calloc(__n, __size);), (return _CUDA_VSTD::__calloc_device(__n, __size);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CSTDLIB_MALLOC_H
