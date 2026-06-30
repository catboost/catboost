// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_ALIGN_H
#define _LIBCUDACXX___MEMORY_ALIGN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__memory/runtime_assume_aligned.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4146) // unary minus operator applied to unsigned type, result still unsigned

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_API inline void* align(size_t __alignment, size_t __size, void*& __ptr, size_t& __space)
{
  _CCCL_ASSERT(::cuda::is_power_of_two(__alignment), "cuda::std::align: alignment must be a power of two!");
  if (__space < __size)
  {
    return nullptr;
  }

  char* __char_ptr = static_cast<char*>(__ptr);
  char* __aligned_ptr =
    reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(__char_ptr + (__alignment - 1)) & -__alignment);
  const size_t __diff = static_cast<size_t>(__aligned_ptr - __char_ptr);
  if (__diff > (__space - __size))
  {
    return nullptr;
  }

  //! We need to avoid using __aligned_ptr here, as nvcc looses track of the execution space otherwise
  __ptr = reinterpret_cast<void*>(__char_ptr + __diff);
  __space -= __diff;
  return _CUDA_VSTD::__runtime_assume_aligned(__ptr, __alignment);
}

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_ALIGN_H
