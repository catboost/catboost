//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_RUNTIME_ASSUME_ALIGNED_H
#define _LIBCUDACXX___MEMORY_RUNTIME_ASSUME_ALIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/remove_volatile.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
[[nodiscard]] _CCCL_API _Tp* __runtime_assume_aligned(_Tp* __ptr, _CUDA_VSTD::size_t __alignment) noexcept
{
#if defined(_CCCL_BUILTIN_ASSUME_ALIGNED)
  using _Up = remove_volatile_t<_Tp>;
  switch (__alignment)
  {
    case 1:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(const_cast<_Up*>(__ptr), 1));
    case 2:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(const_cast<_Up*>(__ptr), 2));
    case 4:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(const_cast<_Up*>(__ptr), 4));
    case 8:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(const_cast<_Up*>(__ptr), 8));
    case 16:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(const_cast<_Up*>(__ptr), 16));
    default:
      return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(const_cast<_Up*>(__ptr), 32));
  }
#else
  _CCCL_ASSUME(reinterpret_cast<uintptr_t>(__ptr) % __alignment == 0);
  return __ptr;
#endif // defined(_CCCL_BUILTIN_ASSUME_ALIGNED)
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_RUNTIME_ASSUME_ALIGNED_H
