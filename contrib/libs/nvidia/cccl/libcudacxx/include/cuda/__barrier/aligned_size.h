//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_ALIGNED_SIZE_H
#define _CUDA___BARRIER_ALIGNED_SIZE_H

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

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <_CUDA_VSTD::size_t _Alignment>
struct aligned_size_t
{
  static_assert(::cuda::is_power_of_two(_Alignment), "alignment must be a power of two");

  static constexpr _CUDA_VSTD::size_t align = _Alignment;
  _CUDA_VSTD::size_t value;

  _CCCL_API explicit constexpr aligned_size_t(_CUDA_VSTD::size_t __s)
      : value(__s)
  {
    _CCCL_ASSERT(value % align == 0,
                 "aligned_size_t must be constructed with a size that is a multiple of the alignment");
  }
  _CCCL_API constexpr operator _CUDA_VSTD::size_t() const
  {
    return value;
  }
};

template <class, class = void>
inline constexpr _CUDA_VSTD::size_t __get_size_align_v = 1;

template <class _Tp>
inline constexpr _CUDA_VSTD::size_t __get_size_align_v<_Tp, _CUDA_VSTD::void_t<decltype(_Tp::align)>> = _Tp::align;

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BARRIER_ALIGNED_SIZE_H
