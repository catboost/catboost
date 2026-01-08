// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_ASSUME_ALIGNED_H
#define _LIBCUDACXX___MEMORY_ASSUME_ALIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/cstddef> // size_t
#include <cuda/std/cstdint> // uintptr_t

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Align, class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp* assume_aligned(_Tp* __ptr) noexcept
{
  static_assert(_CUDA_VSTD::has_single_bit(_Align), "std::assume_aligned requires the alignment to be a power of 2");
  static_assert(_Align >= alignof(_Tp), "Alignment must be greater than or equal to the alignment of the input type");
#if !defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  return __ptr;
#else
  if (!_CUDA_VSTD::is_constant_evaluated())
  {
#  if !_CCCL_COMPILER(MSVC) // MSVC checks within the builtin
    _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % _Align == 0, "Alignment assumption is violated");
#  endif // !_CCCL_COMPILER(MSVC) && defined(_CCCL_BUILTIN_ASSUME_ALIGNED)
#  if defined(_CCCL_BUILTIN_ASSUME_ALIGNED)
    return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(__ptr, _Align));
#  endif // defined(_CCCL_BUILTIN_ASSUME_ALIGNED)
  }
  return __ptr;
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_ASSUME_ALIGNED_H
