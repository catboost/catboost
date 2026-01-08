// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_NVFP16_H
#define _LIBCUDACXX___CMATH_NVFP16_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_NVFP16()

#  include <cuda/std/__cmath/copysign.h>
#  include <cuda/std/__floating_point/nvfp_types.h>
#  include <cuda/std/cstdint>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// floating point helper
_CCCL_API constexpr __half __constexpr_copysign(__half __x, __half __y) noexcept
{
  return _CUDA_VSTD::copysign(__x, __y);
}

_CCCL_API inline __half __constexpr_fabs(__half __x) noexcept
{
  return ::__habs(__x);
}

_CCCL_API inline __half __constexpr_fmax(__half __x, __half __y) noexcept
{
  return ::__hmax(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX_HAS_NVFP16()

#endif // _LIBCUDACXX___CMATH_NVFP16_H
