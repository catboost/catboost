//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_RTTI_H
#define __CCCL_RTTI_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/builtin.h>

// NOTE: some compilers support the `typeid` feature but not the `dynamic_cast`
// feature. This is why we have separate macros for each.

#ifndef _CCCL_NO_RTTI
#  if defined(CCCL_DISABLE_RTTI) // Escape hatch for users to manually disable RTTI
#    define _CCCL_NO_RTTI
#  elif defined(__CUDA_ARCH__)
#    define _CCCL_NO_RTTI // No RTTI in CUDA device code
#  elif _CCCL_COMPILER(NVRTC)
#    define _CCCL_NO_RTTI
#  elif _CCCL_COMPILER(MSVC)
#    if _CPPRTTI == 0
#      define _CCCL_NO_RTTI
#    endif
#  elif _CCCL_COMPILER(CLANG)
#    if !_CCCL_HAS_FEATURE(cxx_rtti)
#      define _CCCL_NO_RTTI
#    endif
#  else
#    if __GXX_RTTI == 0 && __cpp_rtti == 0
#      define _CCCL_NO_RTTI
#    endif
#  endif
#endif // !_CCCL_NO_RTTI

#ifndef _CCCL_NO_TYPEID
#  if defined(CCCL_DISABLE_RTTI) // CCCL_DISABLE_RTTI disables typeid also
#    define _CCCL_NO_TYPEID
#  elif defined(__CUDA_ARCH__)
#    define _CCCL_NO_TYPEID // No typeid in CUDA device code
#  elif _CCCL_COMPILER(NVRTC)
#    define _CCCL_NO_TYPEID
#  elif _CCCL_COMPILER(MSVC)
// No-op, MSVC always supports typeid even when RTTI is disabled
#  elif _CCCL_COMPILER(CLANG)
#    if !_CCCL_HAS_FEATURE(cxx_rtti)
#      define _CCCL_NO_TYPEID
#    endif
#  else
#    if __GXX_RTTI == 0 && __cpp_rtti == 0
#      define _CCCL_NO_TYPEID
#    endif
#  endif
#endif // !_CCCL_NO_TYPEID

#endif // __CCCL_RTTI_H
