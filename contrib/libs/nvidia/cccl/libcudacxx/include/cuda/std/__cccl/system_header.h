//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_SYSTEM_HEADER_H
#define __CCCL_SYSTEM_HEADER_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/is_non_narrowing_convertible.h> // IWYU pragma: export

// Enforce that cccl headers are treated as system headers
#if _CCCL_COMPILER(GCC) || _CCCL_COMPILER(NVHPC)
#  define _CCCL_FORCE_SYSTEM_HEADER_GCC
#elif _CCCL_COMPILER(CLANG)
#  define _CCCL_FORCE_SYSTEM_HEADER_CLANG
#elif _CCCL_COMPILER(MSVC)
#  define _CCCL_FORCE_SYSTEM_HEADER_MSVC
#endif // other compilers

// Potentially enable that cccl headers are treated as system headers
#if !defined(_CCCL_NO_SYSTEM_HEADER) && !(_CCCL_COMPILER(MSVC) && defined(_LIBCUDACXX_DISABLE_PRAGMA_MSVC_WARNING)) \
  && !_CCCL_COMPILER(NVRTC) && !defined(_LIBCUDACXX_DISABLE_PRAGMA_GCC_SYSTEM_HEADER)
#  if _CCCL_COMPILER(GCC) || _CCCL_COMPILER(NVHPC)
#    define _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  elif _CCCL_COMPILER(CLANG)
#    define _CCCL_IMPLICIT_SYSTEM_HEADER_CLANG
#  elif _CCCL_COMPILER(MSVC)
#    define _CCCL_IMPLICIT_SYSTEM_HEADER_MSVC
#  endif // other compilers
#endif // Use system header

#endif // __CCCL_SYSTEM_HEADER_H
