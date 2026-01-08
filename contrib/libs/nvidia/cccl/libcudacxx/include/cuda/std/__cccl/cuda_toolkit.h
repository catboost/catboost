//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_CUDA_TOOLKIT_H
#define __CCCL_CUDA_TOOLKIT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION() || _CCCL_HAS_INCLUDE(<cuda_runtime_api.h>)
#  define _CCCL_HAS_CTK() 1
#else // ^^^ has cuda toolkit ^^^ / vvv no cuda toolkit vvv
#  define _CCCL_HAS_CTK() 0
#endif // ^^^ no cuda toolkit ^^^

// CUDA compilers preinclude cuda_runtime.h, so we need to include it here to get the CUDART_VERSION macro
#if _CCCL_HAS_CTK() && !_CCCL_CUDA_COMPILATION()
#  include <cuda_runtime_api.h>
#endif // _CCCL_HAS_CTK() && !_CCCL_CUDA_COMPILATION()

// Check compatibility of the CUDA compiler and CUDA toolkit headers
#if _CCCL_CUDA_COMPILATION()
#  if !_CCCL_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#    error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
#  endif // !_CCCL_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#endif // _CCCL_CUDA_COMPILATION()

#if _CCCL_HAS_CTK()
#  define _CCCL_CTK() (CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10)
#else // ^^^ has cuda toolkit ^^^ / vvv no cuda toolkit vvv
#  define _CCCL_CTK() _CCCL_VERSION_INVALID()
#endif // ^^^ no cuda toolkit ^^^

#define _CCCL_CTK_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)
#define _CCCL_CTK_BELOW(...)                   _CCCL_VERSION_COMPARE(_CCCL_CTK_, _CCCL_CTK, <, __VA_ARGS__)
#define _CCCL_CTK_AT_LEAST(...)                _CCCL_VERSION_COMPARE(_CCCL_CTK_, _CCCL_CTK, >=, __VA_ARGS__)

#endif // __CCCL_CUDA_TOOLKIT_H
