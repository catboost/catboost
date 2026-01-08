
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_DEPRECATED_H
#define __CCCL_DEPRECATED_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Check for deprecation opt outs
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_DIALECT)
#  if !defined(CCCL_IGNORE_DEPRECATED_CPP_DIALECT)
#    define CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#  endif
#endif // suppress all dialect deprecation warnings
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_14) || defined(CCCL_IGNORE_DEPRECATED_CPP_DIALECT)
#  if !defined(CCCL_IGNORE_DEPRECATED_CPP_14)
#    define CCCL_IGNORE_DEPRECATED_CPP_14
#  endif
#endif // suppress all c++14 dialect deprecation warnings
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_11) || defined(CCCL_IGNORE_DEPRECATED_CPP_DIALECT) \
  || defined(CCCL_IGNORE_DEPRECATED_CPP_14)
#  if !defined(CCCL_IGNORE_DEPRECATED_CPP_11)
#    define CCCL_IGNORE_DEPRECATED_CPP_11
#  endif
#endif // suppress all c++11 dialect deprecation warnings
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_COMPILER) || defined(THRUST_IGNORE_DEPRECATED_COMPILER) \
  || defined(CUB_IGNORE_DEPRECATED_COMPILER) || defined(CCCL_IGNORE_DEPRECATED_CPP_DIALECT)      \
  || defined(CCCL_IGNORE_DEPRECATED_CPP_14) || defined(CCCL_IGNORE_DEPRECATED_CPP_11)
#  if !defined(CCCL_IGNORE_DEPRECATED_COMPILER)
#    define CCCL_IGNORE_DEPRECATED_COMPILER
#  endif
#endif // suppress all compiler deprecation warnings
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_API) || defined(THRUST_IGNORE_DEPRECATED_API) \
  || defined(CUB_IGNORE_DEPRECATED_API)
#  if !defined(CCCL_IGNORE_DEPRECATED_API)
#    define CCCL_IGNORE_DEPRECATED_API
#  endif
#endif // suppress all API deprecation warnings

#ifdef CCCL_IGNORE_DEPRECATED_API
//! deprecated [Since 2.8]
#  define CCCL_DEPRECATED
//! deprecated [Since 2.8]
#  define CCCL_DEPRECATED_BECAUSE(MSG)
#else // ^^^ CCCL_IGNORE_DEPRECATED_API ^^^ / vvv !CCCL_IGNORE_DEPRECATED_API vvv
//! deprecated [Since 2.8]
#  define CCCL_DEPRECATED              [[deprecated]]
//! deprecated [Since 2.8]
#  define CCCL_DEPRECATED_BECAUSE(MSG) [[deprecated(MSG)]]
#endif // !CCCL_IGNORE_DEPRECATED_API

#endif // __CCCL_DEPRECATED_H
