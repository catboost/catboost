//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___INTERNAL_CPP_DIALECT_H
#define _LIBCUDACXX___INTERNAL_CPP_DIALECT_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Define LIBCUDACXX_COMPILER_DEPRECATION macro:
#if _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define LIBCUDACXX_COMP_DEPR_IMPL(msg) \
    _CCCL_PRAGMA(message(__FILE__ ":" _CCCL_TO_STRING(__LINE__) ": warning: " #msg))
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define LIBCUDACXX_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(GCC warning #msg)
#endif // !_CCCL_COMPILER(MSVC)

// clang-format off
#define LIBCUDACXX_DIALECT_DEPRECATION(REQ, CUR)                                                           \
  LIBCUDACXX_COMP_DEPR_IMPL(                                                                               \
    libcu++ requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a \
      future release. Define CCCL_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.)
// clang-format on

#ifndef CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#  if _CCCL_STD_VER < 2017
#    error libcu++ requires at least C++ 17. Define CCCL_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
#  endif // _CCCL_STD_VER < 2017
#endif // CCCL_IGNORE_DEPRECATED_CPP_DIALECT

#endif // _LIBCUDACXX___INTERNAL_CPP_DIALECT_H
