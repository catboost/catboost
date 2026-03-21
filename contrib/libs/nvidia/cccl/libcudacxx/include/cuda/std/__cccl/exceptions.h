//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_EXCEPTIONS_H
#define __CCCL_EXCEPTIONS_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(CCCL_DISABLE_EXCEPTIONS) // Escape hatch for users to manually disable exceptions
#  define _CCCL_HAS_EXCEPTIONS() 0
#elif _CCCL_COMPILER(NVRTC) // NVRTC has no exceptions
#  define _CCCL_HAS_EXCEPTIONS() 0
#elif _CCCL_COMPILER(MSVC) // MSVC needs special checks for `_HAS_EXCEPTIONS` and `_CPPUNWIND`
#  define _CCCL_HAS_EXCEPTIONS() (_HAS_EXCEPTIONS != 0) && (_CPPUNWIND != 0)
#else // other compilers use `__EXCEPTIONS`
#  define _CCCL_HAS_EXCEPTIONS() __EXCEPTIONS
#endif // has exceptions

// The following macros are used to conditionally compile exception handling code. They
// are used in the same way as `try` and `catch`, but they allow for different behavior
// based on whether exceptions are enabled or not, and whether the code is being compiled
// for device or not.
//
// Usage:
//   _CCCL_TRY
//   {
//     can_throw();               // Code that may throw an exception
//   }
//   _CCCL_CATCH (cuda_error& e)  // Handle CUDA exceptions
//   {
//     printf("CUDA error: %s\n", e.what());
//   }
//   _CCCL_CATCH_ALL              // Handle any other exceptions
//   {
//     printf("unknown error\n");
//   }
#if !_CCCL_HAS_EXCEPTIONS() || (_CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC))
#  define _CCCL_TRY if constexpr (true)
#  define _CCCL_CATCH(...)                                              \
    else if constexpr (__VA_ARGS__ = ::__cccl_catch_any_lvalue{}; true) \
    {                                                                   \
    }                                                                   \
    else
#  define _CCCL_CATCH_ALL    \
    else if constexpr (true) \
    {                        \
    }                        \
    else
#else // ^^^ !_CCCL_HAS_EXCEPTIONS() || (_CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)) ^^^
      // vvv _CCCL_HAS_EXCEPTIONS() && (!_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC)) vvv
#  define _CCCL_TRY       try
#  define _CCCL_CATCH     catch
#  define _CCCL_CATCH_ALL catch (...)
#endif // ^^^ _CCCL_HAS_EXCEPTIONS() && (!_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC)) ^^^

struct __cccl_catch_any_lvalue
{
  template <class _Tp>
  _CCCL_HOST_DEVICE operator _Tp&() const noexcept;
};

#endif // __CCCL_EXCEPTIONS_H
