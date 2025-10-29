//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_ASSERT_H
#define __CCCL_ASSERT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/attributes.h>
#include <cuda/std/__cccl/builtin.h>
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__cccl/preprocessor.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <assert.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

#if defined(_DEBUG) || defined(DEBUG)
#  ifndef _CCCL_ENABLE_DEBUG_MODE
#    define _CCCL_ENABLE_DEBUG_MODE
#  endif // !_CCCL_ENABLE_DEBUG_MODE
#endif // _DEBUG || DEBUG

// Automatically enable assertions when debug mode is enabled
#ifdef _CCCL_ENABLE_DEBUG_MODE
#  ifndef CCCL_ENABLE_ASSERTIONS
#    define CCCL_ENABLE_ASSERTIONS
#  endif // !CCCL_ENABLE_ASSERTIONS
#endif // _CCCL_ENABLE_DEBUG_MODE

//! Ensure that we switch on host assertions when all assertions are enabled
#ifndef CCCL_ENABLE_HOST_ASSERTIONS
#  ifdef CCCL_ENABLE_ASSERTIONS
#    define CCCL_ENABLE_HOST_ASSERTIONS
#  endif // CCCL_ENABLE_ASSERTIONS
#endif // !CCCL_ENABLE_HOST_ASSERTIONS

//! Ensure that we switch on device assertions when all assertions are enabled
#ifndef CCCL_ENABLE_DEVICE_ASSERTIONS
#  if defined(CCCL_ENABLE_ASSERTIONS) || defined(__CUDACC_DEBUG__)
#    define CCCL_ENABLE_DEVICE_ASSERTIONS
#  endif // CCCL_ENABLE_ASSERTIONS
#endif // !CCCL_ENABLE_DEVICE_ASSERTIONS

//! Use the different standard library implementations to implement host side asserts
//! _CCCL_ASSERT_IMPL_HOST should never be used directly
#if _CCCL_OS(QNX)
#  define _CCCL_ASSERT_IMPL_HOST(expression, message) ((void) 0)
#elif _CCCL_COMPILER(NVRTC) // There is no host standard library in nvrtc
#  define _CCCL_ASSERT_IMPL_HOST(expression, message) ((void) 0)
#elif _CCCL_HAS_INCLUDE(<yvals.h>) && _CCCL_COMPILER(MSVC) // MSVC uses _STL_VERIFY from <yvals.h>
#  include <yvals.h>
#  define _CCCL_ASSERT_IMPL_HOST(expression, message) _STL_VERIFY(expression, message)
#else // ^^^ MSVC STL ^^^ / vvv !MSVC STL vvv
#  ifdef NDEBUG
// Reintroduce the __assert_fail declaration
extern "C" {
#    if !_CCCL_CUDA_COMPILER(CLANG)
_CCCL_HOST_DEVICE
#    endif // !_CCCL_CUDA_COMPILER(CLANG)
  void
  __assert_fail(const char* __assertion, const char* __file, unsigned int __line, const char* __function) noexcept
  __attribute__((__noreturn__));
}
#  endif // NDEBUG
#  define _CCCL_ASSERT_IMPL_HOST(expression, message)      \
    _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
    ? (void) 0 : __assert_fail(message, __FILE__, __LINE__, __func__)
#endif // !MSVC STL

//! Use custom implementations with nvcc on device and the host ones with clang-cuda and nvhpc
//! _CCCL_ASSERT_IMPL_DEVICE should never be used directly
#if _CCCL_OS(QNX)
#  define _CCCL_ASSERT_IMPL_DEVICE(expression, message) ((void) 0)
#elif _CCCL_COMPILER(NVRTC)
#  define _CCCL_ASSERT_IMPL_DEVICE(expression, message)    \
    _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
    ? (void) 0 : __assertfail(message, __FILE__, __LINE__, __func__, sizeof(char))
#elif _CCCL_CUDA_COMPILER(NVCC) //! Use __assert_fail to implement device side asserts
#  if _CCCL_COMPILER(MSVC)
#    define _CCCL_ASSERT_IMPL_DEVICE(expression, message)    \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : _wassert(_CRT_WIDE(#message), __FILEW__, __LINE__)
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#    define _CCCL_ASSERT_IMPL_DEVICE(expression, message)    \
      _CCCL_BUILTIN_EXPECT(static_cast<bool>(expression), 1) \
      ? (void) 0 : __assert_fail(message, __FILE__, __LINE__, __func__)
#  endif // !_CCCL_COMPILER(MSVC)
#elif _CCCL_CUDA_COMPILATION()
#  define _CCCL_ASSERT_IMPL_DEVICE(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
#  define _CCCL_ASSERT_IMPL_DEVICE(expression, message) ((void) 0)
#endif // !_CCCL_CUDA_COMPILATION()

//! _CCCL_ASSERT_HOST is enabled conditionally depending on CCCL_ENABLE_HOST_ASSERTIONS
#ifdef CCCL_ENABLE_HOST_ASSERTIONS
#  define _CCCL_ASSERT_HOST(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#else // ^^^ CCCL_ENABLE_HOST_ASSERTIONS ^^^ / vvv !CCCL_ENABLE_HOST_ASSERTIONS vvv
#  define _CCCL_ASSERT_HOST(expression, message) ((void) 0)
#endif // !CCCL_ENABLE_HOST_ASSERTIONS

//! _CCCL_ASSERT_DEVICE is enabled conditionally depending on CCCL_ENABLE_DEVICE_ASSERTIONS
#ifdef CCCL_ENABLE_DEVICE_ASSERTIONS
#  define _CCCL_ASSERT_DEVICE(expression, message) _CCCL_ASSERT_IMPL_DEVICE(expression, message)
#else // ^^^ CCCL_ENABLE_DEVICE_ASSERTIONS ^^^ / vvv !CCCL_ENABLE_DEVICE_ASSERTIONS vvv
#  define _CCCL_ASSERT_DEVICE(expression, message) ((void) 0)
#endif // !CCCL_ENABLE_DEVICE_ASSERTIONS

//! _CCCL_VERIFY is enabled unconditionally and reserved for critical checks that are required to always be on
//! _CCCL_ASSERT is enabled conditionally depending on CCCL_ENABLE_HOST_ASSERTIONS and CCCL_ENABLE_DEVICE_ASSERTIONS
#if _CCCL_CUDA_COMPILER(NVHPC) // NVHPC can't have different behavior for host and device.
                               // The host version of the assert will also work in device code.
#  define _CCCL_VERIFY(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#  if defined(CCCL_ENABLE_HOST_ASSERTIONS) || defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
#    define _CCCL_ASSERT(expression, message) _CCCL_ASSERT_HOST(expression, message)
#  else
#    define _CCCL_ASSERT(expression, message) ((void) 0)
#  endif
#elif _CCCL_CUDA_COMPILATION()
#  if _CCCL_DEVICE_COMPILATION()
#    define _CCCL_VERIFY(expression, message) _CCCL_ASSERT_IMPL_DEVICE(expression, message)
#    define _CCCL_ASSERT(expression, message) _CCCL_ASSERT_DEVICE(expression, message)
#  else // ^^^ _CCCL_DEVICE_COMPILATION() ^^^ / vvv !_CCCL_DEVICE_COMPILATION() vvv
#    define _CCCL_VERIFY(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#    define _CCCL_ASSERT(expression, message) _CCCL_ASSERT_HOST(expression, message)
#  endif // !_CCCL_DEVICE_COMPILATION()
#else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
#  define _CCCL_VERIFY(expression, message) _CCCL_ASSERT_IMPL_HOST(expression, message)
#  define _CCCL_ASSERT(expression, message) _CCCL_ASSERT_HOST(expression, message)
#endif // !_CCCL_CUDA_COMPILATION()

#endif // __CCCL_ASSERT_H
