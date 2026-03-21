//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_DIALECT_H
#define __CCCL_DIALECT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

///////////////////////////////////////////////////////////////////////////////
// Determine the C++ standard dialect
///////////////////////////////////////////////////////////////////////////////
#if _CCCL_COMPILER(MSVC)
#  if _MSVC_LANG <= 201103L
#    define _CCCL_STD_VER 2011
#  elif _MSVC_LANG <= 201402L
#    define _CCCL_STD_VER 2014
#  elif _MSVC_LANG <= 201703L
#    define _CCCL_STD_VER 2017
#  elif _MSVC_LANG <= 202002L
#    define _CCCL_STD_VER 2020
#  else
#    define _CCCL_STD_VER 2023 // current year, or date of c++2b ratification
#  endif
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  if __cplusplus <= 199711L
#    define _CCCL_STD_VER 2003
#  elif __cplusplus <= 201103L
#    define _CCCL_STD_VER 2011
#  elif __cplusplus <= 201402L
#    define _CCCL_STD_VER 2014
#  elif __cplusplus <= 201703L
#    define _CCCL_STD_VER 2017
#  elif __cplusplus <= 202002L
#    define _CCCL_STD_VER 2020
#  elif __cplusplus <= 202302L
#    define _CCCL_STD_VER 2023
#  else
#    define _CCCL_STD_VER 2024 // current year, or date of c++2c ratification
#  endif
#endif // !_CCCL_COMPILER(MSVC)

///////////////////////////////////////////////////////////////////////////////
// Conditionally enable constexpr per standard dialect
///////////////////////////////////////////////////////////////////////////////

#if _CCCL_STD_VER >= 2020
#  define _CCCL_CONSTEXPR_CXX20 constexpr
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
#  define _CCCL_CONSTEXPR_CXX20
#endif // _CCCL_STD_VER <= 2017

#if _CCCL_STD_VER >= 2023
#  define _CCCL_CONSTEXPR_CXX23 constexpr
#else // ^^^ C++23 ^^^ / vvv C++20 vvv
#  define _CCCL_CONSTEXPR_CXX23
#endif // _CCCL_STD_VER <= 2020

///////////////////////////////////////////////////////////////////////////////
// Detect whether we can use some language features based on standard dialect
///////////////////////////////////////////////////////////////////////////////

// concepts are only available from C++20 onwards
#if _CCCL_STD_VER <= 2017 || __cpp_concepts < 201907L
#  define _CCCL_HAS_CONCEPTS() 0
#else // ^^^ no concepts ^^^ / vvv has concepts vvv
#  define _CCCL_HAS_CONCEPTS() 1
#endif // ^^^ has concepts ^^^

// Three way comparison is only available from C++20 onwards
#if _CCCL_STD_VER <= 2017 || __cpp_impl_three_way_comparison < 201907L
#  define _CCCL_NO_THREE_WAY_COMPARISON
#endif // _CCCL_STD_VER <= 2017 || __cpp_impl_three_way_comparison < 201907L

// Some compilers turn on pack indexing in pre-C++26 code. We want to use it if it is
// available.
#if defined(__cpp_pack_indexing) && !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(CLANG, <, 20)
#  define _CCCL_HAS_PACK_INDEXING() 1
#else // ^^^ has pack indexing ^^^ / vvv no pack indexing vvv
#  define _CCCL_HAS_PACK_INDEXING() 0
#endif // no pack indexing

#if _CCCL_STD_VER <= 2017 || __cpp_consteval < 201811L
#  define _CCCL_NO_CONSTEVAL
#  define _CCCL_CONSTEVAL constexpr
#else
#  define _CCCL_CONSTEVAL consteval
#endif

///////////////////////////////////////////////////////////////////////////////
// Conditionally use certain language features depending on availability
///////////////////////////////////////////////////////////////////////////////

// Variable templates are more efficient most of the time, so we want to use them rather than structs when possible
#define _CCCL_TRAIT(__TRAIT, ...) __TRAIT##_v<__VA_ARGS__>

// We need to treat host and device separately
#if _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_GLOBAL_CONSTANT _CCCL_DEVICE constexpr
#else // ^^^ _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC) ^^^ /
      // vvv !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC) vvv
#  define _CCCL_GLOBAL_CONSTANT inline constexpr
#endif // !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC)

#if _CCCL_STD_VER >= 2020 && __cpp_constinit >= 201907L
#  define _CCCL_CONSTINIT constinit
#else // ^^^ has constinit ^^^ / vvv no constinit vvv
#  define _CCCL_CONSTINIT _CCCL_REQUIRE_CONSTANT_INITIALIZATION
#endif // ^^^ no constinit ^^^

#endif // __CCCL_DIALECT_H
