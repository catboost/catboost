/*
 *  Copyright 2020 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file cpp_dialect.h
 *  \brief Detect the version of the C++ standard used by the compiler.
 */

#pragma once

#include <thrust/detail/config/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config/compiler.h> // IWYU pragma: export

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - CCCL_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with deprecated C++ dialects will still issue warnings.

#define THRUST_CPP_DIALECT _CCCL_STD_VER

// Define THRUST_COMPILER_DEPRECATION macro:
#define THRUST_COMP_DEPR_IMPL(msg) _CCCL_WARNING(#msg)

// Compiler checks:
// clang-format off
#define THRUST_COMPILER_DEPRECATION(REQ) \
  THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.)

#define THRUST_COMPILER_DEPRECATION_SOFT(REQ, CUR)                                                        \
  THRUST_COMP_DEPR_IMPL(                                                                                  \
    Thrust requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a \
      future release. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.)
// clang-format on

#ifndef CCCL_IGNORE_DEPRECATED_COMPILER
#  if _CCCL_COMPILER(GCC, <, 7)
THRUST_COMPILER_DEPRECATION(GCC 7.0);
#  elif _CCCL_COMPILER(CLANG, <, 7)
THRUST_COMPILER_DEPRECATION(Clang 7.0);
#  elif _CCCL_COMPILER(MSVC, <, 19, 10)
// <2017. Hard upgrade message:
THRUST_COMPILER_DEPRECATION(MSVC 2019(19.20 / 16.0 / 14.20));
#  endif
#endif // CCCL_IGNORE_DEPRECATED_COMPILER

#undef THRUST_COMPILER_DEPRECATION_SOFT
#undef THRUST_COMPILER_DEPRECATION

// C++17 dialect check:
#ifndef CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#  if _CCCL_STD_VER < 2017
#    error Thrust requires at least C++17. Define CCCL_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
#  endif // _CCCL_STD_VER >= 2017
#endif

#undef THRUST_COMP_DEPR_IMPL
