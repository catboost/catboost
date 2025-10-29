/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//! @file
//! Detect the version of the C++ standard used by the compiler.

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - CCCL_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with deprecated C++ dialects will still issue warnings.

//! Deprecated [Since 3.0]
#  define CUB_CPP_DIALECT _CCCL_STD_VER

// Define CUB_COMPILER_DEPRECATION macro:
#  if _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#    define CUB_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(message(__FILE__ ":" _CCCL_TO_STRING(__LINE__) ": warning: " #msg))
#  else // clang / gcc:
#    define CUB_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(GCC warning #msg)
#  endif

// Compiler checks:
// clang-format off
#  define CUB_COMPILER_DEPRECATION(REQ) \
    CUB_COMP_DEPR_IMPL(CUB requires at least REQ. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.)

#  define CUB_COMPILER_DEPRECATION_SOFT(REQ, CUR)                                                        \
    CUB_COMP_DEPR_IMPL(                                                                                  \
      CUB requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a \
        future release. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.)
// clang-format on

#  ifndef CCCL_IGNORE_DEPRECATED_COMPILER
#    if _CCCL_COMPILER(GCC, <, 7)
CUB_COMPILER_DEPRECATION(GCC 7.0);
#    elif _CCCL_COMPILER(CLANG, <, 7)
CUB_COMPILER_DEPRECATION(Clang 7.0);
#    elif _CCCL_COMPILER(MSVC, <, 19, 10)
// <2017. Hard upgrade message:
CUB_COMPILER_DEPRECATION(MSVC 2019(19.20 / 16.0 / 14.20));
#    endif
#  endif // CCCL_IGNORE_DEPRECATED_COMPILER

#  undef CUB_COMPILER_DEPRECATION_SOFT
#  undef CUB_COMPILER_DEPRECATION

// C++17 dialect check:
#  ifndef CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#    if _CCCL_STD_VER < 2017
#      error CUB requires at least C++17. Define CCCL_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
#    endif // _CCCL_STD_VER >= 2017
#  endif

#  undef CUB_COMP_DEPR_IMPL

#endif // !_CCCL_DOXYGEN_INVOKED
