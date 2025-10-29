//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_COMPILER_H
#define __CCCL_COMPILER_H

#include <cuda/std/__cccl/preprocessor.h>

// Utility to compare version numbers. To use:
// 1) Define a macro that makes a pair of (major, minor) numbers:
//    #define MYPRODUCT_MAKE_VERSION(_MAJOR, _MINOR) (_MAJOR * 100 + _MINOR)
// 2) Define a macro that you will use to compare versions, e.g.:
//    #define MYPRODUCT(...) _CCCL_VERSION_COMPARE(MYPRODUCT, MYPRODUCT_##__VA_ARGS__)
//    Signatures:
//       MYPRODUCT(_PROD)                      - is the product _PROD version non-zero?
//       MYPRODUCT(_PROD, _OP, _MAJOR)         - compare the product _PROD major version to _MAJOR using operator _OP
//       MYPRODUCT(_PROD, _OP, _MAJOR, _MINOR) - compare the product _PROD version to _MAJOR._MINOR using operator _OP
// 3) Define the product version macros as a function-like macro that returns the version number or
//    _CCCL_VERSION_INVALID() if the version cannot be determined, e. g.:
//    #define MYPRODUCT_<_PROD>() (1, 2)
//      or
//    #define MYPRODUCT_<_PROD>() _CCCL_VERSION_INVALID()
#define _CCCL_VERSION_MAJOR_(_MAJOR, _MINOR)   _MAJOR
#define _CCCL_VERSION_MAJOR(_PAIR)             _CCCL_VERSION_MAJOR_ _PAIR
#define _CCCL_VERSION_INVALID()                (-1, -1)
#define _CCCL_MAKE_VERSION(_PREFIX, _PAIR)     (_CCCL_PP_EVAL(_CCCL_PP_CAT(_PREFIX, MAKE_VERSION), _CCCL_PP_EXPAND _PAIR))
#define _CCCL_VERSION_IS_INVALID(_PAIR)        (_CCCL_VERSION_MAJOR(_PAIR) == _CCCL_VERSION_MAJOR(_CCCL_VERSION_INVALID()))
#define _CCCL_VERSION_COMPARE_1(_PREFIX, _VER) (!_CCCL_VERSION_IS_INVALID(_VER()))
#define _CCCL_VERSION_COMPARE_3(_PREFIX, _VER, _OP, _MAJOR) \
  (!_CCCL_VERSION_IS_INVALID(_VER()) && (_CCCL_VERSION_MAJOR(_VER()) _OP _MAJOR))
#define _CCCL_VERSION_COMPARE_4(_PREFIX, _VER, _OP, _MAJOR, _MINOR) \
  (!_CCCL_VERSION_IS_INVALID(_VER())                                \
   && (_CCCL_MAKE_VERSION(_PREFIX, _VER()) _OP _CCCL_MAKE_VERSION(_PREFIX, (_MAJOR, _MINOR))))
#define _CCCL_VERSION_SELECT_COUNT(_ARG1, _ARG2, _ARG3, _ARG4, _ARG5, ...) _ARG5
#define _CCCL_VERSION_SELECT2(_ARGS)                                       _CCCL_VERSION_SELECT_COUNT _ARGS
// MSVC traditonal preprocessor requires an extra level of indirection
#define _CCCL_VERSION_SELECT(...)         \
  _CCCL_VERSION_SELECT2(                  \
    (__VA_ARGS__,                         \
     _CCCL_VERSION_COMPARE_4,             \
     _CCCL_VERSION_COMPARE_3,             \
     _CCCL_VERSION_COMPARE_BAD_ARG_COUNT, \
     _CCCL_VERSION_COMPARE_1,             \
     _CCCL_VERSION_COMPARE_BAD_ARG_COUNT))
#define _CCCL_VERSION_COMPARE(_PREFIX, ...) _CCCL_VERSION_SELECT(__VA_ARGS__)(_PREFIX, __VA_ARGS__)

#define _CCCL_COMPILER_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 100 + (_MINOR))
#define _CCCL_COMPILER(...)                         _CCCL_VERSION_COMPARE(_CCCL_COMPILER_, _CCCL_COMPILER_##__VA_ARGS__)

#define _CCCL_COMPILER_NVHPC()    _CCCL_VERSION_INVALID()
#define _CCCL_COMPILER_CLANG()    _CCCL_VERSION_INVALID()
#define _CCCL_COMPILER_GCC()      _CCCL_VERSION_INVALID()
#define _CCCL_COMPILER_MSVC()     _CCCL_VERSION_INVALID()
#define _CCCL_COMPILER_MSVC2019() _CCCL_VERSION_INVALID()
#define _CCCL_COMPILER_MSVC2022() _CCCL_VERSION_INVALID()
#define _CCCL_COMPILER_NVRTC()    _CCCL_VERSION_INVALID()

// Determine the host compiler and its version
#if defined(__INTEL_COMPILER)
#  ifndef CCCL_IGNORE_DEPRECATED_COMPILER
#    warning \
      "The Intel C++ Compiler Classic (icc/icpc) is not supported by CCCL. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message."
#  endif // !CCCL_IGNORE_DEPRECATED_COMPILER
#elif defined(__NVCOMPILER)
#  undef _CCCL_COMPILER_NVHPC
#  define _CCCL_COMPILER_NVHPC() (__NVCOMPILER_MAJOR__, __NVCOMPILER_MINOR__)
#elif defined(__clang__)
#  undef _CCCL_COMPILER_CLANG
#  define _CCCL_COMPILER_CLANG() (__clang_major__, __clang_minor__)
#elif defined(__GNUC__)
#  undef _CCCL_COMPILER_GCC
#  define _CCCL_COMPILER_GCC() (__GNUC__, __GNUC_MINOR__)
#elif defined(_MSC_VER)
#  undef _CCCL_COMPILER_MSVC
#  define _CCCL_COMPILER_MSVC() (_MSC_VER / 100, _MSC_VER % 100)
#  if _CCCL_COMPILER(MSVC, <, 19, 20)
#    ifndef CCCL_IGNORE_DEPRECATED_COMPILER
#      error \
        "Visual Studio 2017 (MSC_VER < 1920) and older are not supported by CCCL. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this error."
#    endif
#  endif // _CCCL_COMPILER(MSVC, <, 19, 20)
#  if _CCCL_COMPILER(MSVC, >=, 19, 20) && _CCCL_COMPILER(MSVC, <, 19, 30)
#    undef _CCCL_COMPILER_MSVC2019
#    define _CCCL_COMPILER_MSVC2019() _CCCL_COMPILER_MSVC()
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 20) && _CCCL_COMPILER(MSVC, <, 19, 30)
#  if _CCCL_COMPILER(MSVC, >=, 19, 30) && _CCCL_COMPILER(MSVC, <, 19, 40)
#    undef _CCCL_COMPILER_MSVC2022
#    define _CCCL_COMPILER_MSVC2022() _CCCL_COMPILER_MSVC()
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 30) && _CCCL_COMPILER(MSVC, <, 19, 40)
#elif defined(__CUDACC_RTC__)
#  undef _CCCL_COMPILER_NVRTC
#  define _CCCL_COMPILER_NVRTC() (__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#endif

// The CUDA compiler version shares the implementation with the C++ compiler
#define _CCCL_CUDA_COMPILER_MAKE_VERSION(_MAJOR, _MINOR) _CCCL_COMPILER_MAKE_VERSION(_MAJOR, _MINOR)
#define _CCCL_CUDA_COMPILER(...)                         _CCCL_VERSION_COMPARE(_CCCL_CUDA_COMPILER_, _CCCL_CUDA_COMPILER_##__VA_ARGS__)

#define _CCCL_CUDA_COMPILER_NVCC()  _CCCL_VERSION_INVALID()
#define _CCCL_CUDA_COMPILER_NVHPC() _CCCL_VERSION_INVALID()
#define _CCCL_CUDA_COMPILER_CLANG() _CCCL_VERSION_INVALID()
#define _CCCL_CUDA_COMPILER_NVRTC() _CCCL_VERSION_INVALID()

// Determine the cuda compiler
#if defined(__NVCC__)
#  undef _CCCL_CUDA_COMPILER_NVCC
#  define _CCCL_CUDA_COMPILER_NVCC() (__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#elif defined(_NVHPC_CUDA)
#  undef _CCCL_CUDA_COMPILER_NVHPC
#  define _CCCL_CUDA_COMPILER_NVHPC() _CCCL_COMPILER_NVHPC()
#elif defined(__CUDA__) && _CCCL_COMPILER(CLANG)
#  undef _CCCL_CUDA_COMPILER_CLANG
#  define _CCCL_CUDA_COMPILER_CLANG() _CCCL_COMPILER_CLANG()
#elif _CCCL_COMPILER(NVRTC)
#  undef _CCCL_CUDA_COMPILER_NVRTC
#  define _CCCL_CUDA_COMPILER_NVRTC() _CCCL_COMPILER_NVRTC()
#endif // ^^^ _CCCL_COMPILER(NVRTC) ^^^

#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(CLANG) || _CCCL_CUDA_COMPILER(NVHPC) || _CCCL_CUDA_COMPILER(NVRTC)
#  define _CCCL_HAS_CUDA_COMPILER() 1
#else // ^^^ has cuda compiler ^^^ / vvv no cuda compiler vvv
#  define _CCCL_HAS_CUDA_COMPILER() 0
#endif // ^^^ no cuda compiler ^^^

#if defined(__CUDACC__) || _CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_CUDA_COMPILATION() 1
#else // ^^^ compiling .cu file ^^^ / vvv not compiling .cu file vvv
#  define _CCCL_CUDA_COMPILATION() 0
#endif // ^^^ not compiling .cu file ^^^

// Determine if we are compiling host code, this includes both CUDA and C++ compilation
// nvc++ does not define __CUDA_ARCH__, but it compiles both host and device code at the same time
#if !defined(__CUDA_ARCH__)
#  define _CCCL_HOST_COMPILATION() 1
#else // ^^^ compiling host code ^^^ / vvv not compiling host code vvv
#  define _CCCL_HOST_COMPILATION() 0
#endif // ^^^ not compiling host code ^^^

#if (_CCCL_CUDA_COMPILATION() && defined(__CUDA_ARCH__)) || _CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_DEVICE_COMPILATION() 1
#else // ^^^ compiling device code ^^^ / vvv not compiling device code vvv
#  define _CCCL_DEVICE_COMPILATION() 0
#endif // ^^^ not compiling device code ^^^

#define _CCCL_CUDACC_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)

// clang-cuda does not define __CUDACC_VER_MAJOR__ and friends. They are instead retrieved from the CUDA_VERSION macro
// defined in "cuda.h". clang-cuda automatically pre-includes "__clang_cuda_runtime_wrapper.h" which includes "cuda.h"
#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(NVHPC) || _CCCL_CUDA_COMPILER(NVRTC)
#  define _CCCL_CUDACC() (__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#elif _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_CUDACC() (CUDA_VERSION / 1000, (CUDA_VERSION % 1000) / 10)
#endif // ^^^ has cuda compiler ^^^

#if !defined(_CCCL_CUDACC) || !_CCCL_CUDA_COMPILATION()
#  undef _CCCL_CUDACC
#  define _CCCL_CUDACC() _CCCL_VERSION_INVALID()
#endif // !_CCCL_CUDACC || !_CCCL_CUDA_COMPILATION()

#define _CCCL_CUDACC_EQUAL(...)    _CCCL_VERSION_COMPARE(_CCCL_CUDACC_, _CCCL_CUDACC, ==, __VA_ARGS__)
#define _CCCL_CUDACC_BELOW(...)    _CCCL_VERSION_COMPARE(_CCCL_CUDACC_, _CCCL_CUDACC, <, __VA_ARGS__)
#define _CCCL_CUDACC_AT_LEAST(...) _CCCL_VERSION_COMPARE(_CCCL_CUDACC_, _CCCL_CUDACC, >=, __VA_ARGS__)

#if _CCCL_CUDA_COMPILATION() && _CCCL_CUDACC_BELOW(12) && !defined(CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12)
#  error "CUDA versions below 12 are not supported." \
"Define CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12 to suppress this message."
#endif

// Define the pragma for the host compiler
#if _CCCL_COMPILER(MSVC)
#  define _CCCL_PRAGMA(_ARG) __pragma(_ARG)
#else
#  define _CCCL_PRAGMA(_ARG) _Pragma(_CCCL_TO_STRING(_ARG))
#endif // _CCCL_COMPILER(MSVC)

// Define the proper object format for NVHPC and NVRTC
#if (_CCCL_COMPILER(NVHPC) && defined(__linux__)) || _CCCL_COMPILER(NVRTC)
#  ifndef __ELF__
#    define __ELF__
#  endif // !__ELF__
#endif // _CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(NVRTC)

#if _CCCL_DEVICE_COMPILATION()
#  define _CCCL_PRAGMA_UNROLL(_N)    _CCCL_PRAGMA(unroll _N)
#  define _CCCL_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA(unroll)
#elif _CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(NVRTC) || _CCCL_COMPILER(CLANG)
#  define _CCCL_PRAGMA_UNROLL(_N)    _CCCL_PRAGMA(unroll _N)
#  define _CCCL_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA(unroll)
#elif _CCCL_COMPILER(GCC, >=, 8)
// gcc supports only #pragma GCC unroll, but that causes problems when compiling with nvcc. So, we use #pragma unroll
// when compiling device code, and #pragma GCC unroll when compiling host code, but we need to suppress the warning
// about the unknown pragma for nvcc.
// #pragma GCC unroll does not support full unrolling, so we use the maximum value that it supports.
#  define _CCCL_PRAGMA_UNROLL(_N) \
    _CCCL_BEGIN_NV_DIAG_SUPPRESS(1675) _CCCL_PRAGMA(GCC unroll _N) _CCCL_END_NV_DIAG_SUPPRESS()
#  define _CCCL_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA_UNROLL(65534)
#else // ^^^ has pragma unroll support ^^^ / vvv no pragma unroll support vvv
#  define _CCCL_PRAGMA_UNROLL(_N)
#  define _CCCL_PRAGMA_UNROLL_FULL()
#endif // ^^^ no pragma unroll support ^^^

#define _CCCL_PRAGMA_NOUNROLL() _CCCL_PRAGMA_UNROLL(1)

#if _CCCL_COMPILER(MSVC)
#  define _CCCL_WARNING(_MSG) _CCCL_PRAGMA(message(__FILE__ ":" _CCCL_TO_STRING(__LINE__) ": warning: " _MSG))
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#  define _CCCL_WARNING(_MSG) _CCCL_PRAGMA(GCC warning _MSG)
#endif // !_CCCL_COMPILER(MSVC)

#endif // __CCCL_COMPILER_H
