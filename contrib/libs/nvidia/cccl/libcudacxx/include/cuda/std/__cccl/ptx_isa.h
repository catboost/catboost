//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_PTX_ISA_H_
#define __CCCL_PTX_ISA_H_

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

/*
 * Targeting macros
 *
 * Information from:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes
 */

// The first define is for future major versions of CUDACC.
// We make sure that these get the highest known PTX ISA version.
// For clang cuda check
// https://github.com/llvm/llvm-project/blob/release/<VER>.x/clang/lib/Driver/ToolChains/Cuda.cpp getNVPTXTargetFeatures
#if _CCCL_CUDACC_AT_LEAST(14, 0) && !_CCCL_CUDA_COMPILER(CLANG)
#  define __cccl_ptx_isa 920ULL
// PTX ISA 9.2 is available from CUDA 13.2
#elif _CCCL_CUDACC_AT_LEAST(13, 2) && !_CCCL_CUDA_COMPILER(CLANG)
#  define __cccl_ptx_isa 920ULL
// PTX ISA 9.1 is available from CUDA 13.1
#elif _CCCL_CUDACC_AT_LEAST(13, 1) && !_CCCL_CUDA_COMPILER(CLANG)
#  define __cccl_ptx_isa 910ULL
// PTX ISA 9.0 is available from CUDA 13.0, driver r580
#elif _CCCL_CUDACC_AT_LEAST(13, 0) && !_CCCL_CUDA_COMPILER(CLANG)
#  define __cccl_ptx_isa 900ULL
// PTX ISA 8.8 is available from CUDA 12.9, driver r575
#elif _CCCL_CUDACC_AT_LEAST(12, 9) && !_CCCL_CUDA_COMPILER(CLANG)
#  define __cccl_ptx_isa 880ULL
// PTX ISA 8.7 is available from CUDA 12.8, driver r570
#elif _CCCL_CUDACC_AT_LEAST(12, 8) && !_CCCL_CUDA_COMPILER(CLANG, <, 20)
#  define __cccl_ptx_isa 870ULL
// PTX ISA 8.5 is available from CUDA 12.5, driver r555
#elif _CCCL_CUDACC_AT_LEAST(12, 5) && !_CCCL_CUDA_COMPILER(CLANG, <, 19)
#  define __cccl_ptx_isa 850ULL
// PTX ISA 8.4 is available from CUDA 12.4, driver r550
#elif _CCCL_CUDACC_AT_LEAST(12, 4) && !_CCCL_CUDA_COMPILER(CLANG, <, 19)
#  define __cccl_ptx_isa 840ULL
// PTX ISA 8.3 is available from CUDA 12.3, driver r545
#elif _CCCL_CUDACC_AT_LEAST(12, 3) && !_CCCL_CUDA_COMPILER(CLANG, <, 18)
#  define __cccl_ptx_isa 830ULL
// PTX ISA 8.2 is available from CUDA 12.2, driver r535
#elif _CCCL_CUDACC_AT_LEAST(12, 2) && !_CCCL_CUDA_COMPILER(CLANG, <, 18)
#  define __cccl_ptx_isa 820ULL
// PTX ISA 8.1 is available from CUDA 12.1, driver r530
#elif _CCCL_CUDACC_AT_LEAST(12, 1) && !_CCCL_CUDA_COMPILER(CLANG, <, 17)
#  define __cccl_ptx_isa 810ULL
// PTX ISA 8.0 is available from CUDA 12.0, driver r525
#elif _CCCL_CUDACC_AT_LEAST(12, 0) && !_CCCL_CUDA_COMPILER(CLANG, <, 17)
#  define __cccl_ptx_isa 800ULL
// PTX ISA 7.8 is available from CUDA 11.8, driver r520
#elif _CCCL_CUDACC_AT_LEAST(11, 8) && !_CCCL_CUDA_COMPILER(CLANG, <, 16)
#  define __cccl_ptx_isa 780ULL
// PTX ISA 7.7 is available from CUDA 11.7, driver r515
#elif _CCCL_CUDACC_AT_LEAST(11, 7) && !_CCCL_CUDA_COMPILER(CLANG, <, 16)
#  define __cccl_ptx_isa 770ULL
// PTX ISA 7.6 is available from CUDA 11.6, driver r510
#elif _CCCL_CUDACC_AT_LEAST(11, 6) && !_CCCL_CUDA_COMPILER(CLANG, <, 16)
#  define __cccl_ptx_isa 760ULL
// PTX ISA 7.5 is available from CUDA 11.5, driver r495
#elif _CCCL_CUDACC_AT_LEAST(11, 5) && !_CCCL_CUDA_COMPILER(CLANG, <, 14)
#  define __cccl_ptx_isa 750ULL
// PTX ISA 7.4 is available from CUDA 11.4, driver r470
#elif _CCCL_CUDACC_AT_LEAST(11, 4) && !_CCCL_CUDA_COMPILER(CLANG, <, 14)
#  define __cccl_ptx_isa 740ULL
// PTX ISA 7.3 is available from CUDA 11.3, driver r465
#elif _CCCL_CUDACC_AT_LEAST(11, 3) && !_CCCL_CUDA_COMPILER(CLANG, <, 14)
#  define __cccl_ptx_isa 730ULL
// PTX ISA 7.2 is available from CUDA 11.2, driver r460
#elif _CCCL_CUDACC_AT_LEAST(11, 2) && !_CCCL_CUDA_COMPILER(CLANG, <, 13)
#  define __cccl_ptx_isa 720ULL
// PTX ISA 7.1 is available from CUDA 11.1, driver r455
#elif _CCCL_CUDACC_AT_LEAST(11, 1) && !_CCCL_CUDA_COMPILER(CLANG, <, 13)
#  define __cccl_ptx_isa 710ULL
// PTX ISA 7.0 is available from CUDA 11.0, driver r445
#elif _CCCL_CUDACC_AT_LEAST(11, 0) && !_CCCL_CUDA_COMPILER(CLANG, <, 12)
#  define __cccl_ptx_isa 700ULL
// Fallback case. Define the ISA version to be zero. This ensures that the macro is always defined.
#else
#  define __cccl_ptx_isa 0ULL
#endif

// We define certain feature test macros depending on availability. When
// __CUDA_MINIMUM_ARCH__ is not available, we define the following features
// depending on PTX ISA. This permits checking for the feature in host code.
// When __CUDA_MINIMUM_ARCH__ is available, we only enable the feature when the
// hardware supports it.
#if __cccl_ptx_isa >= 800
#  if (!defined(__CUDA_MINIMUM_ARCH__)) || (defined(__CUDA_MINIMUM_ARCH__) && 900 <= __CUDA_MINIMUM_ARCH__)
#    define __cccl_lib_local_barrier_arrive_tx
#    define __cccl_lib_experimental_ctk12_cp_async_exposure
#  endif
#endif // __cccl_ptx_isa >= 800

// NVRTC ships a built-in copy of <nv/detail/__target_macros>, so including CCCL's version of this header will omit the
// content since the header guards are already defined. To make older NVRTC versions have a few newer feature macros
// required for the PTX tests, we define them here outside the header guards.
// TODO(bgruber): limit this workaround to NVRTC versions older than the first one shipping those macros
#if _CCCL_COMPILER(NVRTC)

// missing SM_88
#  if !defined(NV_PROVIDES_SM_88)
#    define _NV_TARGET_VAL_SM_88 880
#    define NV_PROVIDES_SM_88    __NV_PROVIDES_SM_88
#    define NV_IS_EXACTLY_SM_88  __NV_IS_EXACTLY_SM_88
#    if (__CUDA_ARCH__ == _NV_TARGET_VAL_SM_88)
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_88 1
#      define _NV_TARGET___NV_IS_EXACTLY_SM_88      1
#    else
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_88 0
#      define _NV_TARGET___NV_IS_EXACTLY_SM_88      0
#    endif
#    if (__CUDA_ARCH__ >= _NV_TARGET_VAL_SM_88)
#      define _NV_TARGET___NV_PROVIDES_SM_88      1
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_88 1
#    else
#      define _NV_TARGET___NV_PROVIDES_SM_88      0
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_88 0
#    endif
#  endif // !NV_PROVIDES_SM_88

// missing SM_90a
#  ifndef NV_HAS_FEATURE_SM_90a
#    define NV_HAS_FEATURE_SM_90a __NV_HAS_FEATURE_SM_90a
#    if defined(__CUDA_ARCH_FEAT_SM90_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 900))
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_90a 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_90a 0
#    endif
#  endif // NV_HAS_FEATURE_SM_90a

// missing SM_100
#  ifndef NV_PROVIDES_SM_100
#    define _NV_TARGET_VAL_SM_100 1000
#    define NV_PROVIDES_SM_100    __NV_PROVIDES_SM_100
#    define NV_IS_EXACTLY_SM_100  __NV_IS_EXACTLY_SM_100
#    if (__CUDA_ARCH__ == _NV_TARGET_VAL_SM_100)
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_100 1
#      define _NV_TARGET___NV_IS_EXACTLY_SM_100      1
#    else
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_100 0
#      define _NV_TARGET___NV_IS_EXACTLY_SM_100      0
#    endif
#    if (__CUDA_ARCH__ >= _NV_TARGET_VAL_SM_100)
#      define _NV_TARGET___NV_PROVIDES_SM_100      1
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_100 1
#    else
#      define _NV_TARGET___NV_PROVIDES_SM_100      0
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_100 0
#    endif
#  endif // !NV_PROVIDES_SM_100

// missing SM_100a
#  ifndef NV_HAS_FEATURE_SM_100a
#    define NV_HAS_FEATURE_SM_100a __NV_HAS_FEATURE_SM_100a
#    if defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000))
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100a 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100a 0
#    endif
#  endif // !NV_HAS_FEATURE_SM_100a

// missing SM_103
#  ifndef NV_PROVIDES_SM_103
#    define _NV_TARGET_VAL_SM_103 1030
#    define NV_PROVIDES_SM_103    __NV_PROVIDES_SM_103
#    define NV_IS_EXACTLY_SM_103  __NV_IS_EXACTLY_SM_103
#    if (__CUDA_ARCH__ == _NV_TARGET_VAL_SM_103)
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_103 1
#      define _NV_TARGET___NV_IS_EXACTLY_SM_103      1
#    else
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_103 0
#      define _NV_TARGET___NV_IS_EXACTLY_SM_103      0
#    endif
#    if (__CUDA_ARCH__ >= _NV_TARGET_VAL_SM_103)
#      define _NV_TARGET___NV_PROVIDES_SM_103      1
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_103 1
#    else
#      define _NV_TARGET___NV_PROVIDES_SM_103      0
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_103 0
#    endif
#  endif // !NV_PROVIDES_SM_103

// missing SM_103
#  ifndef NV_HAS_FEATURE_SM_103a
#    define NV_HAS_FEATURE_SM_103a __NV_HAS_FEATURE_SM_103a
#    if defined(__CUDA_ARCH_FEAT_SM103_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1030))
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_103a 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_103a 0
#    endif
#  endif // !NV_HAS_FEATURE_SM_103a

// missing SM_110
#  ifndef NV_PROVIDES_SM_110
#    define _NV_TARGET_VAL_SM_110 1100
#    define NV_PROVIDES_SM_110    __NV_PROVIDES_SM_110
#    define NV_IS_EXACTLY_SM_110  __NV_IS_EXACTLY_SM_110
#    if (__CUDA_ARCH__ == _NV_TARGET_VAL_SM_110)
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_110 1
#      define _NV_TARGET___NV_IS_EXACTLY_SM_110      1
#    else
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_110 0
#      define _NV_TARGET___NV_IS_EXACTLY_SM_110      0
#    endif
#    if (__CUDA_ARCH__ >= _NV_TARGET_VAL_SM_110)
#      define _NV_TARGET___NV_PROVIDES_SM_110      1
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_110 1
#    else
#      define _NV_TARGET___NV_PROVIDES_SM_110      0
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_110 0
#    endif
#  endif // !NV_PROVIDES_SM_110

// missing SM_110a
#  ifndef NV_HAS_FEATURE_SM_110a
#    define NV_HAS_FEATURE_SM_110a __NV_HAS_FEATURE_SM_110a
#    if defined(__CUDA_ARCH_FEAT_SM110_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100))
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_110a 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_110a 0
#    endif
#  endif // NV_HAS_FEATURE_SM_110a

// missing SM_120
#  ifndef NV_PROVIDES_SM_120
#    define _NV_TARGET_VAL_SM_120 1200
#    define NV_PROVIDES_SM_120    __NV_PROVIDES_SM_120
#    define NV_IS_EXACTLY_SM_120  __NV_IS_EXACTLY_SM_120
#    if (__CUDA_ARCH__ == _NV_TARGET_VAL_SM_120)
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_120 1
#      define _NV_TARGET___NV_IS_EXACTLY_SM_120      1
#    else
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_120 0
#      define _NV_TARGET___NV_IS_EXACTLY_SM_120      0
#    endif
#    if (__CUDA_ARCH__ >= _NV_TARGET_VAL_SM_120)
#      define _NV_TARGET___NV_PROVIDES_SM_120      1
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_120 1
#    else
#      define _NV_TARGET___NV_PROVIDES_SM_120      0
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_120 0
#    endif
#  endif // !NV_PROVIDES_SM_120

// missing SM_120a
#  ifndef NV_HAS_FEATURE_SM_120a
#    define NV_HAS_FEATURE_SM_120a __NV_HAS_FEATURE_SM_120a
#    if defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200))
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_120a 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_120a 0
#    endif
#  endif // _CCCL_COMPILER(NVRTC)

// missing SM_121
#  if !defined(NV_PROVIDES_SM_121)
#    define _NV_TARGET_VAL_SM_121 1210
#    define NV_PROVIDES_SM_121    __NV_PROVIDES_SM_121
#    define NV_IS_EXACTLY_SM_121  __NV_IS_EXACTLY_SM_121
#    if (__CUDA_ARCH__ == _NV_TARGET_VAL_SM_121)
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_121 1
#      define _NV_TARGET___NV_IS_EXACTLY_SM_121      1
#    else
#      define _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_121 0
#      define _NV_TARGET___NV_IS_EXACTLY_SM_121      0
#    endif
#    if (__CUDA_ARCH__ >= _NV_TARGET_VAL_SM_121)
#      define _NV_TARGET___NV_PROVIDES_SM_121      1
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_121 1
#    else
#      define _NV_TARGET___NV_PROVIDES_SM_121      0
#      define _NV_TARGET_BOOL___NV_PROVIDES_SM_121 0
#    endif
#  endif // !NV_PROVIDES_SM_121

// missing SM_121a
#  ifndef NV_HAS_FEATURE_SM_121a
#    define NV_HAS_FEATURE_SM_121a __NV_HAS_FEATURE_SM_121a
#    if defined(__CUDA_ARCH_FEAT_SM121_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1210))
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_121a 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_121a 0
#    endif
#  endif // NV_HAS_FEATURE_SM_121a

//----------------------------------------------------------------------------------------------------------------------
// family-specific SM versions

// missing SM_100f
#  ifndef NV_HAS_FEATURE_SM_100f
#    define NV_HAS_FEATURE_SM_100f __NV_HAS_FEATURE_SM_100f
#    if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000)
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100f 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100f 0
#    endif
#  endif // NV_HAS_FEATURE_SM_100

// missing SM_103f
#  ifndef NV_HAS_FEATURE_SM_103f
#    define NV_HAS_FEATURE_SM_103f __NV_HAS_FEATURE_SM_103f
#    if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1030)
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_103f 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_103f 0
#    endif
#  endif // NV_HAS_FEATURE_SM_103f

// missing SM_110f
#  ifndef NV_HAS_FEATURE_SM_110f
#    define NV_HAS_FEATURE_SM_110f __NV_HAS_FEATURE_SM_110f
#    if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100)
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_110f 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_110f 0
#    endif
#  endif // NV_HAS_FEATURE_SM_110f

// missing SM_120f
#  ifndef NV_HAS_FEATURE_SM_120f
#    define NV_HAS_FEATURE_SM_120f __NV_HAS_FEATURE_SM_120f
#    if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200)
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_120f 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_120f 0
#    endif
#  endif // NV_HAS_FEATURE_SM_120f

// missing SM_121f
#  ifndef NV_HAS_FEATURE_SM_121f
#    define NV_HAS_FEATURE_SM_121f __NV_HAS_FEATURE_SM_121f
#    if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1210)
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_121f 1
#    else
#      define _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_121f 0
#    endif
#  endif // NV_HAS_FEATURE_SM_121f

#endif // _CCCL_COMPILER(NVRTC)
#endif // __CCCL_PTX_ISA_H_
