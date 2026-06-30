//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_CUDA_CAPABILITIES
#define __CCCL_CUDA_CAPABILITIES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/cuda_toolkit.h>

#include <nv/target>

#ifdef _CCCL_DOXYGEN_INVOKED // Only parse this during doxygen passes:
//! When this macro is defined, Programmatic Dependent Launch (PDL) is disabled across CCCL
#  define CCCL_DISABLE_PDL
#endif // _CCCL_DOXYGEN_INVOKED

#ifdef CCCL_DISABLE_PDL
#  define _CCCL_HAS_PDL() 0
#else // CCCL_DISABLE_PDL
#  define _CCCL_HAS_PDL() _CCCL_CTK_AT_LEAST(12, 0)
#endif // CCCL_DISABLE_PDL

#if _CCCL_HAS_PDL()
// Waits for the previous kernel to complete (when it reaches its final membar). Should be put before the first global
// memory access in a kernel.
#  define _CCCL_PDL_GRID_DEPENDENCY_SYNC() NV_IF_TARGET(NV_PROVIDES_SM_90, ::cudaGridDependencySynchronize();)
// Allows the subsequent kernel in the same stream to launch. Can be put anywhere in a kernel.
// Heuristic(ahendriksen): put it after the last load.
#  define _CCCL_PDL_TRIGGER_NEXT_LAUNCH() NV_IF_TARGET(NV_PROVIDES_SM_90, ::cudaTriggerProgrammaticLaunchCompletion();)
#else // _CCCL_HAS_PDL()
#  define _CCCL_PDL_GRID_DEPENDENCY_SYNC()
#  define _CCCL_PDL_TRIGGER_NEXT_LAUNCH()
#endif // _CCCL_HAS_PDL()

#endif // __CCCL_CUDA_CAPABILITIES
