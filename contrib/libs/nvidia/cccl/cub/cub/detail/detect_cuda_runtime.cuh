/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file
 * Utilities for CUDA dynamic parallelism.
 */

#pragma once

// We cannot use `cub/config.cuh` here due to circular dependencies
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// CUDA headers might not be present when using NVRTC, see NVIDIA/cccl#2095 for detail
#if !_CCCL_COMPILER(NVRTC)
#  include <cuda_runtime_api.h>
#endif // !_CCCL_COMPILER(NVRTC)

#ifdef _CCCL_DOXYGEN_INVOKED // Only parse this during doxygen passes:

/**
 * \def CUB_DISABLE_CDP
 *
 * If defined, support for device-side usage of CUB is disabled.
 */
#  define CUB_DISABLE_CDP

/**
 * \def CUB_RDC_ENABLED
 *
 * Defined if RDC is enabled and CUB_DISABLE_CDP is not defined.
 */
#  define CUB_RDC_ENABLED

/**
 * \def CUB_RUNTIME_FUNCTION
 *
 * Execution space for functions that can use the CUDA runtime API (`__host__`
 * when RDC is off, `__host__ __device__` when RDC is on).
 */
#  define CUB_RUNTIME_FUNCTION

#else // Non-doxygen pass:

#  ifndef CUB_RUNTIME_FUNCTION
#    if defined(__CUDACC_RDC__) && !defined(CUB_DISABLE_CDP)
#      define CUB_RDC_ENABLED
#      define CUB_RUNTIME_FUNCTION _CCCL_HOST_DEVICE
#    else // RDC disabled:
#      define CUB_RUNTIME_FUNCTION _CCCL_HOST
#    endif // RDC enabled
#  endif // CUB_RUNTIME_FUNCTION predefined

#  ifdef CUB_RDC_ENABLED
#    ifdef CUDA_FORCE_CDP1_IF_SUPPORTED
#      error "CUDA Dynamic Parallelism 1 is no longer supported. Please undefine CUDA_FORCE_CDP1_IF_SUPPORTED."
#    endif // CUDA_FORCE_CDP1_IF_SUPPORTED
#  endif

#endif // Do not document
