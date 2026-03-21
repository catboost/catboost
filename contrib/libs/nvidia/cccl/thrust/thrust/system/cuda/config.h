/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#ifdef THRUST_DEBUG_SYNC
#  define THRUST_DEBUG_SYNC_FLAG true
#  define CUB_DEBUG_SYNC
#else
#  define THRUST_DEBUG_SYNC_FLAG false
#endif

#include <thrust/detail/config.h> // IWYU pragma: export

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// We don't directly include <cub/version.cuh> since it doesn't exist in
// older releases. This header will always pull in version info:
#include <cub/detail/detect_cuda_runtime.cuh> // IWYU pragma: export
#include <cub/util_debug.cuh> // IWYU pragma: export
#include <cub/util_namespace.cuh> // IWYU pragma: export

/**
 * \def THRUST_RUNTIME_FUNCTION
 *
 * Execution space for functions that can use the CUDA runtime API (`__host__`
 * when RDC is off, `__host__ __device__` when RDC is on).
 */
#define THRUST_RUNTIME_FUNCTION CUB_RUNTIME_FUNCTION

/**
 * \def THRUST_RDC_ENABLED
 *
 * Defined if RDC is enabled.
 */
#ifdef CUB_RDC_ENABLED
#  define THRUST_RDC_ENABLED
#endif

#ifdef THRUST_AGENT_ENTRY_NOINLINE
#  define THRUST_AGENT_ENTRY_INLINE_ATTR __noinline__
#else
#  define THRUST_AGENT_ENTRY_INLINE_ATTR _CCCL_FORCEINLINE
#endif

#define THRUST_DEVICE_FUNCTION _CCCL_DEVICE _CCCL_FORCEINLINE
#define THRUST_HOST_FUNCTION   _CCCL_HOST _CCCL_FORCEINLINE
#define THRUST_FUNCTION        _CCCL_HOST_DEVICE _CCCL_FORCEINLINE

#if 0
#  define THRUST_ARGS(...)         __VA_ARGS__
#  define THRUST_STRIP_PARENS(X)   X
#  define THRUST_AGENT_ENTRY(ARGS) THRUST_FUNCTION static void entry(THRUST_STRIP_PARENS(THRUST_ARGS ARGS))
#else
#  define THRUST_AGENT_ENTRY(...) THRUST_AGENT_ENTRY_INLINE_ATTR _CCCL_DEVICE static void entry(__VA_ARGS__)
#endif

#ifndef THRUST_IGNORE_CUB_VERSION_CHECK

#  include <thrust/version.h>
#  if THRUST_VERSION != CUB_VERSION
#error The version of CUB in your include path is not compatible with this release of Thrust. CUB is now included in the CUDA Toolkit, so you no longer need to use your own checkout of CUB. Define THRUST_IGNORE_CUB_VERSION_CHECK to ignore this.
#  endif

// Make sure the CUB namespace has been declared using the modern macros:
CUB_NAMESPACE_BEGIN
CUB_NAMESPACE_END

#else // THRUST_IGNORE_CUB_VERSION_CHECK

// Make sure the CUB namespace has been declared. Use the old macros for compat
// with older CUB:
CUB_NS_PREFIX
namespace cub
{
}
CUB_NS_POSTFIX

// Older versions of CUB do not define this. Set it to a reasonable default if
// not provided.
#  ifndef CUB_NS_QUALIFIER
#    define CUB_NS_QUALIFIER ::cub
#  endif

#endif // THRUST_IGNORE_CUB_VERSION_CHECK

// Pull the fully qualified cub:: namespace into the thrust:: namespace so we
// don't have to use CUB_NS_QUALIFIER as long as we're in thrust::.
THRUST_NAMESPACE_BEGIN
namespace cub
{
using namespace CUB_NS_QUALIFIER;
}
THRUST_NAMESPACE_END
