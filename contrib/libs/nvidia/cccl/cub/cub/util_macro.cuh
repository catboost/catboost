/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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

/******************************************************************************
 * Common C/C++ macro utilities
 ******************************************************************************/

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/detect_cuda_runtime.cuh> // IWYU pragma: export
#include <cub/util_namespace.cuh> // IWYU pragma: export

#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

#ifndef CUB_DETAIL_KERNEL_ATTRIBUTES
#  define CUB_DETAIL_KERNEL_ATTRIBUTES CCCL_DETAIL_KERNEL_ATTRIBUTES
#endif

/**
 * @def CUB_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION
 * If defined, the default suppression of kernel visibility attribute warning is disabled.
 */
#if !defined(CUB_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_CLANG("-Wattributes")
#  if !_CCCL_CUDA_COMPILER(NVHPC)
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)
#  endif // !_CCCL_CUDA_COMPILER(NVHPC)
#endif // !CUB_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION

#ifndef CUB_DEFINE_KERNEL_GETTER
#  define CUB_DEFINE_KERNEL_GETTER(name, ...)                                               \
    _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr decltype(&__VA_ARGS__) name() \
    {                                                                                       \
      return &__VA_ARGS__;                                                                  \
    }
#endif

#ifndef CUB_DEFINE_SUB_POLICY_GETTER
#  define CUB_DEFINE_SUB_POLICY_GETTER(name)                            \
    CUB_RUNTIME_FUNCTION static constexpr auto name()                   \
    {                                                                   \
      return MakePolicyWrapper(typename StaticPolicyT::name##Policy()); \
    }
#endif

// RAPIDS cuDF needs to avoid unrolling some loops in sort to prevent compile time issues
#if defined(CCCL_AVOID_SORT_UNROLL)
#  define _CCCL_SORT_MAYBE_UNROLL() _CCCL_PRAGMA_NOUNROLL()
#else // ^^^ CCCL_AVOID_SORT_UNROLL ^^^ / vvv !CCCL_AVOID_SORT_UNROLL vvv
#  define _CCCL_SORT_MAYBE_UNROLL() _CCCL_PRAGMA_UNROLL_FULL()
#endif // !CCCL_AVOID_SORT_UNROLL

#if defined(CUB_DEFINE_RUNTIME_POLICIES)
#  define CUB_DETAIL_STATIC_ISH_ASSERT(expr, msg) _CCCL_ASSERT(expr, msg)
#  define CUB_DETAIL_CONSTEXPR_ISH
#else // ^^^ CUB_DEFINE_RUNTIME_POLICIES ^^^ / vvv !CUB_DEFINE_RUNTIME_POLICIES vvv
#  define CUB_DETAIL_STATIC_ISH_ASSERT(expr, msg) static_assert(expr, msg);
#  define CUB_DETAIL_CONSTEXPR_ISH                constexpr
#endif // !(CUB_DEFINE_RUNTIME_POLICIES)

CUB_NAMESPACE_END
