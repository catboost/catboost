/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * \file
 * Static architectural properties by SM version.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_cpp_dialect.cuh> // IWYU pragma: export
#include <cub/util_macro.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/cmath>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

// Legacy include; this functionality used to be defined in here.
#include <cub/detail/detect_cuda_runtime.cuh>

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/// In device code, CUB_PTX_ARCH expands to the PTX version for which we are
/// compiling. In host code, CUB_PTX_ARCH's value is implementation defined.
#  ifndef CUB_PTX_ARCH
// deprecated in 3.1
#    if _CCCL_CUDA_COMPILER(NVHPC)
// __NVCOMPILER_CUDA_ARCH__ is the target PTX version, and is defined
// when compiling both host code and device code. Currently, only one
// PTX version can be targeted.
#      define CUB_PTX_ARCH __NVCOMPILER_CUDA_ARCH__
#    else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVHPC) vvv
#      define CUB_PTX_ARCH _CCCL_PTX_ARCH()
#    endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) ^^^
#  endif

/// Maximum number of devices supported.
#  ifndef CUB_MAX_DEVICES
//! Deprecated [Since 3.0]
#    define CUB_MAX_DEVICES (128)
#  endif
static_assert(CUB_MAX_DEVICES > 0, "CUB_MAX_DEVICES must be greater than 0.");

/// Number of threads per warp
#  ifndef CUB_LOG_WARP_THREADS
//! Deprecated [Since 3.0]
#    define CUB_LOG_WARP_THREADS(unused) (5)
//! Deprecated [Since 3.0]
#    define CUB_WARP_THREADS(unused) (1 << CUB_LOG_WARP_THREADS(0))

//! Deprecated [Since 3.0]
#    define CUB_PTX_WARP_THREADS CUB_WARP_THREADS(0)
//! Deprecated [Since 3.0]
#    define CUB_PTX_LOG_WARP_THREADS CUB_LOG_WARP_THREADS(0)
#  endif

/// Number of smem banks
#  ifndef CUB_LOG_SMEM_BANKS
//! Deprecated [Since 3.0]
#    define CUB_LOG_SMEM_BANKS(unused) (5)
//! Deprecated [Since 3.0]
#    define CUB_SMEM_BANKS(unused) (1 << CUB_LOG_SMEM_BANKS(0))

//! Deprecated [Since 3.0]
#    define CUB_PTX_LOG_SMEM_BANKS CUB_LOG_SMEM_BANKS(0)
//! Deprecated [Since 3.0]
#    define CUB_PTX_SMEM_BANKS CUB_SMEM_BANKS
#  endif

/// Oversubscription factor
#  ifndef CUB_SUBSCRIPTION_FACTOR
//! Deprecated [Since 3.0]
#    define CUB_SUBSCRIPTION_FACTOR(unused) (5)
//! Deprecated [Since 3.0]
#    define CUB_PTX_SUBSCRIPTION_FACTOR CUB_SUBSCRIPTION_FACTOR(0)
#  endif

/// Prefer padding overhead vs X-way conflicts greater than this threshold
#  ifndef CUB_PREFER_CONFLICT_OVER_PADDING
//! Deprecated [Since 3.0]
#    define CUB_PREFER_CONFLICT_OVER_PADDING(unused) (1)
//! Deprecated [Since 3.0]
#    define CUB_PTX_PREFER_CONFLICT_OVER_PADDING CUB_PREFER_CONFLICT_OVER_PADDING(0)
#  endif

namespace detail
{

inline constexpr int max_devices       = CUB_MAX_DEVICES;
inline constexpr int warp_threads      = CUB_PTX_WARP_THREADS;
inline constexpr int log2_warp_threads = CUB_PTX_LOG_WARP_THREADS;
inline constexpr int smem_banks        = CUB_SMEM_BANKS(0);
inline constexpr int log2_smem_banks   = CUB_PTX_LOG_SMEM_BANKS;

inline constexpr int subscription_factor           = CUB_PTX_SUBSCRIPTION_FACTOR;
inline constexpr bool prefer_conflict_over_padding = CUB_PTX_PREFER_CONFLICT_OVER_PADDING;

// The maximum amount of static shared memory available per thread block
// Note that in contrast to dynamic shared memory, static shared memory is still limited to 48 KB
static constexpr ::cuda::std::size_t max_smem_per_block = 48 * 1024;

template <int Nominal4ByteBlockThreads, int Nominal4ByteItemsPerThread, typename T>
struct RegBoundScaling
{
  static constexpr int ITEMS_PER_THREAD =
    (::cuda::std::max) (1, Nominal4ByteItemsPerThread * 4 / (::cuda::std::max) (4, int{sizeof(T)}));
  static constexpr int BLOCK_THREADS =
    (::cuda::std::min) (Nominal4ByteBlockThreads,
                        ::cuda::ceil_div(int{detail::max_smem_per_block} / (int{sizeof(T)} * ITEMS_PER_THREAD), 32)
                          * 32);
};

template <int Nominal4ByteBlockThreads, int Nominal4ByteItemsPerThread, typename T>
struct MemBoundScaling
{
  static constexpr int ITEMS_PER_THREAD =
    (::cuda::std::max) (1,
                        (::cuda::std::min) (Nominal4ByteItemsPerThread * 4 / int{sizeof(T)},
                                            Nominal4ByteItemsPerThread * 2));
  static constexpr int BLOCK_THREADS =
    (::cuda::std::min) (Nominal4ByteBlockThreads,
                        ::cuda::ceil_div(int{detail::max_smem_per_block} / (int{sizeof(T)} * ITEMS_PER_THREAD), 32)
                          * 32);
};

} // namespace detail
#endif // Do not document

CUB_NAMESPACE_END
