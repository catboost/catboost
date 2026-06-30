/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * Utilities for device-accessible temporary storages.
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

#include <cub/util_debug.cuh>
#include <cub/util_namespace.cuh>

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{

/**
 * @brief Alias temporaries to externally-allocated device storage (or simply return the amount of storage needed).
 *
 * @param[in] d_temp_storage
 *   Device-accessible allocation of temporary storage.
 *   When nullptr, the required allocation size is written to @p temp_storage_bytes and no work is
 *   done.
 *
 * @param[in,out] temp_storage_bytes
 *   Size in bytes of @p d_temp_storage allocation
 *
 * @param[in,out] allocations
 *   Pointers to device allocations needed
 *
 * @param[in] allocation_sizes
 *   Sizes in bytes of device allocations needed
 */
template <int ALLOCATIONS>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE cudaError_t AliasTemporaries(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  void* (&allocations)[ALLOCATIONS],
  const size_t (&allocation_sizes)[ALLOCATIONS])
{
  constexpr size_t ALIGN_BYTES = 256;
  constexpr size_t ALIGN_MASK  = ~(ALIGN_BYTES - 1);

  // Compute exclusive prefix sum over allocation requests
  size_t allocation_offsets[ALLOCATIONS];
  size_t bytes_needed = 0;
  for (int i = 0; i < ALLOCATIONS; ++i)
  {
    const size_t allocation_bytes = (allocation_sizes[i] + ALIGN_BYTES - 1) & ALIGN_MASK;
    allocation_offsets[i]         = bytes_needed;
    bytes_needed += allocation_bytes;
  }
  bytes_needed += ALIGN_BYTES - 1;

  // Check if the caller is simply requesting the size of the storage allocation
  if (!d_temp_storage)
  {
    temp_storage_bytes = bytes_needed;
    return cudaSuccess;
  }

  // Check if enough storage provided
  if (temp_storage_bytes < bytes_needed)
  {
    return CubDebug(cudaErrorInvalidValue);
  }

  // Alias
  d_temp_storage =
    reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(d_temp_storage) + ALIGN_BYTES - 1) & ALIGN_MASK);
  for (int i = 0; i < ALLOCATIONS; ++i)
  {
    allocations[i] = static_cast<char*>(d_temp_storage) + allocation_offsets[i];
  }

  return cudaSuccess;
}

} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
