/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

//! @file
//! cub::DeviceCopy provides device-wide, parallel operations for copying data.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/dispatch_batch_memcpy.cuh>
#include <cub/device/dispatch/tuning/tuning_batch_memcpy.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

//! @brief cub::DeviceCopy provides device-wide, parallel operations for copying data.
struct DeviceCopy
{
  //! @rst
  //! Copies data from a batch of given source ranges to their corresponding destination ranges.
  //!
  //! .. note::
  //!
  //!    If any input range aliases any output range the behavior is undefined.
  //!    If any output range aliases another output range the behavior is undefined.
  //!    Input ranges can alias one another.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates usage of DeviceCopy::Batched to perform a DeviceRunLength Decode operation.
  //!
  //! .. code-block:: c++
  //!
  //!    struct GetIteratorToRange
  //!    {
  //!      __host__ __device__ __forceinline__ auto operator()(uint32_t index)
  //!      {
  //!        return thrust::make_constant_iterator(d_data_in[index]);
  //!      }
  //!      int32_t *d_data_in;
  //!    };
  //!
  //!    struct GetPtrToRange
  //!    {
  //!      __host__ __device__ __forceinline__ auto operator()(uint32_t index)
  //!      {
  //!        return d_data_out + d_offsets[index];
  //!      }
  //!      int32_t *d_data_out;
  //!      uint32_t *d_offsets;
  //!    };
  //!
  //!    struct GetRunLength
  //!    {
  //!      __host__ __device__ __forceinline__ uint32_t operator()(uint32_t index)
  //!      {
  //!        return d_offsets[index + 1] - d_offsets[index];
  //!      }
  //!      uint32_t *d_offsets;
  //!    };
  //!
  //!    uint32_t num_ranges = 5;
  //!    int32_t *d_data_in;           // e.g., [4, 2, 7, 3, 1]
  //!    int32_t *d_data_out;          // e.g., [0,                ...               ]
  //!    uint32_t *d_offsets;          // e.g., [0, 2, 5, 6, 9, 14]
  //!
  //!    // Returns a constant iterator to the element of the i-th run
  //!    thrust::counting_iterator<uint32_t> iota(0);
  //!    auto iterators_in = thrust::make_transform_iterator(iota, GetIteratorToRange{d_data_in});
  //!
  //!    // Returns the run length of the i-th run
  //!    auto sizes = thrust::make_transform_iterator(iota, GetRunLength{d_offsets});
  //!
  //!    // Returns pointers to the output range for each run
  //!    auto ptrs_out = thrust::make_transform_iterator(iota, GetPtrToRange{d_data_out, d_offsets});
  //!
  //!    // Determine temporary device storage requirements
  //!    void *d_temp_storage      = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, iterators_in, ptrs_out, sizes,
  //!    num_ranges);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run batched copy algorithm (used to perform runlength decoding)
  //!    cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, iterators_in, ptrs_out, sizes,
  //!    num_ranges);
  //!
  //!    // d_data_out       <-- [4, 4, 2, 2, 2, 7, 3, 3, 3, 1, 1, 1, 1, 1]
  //!
  //! @endrst
  //!
  //! @tparam InputIt
  //!   **[inferred]** Device-accessible random-access input iterator type providing the iterators to the source ranges
  //!
  //! @tparam OutputIt
  //!  **[inferred]** Device-accessible random-access input iterator type providing the iterators to
  //!  the destination ranges
  //!
  //! @tparam SizeIteratorT
  //!   **[inferred]** Device-accessible random-access input iterator type providing the number of items to be
  //!   copied for each pair of ranges
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] input_it
  //!   Device-accessible iterator providing the iterators to the source ranges
  //!
  //! @param[in] output_it
  //!   Device-accessible iterator providing the iterators to the destination ranges
  //!
  //! @param[in] sizes
  //!   Device-accessible iterator providing the number of elements to be copied for each pair of ranges
  //!
  //! @param[in] num_ranges
  //!   The total number of range pairs
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIt, typename OutputIt, typename SizeIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t Batched(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIt input_it,
    OutputIt output_it,
    SizeIteratorT sizes,
    ::cuda::std::int64_t num_ranges,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceCopy::Batched");

    // Integer type large enough to hold any offset in [0, num_thread_blocks_launched), where a safe
    // upper bound on num_thread_blocks_launched can be assumed to be given by
    // IDIV_CEIL(num_ranges, 64)
    using BlockOffsetT = uint32_t;

    return detail::DispatchBatchMemcpy<InputIt, OutputIt, SizeIteratorT, BlockOffsetT, CopyAlg::Copy>::Dispatch(
      d_temp_storage, temp_storage_bytes, input_it, output_it, sizes, num_ranges, stream);
  }
};

CUB_NAMESPACE_END
