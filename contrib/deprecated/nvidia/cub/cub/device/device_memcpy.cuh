/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
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
 * cub::DeviceMemcpy provides device-wide, parallel operations for copying data.
 */

#pragma once
#pragma clang system_header


#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_batch_memcpy.cuh>

#include <cstdint>
#include <type_traits>

CUB_NAMESPACE_BEGIN

/**
 * @brief cub::DeviceMemcpy provides device-wide, parallel operations for copying data.
 * \ingroup SingleModule
 */
struct DeviceMemcpy
{
  /**
   * @brief Copies data from a batch of given source buffers to their corresponding destination
   * buffer.
   * @note If any input buffer aliases memory from any output buffer the behavior is undefined. If
   * any output buffer aliases memory of another output buffer the behavior is undefined. Input
   * buffers can alias one another.
   *
   * @par Snippet
   * The code snippet below illustrates usage of DeviceMemcpy::Batched for mutating strings withing
   * a single string buffer.
   * @par
   * @code
   * struct GetPtrToStringItem
   * {
   *   __host__ __device__ __forceinline__ void *operator()(uint32_t index)
   *   {
   *     return &d_string_data_in[d_string_offsets[index]];
   *   }
   *   char *d_string_data_in;
   *   uint32_t *d_string_offsets;
   * };
   *
   * struct GetStringItemSize
   * {
   *   __host__ __device__ __forceinline__ uint32_t operator()(uint32_t index)
   *   {
   *     return d_string_offsets[index + 1] - d_string_offsets[index];
   *   }
   *   uint32_t *d_string_offsets;
   * };
   *
   * uint32_t num_strings = 5;
   * char *d_string_data_in;         // e.g., "TomatoesBananasApplesOrangesGrapes"
   * char *d_string_data_out;        // e.g., "                ...               "
   * uint32_t *d_string_offsets_old; // e.g., [0, 8, 15, 21, 28, 34]
   * uint32_t *d_string_offsets_new; // e.g., [0, 6, 13, 19, 26, 34]
   * uint32_t *d_gather_index;       // e.g., [2, 1, 4, 3, 0]
   *
   * // Initialize an iterator that returns d_gather_index[i] when the i-th item is dereferenced
   * auto gather_iterator = thrust::make_permutation_iterator(thrust::make_counting_iterator(0),
   * d_gather_index);
   *
   * // Returns pointers to the input buffer for each string
   * auto str_ptrs_in = thrust::make_transform_iterator(gather_iterator,
   *                                                    GetPtrToStringItem{d_string_data_in,
   * d_string_offsets_old});
   *
   * // Returns the string size of the i-th string
   * auto str_sizes = thrust::make_transform_iterator(gather_iterator,
   * GetStringItemSize{d_string_offsets_old});
   *
   * // Returns pointers to the output buffer for each string
   * auto str_ptrs_out = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
   *                                                     GetPtrToStringItem{d_string_data_out,
   * d_string_offsets_new});
   *
   * // Determine temporary device storage requirements
   * void *d_temp_storage      = nullptr;
   * size_t temp_storage_bytes = 0;
   * cub::DeviceMemcpy::Batched(d_temp_storage, temp_storage_bytes, str_ptrs_in, str_ptrs_out,
   * str_sizes, num_strings);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run batched copy algorithm (used to permute strings)
   * cub::DeviceMemcpy::Batched(d_temp_storage, temp_storage_bytes, str_ptrs_in, str_ptrs_out,
   * str_sizes, num_strings);
   *
   * // d_string_data_out       <-- "ApplesBananasGrapesOrangesTomatoe"
   * @endcode
   * @tparam InputBufferIt <b>[inferred]</b> Device-accessible random-access input iterator type
   * providing the pointers to the source memory buffers
   * @tparam OutputBufferIt <b>[inferred]</b> Device-accessible random-access input iterator type
   * providing the pointers to the destination memory buffers
   * @tparam BufferSizeIteratorT <b>[inferred]</b> Device-accessible random-access input iterator
   * type providing the number of bytes to be copied for each pair of buffers
   * @param d_temp_storage [in] Device-accessible allocation of temporary storage.  When NULL, the
   * required allocation size is written to \p temp_storage_bytes and no work is done.
   * @param temp_storage_bytes [in,out] Reference to size in bytes of \p d_temp_storage allocation
   * @param input_buffer_it [in] Device-accessible iterator providing the pointers to the source
   * memory buffers
   * @param output_buffer_it [in] Device-accessible iterator providing the pointers to the
   * destination memory buffers
   * @param buffer_sizes [in] Device-accessible iterator providing the number of bytes to be copied
   * for each pair of buffers
   * @param num_buffers [in] The total number of buffer pairs
   * @param stream [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is
   * stream<sub>0</sub>.
   */
  template <typename InputBufferIt, typename OutputBufferIt, typename BufferSizeIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t Batched(void *d_temp_storage,
                                                  size_t &temp_storage_bytes,
                                                  InputBufferIt input_buffer_it,
                                                  OutputBufferIt output_buffer_it,
                                                  BufferSizeIteratorT buffer_sizes,
                                                  uint32_t num_buffers,
                                                  cudaStream_t stream = 0)
  {
    // Integer type large enough to hold any offset in [0, num_buffers)
    using BufferOffsetT = uint32_t;

    // Integer type large enough to hold any offset in [0, num_thread_blocks_launched), where a safe
    // uppper bound on num_thread_blocks_launched can be assumed to be given by
    // IDIV_CEIL(num_buffers, 64)
    using BlockOffsetT = uint32_t;

    return detail::DispatchBatchMemcpy<InputBufferIt,
                                       OutputBufferIt,
                                       BufferSizeIteratorT,
                                       BufferOffsetT,
                                       BlockOffsetT>::Dispatch(d_temp_storage,
                                                               temp_storage_bytes,
                                                               input_buffer_it,
                                                               output_buffer_it,
                                                               buffer_sizes,
                                                               num_buffers,
                                                               stream);
  }
};

CUB_NAMESPACE_END
