/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * @file cub::DeviceRunLengthEncode provides device-wide, parallel operations 
 *       for computing a run-length encoding across a sequence of data items 
 *       residing within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <iterator>
#include <stdio.h>

#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>
#include <cub/device/dispatch/dispatch_rle.cuh>
#include <cub/util_deprecated.cuh>

CUB_NAMESPACE_BEGIN


/**
 * @brief DeviceRunLengthEncode provides device-wide, parallel operations for 
 *        demarcating "runs" of same-valued items within a sequence residing 
 *        within device-accessible memory. ![](run_length_encode_logo.png)
 * @ingroup SingleModule
 *
 * @par Overview
 * A <a href="http://en.wikipedia.org/wiki/Run-length_encoding">*run-length encoding*</a>
 * computes a simple compressed representation of a sequence of input elements 
 * such that each maximal "run" of consecutive same-valued data items is 
 * encoded as a single data value along with a count of the elements in that 
 * run.
 *
 * @par Usage Considerations
 * @cdp_class{DeviceRunLengthEncode}
 *
 * @par Performance
 * @linear_performance{run-length encode}
 *
 * @par
 * The following chart illustrates DeviceRunLengthEncode::RunLengthEncode 
 * performance across different CUDA architectures for `int32` items.
 * Segments have lengths uniformly sampled from `[1, 1000]`.
 *
 * @image html rle_int32_len_500.png
 *
 * @par
 * @plots_below
 */
struct DeviceRunLengthEncode
{
  /**
   * @brief Computes a run-length encoding of the sequence \p d_in.
   *
   * @par
   * - For the *i*<sup>th</sup> run encountered, the first key of the run and 
   *   its length are written to `d_unique_out[i]` and `d_counts_out[i]`,
   *   respectively.
   * - The total number of runs encountered is written to `d_num_runs_out`.
   * - The `==` equality operator is used to determine whether values are 
   *   equivalent
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_unique_out, d_unique_out + *d_num_runs_out)`
   *   - `[d_counts_out, d_counts_out + *d_num_runs_out)`
   *   - `[d_num_runs_out, d_num_runs_out + 1)`
   *   - `[d_in, d_in + num_items)`
   * - @devicestorage
   *
   * @par Performance
   * The following charts illustrate saturated encode performance across 
   * different CUDA architectures for `int32` and `int64` items, respectively.  
   * Segments have lengths uniformly sampled from [1,1000].
   *
   * @image html rle_int32_len_500.png
   * @image html rle_int64_len_500.png
   *
   * @par
   * The following charts are similar, but with segment lengths uniformly 
   * sampled from [1,10]:
   *
   * @image html rle_int32_len_5.png
   * @image html rle_int64_len_5.png
   *
   * @par Snippet
   * The code snippet below illustrates the run-length encoding of a sequence 
   * of `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_run_length_encode.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for 
   * // input and output
   * int          num_items;          // e.g., 8
   * int          *d_in;              // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
   * int          *d_unique_out;      // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   * int          *d_counts_out;      // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   * int          *d_num_runs_out;    // e.g., [ ]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRunLengthEncode::Encode(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run encoding
   * cub::DeviceRunLengthEncode::Encode(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items);
   *
   * // d_unique_out      <-- [0, 2, 9, 5, 8]
   * // d_counts_out      <-- [1, 2, 1, 3, 1]
   * // d_num_runs_out    <-- [5]
   * @endcode
   *
   * @tparam InputIteratorT           
   *   **[inferred]** Random-access input iterator type for reading input 
   *   items \iterator
   *
   * @tparam UniqueOutputIteratorT    
   *   **[inferred]** Random-access output iterator type for writing unique 
   *   output items \iterator
   *
   * @tparam LengthsOutputIteratorT   
   *   **[inferred]** Random-access output iterator type for writing output 
   *   counts \iterator
   *
   * @tparam NumRunsOutputIteratorT   
   *   **[inferred]** Output iterator type for recording the number of runs 
   *   encountered \iterator
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in  
   *   Pointer to the input sequence of keys
   *
   * @param[out] d_unique_out  
   *   Pointer to the output sequence of unique keys (one key per run)
   *
   * @param[out] d_counts_out  
   *   Pointer to the output sequence of run-lengths (one count per run)
   *
   * @param[out] d_num_runs_out  
   *   Pointer to total number of runs
   *
   * @param[in] num_items  
   *   Total number of associated key+value pairs (i.e., the length of 
   *   `d_in_keys` and `d_in_values`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within. 
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT,
            typename UniqueOutputIteratorT,
            typename LengthsOutputIteratorT,
            typename NumRunsOutputIteratorT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Encode(void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         UniqueOutputIteratorT d_unique_out,
         LengthsOutputIteratorT d_counts_out,
         NumRunsOutputIteratorT d_num_runs_out,
         int num_items,
         cudaStream_t stream = 0)
  {
    using OffsetT      = int;        // Signed integer type for global offsets
    using FlagIterator = NullType *; // FlagT iterator type (not used)
    using SelectOp     = NullType;   // Selection op (not used)
    using EqualityOp   = Equality;   // Default == operator
    using ReductionOp  = cub::Sum;   // Value reduction operator

    // The lengths output value type
    using LengthT =
      cub::detail::non_void_value_t<LengthsOutputIteratorT, OffsetT>;

    // Generator type for providing 1s values for run-length reduction
    using LengthsInputIteratorT = ConstantInputIterator<LengthT, OffsetT>;

    return DispatchReduceByKey<InputIteratorT,
                               UniqueOutputIteratorT,
                               LengthsInputIteratorT,
                               LengthsOutputIteratorT,
                               NumRunsOutputIteratorT,
                               EqualityOp,
                               ReductionOp,
                               OffsetT>::Dispatch(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_in,
                                                  d_unique_out,
                                                  LengthsInputIteratorT(
                                                    (LengthT)1),
                                                  d_counts_out,
                                                  d_num_runs_out,
                                                  EqualityOp(),
                                                  ReductionOp(),
                                                  num_items,
                                                  stream);
  }

  template <typename InputIteratorT,
            typename UniqueOutputIteratorT,
            typename LengthsOutputIteratorT,
            typename NumRunsOutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Encode(void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         UniqueOutputIteratorT d_unique_out,
         LengthsOutputIteratorT d_counts_out,
         NumRunsOutputIteratorT d_num_runs_out,
         int num_items,
         cudaStream_t stream,
         bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Encode<InputIteratorT,
                  UniqueOutputIteratorT,
                  LengthsOutputIteratorT,
                  NumRunsOutputIteratorT>(d_temp_storage,
                                          temp_storage_bytes,
                                          d_in,
                                          d_unique_out,
                                          d_counts_out,
                                          d_num_runs_out,
                                          num_items,
                                          stream);
  }

  /**
   * @brief Enumerates the starting offsets and lengths of all non-trivial runs 
   *        (of `length > 1`) of same-valued keys in the sequence `d_in`.
   *
   * @par
   * - For the *i*<sup>th</sup> non-trivial run, the run's starting offset and 
   *   its length are written to `d_offsets_out[i]` and `d_lengths_out[i]`, 
   *   respectively.
   * - The total number of runs encountered is written to `d_num_runs_out`.
   * - The `==` equality operator is used to determine whether values are 
   *   equivalent
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_offsets_out, d_offsets_out + *d_num_runs_out)`
   *   - `[d_lengths_out, d_lengths_out + *d_num_runs_out)`
   *   - `[d_num_runs_out, d_num_runs_out + 1)`
   *   - `[d_in, d_in + num_items)` 
   * - @devicestorage
   *
   * @par Performance
   *
   * @par Snippet
   * The code snippet below illustrates the identification of non-trivial runs 
   * within a sequence of `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_run_length_encode.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int          num_items;          // e.g., 8
   * int          *d_in;              // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
   * int          *d_offsets_out;     // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   * int          *d_lengths_out;     // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   * int          *d_num_runs_out;    // e.g., [ ]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRunLengthEncode::NonTrivialRuns(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run encoding
   * cub::DeviceRunLengthEncode::NonTrivialRuns(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
   *
   * // d_offsets_out         <-- [1, 4]
   * // d_lengths_out         <-- [2, 3]
   * // d_num_runs_out        <-- [2]
   * @endcode
   *
   * @tparam InputIteratorT           
   *   **[inferred]** Random-access input iterator type for reading input 
   *   items \iterator
   *
   * @tparam OffsetsOutputIteratorT   
   *   **[inferred]** Random-access output iterator type for writing run-offset 
   *   values \iterator
   *
   * @tparam LengthsOutputIteratorT   
   *   **[inferred]** Random-access output iterator type for writing run-length 
   *   values \iterator
   *
   * @tparam NumRunsOutputIteratorT   
   *   **[inferred]** Output iterator type for recording the number of runs 
   *   encountered \iterator
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in  
   *   Pointer to input sequence of data items
   *
   * @param[out] d_offsets_out  
   *   Pointer to output sequence of run-offsets 
   *   (one offset per non-trivial run)
   *
   * @param[out] d_lengths_out  
   *   Pointer to output sequence of run-lengths 
   *   (one count per non-trivial run)
   *
   * @param[out] d_num_runs_out  
   *   Pointer to total number of runs (i.e., length of `d_offsets_out`)
   *
   * @param[in] num_items  
   *   Total number of associated key+value pairs (i.e., the length of 
   *   `d_in_keys` and `d_in_values`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT,
            typename OffsetsOutputIteratorT,
            typename LengthsOutputIteratorT,
            typename NumRunsOutputIteratorT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  NonTrivialRuns(void *d_temp_storage,
                 size_t &temp_storage_bytes,
                 InputIteratorT d_in,
                 OffsetsOutputIteratorT d_offsets_out,
                 LengthsOutputIteratorT d_lengths_out,
                 NumRunsOutputIteratorT d_num_runs_out,
                 int num_items,
                 cudaStream_t stream = 0)
  {
    using OffsetT    = int;      // Signed integer type for global offsets
    using EqualityOp = Equality; // Default == operator

    return DeviceRleDispatch<InputIteratorT,
                             OffsetsOutputIteratorT,
                             LengthsOutputIteratorT,
                             NumRunsOutputIteratorT,
                             EqualityOp,
                             OffsetT>::Dispatch(d_temp_storage,
                                                temp_storage_bytes,
                                                d_in,
                                                d_offsets_out,
                                                d_lengths_out,
                                                d_num_runs_out,
                                                EqualityOp(),
                                                num_items,
                                                stream);
  }

  template <typename InputIteratorT,
            typename OffsetsOutputIteratorT,
            typename LengthsOutputIteratorT,
            typename NumRunsOutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  NonTrivialRuns(void *d_temp_storage,
                 size_t &temp_storage_bytes,
                 InputIteratorT d_in,
                 OffsetsOutputIteratorT d_offsets_out,
                 LengthsOutputIteratorT d_lengths_out,
                 NumRunsOutputIteratorT d_num_runs_out,
                 int num_items,
                 cudaStream_t stream,
                 bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return NonTrivialRuns<InputIteratorT,
                          OffsetsOutputIteratorT,
                          LengthsOutputIteratorT,
                          NumRunsOutputIteratorT>(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_in,
                                                  d_offsets_out,
                                                  d_lengths_out,
                                                  d_num_runs_out,
                                                  num_items,
                                                  stream);
  }
};

CUB_NAMESPACE_END
