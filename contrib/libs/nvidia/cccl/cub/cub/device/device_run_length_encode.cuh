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

//! @file
//! cub::DeviceRunLengthEncode provides device-wide, parallel operations for computing a run-length encoding across a
//! sequence of data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#include <cuda/std/__functional/invoke.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>
#include <cub/device/dispatch/dispatch_rle.cuh>
#include <cub/device/dispatch/dispatch_streaming_reduce_by_key.cuh>
#include <cub/device/dispatch/tuning/tuning_run_length_encode.cuh>

#include <thrust/iterator/constant_iterator.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceRunLengthEncode provides device-wide, parallel operations for
//! demarcating "runs" of same-valued items within a sequence residing
//! within device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! A `run-length encoding <http://en.wikipedia.org/wiki/Run-length_encoding>`_
//! computes a simple compressed representation of a sequence of input elements
//! such that each maximal "run" of consecutive same-valued data items is
//! encoded as a single data value along with a count of the elements in that
//! run.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceRunLengthEncode}
//!
//! Performance
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @linear_performance{run-length encode}
//!
//! @endrst
struct DeviceRunLengthEncode
{
  //! @rst
  //! Computes a run-length encoding of the sequence ``d_in``.
  //!
  //! - For the *i*\ :sup:`th` run encountered, the first key of the run and
  //!   its length are written to ``d_unique_out[i]`` and ``d_counts_out[i]``, respectively.
  //! - The total number of runs encountered is written to ``d_num_runs_out``.
  //! - The ``==`` equality operator is used to determine whether values are equivalent
  //! - In-place operations are not supported. There must be no overlap between any of the provided ranges:
  //!
  //!   - ``[d_unique_out, d_unique_out + *d_num_runs_out)``
  //!   - ``[d_counts_out, d_counts_out + *d_num_runs_out)``
  //!   - ``[d_num_runs_out, d_num_runs_out + 1)``
  //!   - ``[d_in, d_in + num_items)``
  //!
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the run-length encoding of a sequence of ``int`` values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_run_length_encode.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;          // e.g., 8
  //!    int          *d_in;              // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
  //!    int          *d_unique_out;      // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int          *d_counts_out;      // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int          *d_num_runs_out;    // e.g., [ ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceRunLengthEncode::Encode(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run encoding
  //!    cub::DeviceRunLengthEncode::Encode(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items);
  //!
  //!    // d_unique_out      <-- [0, 2, 9, 5, 8]
  //!    // d_counts_out      <-- [1, 2, 1, 3, 1]
  //!    // d_num_runs_out    <-- [5]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam UniqueOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing unique output items @iterator
  //!
  //! @tparam LengthsOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output counts @iterator
  //!
  //! @tparam NumRunsOutputIteratorT
  //!   **[inferred]** Output iterator type for recording the number of runs encountered @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of keys
  //!
  //! @param[out] d_unique_out
  //!   Pointer to the output sequence of unique keys (one key per run)
  //!
  //! @param[out] d_counts_out
  //!   Pointer to the output sequence of run-lengths (one count per run)
  //!
  //! @param[out] d_num_runs_out
  //!   Pointer to total number of runs
  //!
  //! @param[in] num_items
  //!   Total number of associated key+value pairs (i.e., the length of `d_in_keys` and `d_in_values`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename UniqueOutputIteratorT,
            typename LengthsOutputIteratorT,
            typename NumRunsOutputIteratorT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Encode(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    UniqueOutputIteratorT d_unique_out,
    LengthsOutputIteratorT d_counts_out,
    NumRunsOutputIteratorT d_num_runs_out,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceRunLengthEncode::Encode");

    using equality_op  = ::cuda::std::equal_to<>; // Default == operator
    using reduction_op = ::cuda::std::plus<>; // Value reduction operator

    // Offset type used for global offsets
    using offset_t = detail::choose_signed_offset_t<NumItemsT>;

    // The lengths output value type
    using length_t = cub::detail::non_void_value_t<LengthsOutputIteratorT, offset_t>;

    // Generator type for providing 1s values for run-length reduction
    using lengths_input_iterator_t = THRUST_NS_QUALIFIER::constant_iterator<length_t, offset_t>;

    using accum_t = ::cuda::std::__accumulator_t<reduction_op, length_t, length_t>;

    using key_t = cub::detail::non_void_value_t<UniqueOutputIteratorT, cub::detail::it_value_t<InputIteratorT>>;

    using policy_t = detail::rle::encode::policy_hub<accum_t, key_t>;

    return detail::reduce::DispatchStreamingReduceByKey<
      InputIteratorT,
      UniqueOutputIteratorT,
      lengths_input_iterator_t,
      LengthsOutputIteratorT,
      NumRunsOutputIteratorT,
      equality_op,
      reduction_op,
      offset_t,
      accum_t,
      policy_t>::Dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_unique_out,
                          lengths_input_iterator_t((length_t) 1),
                          d_counts_out,
                          d_num_runs_out,
                          equality_op(),
                          reduction_op(),
                          num_items,
                          stream);
  }

  //! @rst
  //! Enumerates the starting offsets and lengths of all non-trivial runs
  //! (of ``length > 1``) of same-valued keys in the sequence ``d_in``.
  //!
  //! - For the *i*\ :sup:`th` non-trivial run, the run's starting offset and
  //!   its length are written to ``d_offsets_out[i]`` and ``d_lengths_out[i]``, respectively.
  //! - The total number of runs encountered is written to ``d_num_runs_out``.
  //! - The ``==`` equality operator is used to determine whether values are equivalent
  //! - In-place operations are not supported. There must be no overlap between any of the provided ranges:
  //!
  //!   - ``[d_offsets_out, d_offsets_out + *d_num_runs_out)``
  //!   - ``[d_lengths_out, d_lengths_out + *d_num_runs_out)``
  //!   - ``[d_num_runs_out, d_num_runs_out + 1)``
  //!   - ``[d_in, d_in + num_items)``
  //!
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the identification of non-trivial runs
  //! within a sequence of ``int`` values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_run_length_encode.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int          num_items;          // e.g., 8
  //!    int          *d_in;              // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
  //!    int          *d_offsets_out;     // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int          *d_lengths_out;     // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int          *d_num_runs_out;    // e.g., [ ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceRunLengthEncode::NonTrivialRuns(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run encoding
  //!    cub::DeviceRunLengthEncode::NonTrivialRuns(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);
  //!
  //!    // d_offsets_out         <-- [1, 4]
  //!    // d_lengths_out         <-- [2, 3]
  //!    // d_num_runs_out        <-- [2]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OffsetsOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing run-offset values @iterator
  //!
  //! @tparam LengthsOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing run-length values @iterator
  //!
  //! @tparam NumRunsOutputIteratorT
  //!   **[inferred]** Output iterator type for recording the number of runs encountered @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to input sequence of data items
  //!
  //! @param[out] d_offsets_out
  //!   Pointer to output sequence of run-offsets
  //!   (one offset per non-trivial run)
  //!
  //! @param[out] d_lengths_out
  //!   Pointer to output sequence of run-lengths (one count per non-trivial run)
  //!
  //! @param[out] d_num_runs_out
  //!   Pointer to total number of runs (i.e., length of `d_offsets_out`)
  //!
  //! @param[in] num_items
  //!   Total number of associated key+value pairs (i.e., the length of `d_in_keys` and `d_in_values`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OffsetsOutputIteratorT,
            typename LengthsOutputIteratorT,
            typename NumRunsOutputIteratorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t NonTrivialRuns(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OffsetsOutputIteratorT d_offsets_out,
    LengthsOutputIteratorT d_lengths_out,
    NumRunsOutputIteratorT d_num_runs_out,
    int num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceRunLengthEncode::NonTrivialRuns");

    using OffsetT    = int; // Signed integer type for global offsets
    using EqualityOp = ::cuda::std::equal_to<>; // Default == operator

    return DeviceRleDispatch<
      InputIteratorT,
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
};

CUB_NAMESPACE_END
