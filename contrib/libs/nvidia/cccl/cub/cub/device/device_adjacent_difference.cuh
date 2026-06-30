/******************************************************************************
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

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/device/dispatch/dispatch_adjacent_difference.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceAdjacentDifference provides device-wide, parallel operations for
//! computing the differences of adjacent elements residing within
//! device-accessible memory.
//!
//! Overview
//! ++++++++++++++++++++++++++
//!
//! - DeviceAdjacentDifference calculates the differences of adjacent elements in
//!   d_input. Because the binary operation could be noncommutative, there
//!   are two sets of methods. Methods named SubtractLeft subtract left element
//!   ``*(i - 1)`` of input sequence from current element ``*i``.
//!   Methods named ``SubtractRight`` subtract current element ``*i`` from the
//!   right one ``*(i + 1)``:
//!
//!   .. code-block:: c++
//!
//!      int *d_values; // [1, 2, 3, 4]
//!      //...
//!      int *d_subtract_left_result  <-- [  1,  1,  1,  1 ]
//!      int *d_subtract_right_result <-- [ -1, -1, -1,  4 ]
//!
//! - For SubtractLeft, if the left element is out of bounds, the iterator is
//!   assigned to ``*(result + (i - first))`` without modification.
//! - For SubtractRight, if the right element is out of bounds, the iterator is
//!   assigned to ``*(result + (i - first))`` without modification.
//!
//! Snippet
//! ++++++++++++++++++++++++++
//!
//! The code snippet below illustrates how to use ``DeviceAdjacentDifference`` to
//! compute the left difference between adjacent elements.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!    // or equivalently <cub/device/device_adjacent_difference.cuh>
//!
//!    // Declare, allocate, and initialize device-accessible pointers
//!    int  num_items;       // e.g., 8
//!    int  *d_values;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
//!    //...
//!
//!    // Determine temporary device storage requirements
//!    void     *d_temp_storage = nullptr;
//!    size_t   temp_storage_bytes = 0;
//!
//!    cub::DeviceAdjacentDifference::SubtractLeft(
//!      d_temp_storage, temp_storage_bytes, d_values, num_items);
//!
//!    // Allocate temporary storage
//!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
//!
//!    // Run operation
//!    cub::DeviceAdjacentDifference::SubtractLeft(
//!      d_temp_storage, temp_storage_bytes, d_values, num_items);
//!
//!    // d_values <-- [1, 1, -1, 1, -1, 1, -1, 1]
//!
//! @endrst
struct DeviceAdjacentDifference
{
private:
  template <MayAlias AliasOpt,
            ReadOption ReadOpt,
            typename NumItemsT,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename DifferenceOpT>
  static CUB_RUNTIME_FUNCTION cudaError_t AdjacentDifference(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_input,
    OutputIteratorT d_output,
    NumItemsT num_items,
    DifferenceOpT difference_op,
    cudaStream_t stream)
  {
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    using DispatchT =
      DispatchAdjacentDifference<InputIteratorT, OutputIteratorT, DifferenceOpT, OffsetT, AliasOpt, ReadOpt>;

    return DispatchT::Dispatch(
      d_temp_storage, temp_storage_bytes, d_input, d_output, static_cast<OffsetT>(num_items), difference_op, stream);
  }

public:
  //! @rst
  //! Subtracts the left element of each adjacent pair of elements residing within device-accessible memory
  //!
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! - Calculates the differences of adjacent elements in ``d_input``.
  //!   That is, ``*d_input`` is assigned to ``*d_output``, and, for each iterator ``i`` in the
  //!   range ``[d_input + 1, d_input + num_items)``, the result of
  //!   ``difference_op(*i, *(i - 1))`` is assigned to ``*(d_output + (i - d_input))``.
  //! - Note that the behavior is undefined if the input and output ranges
  //!   overlap in any way.
  //!
  //! Snippet
  //! ++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``DeviceAdjacentDifference``
  //! to compute the difference between adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_adjacent_difference.cuh>
  //!
  //!    struct CustomDifference
  //!    {
  //!      template <typename DataType>
  //!      __host__ DataType operator()(DataType &lhs, DataType &rhs)
  //!      {
  //!        return lhs - rhs;
  //!      }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    int  num_items;      // e.g., 8
  //!    int  *d_input;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
  //!    int  *d_output;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!
  //!    cub::DeviceAdjacentDifference::SubtractLeftCopy(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_input, d_output,
  //!      num_items, CustomDifference());
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run operation
  //!    cub::DeviceAdjacentDifference::SubtractLeftCopy(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_input, d_output,
  //!      num_items, CustomDifference());
  //!
  //!    // d_input  <-- [1, 2, 1, 2, 1, 2, 1, 2]
  //!    // d_output <-- [1, 1, -1, 1, -1, 1, -1, 1]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   @rst
  //!   is a model of `Input Iterator <https://en.cppreference.com/w/cpp/iterator/input_iterator>`_,
  //!   and ``x`` and ``y`` are objects of ``InputIteratorT``'s ``value_type``, then
  //!   ``x - y`` is defined, and ``InputIteratorT``'s ``value_type`` is convertible to
  //!   a type in ``OutputIteratorT``'s set of ``value_types``, and the return type
  //!   of ``x - y`` is convertible to a type in ``OutputIteratorT``'s set of
  //!   ``value_types``.
  //!   @endrst
  //!
  //! @tparam OutputIteratorT
  //!   @rst
  //!   is a model of `Output Iterator <https://en.cppreference.com/w/cpp/iterator/output_iterator>`_.
  //!   @endrst
  //!
  //! @tparam DifferenceOpT
  //!   Its `result_type` is convertible to a type in `OutputIteratorT`'s set of `value_types`.
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_input
  //!   Pointer to the input sequence
  //!
  //! @param[out] d_output
  //!   Pointer to the output sequence
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename DifferenceOpT = ::cuda::std::minus<>,
            typename NumItemsT     = uint32_t>
  static CUB_RUNTIME_FUNCTION cudaError_t SubtractLeftCopy(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_input,
    OutputIteratorT d_output,
    NumItemsT num_items,
    DifferenceOpT difference_op = {},
    cudaStream_t stream         = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceAdjacentDifference::SubtractLeftCopy");

    return AdjacentDifference<MayAlias::No, ReadOption::Left>(
      d_temp_storage, temp_storage_bytes, d_input, d_output, num_items, difference_op, stream);
  }

  //! @rst
  //! Subtracts the left element of each adjacent pair of elements residing within device-accessible memory.
  //!
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Calculates the differences of adjacent elements in ``d_input``. That is, for
  //! each iterator ``i`` in the range ``[d_input + 1, d_input + num_items)``, the
  //! result of ``difference_op(*i, *(i - 1))`` is assigned to
  //! ``*(d_input + (i - d_input))``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``DeviceAdjacentDifference``
  //! to compute the difference between adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_adjacent_difference.cuh>
  //!
  //!    struct CustomDifference
  //!    {
  //!      template <typename DataType>
  //!      __host__ DataType operator()(DataType &lhs, DataType &rhs)
  //!      {
  //!        return lhs - rhs;
  //!      }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    int  num_items;     // e.g., 8
  //!    int  *d_data;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceAdjacentDifference::SubtractLeft(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items, CustomDifference());
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run operation
  //!    cub::DeviceAdjacentDifference::SubtractLeft(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items, CustomDifference());
  //!
  //!    // d_data <-- [1, 1, -1, 1, -1, 1, -1, 1]
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   @rst
  //!   is a model of `Random Access Iterator <https://en.cppreference.com/w/cpp/iterator/random_access_iterator>`_,
  //!   ``RandomAccessIteratorT`` is mutable. If ``x`` and ``y`` are objects of
  //!   ``RandomAccessIteratorT``'s ``value_type``, and ``x - y`` is defined, then the
  //!   return type of ``x - y`` should be convertible to a type in
  //!   ``RandomAccessIteratorT``'s set of ``value_types``.
  //!   @endrst
  //!
  //! @tparam DifferenceOpT
  //!   Its `result_type` is convertible to a type in `RandomAccessIteratorT`'s
  //!   set of `value_types`.
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of `num_items`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_input
  //!   Pointer to the input sequence and the result
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename RandomAccessIteratorT, typename DifferenceOpT = ::cuda::std::minus<>, typename NumItemsT = uint32_t>
  static CUB_RUNTIME_FUNCTION cudaError_t SubtractLeft(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorT d_input,
    NumItemsT num_items,
    DifferenceOpT difference_op = {},
    cudaStream_t stream         = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceAdjacentDifference::SubtractLeft");

    return AdjacentDifference<MayAlias::Yes, ReadOption::Left>(
      d_temp_storage, temp_storage_bytes, d_input, d_input, num_items, difference_op, stream);
  }

  //! @rst
  //! Subtracts the right element of each adjacent pair of elements residing within device-accessible memory.
  //!
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! - Calculates the right differences of adjacent elements in ``d_input``.
  //!   That is, ``*(d_input + num_items - 1)`` is assigned to
  //!   ``*(d_output + num_items - 1)``, and, for each iterator ``i`` in the range
  //!   ``[d_input, d_input + num_items - 1)``, the result of
  //!   ``difference_op(*i, *(i + 1))`` is assigned to
  //!   ``*(d_output + (i - d_input))``.
  //! - Note that the behavior is undefined if the input and output ranges
  //!   overlap in any way.
  //!
  //! Snippet
  //! ++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``DeviceAdjacentDifference``
  //! to compute the difference between adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_adjacent_difference.cuh>
  //!
  //!    struct CustomDifference
  //!    {
  //!      template <typename DataType>
  //!      __host__ DataType operator()(DataType &lhs, DataType &rhs)
  //!      {
  //!        return lhs - rhs;
  //!      }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    int  num_items;     // e.g., 8
  //!    int  *d_input;      // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
  //!    int  *d_output;
  //!    ..
  //!
  //!    // Determine temporary device storage requirements
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceAdjacentDifference::SubtractRightCopy(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_input, d_output, num_items, CustomDifference());
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run operation
  //!    cub::DeviceAdjacentDifference::SubtractRightCopy(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_input, d_output, num_items, CustomDifference());
  //!
  //!    // d_input <-- [1, 2, 1, 2, 1, 2, 1, 2]
  //!    // d_data  <-- [-1, 1, -1, 1, -1, 1, -1, 2]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   @rst
  //!   is a model of `Input Iterator <https://en.cppreference.com/w/cpp/iterator/input_iterator>`_,
  //!   and ``x`` and ``y`` are objects of ``InputIteratorT``'s ``value_type``, then
  //!   ``x - y`` is defined, and ``InputIteratorT``'s ``value_type`` is convertible to
  //!   a type in ``OutputIteratorT``'s set of ``value_types``, and the return type
  //!   of ``x - y`` is convertible to a type in ``OutputIteratorT``'s set of
  //!   ``value_types``.
  //!   @endrst
  //!
  //! @tparam OutputIteratorT
  //!   @rst
  //!   is a model of `Output Iterator <https://en.cppreference.com/w/cpp/iterator/output_iterator>`_.
  //!   @endrst
  //!
  //! @tparam DifferenceOpT
  //!   Its `result_type` is convertible to a type in `RandomAccessIteratorT`'s
  //!   set of `value_types`.
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_input
  //!   Pointer to the input sequence
  //!
  //! @param[out] d_output
  //!   Pointer to the output sequence
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename DifferenceOpT = ::cuda::std::minus<>,
            typename NumItemsT     = uint32_t>
  static CUB_RUNTIME_FUNCTION cudaError_t SubtractRightCopy(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_input,
    OutputIteratorT d_output,
    NumItemsT num_items,
    DifferenceOpT difference_op = {},
    cudaStream_t stream         = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceAdjacentDifference::SubtractRightCopy");

    return AdjacentDifference<MayAlias::No, ReadOption::Right>(
      d_temp_storage, temp_storage_bytes, d_input, d_output, num_items, difference_op, stream);
  }

  //! @rst
  //! Subtracts the right element of each adjacent pair of elements residing within device-accessible memory.
  //!
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! Calculates the right differences of adjacent elements in ``d_input``.
  //! That is, for each iterator ``i`` in the range
  //! ``[d_input, d_input + num_items - 1)``, the result of
  //! ``difference_op(*i, *(i + 1))`` is assigned to ``*(d_input + (i - d_input))``.
  //!
  //! Snippet
  //! ++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``DeviceAdjacentDifference``
  //! to compute the difference between adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_adjacent_difference.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    int  num_items;    // e.g., 8
  //!    int  *d_data;      // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceAdjacentDifference::SubtractRight(
  //!      d_temp_storage, temp_storage_bytes, d_data, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run operation
  //!    cub::DeviceAdjacentDifference::SubtractRight(
  //!      d_temp_storage, temp_storage_bytes, d_data, num_items);
  //!
  //!    // d_data  <-- [-1, 1, -1, 1, -1, 1, -1, 2]
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   @rst
  //!   is a model of `Random Access Iterator <https://en.cppreference.com/w/cpp/iterator/random_access_iterator>`_,
  //!   ``RandomAccessIteratorT`` is mutable. If ``x`` and ``y`` are objects of
  //!   ``RandomAccessIteratorT``'s `value_type`, and ``x - y`` is defined, then the
  //!   return type of ``x - y`` should be convertible to a type in
  //!   ``RandomAccessIteratorT``'s set of ``value_types``.
  //!   @endrst
  //!
  //! @tparam DifferenceOpT
  //!   Its `result_type` is convertible to a type in `RandomAccessIteratorT`'s
  //!   set of `value_types`.
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_input
  //!   Pointer to the input sequence
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename RandomAccessIteratorT, typename DifferenceOpT = ::cuda::std::minus<>, typename NumItemsT = uint32_t>
  static CUB_RUNTIME_FUNCTION cudaError_t SubtractRight(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorT d_input,
    NumItemsT num_items,
    DifferenceOpT difference_op = {},
    cudaStream_t stream         = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceAdjacentDifference::SubtractRight");

    return AdjacentDifference<MayAlias::Yes, ReadOption::Right>(
      d_temp_storage, temp_storage_bytes, d_input, d_input, num_items, difference_op, stream);
  }
};

CUB_NAMESPACE_END
