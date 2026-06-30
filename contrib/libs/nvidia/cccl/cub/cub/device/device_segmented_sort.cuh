/******************************************************************************
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
//! cub::DeviceSegmentedSort provides device-wide, parallel operations for computing a batched sort across multiple,
//! non-overlapping sequences of data items residing within device-accessible memory.

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
#include <cub/device/dispatch/dispatch_segmented_sort.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceSegmentedSort provides device-wide, parallel operations for
//! computing a batched sort across multiple, non-overlapping sequences of
//! data items residing within device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The algorithm arranges items into ascending (or descending) order.
//! The underlying sorting algorithm is undefined. Depending on the segment size,
//! it might be radix sort, merge sort or something else. Therefore, no
//! assumptions on the underlying implementation should be made.
//!
//! Differences from DeviceSegmentedRadixSort
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! DeviceSegmentedRadixSort is optimized for significantly large segments (tens
//! of thousands of items and more). Nevertheless, some domains produce a wide
//! range of segment sizes. DeviceSegmentedSort partitions segments into size
//! groups and specialize sorting algorithms for each group. This approach leads
//! to better resource utilization in the presence of segment size imbalance or
//! moderate segment sizes (up to thousands of items).
//! This algorithm is more complex and consists of multiple kernels. This fact
//! leads to longer compilation times as well as larger binaries sizes.
//!
//! Supported Types
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The algorithm has to satisfy the underlying algorithms restrictions. Radix
//! sort usage restricts the list of supported types. Therefore,
//! DeviceSegmentedSort can sort all of the built-in C++ numeric primitive types
//! (``unsigned char``, ``int``, ``double``, etc.) as well as CUDA's ``__half`` and
//! ``__nv_bfloat16`` 16-bit floating-point types.
//!
//! Segments are not required to be contiguous. Any element of input(s) or
//! output(s) outside the specified segments will not be accessed nor modified.
//!
//! A simple example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!    // or equivalently <cub/device/device_segmented_sort.cuh>
//!
//!    // Declare, allocate, and initialize device-accessible pointers
//!    // for sorting data
//!    int  num_items;          // e.g., 7
//!    int  num_segments;       // e.g., 3
//!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
//!    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
//!    int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
//!    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
//!    int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
//!    ...
//!
//!    // Determine temporary device storage requirements
//!    void     *d_temp_storage = nullptr;
//!    size_t   temp_storage_bytes = 0;
//!    cub::DeviceSegmentedSort::SortPairs(
//!        d_temp_storage, temp_storage_bytes,
//!        d_keys_in, d_keys_out, d_values_in, d_values_out,
//!        num_items, num_segments, d_offsets, d_offsets + 1);
//!
//!    // Allocate temporary storage
//!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
//!
//!    // Run sorting operation
//!    cub::DeviceSegmentedSort::SortPairs(
//!        d_temp_storage, temp_storage_bytes,
//!        d_keys_in, d_keys_out, d_values_in, d_values_out,
//!        num_items, num_segments, d_offsets, d_offsets + 1);
//!
//!    // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
//!    // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
//!
//! @endrst
struct DeviceSegmentedSort
{
private:
  // Name reported for NVTX ranges
  _CCCL_HOST_DEVICE static constexpr auto GetName() -> const char*
  {
    return "cub::DeviceSegmentedSort";
  }

  // Internal version without NVTX range
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeysNoNVTX(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    constexpr bool is_overwrite_okay = false;

    using OffsetT =
      detail::choose_signed_offset_t<detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>>;
    using DispatchT =
      DispatchSegmentedSort<SortOrder::Ascending, KeyT, cub::NullType, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_okay,
      stream);
  }

public:
  //! @name Keys-only
  //! @{

  //! @rst
  //! Sorts segments of keys into ascending order.
  //! Approximately ``num_items + 2 * num_segments`` auxiliary storage required.
  //!
  //! - The contents of the input data are not altered by the sorting operation.
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as `segment_offsets+1`).
  //! - SortKeys is not guaranteed to be stable. That is, suppose that ``i`` and
  //!   ``j`` are equivalent: neither one is less than the other. It is not
  //!   guaranteed that the relative order of these two elements will be
  //!   preserved by sort.
  //! - The range ``[d_keys_out, d_keys_out + num_items)`` shall not overlap
  //!   ``[d_keys_in, d_keys_in + num_items)``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys_in[i]``, ``d_keys_out[i]`` will not
  //!   be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``int`` keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible
  //!    // pointers for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void    *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::SortKeys(
  //!        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::SortKeys(
  //!        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Device-accessible pointer to the input data of key data to sort
  //!
  //! @param[out] d_keys_out
  //!   Device-accessible pointer to the sorted output sequence of key data
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the i-th segment is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortKeysNoNVTX(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

private:
  // Internal version without NVTX range
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeysDescendingNoNVTX(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    constexpr bool is_overwrite_okay = false;

    using OffsetT =
      detail::choose_signed_offset_t<detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>>;
    using DispatchT =
      DispatchSegmentedSort<SortOrder::Descending, KeyT, cub::NullType, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_okay,
      stream);
  }

public:
  //! @rst
  //! Sorts segments of keys into descending order. Approximately
  //! ``num_items + 2 * num_segments`` auxiliary storage required.
  //!
  //! - The contents of the input data are not altered by the sorting operation.
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - SortKeysDescending is not guaranteed to be stable. That is, suppose that
  //!   ``i`` and ``j`` are equivalent: neither one is less than the other. It is
  //!   not guaranteed that the relative order of these two elements will be
  //!   preserved by sort.
  //! - The range ``[d_keys_out, d_keys_out + num_items)`` shall not overlap
  //!   ``[d_keys_in, d_keys_in + num_items)``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys_in[i]``, ``d_keys_out[i]`` will not
  //!   be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void    *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::SortKeysDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::SortKeysDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Device-accessible pointer to the input data of key data to sort
  //!
  //! @param[out] d_keys_out
  //!   Device-accessible pointer to the sorted output sequence of key data
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the ``i``-th segment is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeysDescending(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortKeysDescendingNoNVTX(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

private:
  // Internal version without NVTX range
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeysNoNVTX(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    constexpr bool is_overwrite_okay = true;
    using OffsetT =
      detail::choose_signed_offset_t<detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>>;
    using DispatchT =
      DispatchSegmentedSort<SortOrder::Ascending, KeyT, cub::NullType, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_okay,
      stream);
  }

public:
  //! @rst
  //! Sorts segments of keys into ascending order. Approximately ``2 * num_segments`` auxiliary storage required.
  //!
  //! - The sorting operation is given a pair of key buffers managed by a
  //!   DoubleBuffer structure that indicates which of the two buffers is
  //!   "current" (and thus contains the input data to be sorted).
  //! - The contents of both buffers may be altered by the sorting operation.
  //! - Upon completion, the sorting operation will update the "current"
  //!   indicator within the DoubleBuffer wrapper to reference which of the two
  //!   buffers now contains the sorted output sequence (a function of the number
  //!   of key bits and the targeted device architecture).
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets  +1``).
  //! - SortKeys is not guaranteed to be stable. That is, suppose that
  //!   ``i`` and ``j`` are equivalent: neither one is less than the other. It is
  //!   not guaranteed that the relative order of these two elements will be
  //!   preserved by sort.
  //! - Let ``cur = d_keys.Current()`` and ``alt = d_keys.Alternate()``.
  //!   The range ``[cur, cur + num_items)`` shall not overlap
  //!   ``[alt, alt + num_items)``. Both ranges shall not overlap
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys.Current()[i]``,
  //!   ``d_keys[i].Alternate()[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible
  //!    // pointers for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Create a DoubleBuffer to wrap the pair of device pointers
  //!    cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::SortKeys(
  //!        d_temp_storage, temp_storage_bytes, d_keys,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::SortKeys(
  //!        d_temp_storage, temp_storage_bytes, d_keys,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no
  //!   work is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_keys
  //!   Reference to the double-buffer of keys whose "current" device-accessible
  //!   buffer contains the unsorted input keys and, upon return, is updated to
  //!   point to the sorted output keys
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the ``i``-th segment is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortKeysNoNVTX(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, stream);
  }

private:
  // Internal version without NVTX range
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeysDescendingNoNVTX(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    constexpr bool is_overwrite_okay = true;
    using OffsetT =
      detail::choose_signed_offset_t<detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>>;
    using DispatchT =
      DispatchSegmentedSort<SortOrder::Descending, KeyT, cub::NullType, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_okay,
      stream);
  }

public:
  //! @rst
  //! Sorts segments of keys into descending order. Approximately
  //! ``2 * num_segments`` auxiliary storage required.
  //!
  //! - The sorting operation is given a pair of key buffers managed by a
  //!   DoubleBuffer structure that indicates which of the two buffers is
  //!   "current" (and thus contains the input data to be sorted).
  //! - The contents of both buffers may be altered by the sorting operation.
  //! - Upon completion, the sorting operation will update the "current"
  //!   indicator within the DoubleBuffer wrapper to reference which of the two
  //!   buffers now contains the sorted output sequence (a function of the number
  //!   of key bits and the targeted device architecture).
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - SortKeysDescending is not guaranteed to be stable. That is, suppose that
  //!   ``i`` and ``j`` are equivalent: neither one is less than the other. It is
  //!   not guaranteed that the relative order of these two elements will be
  //!   preserved by sort.
  //! - Let ``cur = d_keys.Current()`` and ``alt = d_keys.Alternate()``.
  //!   The range ``[cur, cur + num_items)`` shall not overlap
  //!   ``[alt, alt + num_items)``. Both ranges shall not overlap
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys.Current()[i]``,
  //!   ``d_keys[i].Alternate()[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Create a DoubleBuffer to wrap the pair of device pointers
  //!    cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::SortKeysDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::SortKeysDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_keys
  //!   Reference to the double-buffer of keys whose "current" device-accessible
  //!   buffer contains the unsorted input keys and, upon return, is updated to
  //!   point to the sorted output keys
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1<= d_begin_offsets[i]``, the ``i``-th segment is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortKeysDescending(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortKeysDescendingNoNVTX(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, stream);
  }

  //! @rst
  //! Sorts segments of keys into ascending order. Approximately
  //! ``num_items +  2 * num_segments`` auxiliary storage required.
  //!
  //! - The contents of the input data are not altered by the sorting operation.
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - StableSortKeys is stable: it preserves the relative ordering of
  //!   equivalent elements. That is, if ``x`` and ``y`` are elements such that
  //!   ``x`` precedes ``y``, and if the two elements are equivalent (neither
  //!   ``x < y`` nor ``y < x``) then a postcondition of stable sort is that
  //!   ``x`` still precedes ``y``.
  //! - The range ``[d_keys_out, d_keys_out + num_items)`` shall not overlap
  //!   ``[d_keys_in, d_keys_in + num_items)``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys_in[i]``, ``d_keys_out[i]`` will not
  //!   be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void    *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::StableSortKeys(
  //!        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::StableSortKeys(
  //!        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Device-accessible pointer to the input data of key data to sort
  //!
  //! @param[out] d_keys_out
  //!   Device-accessible pointer to the sorted output sequence of key data
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the ``i``-th segment is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t StableSortKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortKeysNoNVTX<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  //! @rst
  //! Sorts segments of keys into descending order.
  //! Approximately ``num_items + 2 * num_segments`` auxiliary storage required.
  //!
  //! - The contents of the input data are not altered by the sorting operation.
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - StableSortKeysDescending is stable: it preserves the relative ordering of
  //!   equivalent elements. That is, if ``x`` and ``y`` are elements such that
  //!   ``x`` precedes ``y``, and if the two elements are equivalent (neither ``x < y`` nor ``y < x``)
  //!   then a postcondition of stable sort is that ``x`` still precedes ``y``.
  //! - The range ``[d_keys_out, d_keys_out + num_items)`` shall not overlap
  //!   ``[d_keys_in, d_keys_in + num_items)``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys_in[i]``, ``d_keys_out[i]`` will not
  //!   be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void    *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::StableSortKeysDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::StableSortKeysDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Device-accessible pointer to the input data of key data to sort
  //!
  //! @param[out] d_keys_out
  //!   Device-accessible pointer to the sorted output sequence of key data
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and
  //!   ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the ``i``-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t StableSortKeysDescending(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortKeysDescendingNoNVTX<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  //! @rst
  //! Sorts segments of keys into ascending order.
  //! Approximately ``2 * num_segments`` auxiliary storage required.
  //!
  //! - The sorting operation is given a pair of key buffers managed by a
  //!   DoubleBuffer structure that indicates which of the two buffers is
  //!   "current" (and thus contains the input data to be sorted).
  //! - The contents of both buffers may be altered by the sorting operation.
  //! - Upon completion, the sorting operation will update the "current"
  //!   indicator within the DoubleBuffer wrapper to reference which of the two
  //!   buffers now contains the sorted output sequence (a function of the number
  //!   of key bits and the targeted device architecture).
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - StableSortKeys is stable: it preserves the relative ordering of
  //!   equivalent elements. That is, if ``x`` and ``y`` are elements such that
  //!   ``x`` precedes ``y``, and if the two elements are equivalent (neither
  //!   ``x < y`` nor ``y < x``) then a postcondition of stable sort is that
  //!   ``x`` still precedes ``y``.
  //! - Let ``cur = d_keys.Current()`` and ``alt = d_keys.Alternate()``.
  //!   The range ``[cur, cur + num_items)`` shall not overlap
  //!   ``[alt, alt + num_items)``. Both ranges shall not overlap
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys.Current()[i]``,
  //!   ``d_keys[i].Alternate()[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Create a DoubleBuffer to wrap the pair of device pointers
  //!    cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::StableSortKeys(
  //!        d_temp_storage, temp_storage_bytes, d_keys,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::StableSortKeys(
  //!        d_temp_storage, temp_storage_bytes, d_keys,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_keys
  //!   Reference to the double-buffer of keys whose "current" device-accessible
  //!   buffer contains the unsorted input keys and, upon return, is updated to
  //!   point to the sorted output keys
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the ``i``-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t StableSortKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortKeysNoNVTX<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, stream);
  }

  //! @rst
  //! Sorts segments of keys into descending order.
  //! Approximately ``2 * num_segments`` auxiliary storage required.
  //!
  //! - The sorting operation is given a pair of key buffers managed by a
  //!   DoubleBuffer structure that indicates which of the two buffers is
  //!   "current" (and thus contains the input data to be sorted).
  //! - The contents of both buffers may be altered by the sorting operation.
  //! - Upon completion, the sorting operation will update the "current"
  //!   indicator within the DoubleBuffer wrapper to reference which of the two
  //!   buffers now contains the sorted output sequence (a function of the number
  //!   of key bits and the targeted device architecture).
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - StableSortKeysDescending is stable: it preserves the relative ordering of
  //!   equivalent elements. That is, if ``x`` and ``y`` are elements such that
  //!   ``x`` precedes ``y``, and if the two elements are equivalent (neither
  //!   ``x < y`` nor ``y < x``) then a postcondition of stable sort is that
  //!   ``x`` still precedes ``y``.
  //! - Let ``cur = d_keys.Current()`` and ``alt = d_keys.Alternate()``.
  //!   The range ``[cur, cur + num_items)`` shall not overlap
  //!   ``[alt, alt + num_items)``. Both ranges shall not overlap
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ```i`
  //!   outside the specified segments ``d_keys.Current()[i]``,
  //!   ``d_keys[i].Alternate()[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Create a DoubleBuffer to wrap the pair of device pointers
  //!    cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::StableSortKeysDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::StableSortKeysDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_keys
  //!   Reference to the double-buffer of keys whose "current" device-accessible
  //!   buffer contains the unsorted input keys and, upon return, is updated to
  //!   point to the sorted output keys
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and
  //!   ``d_values_*``. If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the
  //!   ``i``-th segment is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t StableSortKeysDescending(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortKeysDescendingNoNVTX<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments, d_begin_offsets, d_end_offsets, stream);
  }

private:
  // Internal version without NVTX range
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairsNoNVTX(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    constexpr bool is_overwrite_okay = false;

    using OffsetT =
      detail::choose_signed_offset_t<detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>>;
    using DispatchT =
      DispatchSegmentedSort<SortOrder::Ascending, KeyT, ValueT, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT*>(d_values_in), d_values_out);

    return DispatchT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_okay,
      stream);
  }

public:
  //! @}  end member group
  //! @name Key-value pairs
  //! @{

  //! @rst
  //! Sorts segments of key-value pairs into ascending order.
  //! Approximately ``2 * num_items + 2 * num_segments`` auxiliary storage required.
  //!
  //! - The contents of the input data are not altered by the sorting operation.
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - SortPairs is not guaranteed to be stable. That is, suppose that ``i`` and
  //!   ``j`` are equivalent: neither one is less than the other. It is not
  //!   guaranteed that the relative order of these two elements will be
  //!   preserved by sort.
  //! - Let ``in`` be one of ``{d_keys_in, d_values_in}`` and ``out`` be any of
  //!   ``{d_keys_out, d_values_out}``. The range ``[out, out + num_items)`` shall
  //!   not overlap ``[in, in + num_items)``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys_in[i]``, ``d_values_in[i]``,
  //!   ``d_keys_out[i]``, ``d_values_out[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys with associated vector of
  //! ``i`` nt values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
  //!    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
  //!    int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::SortPairs(
  //!        d_temp_storage, temp_storage_bytes,
  //!        d_keys_in, d_keys_out, d_values_in, d_values_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::SortPairs(
  //!        d_temp_storage, temp_storage_bytes,
  //!        d_keys_in, d_keys_out, d_values_in, d_values_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
  //!    // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam ValueT
  //!   **[inferred]** Value type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Device-accessible pointer to the input data of key data to sort
  //!
  //! @param[out] d_keys_out
  //!   Device-accessible pointer to the sorted output sequence of key data
  //!
  //! @param[in] d_values_in
  //!   Device-accessible pointer to the corresponding input sequence of
  //!   associated value items
  //!
  //! @param[out] d_values_out
  //!   Device-accessible pointer to the correspondingly-reordered output
  //!   sequence of associated value items
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i]-1 <= d_begin_offsets[i]``, the ``i``-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortPairsNoNVTX(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

private:
  // Internal version without NVTX range
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairsDescendingNoNVTX(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    constexpr bool is_overwrite_okay = false;

    using OffsetT =
      detail::choose_signed_offset_t<detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>>;
    using DispatchT =
      DispatchSegmentedSort<SortOrder::Descending, KeyT, ValueT, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT*>(d_values_in), d_values_out);

    return DispatchT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_okay,
      stream);
  }

public:
  //! @rst
  //! Sorts segments of key-value pairs into descending order.
  //! Approximately ``2 * num_items + 2 * num_segments`` auxiliary storage required.
  //!
  //! - The contents of the input data are not altered by the sorting operation.
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - SortPairs is not guaranteed to be stable. That is, suppose that ``i`` and
  //!   ``j`` are equivalent: neither one is less than the other. It is not
  //!   guaranteed that the relative order of these two elements will be
  //!   preserved by sort.
  //! - Let ``in`` be one of ``{d_keys_in, d_values_in}`` and ``out`` be any of
  //!   ``{d_keys_out, d_values_out}``. The range ``[out, out + num_items)`` shall
  //!   not overlap ``[in, in + num_items)``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys_in[i]``, ``d_values_in[i]``,
  //!   ``d_keys_out[i]``, ``d_values_out[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys with associated vector of
  //! ``i`` nt values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
  //!    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
  //!    int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void    *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::SortPairsDescending(
  //!        d_temp_storage, temp_storage_bytes,
  //!        d_keys_in, d_keys_out, d_values_in, d_values_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::SortPairsDescending(
  //!        d_temp_storage, temp_storage_bytes,
  //!        d_keys_in, d_keys_out, d_values_in, d_values_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
  //!    // d_values_out          <-- [0, 2, 1, 6, 3, 4, 5]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam ValueT
  //!   **[inferred]** Value type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Device-accessible pointer to the input data of key data to sort
  //!
  //! @param[out] d_keys_out
  //!   Device-accessible pointer to the sorted output sequence of key data
  //!
  //! @param[in] d_values_in
  //!   Device-accessible pointer to the corresponding input sequence of
  //!   associated value items
  //!
  //! @param[out] d_values_out
  //!   Device-accessible pointer to the correspondingly-reordered output
  //!   sequence of associated value items
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the i-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairsDescending(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortPairsDescendingNoNVTX(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

private:
  // Internal version without NVTX range
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairsNoNVTX(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    constexpr bool is_overwrite_okay = true;

    using OffsetT =
      detail::choose_signed_offset_t<detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>>;
    using DispatchT =
      DispatchSegmentedSort<SortOrder::Ascending, KeyT, ValueT, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

    return DispatchT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_okay,
      stream);
  }

public:
  //! @rst
  //! Sorts segments of key-value pairs into ascending order.
  //! Approximately ``2 * num_segments`` auxiliary storage required.
  //!
  //! - The sorting operation is given a pair of key buffers and a corresponding
  //!   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
  //!   structure that indicates which of the two buffers is "current" (and thus
  //!   contains the input data to be sorted).
  //! - The contents of both buffers within each pair may be altered by the sorting
  //!   operation.
  //! - Upon completion, the sorting operation will update the "current" indicator
  //!   within each DoubleBuffer wrapper to reference which of the two buffers
  //!   now contains the sorted output sequence (a function of the number of key bits
  //!   specified and the targeted device architecture).
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - SortPairs is not guaranteed to be stable. That is, suppose that ``i`` and
  //!   ``j`` are equivalent: neither one is less than the other. It is not
  //!   guaranteed that the relative order of these two elements will be
  //!   preserved by sort.
  //! - Let ``cur`` be one of ``{d_keys.Current(), d_values.Current()}`` and ``alt``
  //!   be any of ``{d_keys.Alternate(), d_values.Alternate()}``. The range
  //!   ``[cur, cur + num_items)`` shall not overlap
  //!   ``[alt, alt + num_items)``. Both ranges shall not overlap
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys.Current()[i]``,
  //!   ``d_values.Current()[i]``, ``d_keys.Alternate()[i]``,
  //!   ``d_values.Alternate()[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys with associated vector of
  //! ``i`` nt values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
  //!    int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
  //!    int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Create a set of DoubleBuffers to wrap pairs of device pointers
  //!    cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
  //!    cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::SortPairs(
  //!        d_temp_storage, temp_storage_bytes, d_keys, d_values,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::SortPairs(
  //!        d_temp_storage, temp_storage_bytes, d_keys, d_values,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
  //!    // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam ValueT
  //!   **[inferred]** Value type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_keys
  //!   Reference to the double-buffer of keys whose "current" device-accessible
  //!   buffer contains the unsorted input keys and, upon return, is updated to
  //!   point to the sorted output keys
  //!
  //! @param[in,out] d_values
  //!   Double-buffer of values whose "current" device-accessible buffer contains
  //!   the unsorted input values and, upon return, is updated to point to the
  //!   sorted output values
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the i-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortPairsNoNVTX(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

private:
  // Internal version without NVTX range
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairsDescendingNoNVTX(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    constexpr bool is_overwrite_okay = true;

    using OffsetT =
      detail::choose_signed_offset_t<detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>>;
    using DispatchT =
      DispatchSegmentedSort<SortOrder::Descending, KeyT, ValueT, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

    return DispatchT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      is_overwrite_okay,
      stream);
  }

public:
  //! @rst
  //! Sorts segments of key-value pairs into descending order.
  //! Approximately ``2 * num_segments`` auxiliary storage required.
  //!
  //! - The sorting operation is given a pair of key buffers and a corresponding
  //!   pair of associated value buffers. Each pair is managed by a DoubleBuffer
  //!   structure that indicates which of the two buffers is "current" (and thus
  //!   contains the input data to be sorted).
  //! - The contents of both buffers within each pair may be altered by the
  //!   sorting operation.
  //! - Upon completion, the sorting operation will update the "current"
  //!   indicator within each DoubleBuffer wrapper to reference which of the two
  //!   buffers now contains the sorted output sequence (a function of the number
  //!   of key bits specified and the targeted device architecture).
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - SortPairsDescending is not guaranteed to be stable. That is, suppose that
  //!   ``i`` and ``j`` are equivalent: neither one is less than the other. It is
  //!   not guaranteed that the relative order of these two elements will be
  //!   preserved by sort.
  //! - Let ``cur`` be one of ``{d_keys.Current(), d_values.Current()}`` and ``alt``
  //!   be any of ``{d_keys.Alternate(), d_values.Alternate()}``. The range
  //!   ``[cur, cur + num_items)`` shall not overlap
  //!   ``[alt, alt + num_items)``. Both ranges shall not overlap
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys.Current()[i]``,
  //!   ``d_values.Current()[i]``, ``d_keys.Alternate()[i]``,
  //!   ``d_values.Alternate()[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys with associated vector of
  //! ``i`` nt values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
  //!    int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
  //!    int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Create a set of DoubleBuffers to wrap pairs of device pointers
  //!    cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
  //!    cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::SortPairsDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys, d_values,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::SortPairsDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys, d_values,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
  //!    // d_values.Current()    <-- [0, 2, 1, 6, 3, 4, 5]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam ValueT
  //!   **[inferred]** Value type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_keys
  //!   Reference to the double-buffer of keys whose "current" device-accessible
  //!   buffer contains the unsorted input keys and, upon return, is updated to
  //!   point to the sorted output keys
  //!
  //! @param[in,out] d_values
  //!   Double-buffer of values whose "current" device-accessible buffer contains
  //!   the unsorted input values and, upon return, is updated to point to the
  //!   sorted output values
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the ``i``-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t SortPairsDescending(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortPairsDescendingNoNVTX(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  //! @rst
  //! Sorts segments of key-value pairs into ascending order.
  //! Approximately ``2 * num_items + 2 * num_segments`` auxiliary storage required.
  //!
  //! - The contents of the input data are not altered by the sorting operation.
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - StableSortPairs is stable: it preserves the relative ordering of
  //!   equivalent elements. That is, if ``x`` and ``y`` are elements such that
  //!   ``x`` precedes ``y``, and if the two elements are equivalent (neither
  //!   ``x < y`` nor ``y < x``) then a postcondition of stable sort is that
  //!   ``x`` still precedes ``y``.
  //! - Let ``in`` be one of ``{d_keys_in, d_values_in}`` and ``out`` be any of
  //!   ``{d_keys_out, d_values_out}``. The range ``[out, out + num_items)`` shall
  //!   not overlap ``[in, in + num_items)``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys_in[i]``, ``d_values_in[i]``,
  //!   ``d_keys_out[i]``, ``d_values_out[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys with associated vector of
  //! ``i`` nt values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
  //!    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
  //!    int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::StableSortPairs(
  //!        d_temp_storage, temp_storage_bytes,
  //!        d_keys_in, d_keys_out, d_values_in, d_values_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::StableSortPairs(
  //!        d_temp_storage, temp_storage_bytes,
  //!        d_keys_in, d_keys_out, d_values_in, d_values_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
  //!    // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam ValueT
  //!   **[inferred]** Value type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When nullptr, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Device-accessible pointer to the input data of key data to sort
  //!
  //! @param[out] d_keys_out
  //!   Device-accessible pointer to the sorted output sequence of key data
  //!
  //! @param[in] d_values_in
  //!   Device-accessible pointer to the corresponding input sequence of
  //!   associated value items
  //!
  //! @param[out] d_values_out
  //!   Device-accessible pointer to the correspondingly-reordered output
  //!   sequence of associated value items
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the ``i``-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t StableSortPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortPairsNoNVTX<KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  //! @rst
  //! Sorts segments of key-value pairs into descending order.
  //! Approximately ``2 * num_items + 2 * num_segments`` auxiliary storage required.
  //!
  //! - The contents of the input data are not altered by the sorting operation.
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - StableSortPairsDescending is stable: it preserves the relative ordering
  //!   of equivalent elements. That is, if ``x`` and ``y`` are elements such that
  //!   ``x`` precedes ``y``, and if the two elements are equivalent (neither
  //!   ``x < y`` nor ``y < x``) then a postcondition of stable sort is that
  //!   ``x`` still precedes ``y``.
  //! - Let `in` be one of ``{d_keys_in, d_values_in}`` and ``out`` be any of
  //!   ``{d_keys_out, d_values_out}``. The range ``[out, out + num_items)`` shall
  //!   not overlap ``[in, in + num_items)``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys_in[i]``, ``d_values_in[i]``,
  //!   ``d_keys_out[i]``, ``d_values_out[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys with associated vector of
  //! ``i`` nt values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
  //!    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
  //!    int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::StableSortPairsDescending(
  //!        d_temp_storage, temp_storage_bytes,
  //!        d_keys_in, d_keys_out, d_values_in, d_values_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::StableSortPairsDescending(
  //!        d_temp_storage, temp_storage_bytes,
  //!        d_keys_in, d_keys_out, d_values_in, d_values_out,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
  //!    // d_values_out          <-- [0, 2, 1, 6, 3, 4, 5]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam ValueT
  //!   **[inferred]** Value type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Device-accessible pointer to the input data of key data to sort
  //!
  //! @param[out] d_keys_out
  //!   Device-accessible pointer to the sorted output sequence of key data
  //!
  //! @param[in] d_values_in
  //!   Device-accessible pointer to the corresponding input sequence of
  //!   associated value items
  //!
  //! @param[out] d_values_out
  //!   Device-accessible pointer to the correspondingly-reordered output
  //!   sequence of associated value items
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the ``i``-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t StableSortPairsDescending(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortPairsDescendingNoNVTX<KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  //! @rst
  //! Sorts segments of key-value pairs into ascending order.
  //! Approximately ``2 * num_segments`` auxiliary storage required.
  //!
  //! - The sorting operation is given a pair of key buffers and a corresponding
  //!   pair of associated value buffers. Each pair is managed by a DoubleBuffer
  //!   structure that indicates which of the two buffers is "current" (and thus
  //!   contains the input data to be sorted).
  //! - The contents of both buffers within each pair may be altered by the
  //!   sorting operation.
  //! - Upon completion, the sorting operation will update the "current"
  //!   indicator within each DoubleBuffer wrapper to reference which of the two
  //!   buffers now contains the sorted output sequence (a function of the number
  //!   of key bits specified and the targeted device architecture).
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - StableSortPairs is stable: it preserves the relative ordering
  //!   of equivalent elements. That is, if ``x`` and ``y`` are elements such that
  //!   ``x`` precedes `y`, and if the two elements are equivalent (neither
  //!   ``x < y`` nor ``y < x``) then a postcondition of stable sort is that
  //!   ``x`` still precedes ``y``.
  //! - Let ``cur`` be one of ``{d_keys.Current(), d_values.Current()}`` and ``alt``
  //!   be any of ``{d_keys.Alternate(), d_values.Alternate()}``. The range
  //!   ``[cur, cur + num_items)`` shall not overlap
  //!   ``[alt, alt + num_items)``. Both ranges shall not overlap
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys.Current()[i]``,
  //!   ``d_values.Current()[i]``, ``d_keys.Alternate()[i]``,
  //!   ``d_values.Alternate()[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys with associated vector of
  //! ``i`` nt values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
  //!    int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
  //!    int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Create a set of DoubleBuffers to wrap pairs of device pointers
  //!    cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
  //!    cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::StableSortPairs(
  //!        d_temp_storage, temp_storage_bytes, d_keys, d_values,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::StableSortPairs(
  //!        d_temp_storage, temp_storage_bytes, d_keys, d_values,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
  //!    // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam ValueT
  //!   **[inferred]** Value type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_keys
  //!   Reference to the double-buffer of keys whose "current" device-accessible
  //!   buffer contains the unsorted input keys and, upon return, is updated to
  //!   point to the sorted output keys
  //!
  //! @param[in,out] d_values
  //!   Double-buffer of values whose "current" device-accessible buffer contains
  //!   the unsorted input values and, upon return, is updated to point to the
  //!   sorted output values
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i]-1 <= d_begin_offsets[i]``, the ``i``-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t StableSortPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortPairsNoNVTX<KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  //! @rst
  //! Sorts segments of key-value pairs into descending order.
  //! Approximately ``2 * num_segments`` auxiliary storage required.
  //!
  //! - The sorting operation is given a pair of key buffers and a corresponding
  //!   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
  //!   structure that indicates which of the two buffers is "current" (and thus
  //!   contains the input data to be sorted).
  //! - The contents of both buffers within each pair may be altered by the sorting
  //!   operation.
  //! - Upon completion, the sorting operation will update the "current" indicator
  //!   within each DoubleBuffer wrapper to reference which of the two buffers
  //!   now contains the sorted output sequence (a function of the number of key bits
  //!   specified and the targeted device architecture).
  //! - When the input is a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - StableSortPairsDescending is stable: it preserves the relative ordering
  //!   of equivalent elements. That is, if ``x`` and ``y`` are elements such that
  //!   ``x`` precedes ``y``, and if the two elements are equivalent (neither
  //!   ``x < y`` nor ``y < x``) then a postcondition of stable sort is that
  //!   ``x`` still precedes ``y``.
  //! - Let ``cur`` be one of ``{d_keys.Current(), d_values.Current()}`` and ``alt``
  //!   be any of ``{d_keys.Alternate(), d_values.Alternate()}``. The range
  //!   ``[cur, cur + num_items)`` shall not overlap
  //!   ``[alt, alt + num_items)``. Both ranges shall not overlap
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)`` in any way.
  //! - Segments are not required to be contiguous. For all index values ``i``
  //!   outside the specified segments ``d_keys.Current()[i]``,
  //!   ``d_values.Current()[i]``, ``d_keys.Alternate()[i]``,
  //!   ``d_values.Alternate()[i]`` will not be accessed nor modified.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the batched sorting of three segments
  //! (with one zero-length segment) of ``i`` nt keys with associated vector of
  //! ``i`` nt values.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_segmented_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for sorting data
  //!    int  num_items;          // e.g., 7
  //!    int  num_segments;       // e.g., 3
  //!    int  *d_offsets;         // e.g., [0, 3, 3, 7]
  //!    int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
  //!    int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
  //!    int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
  //!    ...
  //!
  //!    // Create a set of DoubleBuffers to wrap pairs of device pointers
  //!    cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
  //!    cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSegmentedSort::StableSortPairsDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys, d_values,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sorting operation
  //!    cub::DeviceSegmentedSort::StableSortPairsDescending(
  //!        d_temp_storage, temp_storage_bytes, d_keys, d_values,
  //!        num_items, num_segments, d_offsets, d_offsets + 1);
  //!
  //!    // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
  //!    // d_values.Current()    <-- [0, 2, 1, 6, 3, 4, 5]
  //!
  //! @endrst
  //!
  //! @tparam KeyT
  //!   **[inferred]** Key type
  //!
  //! @tparam ValueT
  //!   **[inferred]** Value type
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_keys
  //!   Reference to the double-buffer of keys whose "current" device-accessible
  //!   buffer contains the unsorted input keys and, upon return, is updated to
  //!   point to the sorted output keys
  //!
  //! @param[in,out] d_values
  //!   Double-buffer of values whose "current" device-accessible buffer contains
  //!   the unsorted input values and, upon return, is updated to point to the
  //!   sorted output values
  //!
  //! @param[in] num_items
  //!   The total number of items to sort (across all segments)
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the sorting data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_keys_*`` and ``d_values_*``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the ``i``-th segment is
  //!   considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t StableSortPairsDescending(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, GetName());
    return SortPairsDescendingNoNVTX<KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  //! @}  end member group
};

CUB_NAMESPACE_END
