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

/**
 * @file
 * cub::DeviceSegmentedSort provides device-wide, parallel operations for
 * computing a batched sort across multiple, non-overlapping sequences of
 * data items residing within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_segmented_sort.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_namespace.cuh>

CUB_NAMESPACE_BEGIN


/**
 * @brief DeviceSegmentedSort provides device-wide, parallel operations for
 *        computing a batched sort across multiple, non-overlapping sequences of
 *        data items residing within device-accessible memory.
 *        ![](segmented_sorting_logo.png)
 * @ingroup SegmentedModule
 *
 * @par Overview
 * The algorithm arranges items into ascending (or descending) order.
 * The underlying sorting algorithm is undefined. Depending on the segment size,
 * it might be radix sort, merge sort or something else. Therefore, no
 * assumptions on the underlying implementation should be made.
 *
 * @par Differences from DeviceSegmentedRadixSort
 * DeviceSegmentedRadixSort is optimized for significantly large segments (tens
 * of thousands of items and more). Nevertheless, some domains produce a wide
 * range of segment sizes. DeviceSegmentedSort partitions segments into size
 * groups and specialize sorting algorithms for each group. This approach leads
 * to better resource utilization in the presence of segment size imbalance or
 * moderate segment sizes (up to thousands of items).
 * This algorithm is more complex and consists of multiple kernels. This fact
 * leads to longer compilation times as well as larger binaries sizes.
 *
 * @par Supported Types
 * The algorithm has to satisfy the underlying algorithms restrictions. Radix
 * sort usage restricts the list of supported types. Therefore,
 * DeviceSegmentedSort can sort all of the built-in C++ numeric primitive types
 * (`unsigned char`, `int`, `double`, etc.) as well as CUDA's `__half` and
 * `__nv_bfloat16` 16-bit floating-point types.
 *
 * @par Segments are not required to be contiguous. Any element of input(s) or 
 * output(s) outside the specified segments will not be accessed nor modified.  
 *
 * @par A simple example
 * @code
 * #include <cub/cub.cuh>
 * // or equivalently <cub/device/device_segmented_sort.cuh>
 *
 * // Declare, allocate, and initialize device-accessible pointers
 * // for sorting data
 * int  num_items;          // e.g., 7
 * int  num_segments;       // e.g., 3
 * int  *d_offsets;         // e.g., [0, 3, 3, 7]
 * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
 * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
 * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
 * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
 * ...
 *
 * // Determine temporary device storage requirements
 * void     *d_temp_storage = NULL;
 * size_t   temp_storage_bytes = 0;
 * cub::DeviceSegmentedSort::SortPairs(
 *     d_temp_storage, temp_storage_bytes,
 *     d_keys_in, d_keys_out, d_values_in, d_values_out,
 *     num_items, num_segments, d_offsets, d_offsets + 1);
 *
 * // Allocate temporary storage
 * cudaMalloc(&d_temp_storage, temp_storage_bytes);
 *
 * // Run sorting operation
 * cub::DeviceSegmentedSort::SortPairs(
 *     d_temp_storage, temp_storage_bytes,
 *     d_keys_in, d_keys_out, d_values_in, d_values_out,
 *     num_items, num_segments, d_offsets, d_offsets + 1);
 *
 * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
 * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
 * @endcode
 */
struct DeviceSegmentedSort
{

  /*************************************************************************//**
   * @name Keys-only
   ****************************************************************************/
  //@{

  /**
   * @brief Sorts segments of keys into ascending order. Approximately
   *        `num_items + 2*num_segments` auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - SortKeys is not guaranteed to be stable. That is, suppose that @p i and
   *   @p j are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   * - The range `[d_keys_out, d_keys_out + num_items)` shall not overlap
   *   `[d_keys_in, d_keys_in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_keys_out[i]` will not 
   *   be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible
   * // pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in] d_keys_in
   *   Device-accessible pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out
   *   Device-accessible pointer to the sorted output sequence of key data
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length `num_segments`, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream = 0)
  {
    constexpr bool is_descending = false;
    constexpr bool is_overwrite_okay = false;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            cub::NullType,
                                            int,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(d_temp_storage,
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

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeys<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
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

  /**
   * @brief Sorts segments of keys into descending order. Approximately
   *        `num_items + 2*num_segments` auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments + 1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets + 1`).
   * - SortKeysDescending is not guaranteed to be stable. That is, suppose that
   *   @p i and @p j are equivalent: neither one is less than the other. It is
   *   not guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   * - The range `[d_keys_out, d_keys_out + num_items)` shall not overlap
   *   `[d_keys_in, d_keys_in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_keys_out[i]` will not 
   *   be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no
   *   work is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in] d_keys_in
   *   Device-accessible pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out
   *   Device-accessible pointer to the sorted output sequence of key data
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i] - 1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     std::size_t &temp_storage_bytes,
                     const KeyT *d_keys_in,
                     KeyT *d_keys_out,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     cudaStream_t stream = 0)
  {
    constexpr bool is_descending = true;
    constexpr bool is_overwrite_okay = false;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            cub::NullType,
                                            int,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(d_temp_storage,
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

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     std::size_t &temp_storage_bytes,
                     const KeyT *d_keys_in,
                     KeyT *d_keys_out,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     cudaStream_t stream,
                     bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeysDescending<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
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

  /**
   * @brief Sorts segments of keys into ascending order. Approximately
   *        `2*num_segments` auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within the DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits and the targeted device architecture).
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - SortKeys is not guaranteed to be stable. That is, suppose that
   *   @p i and @p j are equivalent: neither one is less than the other. It is
   *   not guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   * - Let `cur = d_keys.Current()` and `alt = d_keys.Alternate()`.
   *   The range `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_keys[i].Alternate()[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible
   * // pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no
   *   work is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in,out] d_keys
   *   Reference to the double-buffer of keys whose "current" device-accessible
   *   buffer contains the unsorted input keys and, upon return, is updated to
   *   point to the sorted output keys
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*`
   *   and `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i] - 1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream = 0)
  {
    constexpr bool is_descending     = false;
    constexpr bool is_overwrite_okay = true;

    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            cub::NullType,
                                            int,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(d_temp_storage,
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

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeys<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  /**
   * @brief Sorts segments of keys into descending order. Approximately
   *        `2*num_segments` auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within the DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits and the targeted device architecture).
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments + 1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets + 1`).
   * - SortKeysDescending is not guaranteed to be stable. That is, suppose that
   *   @p i and @p j are equivalent: neither one is less than the other. It is
   *   not guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   * - Let `cur = d_keys.Current()` and `alt = d_keys.Alternate()`.
   *   The range `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_keys[i].Alternate()[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for
   * // sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in,out] d_keys
   *   Reference to the double-buffer of keys whose "current" device-accessible
   *   buffer contains the unsorted input keys and, upon return, is updated to
   *   point to the sorted output keys
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i] - 1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i] - 1<= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     std::size_t &temp_storage_bytes,
                     DoubleBuffer<KeyT> &d_keys,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     cudaStream_t stream = 0)
  {
    constexpr bool is_descending = true;
    constexpr bool is_overwrite_okay = true;

    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            cub::NullType,
                                            int,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(d_temp_storage,
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

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     std::size_t &temp_storage_bytes,
                     DoubleBuffer<KeyT> &d_keys,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     cudaStream_t stream,
                     bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeysDescending<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  /**
   * @brief Sorts segments of keys into ascending order. Approximately
   *        `num_items + 2*num_segments` auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - StableSortKeys is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   * - The range `[d_keys_out, d_keys_out + num_items)` shall not overlap
   *   `[d_keys_in, d_keys_in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_keys_out[i]` will not 
   *   be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in] d_keys_in
   *   Device-accessible pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out
   *   Device-accessible pointer to the sorted output sequence of key data
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeys(void *d_temp_storage,
                 std::size_t &temp_storage_bytes,
                 const KeyT *d_keys_in,
                 KeyT *d_keys_out,
                 int num_items,
                 int num_segments,
                 BeginOffsetIteratorT d_begin_offsets,
                 EndOffsetIteratorT d_end_offsets,
                 cudaStream_t stream = 0)
  {
    return SortKeys<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
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

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeys(void *d_temp_storage,
                 std::size_t &temp_storage_bytes,
                 const KeyT *d_keys_in,
                 KeyT *d_keys_out,
                 int num_items,
                 int num_segments,
                 BeginOffsetIteratorT d_begin_offsets,
                 EndOffsetIteratorT d_end_offsets,
                 cudaStream_t stream,
                 bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortKeys<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
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

  /**
   * @brief Sorts segments of keys into descending order. Approximately
   *        `num_items + 2*num_segments` auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - StableSortKeysDescending is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   * - The range `[d_keys_out, d_keys_out + num_items)` shall not overlap
   *   `[d_keys_in, d_keys_in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_keys_out[i]` will not 
   *   be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in] d_keys_in
   *   Device-accessible pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out
   *   Device-accessible pointer to the sorted output sequence of key data
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeysDescending(void *d_temp_storage,
                           std::size_t &temp_storage_bytes,
                           const KeyT *d_keys_in,
                           KeyT *d_keys_out,
                           int num_items,
                           int num_segments,
                           BeginOffsetIteratorT d_begin_offsets,
                           EndOffsetIteratorT d_end_offsets,
                           cudaStream_t stream = 0)
  {
    return SortKeysDescending<KeyT,
                              BeginOffsetIteratorT,
                              EndOffsetIteratorT>(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_keys_in,
                                                  d_keys_out,
                                                  num_items,
                                                  num_segments,
                                                  d_begin_offsets,
                                                  d_end_offsets,
                                                  stream);
  }

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeysDescending(void *d_temp_storage,
                           std::size_t &temp_storage_bytes,
                           const KeyT *d_keys_in,
                           KeyT *d_keys_out,
                           int num_items,
                           int num_segments,
                           BeginOffsetIteratorT d_begin_offsets,
                           EndOffsetIteratorT d_end_offsets,
                           cudaStream_t stream,
                           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortKeysDescending<KeyT,
                                    BeginOffsetIteratorT,
                                    EndOffsetIteratorT>(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys_in,
                                                        d_keys_out,
                                                        num_items,
                                                        num_segments,
                                                        d_begin_offsets,
                                                        d_end_offsets,
                                                        stream);
  }

  /**
   * @brief Sorts segments of keys into ascending order. Approximately
   *        `2*num_segments` auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within the DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits and the targeted device architecture).
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - StableSortKeys is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   * - Let `cur = d_keys.Current()` and `alt = d_keys.Alternate()`.
   *   The range `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_keys[i].Alternate()[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in,out] d_keys
   *   Reference to the double-buffer of keys whose "current" device-accessible
   *   buffer contains the unsorted input keys and, upon return, is updated to
   *   point to the sorted output keys
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i] - 1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeys(void *d_temp_storage,
                 std::size_t &temp_storage_bytes,
                 DoubleBuffer<KeyT> &d_keys,
                 int num_items,
                 int num_segments,
                 BeginOffsetIteratorT d_begin_offsets,
                 EndOffsetIteratorT d_end_offsets,
                 cudaStream_t stream = 0)
  {
    return SortKeys<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeys(void *d_temp_storage,
                 std::size_t &temp_storage_bytes,
                 DoubleBuffer<KeyT> &d_keys,
                 int num_items,
                 int num_segments,
                 BeginOffsetIteratorT d_begin_offsets,
                 EndOffsetIteratorT d_end_offsets,
                 cudaStream_t stream,
                 bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortKeys<KeyT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream);
  }

  /**
   * @brief Sorts segments of keys into descending order. Approximately
   *        `2*num_segments` auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within the DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits and the targeted device architecture).
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - StableSortKeysDescending is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   * - Let `cur = d_keys.Current()` and `alt = d_keys.Alternate()`.
   *   The range `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_keys[i].Alternate()[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in,out] d_keys
   *   Reference to the double-buffer of keys whose "current" device-accessible
   *   buffer contains the unsorted input keys and, upon return, is updated to
   *   point to the sorted output keys
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`. If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the
   *   i-th segment is considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeysDescending(void *d_temp_storage,
                           std::size_t &temp_storage_bytes,
                           DoubleBuffer<KeyT> &d_keys,
                           int num_items,
                           int num_segments,
                           BeginOffsetIteratorT d_begin_offsets,
                           EndOffsetIteratorT d_end_offsets,
                           cudaStream_t stream = 0)
  {
    return SortKeysDescending<KeyT,
                              BeginOffsetIteratorT,
                              EndOffsetIteratorT>(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_keys,
                                                  num_items,
                                                  num_segments,
                                                  d_begin_offsets,
                                                  d_end_offsets,
                                                  stream);
  }

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeysDescending(void *d_temp_storage,
                           std::size_t &temp_storage_bytes,
                           DoubleBuffer<KeyT> &d_keys,
                           int num_items,
                           int num_segments,
                           BeginOffsetIteratorT d_begin_offsets,
                           EndOffsetIteratorT d_end_offsets,
                           cudaStream_t stream,
                           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortKeysDescending<KeyT,
                                    BeginOffsetIteratorT,
                                    EndOffsetIteratorT>(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys,
                                                        num_items,
                                                        num_segments,
                                                        d_begin_offsets,
                                                        d_end_offsets,
                                                        stream);
  }

  //@}  end member group
  /*************************************************************************//**
   * @name Key-value pairs
   ****************************************************************************/
  //@{

  /**
   * @brief Sorts segments of key-value pairs into ascending order.
   *        Approximately `2*num_items + 2*num_segments` auxiliary storage
   *        required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - SortPairs is not guaranteed to be stable. That is, suppose that @p i and
   *   @p j are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   * - Let `in` be one of `{d_keys_in, d_values_in}` and `out` be any of
   *   `{d_keys_out, d_values_out}`. The range `[out, out + num_items)` shall 
   *   not overlap `[in, in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_values_in[i]`, 
   *   `d_keys_out[i]`, `d_values_out[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam ValueT
   *   <b>[inferred]</b> Value type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in] d_keys_in
   *   Device-accessible pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out
   *   Device-accessible pointer to the sorted output sequence of key data
   *
   * @param[in] d_values_in
   *   Device-accessible pointer to the corresponding input sequence of
   *   associated value items
   *
   * @param[out] d_values_out
   *   Device-accessible pointer to the correspondingly-reordered output
   *   sequence of associated value items
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           const ValueT *d_values_in,
           ValueT *d_values_out,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream = 0)
  {
    constexpr bool is_descending = false;
    constexpr bool is_overwrite_okay = false;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            ValueT,
                                            int,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in), d_values_out);

    return DispatchT::Dispatch(d_temp_storage,
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

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           const ValueT *d_values_in,
           ValueT *d_values_out,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairs<KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT>(
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

  /**
   * @brief Sorts segments of key-value pairs into descending order. Approximately
   *        `2*num_items + 2*num_segments` auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - SortPairs is not guaranteed to be stable. That is, suppose that @p i and
   *   @p j are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   * - Let `in` be one of `{d_keys_in, d_values_in}` and `out` be any of
   *   `{d_keys_out, d_values_out}`. The range `[out, out + num_items)` shall 
   *   not overlap `[in, in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_values_in[i]`, 
   *   `d_keys_out[i]`, `d_values_out[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for
   * // sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
   * // d_values_out          <-- [0, 2, 1, 6, 3, 4, 5]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam ValueT
   *   <b>[inferred]</b> Value type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in] d_keys_in
   *   Device-accessible pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out
   *   Device-accessible pointer to the sorted output sequence of key data
   *
   * @param[in] d_values_in
   *   Device-accessible pointer to the corresponding input sequence of
   *   associated value items
   *
   * @param[out] d_values_out
   *   Device-accessible pointer to the correspondingly-reordered output
   *   sequence of associated value items
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      std::size_t &temp_storage_bytes,
                      const KeyT *d_keys_in,
                      KeyT *d_keys_out,
                      const ValueT *d_values_in,
                      ValueT *d_values_out,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      cudaStream_t stream = 0)
  {
    constexpr bool is_descending = true;
    constexpr bool is_overwrite_okay = false;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            ValueT,
                                            int,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in), d_values_out);

    return DispatchT::Dispatch(d_temp_storage,
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

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      std::size_t &temp_storage_bytes,
                      const KeyT *d_keys_in,
                      KeyT *d_keys_out,
                      const ValueT *d_values_in,
                      ValueT *d_values_out,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      cudaStream_t stream,
                      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairsDescending<KeyT,
                               ValueT,
                               BeginOffsetIteratorT,
                               EndOffsetIteratorT>(d_temp_storage,
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

  /**
   * @brief Sorts segments of key-value pairs into ascending order.
   *        Approximately `2*num_segments` auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the sorting
   *   operation.
   * - Upon completion, the sorting operation will update the "current" indicator
   *   within each DoubleBuffer wrapper to reference which of the two buffers
   *   now contains the sorted output sequence (a function of the number of key bits
   *   specified and the targeted device architecture).
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - SortPairs is not guaranteed to be stable. That is, suppose that @p i and
   *   @p j are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   * - Let `cur` be one of `{d_keys.Current(), d_values.Current()}` and `alt` 
   *   be any of `{d_keys.Alternate(), d_values.Alternate()}`. The range 
   *   `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_values.Current()[i]`, `d_keys.Alternate()[i]`, 
   *   `d_values.Alternate()[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam ValueT
   *   <b>[inferred]</b> Value type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in,out] d_keys
   *   Reference to the double-buffer of keys whose "current" device-accessible
   *   buffer contains the unsorted input keys and, upon return, is updated to
   *   point to the sorted output keys
   *
   * @param[in,out] d_values
   *   Double-buffer of values whose "current" device-accessible buffer contains
   *   the unsorted input values and, upon return, is updated to point to the
   *   sorted output values
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            std::size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            int num_items,
            int num_segments,
            BeginOffsetIteratorT d_begin_offsets,
            EndOffsetIteratorT d_end_offsets,
            cudaStream_t stream = 0)
  {
    constexpr bool is_descending = false;
    constexpr bool is_overwrite_okay = true;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            ValueT,
                                            int,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    return DispatchT::Dispatch(d_temp_storage,
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

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            std::size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            int num_items,
            int num_segments,
            BeginOffsetIteratorT d_begin_offsets,
            EndOffsetIteratorT d_end_offsets,
            cudaStream_t stream,
            bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairs<KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT>(
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

  /**
   * @brief Sorts segments of key-value pairs into descending order.
   *        Approximately `2*num_segments` auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers. Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the
   *   sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within each DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits specified and the targeted device architecture).
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - SortPairsDescending is not guaranteed to be stable. That is, suppose that
   *   @p i and @p j are equivalent: neither one is less than the other. It is
   *   not guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   * - Let `cur` be one of `{d_keys.Current(), d_values.Current()}` and `alt` 
   *   be any of `{d_keys.Alternate(), d_values.Alternate()}`. The range 
   *   `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_values.Current()[i]`, `d_keys.Alternate()[i]`, 
   *   `d_values.Alternate()[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for
   * // sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
   * // d_values.Current()    <-- [0, 2, 1, 6, 3, 4, 5]
   *
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam ValueT
   *   <b>[inferred]</b> Value type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in,out] d_keys
   *   Reference to the double-buffer of keys whose "current" device-accessible
   *   buffer contains the unsorted input keys and, upon return, is updated to
   *   point to the sorted output keys
   *
   * @param[in,out] d_values
   *   Double-buffer of values whose "current" device-accessible buffer contains
   *   the unsorted input values and, upon return, is updated to point to the
   *   sorted output values
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      std::size_t &temp_storage_bytes,
                      DoubleBuffer<KeyT> &d_keys,
                      DoubleBuffer<ValueT> &d_values,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      cudaStream_t stream = 0)
  {
    constexpr bool is_descending = true;
    constexpr bool is_overwrite_okay = true;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            ValueT,
                                            int,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    return DispatchT::Dispatch(d_temp_storage,
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

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      std::size_t &temp_storage_bytes,
                      DoubleBuffer<KeyT> &d_keys,
                      DoubleBuffer<ValueT> &d_values,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      cudaStream_t stream,
                      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairsDescending<KeyT,
                               ValueT,
                               BeginOffsetIteratorT,
                               EndOffsetIteratorT>(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_keys,
                                                   d_values,
                                                   num_items,
                                                   num_segments,
                                                   d_begin_offsets,
                                                   d_end_offsets,
                                                   stream);
  }

  /**
   * @brief Sorts segments of key-value pairs into ascending order. Approximately
   *        `2*num_items + 2*num_segments` auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - StableSortPairs is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   * - Let `in` be one of `{d_keys_in, d_values_in}` and `out` be any of
   *   `{d_keys_out, d_values_out}`. The range `[out, out + num_items)` shall 
   *   not overlap `[in, in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_values_in[i]`, 
   *   `d_keys_out[i]`, `d_values_out[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam ValueT
   *   <b>[inferred]</b> Value type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the
   *   required allocation size is written to @p temp_storage_bytes and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in] d_keys_in
   *   Device-accessible pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out
   *   Device-accessible pointer to the sorted output sequence of key data
   *
   * @param[in] d_values_in
   *   Device-accessible pointer to the corresponding input sequence of
   *   associated value items
   *
   * @param[out] d_values_out
   *   Device-accessible pointer to the correspondingly-reordered output
   *   sequence of associated value items
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairs(void *d_temp_storage,
                  std::size_t &temp_storage_bytes,
                  const KeyT *d_keys_in,
                  KeyT *d_keys_out,
                  const ValueT *d_values_in,
                  ValueT *d_values_out,
                  int num_items,
                  int num_segments,
                  BeginOffsetIteratorT d_begin_offsets,
                  EndOffsetIteratorT d_end_offsets,
                  cudaStream_t stream = 0)
  {
    return SortPairs<KeyT,
                     ValueT,
                     BeginOffsetIteratorT,
                     EndOffsetIteratorT>(d_temp_storage,
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

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairs(void *d_temp_storage,
                  std::size_t &temp_storage_bytes,
                  const KeyT *d_keys_in,
                  KeyT *d_keys_out,
                  const ValueT *d_values_in,
                  ValueT *d_values_out,
                  int num_items,
                  int num_segments,
                  BeginOffsetIteratorT d_begin_offsets,
                  EndOffsetIteratorT d_end_offsets,
                  cudaStream_t stream,
                  bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortPairs<KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT>(
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

  /**
   * @brief Sorts segments of key-value pairs into descending order.
   *        Approximately `2*num_items + 2*num_segments` auxiliary
   *        storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - StableSortPairsDescending is stable: it preserves the relative ordering
   *   of equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   * - Let `in` be one of `{d_keys_in, d_values_in}` and `out` be any of
   *   `{d_keys_out, d_values_out}`. The range `[out, out + num_items)` shall 
   *   not overlap `[in, in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_values_in[i]`, 
   *   `d_keys_out[i]`, `d_values_out[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
   * // d_values_out          <-- [0, 2, 1, 6, 3, 4, 5]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam ValueT
   *   <b>[inferred]</b> Value type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in] d_keys_in
   *   Device-accessible pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out
   *   Device-accessible pointer to the sorted output sequence of key data
   *
   * @param[in] d_values_in
   *   Device-accessible pointer to the corresponding input sequence of
   *   associated value items
   *
   * @param[out] d_values_out
   *   Device-accessible pointer to the correspondingly-reordered output
   *   sequence of associated value items
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairsDescending(void *d_temp_storage,
                            std::size_t &temp_storage_bytes,
                            const KeyT *d_keys_in,
                            KeyT *d_keys_out,
                            const ValueT *d_values_in,
                            ValueT *d_values_out,
                            int num_items,
                            int num_segments,
                            BeginOffsetIteratorT d_begin_offsets,
                            EndOffsetIteratorT d_end_offsets,
                            cudaStream_t stream = 0)
  {
    return SortPairsDescending<KeyT,
                               ValueT,
                               BeginOffsetIteratorT,
                               EndOffsetIteratorT>(d_temp_storage,
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

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairsDescending(void *d_temp_storage,
                            std::size_t &temp_storage_bytes,
                            const KeyT *d_keys_in,
                            KeyT *d_keys_out,
                            const ValueT *d_values_in,
                            ValueT *d_values_out,
                            int num_items,
                            int num_segments,
                            BeginOffsetIteratorT d_begin_offsets,
                            EndOffsetIteratorT d_end_offsets,
                            cudaStream_t stream,
                            bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortPairsDescending<KeyT,
                                     ValueT,
                                     BeginOffsetIteratorT,
                                     EndOffsetIteratorT>(d_temp_storage,
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

  /**
   * @brief Sorts segments of key-value pairs into ascending order.
   *        Approximately `2*num_segments` auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the
   *   sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within each DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits specified and the targeted device architecture).
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - StableSortPairs is stable: it preserves the relative ordering
   *   of equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   * - Let `cur` be one of `{d_keys.Current(), d_values.Current()}` and `alt` 
   *   be any of `{d_keys.Alternate(), d_values.Alternate()}`. The range 
   *   `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_values.Current()[i]`, `d_keys.Alternate()[i]`, 
   *   `d_values.Alternate()[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam ValueT
   *   <b>[inferred]</b> Value type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in,out] d_keys
   *   Reference to the double-buffer of keys whose "current" device-accessible
   *   buffer contains the unsorted input keys and, upon return, is updated to
   *   point to the sorted output keys
   *
   * @param[in,out] d_values
   *   Double-buffer of values whose "current" device-accessible buffer contains
   *   the unsorted input values and, upon return, is updated to point to the
   *   sorted output values
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairs(void *d_temp_storage,
                  std::size_t &temp_storage_bytes,
                  DoubleBuffer<KeyT> &d_keys,
                  DoubleBuffer<ValueT> &d_values,
                  int num_items,
                  int num_segments,
                  BeginOffsetIteratorT d_begin_offsets,
                  EndOffsetIteratorT d_end_offsets,
                  cudaStream_t stream = 0)
  {
    return SortPairs<KeyT,
                     ValueT,
                     BeginOffsetIteratorT,
                     EndOffsetIteratorT>(d_temp_storage,
                                         temp_storage_bytes,
                                         d_keys,
                                         d_values,
                                         num_items,
                                         num_segments,
                                         d_begin_offsets,
                                         d_end_offsets,
                                         stream);
  }

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairs(void *d_temp_storage,
                  std::size_t &temp_storage_bytes,
                  DoubleBuffer<KeyT> &d_keys,
                  DoubleBuffer<ValueT> &d_values,
                  int num_items,
                  int num_segments,
                  BeginOffsetIteratorT d_begin_offsets,
                  EndOffsetIteratorT d_end_offsets,
                  cudaStream_t stream,
                  bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortPairs<KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT>(
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

  /**
   * @brief Sorts segments of key-value pairs into descending order.
   *        Approximately `2*num_segments` auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the sorting
   *   operation.
   * - Upon completion, the sorting operation will update the "current" indicator
   *   within each DoubleBuffer wrapper to reference which of the two buffers
   *   now contains the sorted output sequence (a function of the number of key bits
   *   specified and the targeted device architecture).
   * - When the input is a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length `num_segments+1`) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as `segment_offsets+1`).
   * - StableSortPairsDescending is stable: it preserves the relative ordering
   *   of equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   * - Let `cur` be one of `{d_keys.Current(), d_values.Current()}` and `alt` 
   *   be any of `{d_keys.Alternate(), d_values.Alternate()}`. The range 
   *   `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_values.Current()[i]`, `d_keys.Alternate()[i]`, 
   *   `d_values.Alternate()[i]` will not be accessed nor modified.   
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>
   * // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
   * // d_values.Current()    <-- [0, 2, 1, 6, 3, 4, 5]
   * @endcode
   *
   * @tparam KeyT
   *   <b>[inferred]</b> Key type
   *
   * @tparam ValueT
   *   <b>[inferred]</b> Value type
   *
   * @tparam BeginOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT
   *   <b>[inferred]</b> Random-access input iterator type for reading segment
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to @p temp_storage_bytes and no work
   *   is done
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of @p d_temp_storage allocation
   *
   * @param[in,out] d_keys
   *   Reference to the double-buffer of keys whose "current" device-accessible
   *   buffer contains the unsorted input keys and, upon return, is updated to
   *   point to the sorted output keys
   *
   * @param[in,out] d_values
   *   Double-buffer of values whose "current" device-accessible buffer contains
   *   the unsorted input values and, upon return, is updated to point to the
   *   sorted output values
   *
   * @param[in] num_items
   *   The total number of items to sort (across all segments)
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length @p num_segments, such that `d_begin_offsets[i]` is the first
   *   element of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   @p num_segments, such that `d_end_offsets[i]-1` is the last element of
   *   the <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
   *   considered empty.
   *
   * @param[in] stream
   *   <b>[optional]</b> CUDA stream to launch kernels within. Default is
   *   stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairsDescending(void *d_temp_storage,
                            std::size_t &temp_storage_bytes,
                            DoubleBuffer<KeyT> &d_keys,
                            DoubleBuffer<ValueT> &d_values,
                            int num_items,
                            int num_segments,
                            BeginOffsetIteratorT d_begin_offsets,
                            EndOffsetIteratorT d_end_offsets,
                            cudaStream_t stream = 0)
  {
    return SortPairsDescending<KeyT,
                               ValueT,
                               BeginOffsetIteratorT,
                               EndOffsetIteratorT>(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_keys,
                                                   d_values,
                                                   num_items,
                                                   num_segments,
                                                   d_begin_offsets,
                                                   d_end_offsets,
                                                   stream);
  }

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairsDescending(void *d_temp_storage,
                            std::size_t &temp_storage_bytes,
                            DoubleBuffer<KeyT> &d_keys,
                            DoubleBuffer<ValueT> &d_values,
                            int num_items,
                            int num_segments,
                            BeginOffsetIteratorT d_begin_offsets,
                            EndOffsetIteratorT d_end_offsets,
                            cudaStream_t stream,
                            bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return StableSortPairsDescending<KeyT,
                                     ValueT,
                                     BeginOffsetIteratorT,
                                     EndOffsetIteratorT>(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys,
                                                         d_values,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         stream);
  }

  //@}  end member group

};


CUB_NAMESPACE_END
