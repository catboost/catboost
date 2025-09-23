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
 * @file cub::DeviceSegmentedRadixSort provides device-wide, parallel 
 *       operations for computing a batched radix sort across multiple, 
 *       non-overlapping sequences of data items residing within 
 *       device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <stdio.h>
#include <iterator>

#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_radix_sort.cuh>
#include <cub/util_deprecated.cuh>

CUB_NAMESPACE_BEGIN


/**
 * @brief DeviceSegmentedRadixSort provides device-wide, parallel operations 
 *        for computing a batched radix sort across multiple, non-overlapping 
 *        sequences of data items residing within device-accessible memory. 
 *        ![](segmented_sorting_logo.png)
 * @ingroup SegmentedModule
 *
 * @par Overview
 * The [*radix sorting method*](http://en.wikipedia.org/wiki/Radix_sort) 
 * arranges items into ascending (or descending) order. The algorithm relies 
 * upon a positional representation for keys, i.e., each key is comprised of an 
 * ordered sequence of symbols (e.g., digits, characters, etc.) specified from 
 * least-significant to most-significant.  For a given input sequence of keys 
 * and a set of rules specifying a total ordering of the symbolic alphabet, the 
 * radix sorting method produces a lexicographic ordering of those keys.
 *
 * @par See Also
 * DeviceSegmentedRadixSort shares its implementation with DeviceRadixSort. See
 * that algorithm's documentation for more information.
 *
 * @par Segments are not required to be contiguous. Any element of input(s) or 
 * output(s) outside the specified segments will not be accessed nor modified.  
 *
 * @par Usage Considerations
 * @cdp_class{DeviceSegmentedRadixSort}
 *
 */
struct DeviceSegmentedRadixSort
{
  /******************************************************************//**
   * @name Key-value pairs
   *********************************************************************/
  //@{

  /**
   * @brief Sorts segments of key-value pairs into ascending order. 
   *        (`~2N` auxiliary storage required)
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   `segment_offsets` (of length `num_segments + 1`) can be aliased
   *   for both the `d_begin_offsets` and `d_end_offsets` parameters (where
   *   the latter is specified as `segment_offsets + 1`).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - Let `in` be one of `{d_keys_in, d_values_in}` and `out` be any of
   *   `{d_keys_out, d_values_out}`. The range `[out, out + num_items)` shall 
   *   not overlap `[in, in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - @devicestorageNP For sorting using only `O(P)` temporary storage, see 
   *   the sorting interface using DoubleBuffer wrappers below.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_values_in[i]`, 
   *   `d_keys_out[i]`, `d_values_out[i]` will not be accessed nor modified.   
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments 
   * (with one zero-length segment) of `int` keys with associated vector of 
   * `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>  
   * // or equivalently <cub/device/device_segmented_radix_sort.cuh>
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
   * cub::DeviceSegmentedRadixSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedRadixSort::SortPairs(
         d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
   * @endcode
   *
   * @tparam KeyT                  
   *   **[inferred]** Key type
   *
   * @tparam ValueT                
   *   **[inferred]** Value type
   *
   * @tparam BeginOffsetIteratorT  
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT    
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
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
   *   length `num_segments`, such that `d_begin_offsets[i]` is the first 
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets 
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`. If 
   *   `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is 
   *   considered empty.
   *
   * @param[in] begin_bit 
   *   **[optional]** The least-significant bit index (inclusive) needed for 
   *   key comparison
   *
   * @param[in] end_bit 
   *   **[optional]** The most-significant bit index (exclusive) needed for key 
   *   comparison (e.g., `sizeof(unsigned int) * 8`)
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            size_t &temp_storage_bytes,
            const KeyT *d_keys_in,
            KeyT *d_keys_out,
            const ValueT *d_values_in,
            ValueT *d_values_out,
            int num_items,
            int num_segments,
            BeginOffsetIteratorT d_begin_offsets,
            EndOffsetIteratorT d_end_offsets,
            int begin_bit       = 0,
            int end_bit         = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in),
                                  d_values_out);

    return DispatchSegmentedRadixSort<false,
                                      KeyT,
                                      ValueT,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT>::Dispatch(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys,
                                                         d_values,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         begin_bit,
                                                         end_bit,
                                                         false,
                                                         stream);
  }

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            size_t &temp_storage_bytes,
            const KeyT *d_keys_in,
            KeyT *d_keys_out,
            const ValueT *d_values_in,
            ValueT *d_values_out,
            int num_items,
            int num_segments,
            BeginOffsetIteratorT d_begin_offsets,
            EndOffsetIteratorT d_end_offsets,
            int begin_bit,
            int end_bit,
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
      begin_bit,
      end_bit,
      stream);
  }

  /**
   * @brief Sorts segments of key-value pairs into ascending order. 
   *        (`~N` auxiliary storage required)
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
   * - When input a contiguous sequence of segments, a single sequence
   *   `segment_offsets` (of length `num_segments + 1`) can be aliased for both 
   *   the `d_begin_offsets` and `d_end_offsets` parameters (where the latter is 
   *   specified as `segment_offsets + 1`).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and yield 
   *   a corresponding performance improvement.
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
   * - @devicestorageP
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments 
   * (with one zero-length segment) of `int` keys with associated vector of 
   * `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_segmented_radix_sort.cuh>
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
   * cub::DeviceSegmentedRadixSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedRadixSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyT             
   *   **[inferred]** Key type
   *
   * @tparam ValueT           
   *   **[inferred]** Value type
   *
   * @tparam BeginOffsetIteratorT  
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT    
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_keys 
   *   Reference to the double-buffer of keys whose "current" device-accessible 
   *   buffer contains the unsorted input keys and, upon return, is updated to 
   *   point to the sorted output keys
   *
   * @param[in,out] d_values 
   *   Double-buffer of values whose "current" device-accessible buffer 
   *   contains the unsorted input values and, upon return, is updated to point 
   *   to the sorted output values
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
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets 
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`. 
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is 
   *   considered empty.
   *
   * @param[in] begin_bit 
   *   **[optional]** The least-significant bit index (inclusive) needed for 
   *   key comparison
   *
   * @param[in] end_bit 
   *   **[optional]** The most-significant bit index (exclusive) needed for key 
   *   comparison (e.g., `sizeof(unsigned int) * 8`)
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            int num_items,
            int num_segments,
            BeginOffsetIteratorT d_begin_offsets,
            EndOffsetIteratorT d_end_offsets,
            int begin_bit       = 0,
            int end_bit         = sizeof(KeyT) * 8,
            cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    return DispatchSegmentedRadixSort<false,
                                      KeyT,
                                      ValueT,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT>::Dispatch(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys,
                                                         d_values,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         begin_bit,
                                                         end_bit,
                                                         true,
                                                         stream);
  }

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            int num_items,
            int num_segments,
            BeginOffsetIteratorT d_begin_offsets,
            EndOffsetIteratorT d_end_offsets,
            int begin_bit,
            int end_bit,
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
      begin_bit,
      end_bit,
      stream);
  }

  /**
   * @brief Sorts segments of key-value pairs into descending order. 
   *        (`~2N` auxiliary storage required).
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   `segment_offsets` (of length `num_segments + 1`) can be aliased for both 
   *   the `d_begin_offsets` and `d_end_offsets` parameters (where the latter is 
   *   specified as `segment_offsets + 1`).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - Let `in` be one of `{d_keys_in, d_values_in}` and `out` be any of
   *   `{d_keys_out, d_values_out}`. The range `[out, out + num_items)` shall 
   *   not overlap `[in, in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - @devicestorageNP For sorting using only `O(P)` temporary storage, see 
   *   the sorting interface using DoubleBuffer wrappers below.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_values_in[i]`, 
   *   `d_keys_out[i]`, `d_values_out[i]` will not be accessed nor modified.   
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments 
   * (with one zero-length segment) of `int` keys with associated vector of 
   * `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_segmented_radix_sort.cuh>
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
   * cub::DeviceSegmentedRadixSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedRadixSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
   * // d_values_out          <-- [0, 2, 1, 6, 3, 4, 5]
   * @endcode
   *
   * @tparam KeyT             
   *   **[inferred]** Key type
   *
   * @tparam ValueT           
   *   **[inferred]** Value type
   *
   * @tparam BeginOffsetIteratorT  
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT    
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
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
   *   length `num_segments`, such that `d_begin_offsets[i]` is the first 
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets 
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`. 
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> 
   *   is considered empty.
   *
   * @param[in] begin_bit 
   *   **[optional]** The least-significant bit index (inclusive) needed for 
   *   key comparison
   *
   * @param[in] end_bit 
   *   **[optional]** The most-significant bit index (exclusive) needed for key 
   *   comparison (e.g., `sizeof(unsigned int) * 8`)
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      const KeyT *d_keys_in,
                      KeyT *d_keys_out,
                      const ValueT *d_values_in,
                      ValueT *d_values_out,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      int begin_bit          = 0,
                      int end_bit            = sizeof(KeyT) * 8,
                      cudaStream_t stream    = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in),
                                  d_values_out);

    return DispatchSegmentedRadixSort<true,
                                      KeyT,
                                      ValueT,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT>::Dispatch(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys,
                                                         d_values,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         begin_bit,
                                                         end_bit,
                                                         false,
                                                         stream);
  }

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      const KeyT *d_keys_in,
                      KeyT *d_keys_out,
                      const ValueT *d_values_in,
                      ValueT *d_values_out,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      int begin_bit,
                      int end_bit,
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
                                                   begin_bit,
                                                   end_bit,
                                                   stream);
  }

  /**
   * @brief Sorts segments of key-value pairs into descending order. 
   *        (`~N` auxiliary storage required).
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
   * - When input a contiguous sequence of segments, a single sequence
   *   `segment_offsets` (of length `num_segments + 1`) can be aliased for both 
   *   the `d_begin_offsets` and `d_end_offsets` parameters (where the latter is 
   *   specified as `segment_offsets + 1`).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
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
   *   not to be modified. 
   * - @devicestorageP
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments 
   * (with one zero-length segment) of `int` keys with associated vector of 
   * `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_segmented_radix_sort.cuh>
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
   * cub::DeviceSegmentedRadixSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedRadixSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
   * // d_values.Current()    <-- [0, 2, 1, 6, 3, 4, 5]
   * @endcode
   *
   * @tparam KeyT             
   *   **[inferred]** Key type
   *
   * @tparam ValueT           
   *   **[inferred]** Value type
   *
   * @tparam BeginOffsetIteratorT  
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT    
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_keys 
   *   Reference to the double-buffer of keys whose "current" device-accessible 
   *   buffer contains the unsorted input keys and, upon return, is updated to 
   *   point to the sorted output keys
   *
   * @param[in,out] d_values 
   *   Double-buffer of values whose "current" device-accessible buffer 
   *   contains the unsorted input values and, upon return, is updated to point 
   *   to the sorted output values
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
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets 
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.  
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> 
   *   is considered empty.
   *
   * @param[in] begin_bit 
   *   **[optional]** The least-significant bit index (inclusive) needed for 
   *   key comparison
   *
   * @param[in] end_bit 
   *   **[optional]** The most-significant bit index (exclusive) needed for key 
   *   comparison (e.g., `sizeof(unsigned int) * 8`)
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within. 
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      DoubleBuffer<KeyT> &d_keys,
                      DoubleBuffer<ValueT> &d_values,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      int begin_bit       = 0,
                      int end_bit         = sizeof(KeyT) * 8,
                      cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    return DispatchSegmentedRadixSort<true,
                                      KeyT,
                                      ValueT,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT>::Dispatch(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys,
                                                         d_values,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         begin_bit,
                                                         end_bit,
                                                         true,
                                                         stream);
  }

  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      DoubleBuffer<KeyT> &d_keys,
                      DoubleBuffer<ValueT> &d_values,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      int begin_bit,
                      int end_bit,
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
                                                   begin_bit,
                                                   end_bit,
                                                   stream);
  }

  //@}  end member group
  /******************************************************************//**
   * @name Keys-only
   *********************************************************************/
  //@{


  /**
   * @brief Sorts segments of keys into ascending order. 
   *        (`~2N` auxiliary storage required)
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - When input a contiguous sequence of segments, a single sequence
   *   `segment_offsets` (of length `num_segments + 1`) can be aliased for both 
   *   the `d_begin_offsets` and `d_end_offsets` parameters (where the latter 
   *   is specified as `segment_offsets + 1`).
   * - The range `[d_keys_out, d_keys_out + num_items)` shall not overlap
   *   `[d_keys_in, d_keys_in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - @devicestorageNP For sorting using only `O(P)` temporary storage, see 
   *   the sorting interface using DoubleBuffer wrappers below.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_keys_out[i]` will not 
   *   be accessed nor modified.   
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments 
   * (with one zero-length segment) of `int` keys.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_segmented_radix_sort.cuh>
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
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedRadixSort::SortKeys( 
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedRadixSort::SortKeys( 
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   *
   * @endcode
   *
   * @tparam KeyT             
   *   **[inferred]** Key type
   *
   * @tparam BeginOffsetIteratorT  
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT    
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of \p d_temp_storage allocation
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
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets  
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.  
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is 
   *   considered empty.
   *
   * @param[in] begin_bit  
   *   **[optional]** The least-significant bit index (inclusive) needed for 
   *   key comparison
   *
   * @param[in] end_bit  
   *   **[optional]** The most-significant bit index (exclusive) needed for key 
   *   comparison (e.g., `sizeof(unsigned int) * 8`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           int begin_bit       = 0,
           int end_bit         = sizeof(KeyT) * 8,
           cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    // Null value type
    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchSegmentedRadixSort<false,
                                      KeyT,
                                      NullType,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT>::Dispatch(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys,
                                                         d_values,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         begin_bit,
                                                         end_bit,
                                                         false,
                                                         stream);
  }

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           int begin_bit,
           int end_bit,
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
      begin_bit,
      end_bit,
      stream);
  }

  /**
   * @brief Sorts segments of keys into ascending order. (~<em>N </em>auxiliary storage required).
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a 
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current" 
   *   indicator within the DoubleBuffer wrapper to reference which of the two 
   *   buffers now contains the sorted output sequence (a function of the 
   *   number of key bits specified and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   `segment_offsets` (of length `num_segments + 1`) can be aliased for both 
   *   the `d_begin_offsets` and `d_end_offsets` parameters (where the latter 
   *   is specified as `segment_offsets + 1`).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - Let `cur = d_keys.Current()` and `alt = d_keys.Alternate()`.
   *   The range `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_keys[i].Alternate()[i]` will not be accessed nor modified.   
   * - @devicestorageP
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments 
   * (with one zero-length segment) of `int` keys.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_segmented_radix_sort.cuh>
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
   * cub::DeviceSegmentedRadixSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedRadixSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   *
   * @endcode
   *
   * @tparam KeyT             
   *   **[inferred]** Key type
   *
   * @tparam BeginOffsetIteratorT  
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT    
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
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
   *   length `num_segments`, such that `d_begin_offsets[i]` is the first 
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets  
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`. 
   *   If `d_end_offsets[i] - 1` <= d_begin_offsets[i]`, the *i*<sup>th</sup>
   *   is considered empty.
   *
   * @param[in] begin_bit  
   *   **[optional]** The least-significant bit index (inclusive)  
   *   needed for key comparison
   *
   * @param[in] end_bit  
   *   **[optional]** The most-significant bit index (exclusive) needed for key 
   *   comparison (e.g., `sizeof(unsigned int) * 8`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           int begin_bit       = 0,
           int end_bit         = sizeof(KeyT) * 8,
           cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    // Null value type
    DoubleBuffer<NullType> d_values;

    return DispatchSegmentedRadixSort<false,
                                      KeyT,
                                      NullType,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT>::Dispatch(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys,
                                                         d_values,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         begin_bit,
                                                         end_bit,
                                                         true,
                                                         stream);
  }

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           int begin_bit,
           int end_bit,
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
      begin_bit,
      end_bit,
      stream);
  }

  /**
   * @brief Sorts segments of keys into descending order. 
   * (`~2N` auxiliary storage required).
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   `segment_offsets` (of length `num_segments + 1`) can be aliased for both 
   *   the `d_begin_offsets` and `d_end_offsets` parameters (where the latter 
   *   is specified as `segment_offsets + 1`).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - The range `[d_keys_out, d_keys_out + num_items)` shall not overlap
   *   `[d_keys_in, d_keys_in + num_items)`, 
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - @devicestorageNP For sorting using only `O(P)` temporary storage, see 
   *   the sorting interface using DoubleBuffer wrappers below.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys_in[i]`, `d_keys_out[i]` will not 
   *   be accessed nor modified.   
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments 
   * (with one zero-length segment) of `int` keys.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_segmented_radix_sort.cuh>
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
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedRadixSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedRadixSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
   *
   * @endcode
   *
   * @tparam KeyT             
   *   **[inferred]** Key type
   *
   * @tparam BeginOffsetIteratorT  
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT    
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
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
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets  
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`. 
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is 
   *   considered empty.
   *
   * @param[in] begin_bit  
   *   **[optional]** The least-significant bit index (inclusive) needed for 
   *   key comparison
   *
   * @param[in] end_bit  
   *   **[optional]** The most-significant bit index (exclusive) needed for key 
   *   comparison (e.g., sizeof(unsigned int) * 8)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     const KeyT *d_keys_in,
                     KeyT *d_keys_out,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     int begin_bit       = 0,
                     int end_bit         = sizeof(KeyT) * 8,
                     cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchSegmentedRadixSort<true,
                                      KeyT,
                                      NullType,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT>::Dispatch(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys,
                                                         d_values,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         begin_bit,
                                                         end_bit,
                                                         false,
                                                         stream);
  }

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     const KeyT *d_keys_in,
                     KeyT *d_keys_out,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     int begin_bit,
                     int end_bit,
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
      begin_bit,
      end_bit,
      stream);
  }

  /**
   * @brief Sorts segments of keys into descending order. 
   * (`~N` auxiliary storage required).
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current" 
   *   indicator within the DoubleBuffer wrapper to reference which of the two 
   *   buffers now contains the sorted output sequence (a function of the 
   *   number of key bits specified and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   `segment_offsets` (of length `num_segments + 1`) can be aliased
   *   for both the `d_begin_offsets` and `d_end_offsets` parameters (where
   *   the latter is specified as `segment_offsets + 1`).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - Let `cur = d_keys.Current()` and `alt = d_keys.Alternate()`.
   *   The range `[cur, cur + num_items)` shall not overlap 
   *   `[alt, alt + num_items)`. Both ranges shall not overlap
   *   `[d_begin_offsets, d_begin_offsets + num_segments)` nor
   *   `[d_end_offsets, d_end_offsets + num_segments)` in any way.
   * - Segments are not required to be contiguous. For all index values `i` 
   *   outside the specified segments `d_keys.Current()[i]`, 
   *   `d_keys[i].Alternate()[i]` will not be accessed nor modified.   
   * - @devicestorageP
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments 
   * (with one zero-length segment) of `int` keys.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_segmented_radix_sort.cuh>
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
   * cub::DeviceSegmentedRadixSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedRadixSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
   * @endcode
   *
   * @tparam KeyT             
   *   **[inferred]** Key type
   *
   * @tparam BeginOffsetIteratorT  
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   beginning offsets \iterator
   *
   * @tparam EndOffsetIteratorT    
   *   **[inferred]** Random-access input iterator type for reading segment 
   *   ending offsets \iterator
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
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
   *   length `num_segments`, such that `d_begin_offsets[i]` is the first 
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and 
   *   `d_values_*`
   *
   * @param[in] d_end_offsets  
   *   Random-access input iterator to the sequence of ending offsets of length 
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of 
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.  
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i], the *i*<sup>th</sup> is 
   *   considered empty.
   *
   * @param[in] begin_bit  
   *   **[optional]** The least-significant bit index (inclusive) needed for 
   *   key comparison
   *
   * @param[in] end_bit  
   *   **[optional]** The most-significant bit index (exclusive) needed for key 
   *   comparison (e.g., `sizeof(unsigned int) * 8`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     DoubleBuffer<KeyT> &d_keys,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     int begin_bit       = 0,
                     int end_bit         = sizeof(KeyT) * 8,
                     cudaStream_t stream = 0)
  {
    // Signed integer type for global offsets
    using OffsetT = int;

    // Null value type
    DoubleBuffer<NullType> d_values;

    return DispatchSegmentedRadixSort<true,
                                      KeyT,
                                      NullType,
                                      BeginOffsetIteratorT,
                                      EndOffsetIteratorT,
                                      OffsetT>::Dispatch(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_keys,
                                                         d_values,
                                                         num_items,
                                                         num_segments,
                                                         d_begin_offsets,
                                                         d_end_offsets,
                                                         begin_bit,
                                                         end_bit,
                                                         true,
                                                         stream);
  }

  template <typename KeyT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     DoubleBuffer<KeyT> &d_keys,
                     int num_items,
                     int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     int begin_bit,
                     int end_bit,
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
      begin_bit,
      end_bit,
      stream);
  }

  //@}  end member group
};

CUB_NAMESPACE_END
