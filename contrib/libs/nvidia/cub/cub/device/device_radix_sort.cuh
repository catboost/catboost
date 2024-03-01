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
 * @file cub::DeviceRadixSort provides device-wide, parallel operations for 
 *       computing a radix sort across a sequence of data items residing within 
 *       device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <cub/config.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_radix_sort.cuh>
#include <cub/util_deprecated.cuh>

CUB_NAMESPACE_BEGIN

/**
 * @brief DeviceRadixSort provides device-wide, parallel operations for 
 *        computing a radix sort across a sequence of data items residing 
 *        within device-accessible memory. ![](sorting_logo.png)
 * @ingroup SingleModule
 *
 * @par Overview
 * The [*radix sorting method*](http://en.wikipedia.org/wiki/Radix_sort) 
 * arranges items into ascending (or descending) order. The algorithm relies 
 * upon a positional representation for keys, i.e., each key is comprised of an 
 * ordered sequence of symbols (e.g., digits, characters, etc.) specified from 
 * least-significant to most-significant. For a given input sequence of keys 
 * and a set of rules specifying a total ordering of the symbolic alphabet, the 
 * radix sorting method produces a lexicographic ordering of those keys.
 *
 * @par Supported Types
 * DeviceRadixSort can sort all of the built-in C++ numeric primitive types
 * (`unsigned char`, `int`, `double`, etc.) as well as CUDA's `__half`
 * and `__nv_bfloat16` 16-bit floating-point types.
 *
 * @par Floating-Point Special Cases
 *
 * - Positive and negative zeros are considered equivalent, and will be treated
 *   as such in the output.
 * - No special handling is implemented for NaN values; these are sorted
 *   according to their bit representations after any transformations.
 *
 * @par Transformations
 * Although the direct radix sorting method can only be applied to unsigned
 * integral types, DeviceRadixSort is able to sort signed and floating-point
 * types via simple bit-wise transformations that ensure lexicographic key
 * ordering. Additional transformations occur for descending sorts. These
 * transformations must be considered when restricting the
 * `[begin_bit, end_bit)` range, as the bitwise transformations will occur
 * before the bit-range truncation.
 *
 * Any transformations applied to the keys prior to sorting are reversed
 * while writing to the final output buffer.
 *
 * \par Type Specific Bitwise Transformations
 * To convert the input values into a radix-sortable bitwise representation,
 * the following transformations take place prior to sorting:
 *
 * - For unsigned integral values, the keys are used directly.
 * - For signed integral values, the sign bit is inverted.
 * - For positive floating point values, the sign bit is inverted.
 * - For negative floating point values, the full key is inverted.
 *
 * For floating point types, positive and negative zero are a special case and
 * will be considered equivalent during sorting.
 *
 * @par Descending Sort Bitwise Transformations
 * If descending sort is used, the keys are inverted after performing any
 * type-specific transformations, and the resulting keys are sorted in ascending
 * order.
 *
 * @par Stability
 * DeviceRadixSort is stable. For floating-point types, `-0.0` and `+0.0` are
 * considered equal and appear in the result in the same order as they appear in
 * the input.
 *
 * @par Usage Considerations
 * @cdp_class{DeviceRadixSort}
 *
 * @par Performance
 * @linear_performance{radix sort} The following chart illustrates 
 * DeviceRadixSort::SortKeys performance across different CUDA architectures 
 * for uniform-random `uint32` keys.
 * @plots_below
 *
 * @image html lsb_radix_sort_int32_keys.png
 *
 */
struct DeviceRadixSort
{

  /******************************************************************//**
   * \name KeyT-value pairs
   *********************************************************************/
  //@{

  /**
   * @brief Sorts key-value pairs into ascending order. 
   *        (`~2N` auxiliary storage required)
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - Pointers to contiguous memory must be used; iterators are not currently
   *   supported.
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_keys_in,    d_keys_in    + num_items)`
   *   - `[d_keys_out,   d_keys_out   + num_items)`
   *   - `[d_values_in,  d_values_in  + num_items)`
   *   - `[d_values_out, d_values_out + num_items)`
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - @devicestorageNP For sorting using only `O(P)` temporary storage, see 
   *   the sorting interface using DoubleBuffer wrappers below.
   * - @devicestorage
   *
   * @par Performance
   * The following charts illustrate saturated sorting performance across 
   * different CUDA architectures for uniform-random `uint32, uint32` and
   * `uint64, uint64` pairs, respectively.
   *
   * @image html lsb_radix_sort_int32_pairs.png
   * @image html lsb_radix_sort_int64_pairs.png
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of `int`
   * keys with associated vector of `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [        ...        ]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [        ...        ]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
   *
   * // d_keys_out            <-- [0, 3, 5, 6, 7, 8, 9]
   * // d_values_out          <-- [5, 4, 3, 1, 2, 0, 6]
   * @endcode
   *
   * @tparam KeyT      
   *   **[inferred]** KeyT type
   *
   * @tparam ValueT    
   *   **[inferred]** ValueT type
   *
   * @tparam NumItemsT 
   *   **[inferred]** Type of num_items
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
   *   Pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out 
   *   Pointer to the sorted output sequence of key data
   *
   * @param[in] d_values_in 
   *   Pointer to the corresponding input sequence of associated value items
   *
   * @param[out] d_values_out 
   *   Pointer to the correspondingly-reordered output sequence of associated 
   *   value items
   *
   * @param[in] num_items 
   *   Number of items to sort
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
  template <typename KeyT, typename ValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            size_t &temp_storage_bytes,
            const KeyT *d_keys_in,
            KeyT *d_keys_out,
            const ValueT *d_values_in,
            ValueT *d_values_out,
            NumItemsT num_items,
            int begin_bit          = 0,
            int end_bit            = sizeof(KeyT) * 8,
            cudaStream_t stream    = 0)
  {
    // Unsigned integer type for global offsets.
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    // We cast away const-ness, but will *not* write to these arrays.
    // `DispatchRadixSort::Dispatch` will allocate temporary storage and
    // create a new double-buffer internally when the `is_overwrite_ok` flag
    // is not set.
    constexpr bool is_overwrite_okay = false;
    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in),
                                  d_values_out);

    return DispatchRadixSort<false, KeyT, ValueT, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      static_cast<OffsetT>(num_items),
      begin_bit,
      end_bit,
      is_overwrite_okay,
      stream);
  }

  template <typename KeyT, typename ValueT, typename NumItemsT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            size_t &temp_storage_bytes,
            const KeyT *d_keys_in,
            KeyT *d_keys_out,
            const ValueT *d_values_in,
            ValueT *d_values_out,
            NumItemsT num_items,
            int begin_bit,
            int end_bit,
            cudaStream_t stream,
            bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairs<KeyT, ValueT, NumItemsT>(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys_in,
                                              d_keys_out,
                                              d_values_in,
                                              d_values_out,
                                              num_items,
                                              begin_bit,
                                              end_bit,
                                              stream);
  }

  /**
   * @brief Sorts key-value pairs into ascending order. 
   *        (`~N` auxiliary storage required)
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the 
   *   sorting operation.
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_keys.Current(),     d_keys.Current()     + num_items)`
   *   - `[d_keys.Alternate(),   d_keys.Alternate()   + num_items)`
   *   - `[d_values.Current(),   d_values.Current()   + num_items)`
   *   - `[d_values.Alternate(), d_values.Alternate() + num_items)`
   * - Upon completion, the sorting operation will update the "current" 
   *   indicator within each DoubleBuffer wrapper to reference which of the two 
   *   buffers now contains the sorted output sequence (a function of the 
   *   number of key bits specified and the targeted device architecture).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - @devicestorageP
   * - @devicestorage
   *
   * @par Performance
   * The following charts illustrate saturated sorting performance across 
   * different CUDA architectures for uniform-random `uint32, uint32` and
   * `uint64, uint64` pairs, respectively.
   *
   * @image html lsb_radix_sort_int32_pairs.png
   * @image html lsb_radix_sort_int64_pairs.png
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of `int` 
   * keys with associated vector of `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for 
   * // sorting data
   * int  num_items;          // e.g., 7
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [        ...        ]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [        ...        ]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRadixSort::SortPairs(
   *   d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceRadixSort::SortPairs(
   *   d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
   *
   * // d_keys.Current()      <-- [0, 3, 5, 6, 7, 8, 9]
   * // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyT      
   *   **[inferred]** KeyT type
   *
   * @tparam ValueT    
   *   **[inferred]** ValueT type
   *
   * @tparam NumItemsT 
   *   **[inferred]** Type of num_items
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to \p temp_storage_bytes and no work is done.
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
   *   Number of items to sort
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
  template <typename KeyT, typename ValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            NumItemsT num_items,
            int begin_bit          = 0,
            int end_bit            = sizeof(KeyT) * 8,
            cudaStream_t stream    = 0)
  {
    // Unsigned integer type for global offsets.
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    constexpr bool is_overwrite_okay = true;

    return DispatchRadixSort<false, KeyT, ValueT, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      begin_bit,
      end_bit,
      is_overwrite_okay,
      stream);
  }

  template <typename KeyT, typename ValueT, typename NumItemsT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            NumItemsT num_items,
            int begin_bit,
            int end_bit,
            cudaStream_t stream,
            bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairs<KeyT, ValueT, NumItemsT>(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys,
                                              d_values,
                                              num_items,
                                              begin_bit,
                                              end_bit,
                                              stream);
  }

  /**
   * @brief Sorts key-value pairs into descending order. 
   *        (`~2N` auxiliary storage required).
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - Pointers to contiguous memory must be used; iterators are not currently
   *   supported.
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_keys_in,    d_keys_in    + num_items)`
   *   - `[d_keys_out,   d_keys_out   + num_items)`
   *   - `[d_values_in,  d_values_in  + num_items)`
   *   - `[d_values_out, d_values_out + num_items)`
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - @devicestorageNP  For sorting using only `O(P)` temporary storage, see 
   *   the sorting interface using DoubleBuffer wrappers below.
   * - @devicestorage
   *
   * @par Performance
   * Performance is similar to DeviceRadixSort::SortPairs.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of `int` 
   * keys with associated vector of `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [        ...        ]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [        ...        ]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRadixSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceRadixSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
   *
   * // d_keys_out            <-- [9, 8, 7, 6, 5, 3, 0]
   * // d_values_out          <-- [6, 0, 2, 1, 3, 4, 5]
   * @endcode
   *
   * @tparam KeyT      
   *   **[inferred]** KeyT type
   *
   * @tparam ValueT    
   *   **[inferred]** ValueT type
   *
   * @tparam NumItemsT 
   *   **[inferred]** Type of num_items
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of \p d_temp_storage allocation
   *
   * @param[in] d_keys_in 
   *   Pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out 
   *   Pointer to the sorted output sequence of key data
   *
   * @param[in] d_values_in 
   *   Pointer to the corresponding input sequence of associated value items
   *
   * @param[out] d_values_out 
   *   Pointer to the correspondingly-reordered output sequence of associated 
   *   value items
   *
   * @param[in] num_items 
   *   Number of items to sort
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
  template <typename KeyT, typename ValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      const KeyT *d_keys_in,
                      KeyT *d_keys_out,
                      const ValueT *d_values_in,
                      ValueT *d_values_out,
                      NumItemsT num_items,
                      int begin_bit       = 0,
                      int end_bit         = sizeof(KeyT) * 8,
                      cudaStream_t stream = 0)
  {
    // Unsigned integer type for global offsets.
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    // We cast away const-ness, but will *not* write to these arrays.
    // `DispatchRadixSort::Dispatch` will allocate temporary storage and
    // create a new double-buffer internally when the `is_overwrite_ok` flag
    // is not set.
    constexpr bool is_overwrite_okay = false;
    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in),
                                  d_values_out);

    return DispatchRadixSort<true, KeyT, ValueT, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      begin_bit,
      end_bit,
      is_overwrite_okay,
      stream);
  }

  template <typename KeyT, typename ValueT, typename NumItemsT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      const KeyT *d_keys_in,
                      KeyT *d_keys_out,
                      const ValueT *d_values_in,
                      ValueT *d_values_out,
                      NumItemsT num_items,
                      int begin_bit,
                      int end_bit,
                      cudaStream_t stream,
                      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairsDescending<KeyT, ValueT, NumItemsT>(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys_in,
                                                        d_keys_out,
                                                        d_values_in,
                                                        d_values_out,
                                                        num_items,
                                                        begin_bit,
                                                        end_bit,
                                                        stream);
  }

  /**
   * @brief Sorts key-value pairs into descending order. 
   *        (`~N` auxiliary storage required).
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the 
   *   sorting operation.
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_keys.Current(),     d_keys.Current()     + num_items)`
   *   - `[d_keys.Alternate(),   d_keys.Alternate()   + num_items)`
   *   - `[d_values.Current(),   d_values.Current()   + num_items)`
   *   - `[d_values.Alternate(), d_values.Alternate() + num_items)`
   * - Upon completion, the sorting operation will update the "current" 
   *   indicator within each DoubleBuffer wrapper to reference which of the two 
   *   buffers now contains the sorted output sequence (a function of the number 
   *   of key bits specified and the targeted device architecture).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - @devicestorageP
   * - @devicestorage
   *
   * @par Performance
   * Performance is similar to DeviceRadixSort::SortPairs.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of `int` 
   * keys with associated vector of `int` values.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [        ...        ]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [        ...        ]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRadixSort::SortPairsDescending(
   *   d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceRadixSort::SortPairsDescending(
   *   d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
   *
   * // d_keys.Current()      <-- [9, 8, 7, 6, 5, 3, 0]
   * // d_values.Current()    <-- [6, 0, 2, 1, 3, 4, 5]
   * @endcode
   *
   * @tparam KeyT      
   *   **[inferred]** KeyT type
   *
   * @tparam ValueT    
   *   **[inferred]** ValueT type
   *
   * @tparam NumItemsT 
   *   **[inferred]** Type of num_items
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
   *   Number of items to sort
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
  template <typename KeyT, typename ValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      DoubleBuffer<KeyT> &d_keys,
                      DoubleBuffer<ValueT> &d_values,
                      NumItemsT num_items,
                      int begin_bit          = 0,
                      int end_bit            = sizeof(KeyT) * 8,
                      cudaStream_t stream    = 0)
  {
    // Unsigned integer type for global offsets.
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    constexpr bool is_overwrite_okay = true;

    return DispatchRadixSort<true, KeyT, ValueT, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      begin_bit,
      end_bit,
      is_overwrite_okay,
      stream);
  }

  template <typename KeyT, typename ValueT, typename NumItemsT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      DoubleBuffer<KeyT> &d_keys,
                      DoubleBuffer<ValueT> &d_values,
                      NumItemsT num_items,
                      int begin_bit,
                      int end_bit,
                      cudaStream_t stream,
                      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortPairsDescending<KeyT, ValueT, NumItemsT>(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys,
                                                        d_values,
                                                        num_items,
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
   * @brief Sorts keys into ascending order. 
   *        (`~2N` auxiliary storage required)
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - Pointers to contiguous memory must be used; iterators are not currently
   *   supported.
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_keys_in,    d_keys_in    + num_items)`
   *   - `[d_keys_out,   d_keys_out   + num_items)`
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - @devicestorageNP  For sorting using only `O(P)` temporary storage, see 
   *   the sorting interface using DoubleBuffer wrappers below.
   * - @devicestorage
   *
   * @par Performance
   * The following charts illustrate saturated sorting performance across 
   * different CUDA architectures for uniform-random `uint32` and `uint64` 
   * keys, respectively.
   *
   * @image html lsb_radix_sort_int32_keys.png
   * @image html lsb_radix_sort_int64_keys.png
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of 
   * `int` keys.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [        ...        ]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRadixSort::SortKeys(
   *   d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceRadixSort::SortKeys(
   *   d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
   *
   * // d_keys_out            <-- [0, 3, 5, 6, 7, 8, 9]
   * @endcode
   *
   * @tparam KeyT      
   *   **[inferred]** KeyT type
   *
   * @tparam NumItemsT 
   *   **[inferred]** Type of num_items
   *
   * @tparam NumItemsT 
   *   **[inferred]** Type of num_items
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
   *   Pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out 
   *   Pointer to the sorted output sequence of key data
   *
   * @param[in] num_items 
   *   Number of items to sort
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
  template <typename KeyT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           NumItemsT num_items,
           int begin_bit       = 0,
           int end_bit         = sizeof(KeyT) * 8,
           cudaStream_t stream = 0)
  {
    // Unsigned integer type for global offsets.
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    // We cast away const-ness, but will *not* write to these arrays.
    // `DispatchRadixSort::Dispatch` will allocate temporary storage and
    // create a new double-buffer internally when the `is_overwrite_ok` flag
    // is not set.
    constexpr bool is_overwrite_okay = false;
    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    // Null value type
    DoubleBuffer<NullType> d_values;

    return DispatchRadixSort<false, KeyT, NullType, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      static_cast<OffsetT>(num_items),
      begin_bit,
      end_bit,
      is_overwrite_okay,
      stream);
  }

  template <typename KeyT, typename NumItemsT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           NumItemsT num_items,
           int begin_bit,
           int end_bit,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeys<KeyT, NumItemsT>(d_temp_storage,
                                     temp_storage_bytes,
                                     d_keys_in,
                                     d_keys_out,
                                     num_items,
                                     begin_bit,
                                     end_bit,
                                     stream);
  }

  /**
   * @brief Sorts keys into ascending order. (`~N` auxiliary storage required).
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_keys.Current(),     d_keys.Current()     + num_items)`
   *   - `[d_keys.Alternate(),   d_keys.Alternate()   + num_items)`
   * - Upon completion, the sorting operation will update the "current" 
   *   indicator within the DoubleBuffer wrapper to reference which of the two 
   *   buffers now contains the sorted output sequence (a function of the 
   *   number of key bits specified and the targeted device architecture).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - @devicestorageP
   * - @devicestorage
   *
   * @par Performance
   * The following charts illustrate saturated sorting performance across 
   * different CUDA architectures for uniform-random `uint32` and `uint64` 
   * keys, respectively.
   *
   * @image html lsb_radix_sort_int32_keys.png
   * @image html lsb_radix_sort_int64_keys.png
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of 
   * `int` keys.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [        ...        ]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRadixSort::SortKeys(
   *   d_temp_storage, temp_storage_bytes, d_keys, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceRadixSort::SortKeys(
   *   d_temp_storage, temp_storage_bytes, d_keys, num_items);
   *
   * // d_keys.Current()      <-- [0, 3, 5, 6, 7, 8, 9]
   * @endcode
   *
   * @tparam KeyT      
   *   **[inferred]** KeyT type
   *
   * @tparam NumItemsT 
   *   **[inferred]** Type of num_items
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
   *   Number of items to sort
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
  template <typename KeyT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           NumItemsT num_items,
           int begin_bit       = 0,
           int end_bit         = sizeof(KeyT) * 8,
           cudaStream_t stream = 0)
  {
    // Unsigned integer type for global offsets.
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    constexpr bool is_overwrite_okay = true;

    // Null value type
    DoubleBuffer<NullType> d_values;

    return DispatchRadixSort<false, KeyT, NullType, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      begin_bit,
      end_bit,
      is_overwrite_okay,
      stream);
  }

  template <typename KeyT, typename NumItemsT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           NumItemsT num_items,
           int begin_bit,
           int end_bit,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeys<KeyT, NumItemsT>(d_temp_storage,
                                     temp_storage_bytes,
                                     d_keys,
                                     num_items,
                                     begin_bit,
                                     end_bit,
                                     stream);
  }

  /**
   * @brief Sorts keys into descending order. 
   *        (`~2N` auxiliary storage required).
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation.
   * - Pointers to contiguous memory must be used; iterators are not currently
   *   supported.
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_keys_in,    d_keys_in    + num_items)`
   *   - `[d_keys_out,   d_keys_out   + num_items)`
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - @devicestorageNP For sorting using only `O(P)` temporary storage, see 
   *   the sorting interface using DoubleBuffer wrappers below.
   * - @devicestorage
   *
   * @par Performance
   * Performance is similar to DeviceRadixSort::SortKeys.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of 
   * `int` keys.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [        ...        ]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRadixSort::SortKeysDescending( 
   *   d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceRadixSort::SortKeysDescending(
   *   d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
   *
   * // d_keys_out            <-- [9, 8, 7, 6, 5, 3, 0]s
   *
   * @endcode
   *
   * @tparam KeyT      
   *   **[inferred]** KeyT type
   *
   * @tparam NumItemsT 
   *   **[inferred]** Type of num_items
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
   *   Pointer to the input data of key data to sort
   *
   * @param[out] d_keys_out 
   *   Pointer to the sorted output sequence of key data
   *
   * @param[in] num_items 
   *   Number of items to sort
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
  template <typename KeyT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     const KeyT *d_keys_in,
                     KeyT *d_keys_out,
                     NumItemsT num_items,
                     int begin_bit       = 0,
                     int end_bit         = sizeof(KeyT) * 8,
                     cudaStream_t stream = 0)
  {
    // Unsigned integer type for global offsets.
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    // We cast away const-ness, but will *not* write to these arrays.
    // `DispatchRadixSort::Dispatch` will allocate temporary storage and
    // create a new double-buffer internally when the `is_overwrite_ok` flag
    // is not set.
    constexpr bool is_overwrite_okay = false;
    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchRadixSort<true, KeyT, NullType, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      begin_bit,
      end_bit,
      is_overwrite_okay,
      stream);
  }

  template <typename KeyT, typename NumItemsT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     const KeyT *d_keys_in,
                     KeyT *d_keys_out,
                     NumItemsT num_items,
                     int begin_bit,
                     int end_bit,
                     cudaStream_t stream,
                     bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeysDescending<KeyT, NumItemsT>(d_temp_storage,
                                               temp_storage_bytes,
                                               d_keys_in,
                                               d_keys_out,
                                               num_items,
                                               begin_bit,
                                               end_bit,
                                               stream);
  }

  /**
   * @brief Sorts keys into descending order. 
   *        (`~N` auxiliary storage required).
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_keys.Current(),     d_keys.Current()     + num_items)`
   *   - `[d_keys.Alternate(),   d_keys.Alternate()   + num_items)`
   * - Upon completion, the sorting operation will update the "current" 
   *   indicator within the DoubleBuffer wrapper to reference which of the two 
   *   buffers now contains the sorted output sequence (a function of the 
   *   number of key bits specified and the targeted device architecture).
   * - An optional bit subrange `[begin_bit, end_bit)` of differentiating key 
   *   bits can be specified. This can reduce overall sorting overhead and 
   *   yield a corresponding performance improvement.
   * - @devicestorageP
   * - @devicestorage
   *
   * @par Performance
   * Performance is similar to DeviceRadixSort::SortKeys.
   *
   * @par Snippet
   * The code snippet below illustrates the sorting of a device vector of @p int keys.
   * @par
   * @code
   * #include <cub/cub.cuh>   
   * // or equivalently <cub/device/device_radix_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for sorting data
   * int  num_items;          // e.g., 7
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [        ...        ]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceRadixSort::SortKeysDescending(
   *   d_temp_storage, temp_storage_bytes, d_keys, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceRadixSort::SortKeysDescending(
   *   d_temp_storage, temp_storage_bytes, d_keys, num_items);
   *
   * // d_keys.Current()      <-- [9, 8, 7, 6, 5, 3, 0]
   * @endcode
   *
   * @tparam KeyT      
   *   **[inferred]** KeyT type
   *
   * @tparam NumItemsT 
   *   **[inferred]** Type of num_items
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
   *   Number of items to sort
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
  template <typename KeyT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     DoubleBuffer<KeyT> &d_keys,
                     NumItemsT num_items,
                     int begin_bit       = 0,
                     int end_bit         = sizeof(KeyT) * 8,
                     cudaStream_t stream = 0)
  {
    // Unsigned integer type for global offsets.
    using OffsetT = typename detail::ChooseOffsetT<NumItemsT>::Type;

    constexpr bool is_overwrite_okay = true;

    // Null value type
    DoubleBuffer<NullType> d_values;

    return DispatchRadixSort<true, KeyT, NullType, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_values,
      num_items,
      begin_bit,
      end_bit,
      is_overwrite_okay,
      stream);
  }

  template <typename KeyT, typename NumItemsT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     DoubleBuffer<KeyT> &d_keys,
                     NumItemsT num_items,
                     int begin_bit,
                     int end_bit,
                     cudaStream_t stream,
                     bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return SortKeysDescending<KeyT, NumItemsT>(d_temp_storage,
                                               temp_storage_bytes,
                                               d_keys,
                                               num_items,
                                               begin_bit,
                                               end_bit,
                                               stream);
  }

  //@}  end member group

};

/**
 * @example example_device_radix_sort.cu
 */

CUB_NAMESPACE_END
