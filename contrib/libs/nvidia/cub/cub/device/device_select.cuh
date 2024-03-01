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
 * @file cub::DeviceSelect provides device-wide, parallel operations for 
 *       compacting selected items from sequences of data items residing within 
 *       device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <iterator>
#include <stdio.h>

#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_select_if.cuh>
#include <cub/device/dispatch/dispatch_unique_by_key.cuh>
#include <cub/util_deprecated.cuh>

CUB_NAMESPACE_BEGIN


/**
 * @brief DeviceSelect provides device-wide, parallel operations for compacting 
 *        selected items from sequences of data items residing within 
 *        device-accessible memory. ![](select_logo.png)
 * @ingroup SingleModule
 *
 * @par Overview
 * These operations apply a selection criterion to selectively copy
 * items from a specified input sequence to a compact output sequence.
 *
 * @par Usage Considerations
 * @cdp_class{DeviceSelect}
 *
 * @par Performance
 * @linear_performance{select-flagged, select-if, and select-unique}
 *
 * @par
 * The following chart illustrates DeviceSelect::If performance across 
 * different CUDA architectures for `int32` items, where 50% of the items are 
 * randomly selected.
 *
 * @image html select_if_int32_50_percent.png
 *
 * @par
 * The following chart illustrates DeviceSelect::Unique performance across 
 * different CUDA architectures for `int32` items where segments have lengths 
 * uniformly sampled from `[1, 1000]`.
 *
 * @image html select_unique_int32_len_500.png
 *
 * @par
 * @plots_below
 *
 */
struct DeviceSelect
{
  /**
   * @brief Uses the `d_flags` sequence to selectively copy the corresponding 
   *        items from `d_in` into `d_out`. The total number of items selected 
   *        is written to `d_num_selected_out`. ![](select_flags_logo.png)
   *
   * @par
   * - The value type of `d_flags` must be castable to `bool` (e.g., `bool`, 
   *   `char`, `int`, etc.).
   * - Copies of the selected items are compacted into `d_out` and maintain 
   *   their original relative ordering.
   * - The range `[d_out, d_out + *d_num_selected_out)` shall not overlap 
   *   `[d_in, d_in + num_items)`, `[d_flags, d_flags + num_items)` nor
   *   `d_num_selected_out` in any way.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the compaction of items selected from 
   * an `int` device vector.
   * @par
   * @code
   * #include <cub/cub.cuh>  // or equivalently <cub/device/device_select.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for input, 
   * // flags, and output
   * int  num_items;              // e.g., 8
   * int  *d_in;                  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
   * char *d_flags;               // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
   * int  *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   * int  *d_num_selected_out;    // e.g., [ ]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSelect::Flagged(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_flags, d_out, d_num_selected_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run selection
   * cub::DeviceSelect::Flagged(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_flags, d_out, d_num_selected_out, num_items);
   *
   * // d_out                 <-- [1, 4, 6, 7]
   * // d_num_selected_out    <-- [4]
   *
   * @endcode
   *
   * @tparam InputIteratorT       
   *   **[inferred]** Random-access input iterator type for reading input 
   *   items \iterator
   *
   * @tparam FlagIterator         
   *   **[inferred]** Random-access input iterator type for reading selection 
   *   flags \iterator
   *
   * @tparam OutputIteratorT      
   *   **[inferred]** Random-access output iterator type for writing selected 
   *   items \iterator
   *
   * @tparam NumSelectedIteratorT  
   *   **[inferred]** Output iterator type for recording the number of items 
   *   selected \iterator
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
   *   Pointer to the input sequence of data items
   *
   * @param[in] d_flags  
   *   Pointer to the input sequence of selection flags
   *
   * @param[out] d_out  
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_num_selected_out  
   *   Pointer to the output total number of items selected 
   *   (i.e., length of `d_out`)
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT,
            typename FlagIterator,
            typename OutputIteratorT,
            typename NumSelectedIteratorT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Flagged(void *d_temp_storage,
          size_t &temp_storage_bytes,
          InputIteratorT d_in,
          FlagIterator d_flags,
          OutputIteratorT d_out,
          NumSelectedIteratorT d_num_selected_out,
          int num_items,
          cudaStream_t stream = 0)
  {
    using OffsetT    = int;      // Signed integer type for global offsets
    using SelectOp   = NullType; // Selection op (not used)
    using EqualityOp = NullType; // Equality operator (not used)

    return DispatchSelectIf<InputIteratorT,
                            FlagIterator,
                            OutputIteratorT,
                            NumSelectedIteratorT,
                            SelectOp,
                            EqualityOp,
                            OffsetT,
                            false>::Dispatch(d_temp_storage,
                                             temp_storage_bytes,
                                             d_in,
                                             d_flags,
                                             d_out,
                                             d_num_selected_out,
                                             SelectOp(),
                                             EqualityOp(),
                                             num_items,
                                             stream);
  }

  template <typename InputIteratorT,
            typename FlagIterator,
            typename OutputIteratorT,
            typename NumSelectedIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Flagged(void *d_temp_storage,
          size_t &temp_storage_bytes,
          InputIteratorT d_in,
          FlagIterator d_flags,
          OutputIteratorT d_out,
          NumSelectedIteratorT d_num_selected_out,
          int num_items,
          cudaStream_t stream,
          bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Flagged<InputIteratorT,
                   FlagIterator,
                   OutputIteratorT,
                   NumSelectedIteratorT>(d_temp_storage,
                                         temp_storage_bytes,
                                         d_in,
                                         d_flags,
                                         d_out,
                                         d_num_selected_out,
                                         num_items,
                                         stream);
  }
   
  /**
   * @brief Uses the `d_flags` sequence to selectively compact the items in 
   *        `d_data`. The total number of items selected is written to 
   *        `d_num_selected_out`. ![](select_flags_logo.png)
   *
   * @par
   * - The value type of `d_flags` must be castable to `bool` (e.g., `bool`, 
   *   `char`, `int`, etc.).
   * - Copies of the selected items are compacted in-place and maintain 
   *   their original relative ordering.
   * - The `d_data` may equal `d_flags`. The range 
   *  `[d_data, d_data + num_items)` shall not overlap 
   *  `[d_flags, d_flags + num_items)` in any other way.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the compaction of items selected from 
   * an `int` device vector.
   * @par
   * @code
   * #include <cub/cub.cuh>  // or equivalently <cub/device/device_select.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for input, 
   * // flags, and output
   * int  num_items;              // e.g., 8
   * int  *d_data;                // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
   * char *d_flags;               // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
   * int  *d_num_selected_out;    // e.g., [ ]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSelect::Flagged(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_flags, d_num_selected_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run selection
   * cub::DeviceSelect::Flagged(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_flags, d_num_selected_out, num_items);
   *
   * // d_data                <-- [1, 4, 6, 7]
   * // d_num_selected_out    <-- [4]
   *
   * @endcode
   *
   * @tparam IteratorT       
   *   **[inferred]** Random-access iterator type for reading and writing 
   *   selected items \iterator
   *
   * @tparam FlagIterator         
   *   **[inferred]** Random-access input iterator type for reading selection 
   *   flags \iterator
   *
   * @tparam NumSelectedIteratorT  
   *   **[inferred]** Output iterator type for recording the number of items 
   *   selected \iterator
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_data
   *   Pointer to the sequence of data items
   *
   * @param[in] d_flags  
   *   Pointer to the input sequence of selection flags
   *
   * @param[out] d_num_selected_out  
   *   Pointer to the output total number of items selected 
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_data`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename IteratorT,
            typename FlagIterator,
            typename NumSelectedIteratorT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Flagged(void *d_temp_storage,
          size_t &temp_storage_bytes,
          IteratorT d_data,
          FlagIterator d_flags,
          NumSelectedIteratorT d_num_selected_out,
          int num_items,
          cudaStream_t stream = 0)
  {
    using OffsetT    = int;      // Signed integer type for global offsets
    using SelectOp   = NullType; // Selection op (not used)
    using EqualityOp = NullType; // Equality operator (not used)

    constexpr bool may_alias = true;

    return DispatchSelectIf<IteratorT,
                            FlagIterator,
                            IteratorT,
                            NumSelectedIteratorT,
                            SelectOp,
                            EqualityOp,
                            OffsetT,
                            false, 
                            may_alias>::Dispatch(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_data, // in
                                                 d_flags,
                                                 d_data, // out
                                                 d_num_selected_out,
                                                 SelectOp(),
                                                 EqualityOp(),
                                                 num_items,
                                                 stream);
  }

  template <typename IteratorT,
            typename FlagIterator,
            typename NumSelectedIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Flagged(void *d_temp_storage,
          size_t &temp_storage_bytes,
          IteratorT d_data,
          FlagIterator d_flags,
          NumSelectedIteratorT d_num_selected_out,
          int num_items,
          cudaStream_t stream,
          bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Flagged<IteratorT, FlagIterator, NumSelectedIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_data,
      d_flags,
      d_num_selected_out,
      num_items,
      stream);
  }

  /**
   * @brief Uses the `select_op` functor to selectively copy items from `d_in` 
   *        into `d_out`. The total number of items selected is written to 
   *        `d_num_selected_out`. ![](select_logo.png)
   *
   * @par
   * - Copies of the selected items are compacted into `d_out` and maintain 
   *   their original relative ordering.
   * - The range `[d_out, d_out + *d_num_selected_out)` shall not overlap 
   *   `[d_in, d_in + num_items)` nor `d_num_selected_out` in any way.
   * - @devicestorage
   *
   * @par Performance
   * The following charts illustrate saturated select-if performance across 
   * different CUDA architectures for `int32` and `int64` items, respectively. 
   * Items are selected with 50% probability.
   *
   * @image html select_if_int32_50_percent.png
   * @image html select_if_int64_50_percent.png
   *
   * @par
   * The following charts are similar, but 5% selection probability:
   *
   * @image html select_if_int32_5_percent.png
   * @image html select_if_int64_5_percent.png
   *
   * @par Snippet
   * The code snippet below illustrates the compaction of items selected from 
   * an `int` device vector.
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_select.cuh>
   *
   * // Functor type for selecting values less than some criteria
   * struct LessThan
   * {
   *     int compare;
   *
   *     CUB_RUNTIME_FUNCTION __forceinline__
   *     LessThan(int compare) : compare(compare) {}
   *
   *     CUB_RUNTIME_FUNCTION __forceinline__
   *     bool operator()(const int &a) const {
   *         return (a < compare);
   *     }
   * };
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int      num_items;              // e.g., 8
   * int      *d_in;                  // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
   * int      *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   * int      *d_num_selected_out;    // e.g., [ ]
   * LessThan select_op(7);
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSelect::If(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_out, d_num_selected_out, num_items, select_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run selection
   * cub::DeviceSelect::If(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_out, d_num_selected_out, num_items, select_op);
   *
   * // d_out                 <-- [0, 2, 3, 5, 2]
   * // d_num_selected_out    <-- [5]
   * @endcode
   *
   * @tparam InputIteratorT       
   *   **[inferred]** Random-access input iterator type for reading input 
   *   items \iterator
   *
   * @tparam OutputIteratorT      
   *   **[inferred]** Random-access output iterator type for writing selected 
   *   items \iterator
   *
   * @tparam NumSelectedIteratorT  
   *   **[inferred]** Output iterator type for recording the number of items 
   *   selected \iterator
   *
   * @tparam SelectOp             
   *   **[inferred]** Selection operator type having member 
   *   `bool operator()(const T &a)`
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
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out  
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_num_selected_out  
   *   Pointer to the output total number of items selected 
   *   (i.e., length of `d_out`)
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] select_op  
   *   Unary selection operator
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumSelectedIteratorT,
            typename SelectOp>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  If(void *d_temp_storage,
     size_t &temp_storage_bytes,
     InputIteratorT d_in,
     OutputIteratorT d_out,
     NumSelectedIteratorT d_num_selected_out,
     int num_items,
     SelectOp select_op,
     cudaStream_t stream = 0)
  {
    using OffsetT      = int;        // Signed integer type for global offsets
    using FlagIterator = NullType *; // FlagT iterator type (not used)
    using EqualityOp   = NullType;   // Equality operator (not used)

    return DispatchSelectIf<InputIteratorT,
                            FlagIterator,
                            OutputIteratorT,
                            NumSelectedIteratorT,
                            SelectOp,
                            EqualityOp,
                            OffsetT,
                            false>::Dispatch(d_temp_storage,
                                             temp_storage_bytes,
                                             d_in,
                                             NULL,
                                             d_out,
                                             d_num_selected_out,
                                             select_op,
                                             EqualityOp(),
                                             num_items,
                                             stream);
  }

  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumSelectedIteratorT,
            typename SelectOp>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  If(void *d_temp_storage,
     size_t &temp_storage_bytes,
     InputIteratorT d_in,
     OutputIteratorT d_out,
     NumSelectedIteratorT d_num_selected_out,
     int num_items,
     SelectOp select_op,
     cudaStream_t stream,
     bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return If<InputIteratorT, OutputIteratorT, NumSelectedIteratorT, SelectOp>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      d_num_selected_out,
      num_items,
      select_op,
      stream);
  }

  /**
   * @brief Uses the `select_op` functor to selectively compact items in 
   *        `d_data`. The total number of items selected is written to 
   *        `d_num_selected_out`. ![](select_logo.png)
   *
   * @par
   * - Copies of the selected items are compacted in `d_data` and maintain 
   *   their original relative ordering.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the compaction of items selected from 
   * an `int` device vector.
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_select.cuh>
   *
   * // Functor type for selecting values less than some criteria
   * struct LessThan
   * {
   *     int compare;
   *
   *     CUB_RUNTIME_FUNCTION __forceinline__
   *     LessThan(int compare) : compare(compare) {}
   *
   *     CUB_RUNTIME_FUNCTION __forceinline__
   *     bool operator()(const int &a) const {
   *         return (a < compare);
   *     }
   * };
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int      num_items;              // e.g., 8
   * int      *d_data;                // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
   * int      *d_num_selected_out;    // e.g., [ ]
   * LessThan select_op(7);
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSelect::If(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_data, d_num_selected_out, num_items, select_op);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run selection
   * cub::DeviceSelect::If(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_data, d_num_selected_out, num_items, select_op);
   *
   * // d_data                <-- [0, 2, 3, 5, 2]
   * // d_num_selected_out    <-- [5]
   * @endcode
   *
   * @tparam IteratorT       
   *   **[inferred]** Random-access input iterator type for reading and 
   *   writing items \iterator
   *
   * @tparam NumSelectedIteratorT  
   *   **[inferred]** Output iterator type for recording the number of items 
   *   selected \iterator
   *
   * @tparam SelectOp             
   *   **[inferred]** Selection operator type having member 
   *   `bool operator()(const T &a)`
   *
   * @param[in] d_temp_storage  
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes  
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_data
   *   Pointer to the sequence of data items
   *
   * @param[out] d_num_selected_out  
   *   Pointer to the output total number of items selected 
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_data`)
   *
   * @param[in] select_op  
   *   Unary selection operator
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename IteratorT,
            typename NumSelectedIteratorT,
            typename SelectOp>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  If(void *d_temp_storage,
     size_t &temp_storage_bytes,
     IteratorT d_data,
     NumSelectedIteratorT d_num_selected_out,
     int num_items,
     SelectOp select_op,
     cudaStream_t stream = 0)
  {
    using OffsetT      = int;        // Signed integer type for global offsets
    using FlagIterator = NullType *; // FlagT iterator type (not used)
    using EqualityOp   = NullType;   // Equality operator (not used)

    constexpr bool may_alias = true;

    return DispatchSelectIf<IteratorT,
                            FlagIterator,
                            IteratorT,
                            NumSelectedIteratorT,
                            SelectOp,
                            EqualityOp,
                            OffsetT,
                            false,
                            may_alias>::Dispatch(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_data, // in
                                                 NULL,
                                                 d_data, // out
                                                 d_num_selected_out,
                                                 select_op,
                                                 EqualityOp(),
                                                 num_items,
                                                 stream);
  }

  template <typename IteratorT,
            typename NumSelectedIteratorT,
            typename SelectOp>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  If(void *d_temp_storage,
     size_t &temp_storage_bytes,
     IteratorT d_data,
     NumSelectedIteratorT d_num_selected_out,
     int num_items,
     SelectOp select_op,
     cudaStream_t stream,
     bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return If<IteratorT, NumSelectedIteratorT, SelectOp>(d_temp_storage,
                                                         temp_storage_bytes,
                                                         d_data,
                                                         d_num_selected_out,
                                                         num_items,
                                                         select_op,
                                                         stream);
  }

  /**
   * @brief Given an input sequence `d_in` having runs of consecutive 
   *        equal-valued keys, only the first key from each run is selectively 
   *        copied to `d_out`. The total number of items selected is written to 
   *        `d_num_selected_out`. ![](unique_logo.png)
   *
   * @par
   * - The `==` equality operator is used to determine whether keys are 
   *   equivalent
   * - Copies of the selected items are compacted into `d_out` and maintain 
   *   their original relative ordering.
   * - The range `[d_out, d_out + *d_num_selected_out)` shall not overlap 
   *   `[d_in, d_in + num_items)` nor `d_num_selected_out` in any way.
   * - @devicestorage
   *
   * @par Performance
   * The following charts illustrate saturated select-unique performance across different
   * CUDA architectures for `int32` and `int64` items, respectively. Segments 
   * have lengths uniformly sampled from `[1, 1000]`.
   *
   * @image html select_unique_int32_len_500.png
   * @image html select_unique_int64_len_500.png
   *
   * @par
   * The following charts are similar, but with segment lengths uniformly 
   * sampled from `[1, 10]`:
   *
   * @image html select_unique_int32_len_5.png
   * @image html select_unique_int64_len_5.png
   *
   * @par Snippet
   * The code snippet below illustrates the compaction of items selected from 
   * an `int` device vector.
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_select.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int  num_items;              // e.g., 8
   * int  *d_in;                  // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
   * int  *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   * int  *d_num_selected_out;    // e.g., [ ]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSelect::Unique(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_out, d_num_selected_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run selection
   * cub::DeviceSelect::Unique(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_in, d_out, d_num_selected_out, num_items);
   *
   * // d_out                 <-- [0, 2, 9, 5, 8]
   * // d_num_selected_out    <-- [5]
   * @endcode
   *
   * @tparam InputIteratorT       
   *   **[inferred]** Random-access input iterator type for reading input 
   *   items \iterator
   *
   * @tparam OutputIteratorT      
   *   **[inferred]** Random-access output iterator type for writing selected 
   *   items \iterator
   *
   * @tparam NumSelectedIteratorT  
   *   **[inferred]** Output iterator type for recording the number of items 
   *   selected \iterator
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
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out  
   *   Pointer to the output sequence of selected data items
   *
   * @param[out] d_num_selected_out  
   *   Pointer to the output total number of items selected 
   *   (i.e., length of `d_out`)
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumSelectedIteratorT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Unique(void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT d_out,
         NumSelectedIteratorT d_num_selected_out,
         int num_items,
         cudaStream_t stream = 0)
  {
    using OffsetT      = int;        // Signed integer type for global offsets
    using FlagIterator = NullType *; // FlagT iterator type (not used)
    using SelectOp     = NullType;   // Selection op (not used)
    using EqualityOp   = Equality;   // Default == operator

    return DispatchSelectIf<InputIteratorT,
                            FlagIterator,
                            OutputIteratorT,
                            NumSelectedIteratorT,
                            SelectOp,
                            EqualityOp,
                            OffsetT,
                            false>::Dispatch(d_temp_storage,
                                             temp_storage_bytes,
                                             d_in,
                                             NULL,
                                             d_out,
                                             d_num_selected_out,
                                             SelectOp(),
                                             EqualityOp(),
                                             num_items,
                                             stream);
  }

  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumSelectedIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Unique(void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT d_out,
         NumSelectedIteratorT d_num_selected_out,
         int num_items,
         cudaStream_t stream,
         bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Unique<InputIteratorT, OutputIteratorT, NumSelectedIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      d_num_selected_out,
      num_items,
      stream);
  }

  /**
   * @brief Given an input sequence `d_keys_in` and `d_values_in` with runs of 
   *        key-value pairs with consecutive equal-valued keys, only the first 
   *        key and its value from each run is selectively copied to 
   *        `d_keys_out` and `d_values_out`. The total number of items selected 
   *        is written to `d_num_selected_out`. ![](unique_logo.png)
   *
   * @par
   * - The `==` equality operator is used to determine whether keys are 
   *   equivalent
   * - Copies of the selected items are compacted into `d_out` and maintain 
   *   their original relative ordering.
   * - In-place operations are not supported. There must be no overlap between
   *   any of the provided ranges:
   *   - `[d_keys_in,          d_keys_in    + num_items)`
   *   - `[d_keys_out,         d_keys_out   + *d_num_selected_out)`
   *   - `[d_values_in,        d_values_in  + num_items)`
   *   - `[d_values_out,       d_values_out + *d_num_selected_out)`
   *   - `[d_num_selected_out, d_num_selected_out + 1)`
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the compaction of items selected from 
   * an `int` device vector.
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_select.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers 
   * // for input and output
   * int  num_items;              // e.g., 8
   * int  *d_keys_in;             // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
   * int  *d_values_in;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
   * int  *d_keys_out;            // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   * int  *d_values_out;          // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   * int  *d_num_selected_out;    // e.g., [ ]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSelect::UniqueByKey(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_keys_in, d_values_in, 
   *   d_keys_out, d_values_out, d_num_selected_out, num_items);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run selection
   * cub::DeviceSelect::UniqueByKey(
   *   d_temp_storage, temp_storage_bytes, 
   *   d_keys_in, d_values_in, 
   *   d_keys_out, d_values_out, d_num_selected_out, num_items);
   *
   * // d_keys_out            <-- [0, 2, 9, 5, 8]
   * // d_values_out          <-- [1, 2, 4, 5, 8]
   * // d_num_selected_out    <-- [5]
   * @endcode
   *
   * @tparam KeyInputIteratorT       
   *   **[inferred]** Random-access input iterator type for reading input 
   *   keys \iterator
   *
   * @tparam ValueInputIteratorT     
   *   **[inferred]** Random-access input iterator type for reading input 
   *   values \iterator
   *
   * @tparam KeyOutputIteratorT      
   *   **[inferred]** Random-access output iterator type for writing selected 
   *   keys \iterator
   *
   * @tparam ValueOutputIteratorT    
   *   **[inferred]** Random-access output iterator type for writing selected 
   *   values \iterator
   *
   * @tparam NumSelectedIteratorT    
   *   **[inferred]** Output iterator type for recording the number of items 
   *   selected \iterator
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
   *   Pointer to the input sequence of keys
   *
   * @param[in] d_values_in  
   *   Pointer to the input sequence of values
   *
   * @param[out] d_keys_out  
   *   Pointer to the output sequence of selected keys
   *
   * @param[out] d_values_out  
   *   Pointer to the output sequence of selected values
   *
   * @param[out] d_num_selected_out  
   *   Pointer to the total number of items selected (i.e., length of 
   *   `d_keys_out` or `d_values_out`)
   *
   * @param[in] num_items  
   *   Total number of input items (i.e., length of `d_keys_in` or 
   *   `d_values_in`)
   *
   * @param[in] stream  
   *   **[optional]** CUDA stream to launch kernels within. 
   *   Default is stream<sub>0</sub>.
   */
  template <typename KeyInputIteratorT,
            typename ValueInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueOutputIteratorT,
            typename NumSelectedIteratorT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  UniqueByKey(void *d_temp_storage,
              size_t &temp_storage_bytes,
              KeyInputIteratorT d_keys_in,
              ValueInputIteratorT d_values_in,
              KeyOutputIteratorT d_keys_out,
              ValueOutputIteratorT d_values_out,
              NumSelectedIteratorT d_num_selected_out,
              int num_items,
              cudaStream_t stream = 0)
  {
    using OffsetT    = int;
    using EqualityOp = Equality;

    return DispatchUniqueByKey<KeyInputIteratorT,
                               ValueInputIteratorT,
                               KeyOutputIteratorT,
                               ValueOutputIteratorT,
                               NumSelectedIteratorT,
                               EqualityOp,
                               OffsetT>::Dispatch(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_keys_in,
                                                  d_values_in,
                                                  d_keys_out,
                                                  d_values_out,
                                                  d_num_selected_out,
                                                  EqualityOp(),
                                                  num_items,
                                                  stream);
  }

  template <typename KeyInputIteratorT,
            typename ValueInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueOutputIteratorT,
            typename NumSelectedIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  UniqueByKey(void *d_temp_storage,
              size_t &temp_storage_bytes,
              KeyInputIteratorT d_keys_in,
              ValueInputIteratorT d_values_in,
              KeyOutputIteratorT d_keys_out,
              ValueOutputIteratorT d_values_out,
              NumSelectedIteratorT d_num_selected_out,
              int num_items,
              cudaStream_t stream,
              bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return UniqueByKey<KeyInputIteratorT,
                       ValueInputIteratorT,
                       KeyOutputIteratorT,
                       ValueOutputIteratorT,
                       NumSelectedIteratorT>(d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_in,
                                             d_values_in,
                                             d_keys_out,
                                             d_values_out,
                                             d_num_selected_out,
                                             num_items,
                                             stream);
  }
};

/**
 * @example example_device_select_flagged.cu
 * @example example_device_select_if.cu
 * @example example_device_select_unique.cu
 */

CUB_NAMESPACE_END
