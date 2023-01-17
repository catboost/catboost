/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
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

/**
 * @file
 * cub::DevicePartition provides device-wide, parallel operations for
 * partitioning sequences of data items residing within device-accessible memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "dispatch/dispatch_select_if.cuh"
#include "dispatch/dispatch_three_way_partition.cuh"
#include "../config.cuh"

CUB_NAMESPACE_BEGIN


/**
 * @brief DevicePartition provides device-wide, parallel operations for
 *        partitioning sequences of data items residing within device-accessible
 *        memory. ![](partition_logo.png)
 * @ingroup SingleModule
 *
 * @par Overview
 * These operations apply a selection criterion to construct a partitioned
 * output sequence from items selected/unselected from a specified input
 * sequence.
 *
 * @par Usage Considerations
 * \cdp_class{DevicePartition}
 *
 * @par Performance
 * \linear_performance{partition}
 *
 * @par
 * The following chart illustrates DevicePartition::If
 * performance across different CUDA architectures for @p int32 items,
 * where 50% of the items are randomly selected for the first partition.
 * \plots_below
 *
 * @image html partition_if_int32_50_percent.png
 *
 */
struct DevicePartition
{
    /**
     * @brief Uses the @p d_flags sequence to split the corresponding items from
     *        @p d_in into a partitioned sequence @p d_out. The total number of
     *        items copied into the first partition is written to
     *        @p d_num_selected_out. ![](partition_flags_logo.png)
     *
     * @par
     * - The value type of @p d_flags must be castable to @p bool (e.g.,
     *   @p bool, @p char, @p int, etc.).
     * - Copies of the selected items are compacted into @p d_out and maintain
     *   their original relative ordering, however copies of the unselected
     *   items are compacted into the rear of @p d_out in reverse order.
     * - \devicestorage
     *
     * @par Snippet
     * The code snippet below illustrates the compaction of items selected from
     * an @p int device vector.
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/device/device_partition.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for
     * // input, flags, and output
     * int  num_items;              // e.g., 8
     * int  *d_in;                  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
     * char *d_flags;               // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
     * int  *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * int  *d_num_selected_out;    // e.g., [ ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void *d_temp_storage = nullptr;
     * std::size_t temp_storage_bytes = 0;
     * cub::DevicePartition::Flagged(
     *   d_temp_storage, temp_storage_bytes,
     *   d_in, d_flags, d_out, d_num_selected_out, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run selection
     * cub::DevicePartition::Flagged(
     *   d_temp_storage, temp_storage_bytes,
     *   d_in, d_flags, d_out, d_num_selected_out, num_items);
     *
     * // d_out                 <-- [1, 4, 6, 7, 8, 5, 3, 2]
     * // d_num_selected_out    <-- [4]
     * @endcode
     *
     * @tparam InputIteratorT
     *   **[inferred]** Random-access input iterator type for reading
     *   input items \iterator
     *
     * @tparam FlagIterator
     *   **[inferred]** Random-access input iterator type for reading
     *   selection flags \iterator
     *
     * @tparam OutputIteratorT
     *   **[inferred]** Random-access output iterator type for writing
     *   output items \iterator
     *
     * @tparam NumSelectedIteratorT
     *   **[inferred]** Output iterator type for recording the number
     *   of items selected \iterator
     *
     * @param[in] d_temp_storage
     *   Device-accessible allocation of temporary storage. When `nullptr`, the
     *   required allocation size is written to @p temp_storage_bytes and no
     *   work is done.
     *
     * @param[in,out] temp_storage_bytes
     *   Reference to size in bytes of @p d_temp_storage allocation
     *
     * @param[in] d_in
     *   Pointer to the input sequence of data items
     *
     * @param[in] d_flags
     *   Pointer to the input sequence of selection flags
     *
     * @param[out] d_out
     *   Pointer to the output sequence of partitioned data items
     *
     * @param[out] d_num_selected_out
     *   Pointer to the output total number of items selected (i.e., the
     *   offset of the unselected partition)
     *
     * @param[in] num_items
     *   Total number of items to select from
     *
     * @param[in] stream
     *   **[optional]** CUDA stream to launch kernels within.
     *   Default is stream<sub>0</sub>.
     *
     * @param[in] debug_synchronous
     *   **[optional]** Whether or not to synchronize the stream after every
     *   kernel launch to check for errors. May cause significant slowdown.
     *   Default is @p false.
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
            cudaStream_t stream    = 0,
            bool debug_synchronous = false)
    {
      using OffsetT    = int;      // Signed integer type for global offsets
      using SelectOp   = NullType; // Selection op (not used)
      using EqualityOp = NullType; // Equality operator (not used)
      using DispatchSelectIfT = DispatchSelectIf<InputIteratorT,
                                                 FlagIterator,
                                                 OutputIteratorT,
                                                 NumSelectedIteratorT,
                                                 SelectOp,
                                                 EqualityOp,
                                                 OffsetT,
                                                 true>;

      return DispatchSelectIfT::Dispatch(d_temp_storage,
                                         temp_storage_bytes,
                                         d_in,
                                         d_flags,
                                         d_out,
                                         d_num_selected_out,
                                         SelectOp{},
                                         EqualityOp{},
                                         num_items,
                                         stream,
                                         debug_synchronous);
    }


    /**
     * @brief Uses the @p select_op functor to split the corresponding items
     *        from @p d_in into a partitioned sequence @p d_out. The total
     *        number of items copied into the first partition is written to
     *        @p d_num_selected_out. ![](partition_logo.png)
     *
     * @par
     * - Copies of the selected items are compacted into @p d_out and maintain
     *   their original relative ordering, however copies of the unselected
     *   items are compacted into the rear of @p d_out in reverse order.
     * - \devicestorage
     *
     * @par Performance
     * The following charts illustrate saturated partition-if performance across
     * different CUDA architectures for @p int32 and @p int64 items,
     * respectively. Items are selected for the first partition with 50%
     * probability.
     *
     * @image html partition_if_int32_50_percent.png
     * @image html partition_if_int64_50_percent.png
     *
     * @par
     * The following charts are similar, but 5% selection probability for the
     * first partition:
     *
     * @image html partition_if_int32_5_percent.png
     * @image html partition_if_int64_5_percent.png
     *
     * @par Snippet
     * The code snippet below illustrates the compaction of items selected from
     * an @p int device vector.
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/device/device_partition.cuh>
     *
     * // Functor type for selecting values less than some criteria
     * struct LessThan
     * {
     *     int compare;
     *
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     explicit LessThan(int compare) : compare(compare) {}
     *
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     bool operator()(const int &a) const
     *     {
     *         return (a < compare);
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for
     * // input and output
     * int      num_items;              // e.g., 8
     * int      *d_in;                  // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
     * int      *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * int      *d_num_selected_out;    // e.g., [ ]
     * LessThan select_op(7);
     * ...
     *
     * // Determine temporary device storage requirements
     * void *d_temp_storage = nullptr;
     * std::size_t temp_storage_bytes = 0;
     * cub::DevicePartition::If(
     * d_temp_storage, temp_storage_bytes,
     * d_in, d_out, d_num_selected_out, num_items, select_op);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run selection
     * cub::DevicePartition::If(
     *   d_temp_storage, temp_storage_bytes,
     *   d_in, d_out, d_num_selected_out, num_items, select_op);
     *
     * // d_out                 <-- [0, 2, 3, 5, 2, 8, 81, 9]
     * // d_num_selected_out    <-- [5]
     *
     * @endcode
     *
     * @tparam InputIteratorT
     *   **[inferred]** Random-access input iterator type for reading input
     *   items \iterator
     *
     * @tparam OutputIteratorT
     *   **[inferred]** Random-access output iterator type for writing output
     *   items \iterator
     *
     * @tparam NumSelectedIteratorT
     *   **[inferred]** Output iterator type for recording the number of items
     *   selected \iterator
     *
     * @tparam SelectOp
     *   **[inferred]** Selection functor type having member
     *   `bool operator()(const T &a)`
     *
     * @param[in] d_temp_storage
     *   Device-accessible allocation of temporary storage. When `nullptr`, the
     *   required allocation size is written to `temp_storage_bytes` and no
     *   work is done.
     *
     * @param[in,out] temp_storage_bytes
     *   Reference to size in bytes of @p d_temp_storage allocation
     *
     * @param[in] d_in
     *   Pointer to the input sequence of data items
     *
     * @param[out] d_out
     *   Pointer to the output sequence of partitioned data items
     *
     * @param[out] d_num_selected_out
     *   Pointer to the output total number of items selected (i.e., the
     *   offset of the unselected partition)
     *
     * @param[in] num_items
     *   Total number of items to select from
     *
     * @param[in] select_op
     *   Unary selection operator
     *
     * @param[in] stream
     *   **[optional]** CUDA stream to launch kernels within.
     *   Default is stream<sub>0</sub>.
     *
     * @param[in] debug_synchronous
     *   **[optional]** Whether or not to synchronize the stream after every
     *   kernel launch to check for errors. May cause significant slowdown.
     *   Default is @p false.
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
       cudaStream_t stream    = 0,
       bool debug_synchronous = false)
    {
        using OffsetT      = int; // Signed integer type for global offsets
        using FlagIterator = NullType *; // FlagT iterator type (not used)
        using EqualityOp   = NullType;   // Equality operator (not used)

        using DispatchSelectIfT = DispatchSelectIf<InputIteratorT,
                                                   FlagIterator,
                                                   OutputIteratorT,
                                                   NumSelectedIteratorT,
                                                   SelectOp,
                                                   EqualityOp,
                                                   OffsetT,
                                                   true>;

        return DispatchSelectIfT::Dispatch(d_temp_storage,
                                           temp_storage_bytes,
                                           d_in,
                                           nullptr,
                                           d_out,
                                           d_num_selected_out,
                                           select_op,
                                           EqualityOp{},
                                           num_items,
                                           stream,
                                           debug_synchronous);
    }


    /**
     * @brief Uses two functors to split the corresponding items from @p d_in
     *        into a three partitioned sequences @p d_first_part_out
     *        @p d_second_part_out and @p d_unselected_out.
     *        The total number of items copied into the first partition is written
     *        to `d_num_selected_out[0]`, while the total number of items copied
     *        into the second partition is written to `d_num_selected_out[1]`.
     *
     * @par
     * - Copies of the items selected by @p select_first_part_op are compacted
     *   into @p d_first_part_out and maintain their original relative ordering.
     * - Copies of the items selected by @p select_second_part_op are compacted
     *   into @p d_second_part_out and maintain their original relative ordering.
     * - Copies of the unselected items are compacted into the
     *   @p d_unselected_out in reverse order.
     *
     * @par Snippet
     * The code snippet below illustrates how this algorithm can partition an
     * input vector into small, medium, and large items so that the relative
     * order of items remain deterministic.
     *
     * Let's consider any value that doesn't exceed six a small one. On the
     * other hand, any value that exceeds 50 will be considered a large one.
     * Since the value used to define a small part doesn't match one that
     * defines the large part, the intermediate segment is implied.
     *
     * These definitions partition a value space into three categories. We want
     * to preserve the order of items in which they appear in the input vector.
     * Since the algorithm provides stable partitioning, this is possible.
     *
     * Since the number of items in each category is unknown beforehand, we need
     * three output arrays of num_items elements each. To reduce the memory
     * requirements, we can combine the output storage for two categories.
     *
     * Since each value falls precisely in one category, it's safe to add
     * "large" values into the head of the shared output vector and the "middle"
     * values into its tail. To add items into the tail of the output array, we
     * can use `thrust::reverse_iterator`.
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/device/device_partition.cuh>
     *
     * // Functor type for selecting values less than some criteria
     * struct LessThan
     * {
     *     int compare;
     *
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     explicit LessThan(int compare) : compare(compare) {}
     *
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     bool operator()(const int &a) const
     *     {
     *         return a < compare;
     *     }
     * };
     *
     * // Functor type for selecting values greater than some criteria
     * struct GreaterThan
     * {
     *     int compare;
     *
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     explicit GreaterThan(int compare) : compare(compare) {}
     *
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     bool operator()(const int &a) const
     *     {
     *         return a > compare;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for
     * // input and output
     * int      num_items;                   // e.g., 8
     * int      *d_in;                       // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
     * int      *d_large_and_unselected_out; // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * int      *d_small_out;                // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * int      *d_num_selected_out;         // e.g., [ , ]
     * thrust::reverse_iterator<T> unselected_out(d_large_and_unselected_out + num_items);
     * LessThan small_items_selector(7);
     * GreaterThan large_items_selector(50);
     * ...
     *
     * // Determine temporary device storage requirements
     * void *d_temp_storage = nullptr;
     * std::size_t temp_storage_bytes = 0;
     * cub::DevicePartition::If(
     *      d_temp_storage, temp_storage_bytes,
     *      d_in, d_large_and_medium_out, d_small_out, unselected_out,
     *      d_num_selected_out, num_items,
     *      large_items_selector, small_items_selector);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run selection
     * cub::DevicePartition::If(
     *      d_temp_storage, temp_storage_bytes,
     *      d_in, d_large_and_medium_out, d_small_out, unselected_out,
     *      d_num_selected_out, num_items,
     *      large_items_selector, small_items_selector);
     *
     * // d_large_and_unselected_out  <-- [ 81,  ,  ,  ,  ,  , 8, 9 ]
     * // d_small_out                 <-- [  0, 2, 3, 5, 2,  ,  ,   ]
     * // d_num_selected_out          <-- [  1, 5 ]
     * @endcode
     *
     * @tparam InputIteratorT
     *   **[inferred]** Random-access input iterator type for reading
     *   input items \iterator
     *
     * @tparam FirstOutputIteratorT
     *   **[inferred]** Random-access output iterator type for writing output
     *   items selected by first operator \iterator
     *
     * @tparam SecondOutputIteratorT
     *   **[inferred]** Random-access output iterator type for writing output
     *   items selected by second operator \iterator
     *
     * @tparam UnselectedOutputIteratorT
     *   **[inferred]** Random-access output iterator type for writing
     *   unselected items \iterator
     *
     * @tparam NumSelectedIteratorT
     *   **[inferred]** Output iterator type for recording the number of items
     *   selected \iterator
     *
     * @tparam SelectFirstPartOp
     *   **[inferred]** Selection functor type having member
     *   `bool operator()(const T &a)`
     *
     * @tparam SelectSecondPartOp
     *   **[inferred]** Selection functor type having member
     *   `bool operator()(const T &a)`
     *
     * @param[in] d_temp_storage
     *   Device-accessible allocation of temporary storage. When `nullptr`, the
     *   required allocation size is written to @p temp_storage_bytes and
     *   no work is done.
     *
     * @param[in,out] temp_storage_bytes
     *   Reference to size in bytes of @p d_temp_storage allocation
     *
     * @param[in] d_in
     *   Pointer to the input sequence of data items
     *
     * @param[out] d_first_part_out
     *   Pointer to the output sequence of data items selected by
     *   @p select_first_part_op
     *
     * @param[out] d_second_part_out
     *   Pointer to the output sequence of data items selected by
     *   @p select_second_part_op
     *
     * @param[out] d_unselected_out
     *   Pointer to the output sequence of unselected data items
     *
     * @param[out] d_num_selected_out
     *   Pointer to the output array with two elements, where total number of
     *   items selected by @p select_first_part_op is stored as
     *   `d_num_selected_out[0]` and total number of items selected by
     *   @p select_second_part_op is stored as `d_num_selected_out[1]`,
     *   respectively
     */
    template <typename InputIteratorT,
              typename FirstOutputIteratorT,
              typename SecondOutputIteratorT,
              typename UnselectedOutputIteratorT,
              typename NumSelectedIteratorT,
              typename SelectFirstPartOp,
              typename SelectSecondPartOp>
    CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
    If(void *d_temp_storage,
       std::size_t &temp_storage_bytes,
       InputIteratorT d_in,
       FirstOutputIteratorT d_first_part_out,
       SecondOutputIteratorT d_second_part_out,
       UnselectedOutputIteratorT d_unselected_out,
       NumSelectedIteratorT d_num_selected_out,
       int num_items,
       SelectFirstPartOp select_first_part_op,
       SelectSecondPartOp select_second_part_op,
       cudaStream_t stream    = 0,
       bool debug_synchronous = false)
    {
      using OffsetT = int;
      using DispatchThreeWayPartitionIfT =
        DispatchThreeWayPartitionIf<InputIteratorT,
                                    FirstOutputIteratorT,
                                    SecondOutputIteratorT,
                                    UnselectedOutputIteratorT,
                                    NumSelectedIteratorT,
                                    SelectFirstPartOp,
                                    SelectSecondPartOp,
                                    OffsetT>;

      return DispatchThreeWayPartitionIfT::Dispatch(d_temp_storage,
                                                    temp_storage_bytes,
                                                    d_in,
                                                    d_first_part_out,
                                                    d_second_part_out,
                                                    d_unselected_out,
                                                    d_num_selected_out,
                                                    select_first_part_op,
                                                    select_second_part_op,
                                                    num_items,
                                                    stream,
                                                    debug_synchronous);
    }
};

/**
 * @example example_device_partition_flagged.cu
 * @example example_device_partition_if.cu
 */

CUB_NAMESPACE_END


