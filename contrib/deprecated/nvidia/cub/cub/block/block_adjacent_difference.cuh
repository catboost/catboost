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
 * The cub::BlockAdjacentDifference class provides
 * [<em>collective</em>](index.html#sec0) methods for computing the differences
 * of adjacent elements partitioned across a CUDA thread block.
 */

#pragma once
#pragma clang system_header


#include "../config.cuh"
#include "../util_type.cuh"
#include "../util_ptx.cuh"

CUB_NAMESPACE_BEGIN

/**
 * @brief BlockAdjacentDifference provides
 *        [<em>collective</em>](index.html#sec0) methods for computing the
 *        differences of adjacent elements partitioned across a CUDA thread
 *        block.
 *
 * @ingroup BlockModule
 *
 * @par Overview
 * - BlockAdjacentDifference calculates the differences of adjacent elements in
 *   the elements partitioned across a CUDA thread block. Because the binary
 *   operation could be noncommutative, there are two sets of methods.
 *   Methods named SubtractLeft subtract left element `i - 1` of input sequence
 *   from current element `i`. Methods named SubtractRight subtract current
 *   element `i` from the right one `i + 1`:
 *   @par
 *   @code
 *   int values[4]; // [1, 2, 3, 4]
 *   //...
 *   int subtract_left_result[4];  <-- [  1,  1,  1,  1 ]
 *   int subtract_right_result[4]; <-- [ -1, -1, -1,  4 ]
 *   @endcode
 * - For SubtractLeft, if the left element is out of bounds, the
 *   output value is assigned to `input[0]` without modification.
 * - For SubtractRight, if the right element is out of bounds, the output value
 *   is assigned to the current input value without modification.
 * - The following example under the examples/block folder illustrates usage of
 *   dynamically shared memory with BlockReduce and how to re-purpose
 *   the same memory region:
 *   <a href="../../examples/block/example_block_reduce_dyn_smem.cu">example_block_reduce_dyn_smem.cu</a>
 *   This example can be easily adapted to the storage required by
 *   BlockAdjacentDifference.
 *
 * @par Snippet
 * The code snippet below illustrates how to use @p BlockAdjacentDifference to
 * compute the left difference between adjacent elements.
 *
 * @par
 * @code
 * #include <cub/cub.cuh>
 * // or equivalently <cub/block/block_adjacent_difference.cuh>
 *
 * struct CustomDifference
 * {
 *   template <typename DataType>
 *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
 *   {
 *     return lhs - rhs;
 *   }
 * };
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Specialize BlockAdjacentDifference for a 1D block of
 *     // 128 threads of type int
 *     using BlockAdjacentDifferenceT =
 *        cub::BlockAdjacentDifference<int, 128>;
 *
 *     // Allocate shared memory for BlockDiscontinuity
 *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
 *
 *     // Obtain a segment of consecutive items that are blocked across threads
 *     int thread_data[4];
 *     ...
 *
 *     // Collectively compute adjacent_difference
 *     int result[4];
 *
 *     BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
 *         result,
 *         thread_data,
 *         CustomDifference());
 *
 * @endcode
 * @par
 * Suppose the set of input `thread_data` across the block of threads is
 * <tt>{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }</tt>.
 * The corresponding output `result` in those threads will be
 * <tt>{ [4,-2,-1,0], [0,0,0,0], [1,1,0,0], [0,1,-3,3], ... }</tt>.
 *
 */
template <typename T,
          int BLOCK_DIM_X,
          int BLOCK_DIM_Y     = 1,
          int BLOCK_DIM_Z     = 1,
          int LEGACY_PTX_ARCH = 0>
class BlockAdjacentDifference
{
private:

    /***************************************************************************
     * Constants and type definitions
     **************************************************************************/

    /// Constants

    /// The thread block size in threads
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

    /// Shared memory storage layout type (last element from each thread's input)
    struct _TempStorage
    {
        T first_items[BLOCK_THREADS];
        T last_items[BLOCK_THREADS];
    };


    /***************************************************************************
     * Utility methods
     **************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    /// Specialization for when FlagOp has third index param
    template <typename FlagOp,
              bool HAS_PARAM = BinaryOpHasIdxParam<T, FlagOp>::HAS_PARAM>
    struct ApplyOp
    {
        // Apply flag operator
      static __device__ __forceinline__ T FlagT(FlagOp flag_op,
                                                const T &a,
                                                const T &b,
                                                int idx)
      {
        return flag_op(b, a, idx);
      }
    };

    /// Specialization for when FlagOp does not have a third index param
    template <typename FlagOp>
    struct ApplyOp<FlagOp, false>
    {
      // Apply flag operator
      static __device__ __forceinline__ T FlagT(FlagOp flag_op,
                                                const T &a,
                                                const T &b,
                                                int /*idx*/)
      {
        return flag_op(b, a);
      }
    };

    /// Templated unrolling of item comparison (inductive case)
    struct Iterate
    {
        /**
         * Head flags
         *
         * @param[out] flags Calling thread's discontinuity head_flags
         * @param[in] input Calling thread's input items
         * @param[out] preds Calling thread's predecessor items
         * @param[in] flag_op Binary boolean flag predicate
         */
        template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
        static __device__ __forceinline__ void
        FlagHeads(int linear_tid,
                  FlagT (&flags)[ITEMS_PER_THREAD],
                  T (&input)[ITEMS_PER_THREAD],
                  T (&preds)[ITEMS_PER_THREAD],
                  FlagOp flag_op)
        {
          #pragma unroll
          for (int i = 1; i < ITEMS_PER_THREAD; ++i) {
            preds[i] = input[i - 1];
            flags[i] = ApplyOp<FlagOp>::FlagT(
                flag_op,
                preds[i],
                input[i],
                (linear_tid * ITEMS_PER_THREAD) + i);
          }
        }

        /**
         * Tail flags
         *
         * @param[out] flags Calling thread's discontinuity head_flags
         * @param[in] input Calling thread's input items
         * @param[in] flag_op Binary boolean flag predicate
         */
        template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
        static __device__ __forceinline__ void
        FlagTails(int linear_tid,
                  FlagT (&flags)[ITEMS_PER_THREAD],
                  T (&input)[ITEMS_PER_THREAD],
                  FlagOp flag_op)
        {
          #pragma unroll
          for (int i = 0; i < ITEMS_PER_THREAD - 1; ++i) {
            flags[i] = ApplyOp<FlagOp>::FlagT(
                flag_op,
                input[i],
                input[i + 1],
                (linear_tid * ITEMS_PER_THREAD) + i + 1);
          }
        }
    };

    /***************************************************************************
     * Thread fields
     **************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;


public:

    /// \smemstorage{BlockDiscontinuity}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /***********************************************************************//**
     * @name Collective constructors
     **************************************************************************/
    //@{

    /**
     * @brief Collective constructor using a private static allocation of shared
     *        memory as temporary storage.
     */
    __device__ __forceinline__ BlockAdjacentDifference()
        : temp_storage(PrivateStorage())
        , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}

    /**
     * @brief Collective constructor using the specified memory allocation as
     *        temporary storage.
     *
     * @param[in] temp_storage Reference to memory allocation having layout type TempStorage
     */
    __device__ __forceinline__ BlockAdjacentDifference(TempStorage &temp_storage)
        : temp_storage(temp_storage.Alias())
        , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}

    //@}  end member group
    /***********************************************************************//**
     * @name Read left operations
     **************************************************************************/
    //@{

    /**
     * @brief Subtracts the left element of each adjacent pair of elements
     *        partitioned across a CUDA thread block.
     *
     * @par
     * - \rowmajor
     * - \smemreuse
     *
     * @par Snippet
     * The code snippet below illustrates how to use @p BlockAdjacentDifference
     * to compute the left difference between adjacent elements.
     *
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockAdjacentDifference for a 1D block
     *     // of 128 threads of type int
     *     using BlockAdjacentDifferenceT =
     *        cub::BlockAdjacentDifference<int, 128>;
     *
     *     // Allocate shared memory for BlockDiscontinuity
     *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute adjacent_difference
     *     BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
     *         thread_data,
     *         thread_data,
     *         CustomDifference());
     *
     * @endcode
     * @par
     * Suppose the set of input `thread_data` across the block of threads is
     * `{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }`.
     * The corresponding output `result` in those threads will be
     * `{ [4,-2,-1,0], [0,0,0,0], [1,1,0,0], [0,1,-3,3], ... }`.
     *
     * @param[out] output
     *   Calling thread's adjacent difference result
     *
     * @param[in] input
     *   Calling thread's input items (may be aliased to @p output)
     *
     * @param[in] difference_op
     *   Binary difference operator
     */
    template <int ITEMS_PER_THREAD,
              typename OutputType,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractLeft(T (&input)[ITEMS_PER_THREAD],
                 OutputType (&output)[ITEMS_PER_THREAD],
                 DifferenceOpT difference_op)
    {
      // Share last item
      temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

      CTA_SYNC();

      #pragma unroll
      for (int item = ITEMS_PER_THREAD - 1; item > 0; item--)
      {
        output[item] = difference_op(input[item], input[item - 1]);
      }

      if (linear_tid == 0)
      {
        output[0] = input[0];
      }
      else
      {
        output[0] = difference_op(input[0],
                                  temp_storage.last_items[linear_tid - 1]);
      }
    }

    /**
     * @brief Subtracts the left element of each adjacent pair of elements
     *        partitioned across a CUDA thread block.
     *
     * @par
     * - \rowmajor
     * - \smemreuse
     *
     * @par Snippet
     * The code snippet below illustrates how to use @p BlockAdjacentDifference
     * to compute the left difference between adjacent elements.
     *
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockAdjacentDifference for a 1D block of
     *     // 128 threads of type int
     *     using BlockAdjacentDifferenceT =
     *        cub::BlockAdjacentDifference<int, 128>;
     *
     *     // Allocate shared memory for BlockDiscontinuity
     *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // The last item in the previous tile:
     *     int tile_predecessor_item = ...;
     *
     *     // Collectively compute adjacent_difference
     *     BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
     *         thread_data,
     *         thread_data,
     *         CustomDifference(),
     *         tile_predecessor_item);
     *
     * @endcode
     * @par
     * Suppose the set of input `thread_data` across the block of threads is
     * `{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }`.
     * and that `tile_predecessor_item` is `3`. The corresponding output
     * `result` in those threads will be
     * `{ [1,-2,-1,0], [0,0,0,0], [1,1,0,0], [0,1,-3,3], ... }`.
     *
     * @param[out] output
     *   Calling thread's adjacent difference result
     *
     * @param[in] input
     *   Calling thread's input items (may be aliased to \p output)
     *
     * @param[in] difference_op
     *   Binary difference operator
     *
     * @param[in] tile_predecessor_item
     *   <b>[<em>thread</em><sub>0</sub> only]</b> item which is going to be
     *   subtracted from the first tile item (<tt>input<sub>0</sub></tt> from
     *   <em>thread</em><sub>0</sub>).
     */
    template <int ITEMS_PER_THREAD,
              typename OutputT,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractLeft(T (&input)[ITEMS_PER_THREAD],
                 OutputT (&output)[ITEMS_PER_THREAD],
                 DifferenceOpT difference_op,
                 T tile_predecessor_item)
    {
      // Share last item
      temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

      CTA_SYNC();

      #pragma unroll
      for (int item = ITEMS_PER_THREAD - 1; item > 0; item--)
      {
        output[item] = difference_op(input[item], input[item - 1]);
      }

      // Set flag for first thread-item
      if (linear_tid == 0)
      {
        output[0] = difference_op(input[0], tile_predecessor_item);
      }
      else
      {
        output[0] = difference_op(input[0],
                                  temp_storage.last_items[linear_tid - 1]);
      }
    }

    /**
     * @brief Subtracts the left element of each adjacent pair of elements 
     *        partitioned across a CUDA thread block.
     *
     * @par
     * - \rowmajor
     * - \smemreuse
     *
     * @par Snippet
     * The code snippet below illustrates how to use @p BlockAdjacentDifference 
     * to compute the left difference between adjacent elements.
     *
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *   // Specialize BlockAdjacentDifference for a 1D block of 
     *   // 128 threads of type int
     *   using BlockAdjacentDifferenceT =
     *      cub::BlockAdjacentDifference<int, 128>;
     *
     *   // Allocate shared memory for BlockDiscontinuity
     *   __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *   // Obtain a segment of consecutive items that are blocked across threads
     *   int thread_data[4];
     *   ...
     *   int valid_items = 9;
     *
     *   // Collectively compute adjacent_difference
     *   BlockAdjacentDifferenceT(temp_storage).SubtractLeftPartialTile(
     *       thread_data,
     *       thread_data,
     *       CustomDifference(),
     *       valid_items);
     *
     * @endcode
     * @par
     * Suppose the set of input `thread_data` across the block of threads is
     * `{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }`.
     * The corresponding output `result` in those threads will be
     * `{ [4,-2,-1,0], [0,0,0,0], [1,3,3,3], [3,4,1,4], ... }`.
     *
     * @param[out] output
     *   Calling thread's adjacent difference result
     *
     * @param[in] input
     *   Calling thread's input items (may be aliased to \p output)
     *
     * @param[in] difference_op
     *   Binary difference operator
     *
     * @param[in] valid_items
     *   Number of valid items in thread block
     */
    template <int ITEMS_PER_THREAD,
              typename OutputType,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractLeftPartialTile(T (&input)[ITEMS_PER_THREAD],
                            OutputType (&output)[ITEMS_PER_THREAD],
                            DifferenceOpT difference_op,
                            int valid_items)
    {
      // Share last item
      temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

      CTA_SYNC();

      if ((linear_tid + 1) * ITEMS_PER_THREAD <= valid_items)
      {
        #pragma unroll
        for (int item = ITEMS_PER_THREAD - 1; item > 0; item--)
        {
          output[item] = difference_op(input[item], input[item - 1]);
        }
      }
      else
      {
        #pragma unroll
        for (int item = ITEMS_PER_THREAD - 1; item > 0; item--)
        {
          const int idx = linear_tid * ITEMS_PER_THREAD + item;

          if (idx < valid_items)
          {
            output[item] = difference_op(input[item], input[item - 1]);
          }
          else
          {
            output[item] = input[item];
          }
        }
      }

      if (linear_tid == 0 || valid_items <= linear_tid * ITEMS_PER_THREAD)
      {
        output[0] = input[0];
      }
      else
      {
        output[0] = difference_op(input[0],
                                  temp_storage.last_items[linear_tid - 1]);
      }
    }

    /**
     * @brief Subtracts the left element of each adjacent pair of elements 
     *        partitioned across a CUDA thread block.
     *
     * @par
     * - \rowmajor
     * - \smemreuse
     *
     * @par Snippet
     * The code snippet below illustrates how to use @p BlockAdjacentDifference 
     * to compute the left difference between adjacent elements.
     *
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *   // Specialize BlockAdjacentDifference for a 1D block of 
     *   // 128 threads of type int
     *   using BlockAdjacentDifferenceT =
     *      cub::BlockAdjacentDifference<int, 128>;
     *
     *   // Allocate shared memory for BlockDiscontinuity
     *   __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *   // Obtain a segment of consecutive items that are blocked across threads
     *   int thread_data[4];
     *   ...
     *   int valid_items = 9;
     *   int tile_predecessor_item = 4;
     *
     *   // Collectively compute adjacent_difference
     *   BlockAdjacentDifferenceT(temp_storage).SubtractLeftPartialTile(
     *       thread_data,
     *       thread_data,
     *       CustomDifference(),
     *       valid_items,
     *       tile_predecessor_item);
     *
     * @endcode
     * @par
     * Suppose the set of input `thread_data` across the block of threads is
     * `{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }`.
     * The corresponding output `result` in those threads will be
     * `{ [0,-2,-1,0], [0,0,0,0], [1,3,3,3], [3,4,1,4], ... }`.
     *
     * @param[out] output
     *   Calling thread's adjacent difference result
     *
     * @param[in] input
     *   Calling thread's input items (may be aliased to \p output)
     *
     * @param[in] difference_op
     *   Binary difference operator
     *
     * @param[in] valid_items
     *   Number of valid items in thread block
     *
     * @param[in] tile_predecessor_item
     *   **[<em>thread</em><sub>0</sub> only]** item which is going to be
     *   subtracted from the first tile item (<tt>input<sub>0</sub></tt> from
     *   <em>thread</em><sub>0</sub>).
     */
    template <int ITEMS_PER_THREAD,
              typename OutputType,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractLeftPartialTile(T (&input)[ITEMS_PER_THREAD],
                            OutputType (&output)[ITEMS_PER_THREAD],
                            DifferenceOpT difference_op,
                            int valid_items,
                            T tile_predecessor_item)
    {
      // Share last item
      temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

      CTA_SYNC();

      if ((linear_tid + 1) * ITEMS_PER_THREAD <= valid_items)
      {
        #pragma unroll
        for (int item = ITEMS_PER_THREAD - 1; item > 0; item--)
        {
          output[item] = difference_op(input[item], input[item - 1]);
        }
      }
      else
      {
        #pragma unroll
        for (int item = ITEMS_PER_THREAD - 1; item > 0; item--)
        {
          const int idx = linear_tid * ITEMS_PER_THREAD + item;

          if (idx < valid_items)
          {
            output[item] = difference_op(input[item], input[item - 1]);
          }
          else
          {
            output[item] = input[item];
          }
        }
      }

      if (valid_items <= linear_tid * ITEMS_PER_THREAD)
      {
        output[0] = input[0];
      }
      else if (linear_tid == 0) 
      {
        output[0] = difference_op(input[0], 
                                  tile_predecessor_item);
      }
      else
      {
        output[0] = difference_op(input[0],
                                  temp_storage.last_items[linear_tid - 1]);
      }
    }

    //@}  end member group
    /******************************************************************//**
     * @name Read right operations
     *********************************************************************/
    //@{

    /**
     * @brief Subtracts the right element of each adjacent pair of elements
     *        partitioned across a CUDA thread block.
     *
     * @par
     * - \rowmajor
     * - \smemreuse
     *
     * @par Snippet
     * The code snippet below illustrates how to use @p BlockAdjacentDifference
     * to compute the right difference between adjacent elements.
     *
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockAdjacentDifference for a 1D block of
     *     // 128 threads of type int
     *     using BlockAdjacentDifferenceT =
     *        cub::BlockAdjacentDifference<int, 128>;
     *
     *     // Allocate shared memory for BlockDiscontinuity
     *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute adjacent_difference
     *     BlockAdjacentDifferenceT(temp_storage).SubtractRight(
     *         thread_data,
     *         thread_data,
     *         CustomDifference());
     *
     * @endcode
     * @par
     * Suppose the set of input `thread_data` across the block of threads is
     * `{ ...3], [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4] }`.
     * The corresponding output `result` in those threads will be
     * `{ ..., [-1,2,1,0], [0,0,0,-1], [-1,0,0,0], [-1,3,-3,4] }`.
     *
     * @param[out] output
     *   Calling thread's adjacent difference result
     *
     * @param[in] input
     *   Calling thread's input items (may be aliased to \p output)
     *
     * @param[in] difference_op
     *   Binary difference operator
     */
    template <int ITEMS_PER_THREAD,
              typename OutputT,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractRight(T (&input)[ITEMS_PER_THREAD],
                  OutputT (&output)[ITEMS_PER_THREAD],
                  DifferenceOpT difference_op)
    {
      // Share first item
      temp_storage.first_items[linear_tid] = input[0];

      CTA_SYNC();

      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD - 1; item++)
      {
        output[item] = difference_op(input[item], input[item + 1]);
      }

      if (linear_tid == BLOCK_THREADS - 1)
      {
        output[ITEMS_PER_THREAD - 1] = input[ITEMS_PER_THREAD - 1];
      }
      else
      {
        output[ITEMS_PER_THREAD - 1] =
          difference_op(input[ITEMS_PER_THREAD - 1],
                        temp_storage.first_items[linear_tid + 1]);
      }
    }

    /**
     * @brief Subtracts the right element of each adjacent pair of elements
     *        partitioned across a CUDA thread block.
     *
     * @par
     * - \rowmajor
     * - \smemreuse
     *
     * @par Snippet
     * The code snippet below illustrates how to use @p BlockAdjacentDifference
     * to compute the right difference between adjacent elements.
     *
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockAdjacentDifference for a 1D block of
     *     // 128 threads of type int
     *     using BlockAdjacentDifferenceT =
     *        cub::BlockAdjacentDifference<int, 128>;
     *
     *     // Allocate shared memory for BlockDiscontinuity
     *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // The first item in the nest tile:
     *     int tile_successor_item = ...;
     *
     *     // Collectively compute adjacent_difference
     *     BlockAdjacentDifferenceT(temp_storage).SubtractRight(
     *         thread_data,
     *         thread_data,
     *         CustomDifference(),
     *         tile_successor_item);
     *
     * @endcode
     * @par
     * Suppose the set of input `thread_data` across the block of threads is
     * `{ ...3], [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4] }`,
     * and that `tile_successor_item` is `3`. The corresponding output `result`
     * in those threads will be
     * `{ ..., [-1,2,1,0], [0,0,0,-1], [-1,0,0,0], [-1,3,-3,1] }`.
     *
     * @param[out] output
     *   Calling thread's adjacent difference result
     *
     * @param[in] input
     *   Calling thread's input items (may be aliased to @p output)
     *
     * @param[in] difference_op
     *   Binary difference operator
     *
     * @param[in] tile_successor_item
     *   <b>[<em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub> only]</b> item
     *   which is going to be subtracted from the last tile item
     *   (<tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> from
     *   <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>).
     */
    template <int ITEMS_PER_THREAD,
              typename OutputT,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractRight(T (&input)[ITEMS_PER_THREAD],
                  OutputT (&output)[ITEMS_PER_THREAD],
                  DifferenceOpT difference_op,
                  T tile_successor_item)
    {
      // Share first item
      temp_storage.first_items[linear_tid] = input[0];

      CTA_SYNC();

      // Set flag for last thread-item
      T successor_item = (linear_tid == BLOCK_THREADS - 1)
                           ? tile_successor_item // Last thread
                           : temp_storage.first_items[linear_tid + 1];

      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD - 1; item++)
      {
        output[item] = difference_op(input[item], input[item + 1]);
      }

      output[ITEMS_PER_THREAD - 1] =
        difference_op(input[ITEMS_PER_THREAD - 1], successor_item);
    }

    /**
     * @brief Subtracts the right element of each adjacent pair in range of
     *        elements partitioned across a CUDA thread block.
     *
     * @par
     * - \rowmajor
     * - \smemreuse
     *
     * @par Snippet
     * The code snippet below illustrates how to use @p BlockAdjacentDifference to
     * compute the right difference between adjacent elements.
     *
     * @par
     * @code
     * #include <cub/cub.cuh>
     * // or equivalently <cub/block/block_adjacent_difference.cuh>
     *
     * struct CustomDifference
     * {
     *   template <typename DataType>
     *   __device__ DataType operator()(DataType &lhs, DataType &rhs)
     *   {
     *     return lhs - rhs;
     *   }
     * };
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockAdjacentDifference for a 1D block of
     *     // 128 threads of type int
     *     using BlockAdjacentDifferenceT =
     *        cub::BlockAdjacentDifference<int, 128>;
     *
     *     // Allocate shared memory for BlockDiscontinuity
     *     __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute adjacent_difference
     *     BlockAdjacentDifferenceT(temp_storage).SubtractRightPartialTile(
     *         thread_data,
     *         thread_data,
     *         CustomDifference(),
     *         valid_items);
     *
     * @endcode
     * @par
     * Suppose the set of input `thread_data` across the block of threads is
     * `{ ...3], [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4] }`.
     * and that `valid_items` is `507`. The corresponding output `result` in
     * those threads will be
     * `{ ..., [-1,2,1,0], [0,0,0,-1], [-1,0,3,3], [3,4,1,4] }`.
     *
     * @param[out] output
     *   Calling thread's adjacent difference result
     *
     * @param[in] input
     *   Calling thread's input items (may be aliased to @p output)
     *
     * @param[in] difference_op
     *   Binary difference operator
     *
     * @param[in] valid_items
     *   Number of valid items in thread block
     */
    template <int ITEMS_PER_THREAD,
              typename OutputT,
              typename DifferenceOpT>
    __device__ __forceinline__ void
    SubtractRightPartialTile(T (&input)[ITEMS_PER_THREAD],
                             OutputT (&output)[ITEMS_PER_THREAD],
                             DifferenceOpT difference_op,
                             int valid_items)
    {
      // Share first item
      temp_storage.first_items[linear_tid] = input[0];

      CTA_SYNC();

      if ((linear_tid + 1) * ITEMS_PER_THREAD < valid_items)
      {
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD - 1; item++)
        {
           output[item] = difference_op(input[item], input[item + 1]);
        }

        output[ITEMS_PER_THREAD - 1] =
          difference_op(input[ITEMS_PER_THREAD - 1],
                        temp_storage.first_items[linear_tid + 1]);
      }
      else
      {
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; item++)
        {
          const int idx = linear_tid * ITEMS_PER_THREAD + item;

          // Right element of input[valid_items - 1] is out of bounds.
          // According to the API it's copied into output array
          // without modification.
          if (idx < valid_items - 1)
          {
            output[item] = difference_op(input[item], input[item + 1]);
          }
          else
          {
            output[item] = input[item];
          }
        }
      }
    }

    //@}  end member group
    /******************************************************************//**
     * @name Head flag operations (deprecated)
     *********************************************************************/
    //@{

    #ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeads
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft instead.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeads(
        FlagT           (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&preds)[ITEMS_PER_THREAD],     ///< [out] Calling thread's predecessor items
        FlagOp          flag_op)                        ///< [in] Binary boolean flag predicate
    {
        // Share last item
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        if (linear_tid == 0)
        {
            // Set flag for first thread-item (preds[0] is undefined)
            output[0] = 1;
        }
        else
        {
            preds[0] = temp_storage.last_items[linear_tid - 1];
            output[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);
        }

        // Set output for remaining items
        Iterate::FlagHeads(linear_tid, output, input, preds, flag_op);
    }

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeads
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft instead.
     */
    template <int             ITEMS_PER_THREAD,
              typename        FlagT,
              typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeads(
        FlagT           (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity result
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        T               (&preds)[ITEMS_PER_THREAD],     ///< [out] Calling thread's predecessor items
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        T               tile_predecessor_item)          ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
    {
        // Share last item
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        // Set flag for first thread-item
        preds[0] = (linear_tid == 0) ?
            tile_predecessor_item :              // First thread
            temp_storage.last_items[linear_tid - 1];

        output[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);

        // Set output for remaining items
        Iterate::FlagHeads(linear_tid, output, input, preds, flag_op);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeads
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft instead.
     */
    template <int ITEMS_PER_THREAD,
              typename FlagT,
              typename FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void
    FlagHeads(FlagT           (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity result
              T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
              FlagOp          flag_op)                        ///< [in] Binary boolean flag predicate
    {
        T preds[ITEMS_PER_THREAD];
        FlagHeads(output, input, preds, flag_op);
    }

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeads
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft instead.
     */
    template <int ITEMS_PER_THREAD,
              typename FlagT,
              typename FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void
    FlagHeads(FlagT           (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity result
              T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
              FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
              T               tile_predecessor_item)          ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
    {
        T preds[ITEMS_PER_THREAD];
        FlagHeads(output, input, preds, flag_op, tile_predecessor_item);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagTails(
        FlagT           (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity result
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        FlagOp          flag_op)                        ///< [in] Binary boolean flag predicate
    {
        // Share first item
        temp_storage.first_items[linear_tid] = input[0];

        CTA_SYNC();

        // Set flag for last thread-item
        output[ITEMS_PER_THREAD - 1] = (linear_tid == BLOCK_THREADS - 1) ?
            1 :                             // Last thread
            ApplyOp<FlagOp>::FlagT(
                flag_op,
                input[ITEMS_PER_THREAD - 1],
                temp_storage.first_items[linear_tid + 1],
                (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set output for remaining items
        Iterate::FlagTails(linear_tid, output, input, flag_op);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagTails(
        FlagT           (&output)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity result
        T               (&input)[ITEMS_PER_THREAD],     ///< [in] Calling thread's input items
        FlagOp          flag_op,                        ///< [in] Binary boolean flag predicate
        T               tile_successor_item)            ///< [in] <b>[<em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub> only]</b> Item with which to compare the last tile item (<tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> from <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>).
    {
        // Share first item
        temp_storage.first_items[linear_tid] = input[0];

        CTA_SYNC();

        // Set flag for last thread-item
        T successor_item = (linear_tid == BLOCK_THREADS - 1) ?
            tile_successor_item :              // Last thread
            temp_storage.first_items[linear_tid + 1];

        output[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            input[ITEMS_PER_THREAD - 1],
            successor_item,
            (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set output for remaining items
        Iterate::FlagTails(linear_tid, output, input, flag_op);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeadsAndTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft or
     * cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeadsAndTails(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first and last items
        temp_storage.first_items[linear_tid] = input[0];
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        T preds[ITEMS_PER_THREAD];

        // Set flag for first thread-item
        preds[0] = temp_storage.last_items[linear_tid - 1];
        if (linear_tid == 0)
        {
            head_flags[0] = 1;
        }
        else
        {
            head_flags[0] = ApplyOp<FlagOp>::FlagT(
                flag_op,
                preds[0],
                input[0],
                linear_tid * ITEMS_PER_THREAD);
        }


        // Set flag for last thread-item
        tail_flags[ITEMS_PER_THREAD - 1] = (linear_tid == BLOCK_THREADS - 1) ?
            1 :                             // Last thread
            ApplyOp<FlagOp>::FlagT(
                flag_op,
                input[ITEMS_PER_THREAD - 1],
                temp_storage.first_items[linear_tid + 1],
                (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        Iterate::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

        // Set tail_flags for remaining items
        Iterate::FlagTails(linear_tid, tail_flags, input, flag_op);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeadsAndTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft or
     * cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeadsAndTails(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               tile_successor_item,                ///< [in] <b>[<em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub> only]</b> Item with which to compare the last tile item (<tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> from <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>).
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first and last items
        temp_storage.first_items[linear_tid] = input[0];
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        T preds[ITEMS_PER_THREAD];

        // Set flag for first thread-item
        if (linear_tid == 0)
        {
            head_flags[0] = 1;
        }
        else
        {
            preds[0] = temp_storage.last_items[linear_tid - 1];
            head_flags[0] = ApplyOp<FlagOp>::FlagT(
                flag_op,
                preds[0],
                input[0],
                linear_tid * ITEMS_PER_THREAD);
        }

        // Set flag for last thread-item
        T successor_item = (linear_tid == BLOCK_THREADS - 1) ?
            tile_successor_item :              // Last thread
            temp_storage.first_items[linear_tid + 1];

        tail_flags[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            input[ITEMS_PER_THREAD - 1],
            successor_item,
            (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        Iterate::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

        // Set tail_flags for remaining items
        Iterate::FlagTails(linear_tid, tail_flags, input, flag_op);
    }

    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeadsAndTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft or
     * cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
      int             ITEMS_PER_THREAD,
      typename        FlagT,
      typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeadsAndTails(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        T               tile_predecessor_item,              ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first and last items
        temp_storage.first_items[linear_tid] = input[0];
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        T preds[ITEMS_PER_THREAD];

        // Set flag for first thread-item
        preds[0] = (linear_tid == 0) ?
            tile_predecessor_item :              // First thread
            temp_storage.last_items[linear_tid - 1];

        head_flags[0] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            preds[0],
            input[0],
            linear_tid * ITEMS_PER_THREAD);

        // Set flag for last thread-item
        tail_flags[ITEMS_PER_THREAD - 1] = (linear_tid == BLOCK_THREADS - 1) ?
            1 :                             // Last thread
            ApplyOp<FlagOp>::FlagT(
                flag_op,
                input[ITEMS_PER_THREAD - 1],
                temp_storage.first_items[linear_tid + 1],
                (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        Iterate::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

        // Set tail_flags for remaining items
        Iterate::FlagTails(linear_tid, tail_flags, input, flag_op);
    }


    /**
     * \deprecated [Since 1.14.0] The cub::BlockAdjacentDifference::FlagHeadsAndTails
     * APIs are deprecated. Use cub::BlockAdjacentDifference::SubtractLeft or
     * cub::BlockAdjacentDifference::SubtractRight instead.
     */
    template <
        int             ITEMS_PER_THREAD,
        typename        FlagT,
        typename        FlagOp>
    CUB_DEPRECATED __device__ __forceinline__ void FlagHeadsAndTails(
        FlagT           (&head_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity head_flags
        T               tile_predecessor_item,              ///< [in] <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
        FlagT           (&tail_flags)[ITEMS_PER_THREAD],    ///< [out] Calling thread's discontinuity tail_flags
        T               tile_successor_item,                ///< [in] <b>[<em>thread</em><sub><tt>BLOCK_THREADS</tt>-1</sub> only]</b> Item with which to compare the last tile item (<tt>input</tt><sub><em>ITEMS_PER_THREAD</em>-1</sub> from <em>thread</em><sub><em>BLOCK_THREADS</em>-1</sub>).
        T               (&input)[ITEMS_PER_THREAD],         ///< [in] Calling thread's input items
        FlagOp          flag_op)                            ///< [in] Binary boolean flag predicate
    {
        // Share first and last items
        temp_storage.first_items[linear_tid] = input[0];
        temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

        CTA_SYNC();

        T preds[ITEMS_PER_THREAD];

        // Set flag for first thread-item
        preds[0] = (linear_tid == 0) ?
            tile_predecessor_item :              // First thread
            temp_storage.last_items[linear_tid - 1];

        head_flags[0] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            preds[0],
            input[0],
            linear_tid * ITEMS_PER_THREAD);

        // Set flag for last thread-item
        T successor_item = (linear_tid == BLOCK_THREADS - 1) ?
            tile_successor_item :              // Last thread
            temp_storage.first_items[linear_tid + 1];

        tail_flags[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::FlagT(
            flag_op,
            input[ITEMS_PER_THREAD - 1],
            successor_item,
            (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

        // Set head_flags for remaining items
        Iterate::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

        // Set tail_flags for remaining items
        Iterate::FlagTails(linear_tid, tail_flags, input, flag_op);
    }

};


CUB_NAMESPACE_END
