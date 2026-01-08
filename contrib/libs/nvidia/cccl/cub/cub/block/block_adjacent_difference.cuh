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

//! @file
//! The cub::BlockAdjacentDifference class provides collective methods for computing the differences of adjacent
//! elements partitioned across a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

//! @rst
//! BlockAdjacentDifference provides :ref:`collective <collective-primitives>` methods for computing the
//! differences of adjacent elements partitioned across a CUDA thread block.
//!
//! Overview
//! ++++++++++++++++
//!
//! BlockAdjacentDifference calculates the differences of adjacent elements in the elements partitioned across a CUDA
//! thread block. Because the binary operation could be noncommutative, there are two sets of methods.
//! Methods named SubtractLeft subtract left element ``i - 1`` of input sequence from current element ``i``.
//! Methods named SubtractRight subtract the right element ``i + 1`` from the current one ``i``:
//!
//! .. code-block:: c++
//!
//!    int values[4]; // [1, 2, 3, 4]
//!    //...
//!    int subtract_left_result[4];  <-- [  1,  1,  1,  1 ]
//!    int subtract_right_result[4]; <-- [ -1, -1, -1,  4 ]
//!
//! - For SubtractLeft, if the left element is out of bounds, the input value is assigned to ``output[0]``
//!   without modification.
//! - For SubtractRight, if the right element is out of bounds, the input value is assigned to the current output value
//!   without modification.
//! - The block/example_block_reduce_dyn_smem.cu example under the examples/block folder illustrates usage of
//!   dynamically shared memory with BlockReduce and how to re-purpose the same memory region.
//!   This example can be easily adapted to the storage required by BlockAdjacentDifference.
//!
//! A Simple Example
//! ++++++++++++++++
//!
//! The code snippet below illustrates how to use BlockAdjacentDifference to
//! compute the left difference between adjacent elements.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!    // or equivalently <cub/block/block_adjacent_difference.cuh>
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
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize BlockAdjacentDifference for a 1D block of
//!        // 128 threads of type int
//!        using BlockAdjacentDifferenceT =
//!           cub::BlockAdjacentDifference<int, 128>;
//!
//!        // Allocate shared memory for BlockAdjacentDifference
//!        __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
//!
//!        // Obtain a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        ...
//!
//!        // Collectively compute adjacent_difference
//!        int result[4];
//!
//!        BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
//!            thread_data,
//!            result,
//!            CustomDifference());
//!
//! Suppose the set of input `thread_data` across the block of threads is
//! ``{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }``.
//! The corresponding output ``result`` in those threads will be
//! ``{ [4,-2,-1,0], [0,0,0,0], [1,1,0,0], [0,1,-3,3], ... }``.
//!
//! @endrst
template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1>
class BlockAdjacentDifference
{
private:
  /// The thread block size in threads
  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

  /// Shared memory storage layout type (last element from each thread's input)
  struct _TempStorage
  {
    T first_items[BLOCK_THREADS];
    T last_items[BLOCK_THREADS];
  };

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  /// Specialization for when FlagOp has third index param
  template <typename FlagOp, bool HAS_PARAM = BinaryOpHasIdxParam<T, FlagOp>::value>
  struct ApplyOp
  {
    // Apply flag operator
    static _CCCL_DEVICE _CCCL_FORCEINLINE T FlagT(FlagOp flag_op, const T& a, const T& b, int idx)
    {
      return flag_op(b, a, idx);
    }
  };

  /// Specialization for when FlagOp does not have a third index param
  template <typename FlagOp>
  struct ApplyOp<FlagOp, false>
  {
    // Apply flag operator
    static _CCCL_DEVICE _CCCL_FORCEINLINE T FlagT(FlagOp flag_op, const T& a, const T& b, int /*idx*/)
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
    static _CCCL_DEVICE _CCCL_FORCEINLINE void FlagHeads(
      int linear_tid,
      FlagT (&flags)[ITEMS_PER_THREAD],
      T (&input)[ITEMS_PER_THREAD],
      T (&preds)[ITEMS_PER_THREAD],
      FlagOp flag_op)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < ITEMS_PER_THREAD; ++i)
      {
        preds[i] = input[i - 1];
        flags[i] = ApplyOp<FlagOp>::FlagT(flag_op, preds[i], input[i], (linear_tid * ITEMS_PER_THREAD) + i);
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
    static _CCCL_DEVICE _CCCL_FORCEINLINE void
    FlagTails(int linear_tid, FlagT (&flags)[ITEMS_PER_THREAD], T (&input)[ITEMS_PER_THREAD], FlagOp flag_op)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < ITEMS_PER_THREAD - 1; ++i)
      {
        flags[i] = ApplyOp<FlagOp>::FlagT(flag_op, input[i], input[i + 1], (linear_tid * ITEMS_PER_THREAD) + i + 1);
      }
    }
  };

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  unsigned int linear_tid;

public:
  /// @smemstorage{BlockAdjacentDifference}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared memory as temporary storage
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockAdjacentDifference()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @brief Collective constructor using the specified memory allocation as temporary storage
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockAdjacentDifference(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @} end member group
  //! @name Read left operations
  //! @{

  //! @rst
  //! Subtracts the left element of each adjacent pair of elements partitioned across a CUDA thread block.
  //!
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates how to use BlockAdjacentDifference to compute the left difference between
  //! adjacent elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/block/block_adjacent_difference.cuh>
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
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockAdjacentDifference for a 1D block
  //!        // of 128 threads of type int
  //!        using BlockAdjacentDifferenceT =
  //!           cub::BlockAdjacentDifference<int, 128>;
  //!
  //!        // Allocate shared memory for BlockAdjacentDifference
  //!        __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute adjacent_difference
  //!        BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
  //!            thread_data,
  //!            thread_data,
  //!            CustomDifference());
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }``.
  //! The corresponding output ``result`` in those threads will be
  //! ``{ [4,-2,-1,0], [0,0,0,0], [1,1,0,0], [0,1,-3,3], ... }``.
  //! @endrst
  //!
  //! @param[out] output
  //!   Calling thread's adjacent difference result
  //!
  //! @param[in] input
  //!   Calling thread's input items (may be aliased to `output`)
  //!
  //! @param[in] difference_op
  //!   Binary difference operator
  template <int ITEMS_PER_THREAD, typename OutputType, typename DifferenceOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  SubtractLeft(T (&input)[ITEMS_PER_THREAD], OutputType (&output)[ITEMS_PER_THREAD], DifferenceOpT difference_op)
  {
    // Share last item
    temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
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
      output[0] = difference_op(input[0], temp_storage.last_items[linear_tid - 1]);
    }
  }

  //! @rst
  //! Subtracts the left element of each adjacent pair of elements partitioned across a CUDA thread block.
  //!
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates how to use BlockAdjacentDifference to compute the left difference between
  //! adjacent elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/block/block_adjacent_difference.cuh>
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
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockAdjacentDifference for a 1D block of
  //!        // 128 threads of type int
  //!        using BlockAdjacentDifferenceT =
  //!           cub::BlockAdjacentDifference<int, 128>;
  //!
  //!        // Allocate shared memory for BlockAdjacentDifference
  //!        __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // The last item in the previous tile:
  //!        int tile_predecessor_item = ...;
  //!
  //!        // Collectively compute adjacent_difference
  //!        BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
  //!            thread_data,
  //!            thread_data,
  //!            CustomDifference(),
  //!            tile_predecessor_item);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }``.
  //! and that `tile_predecessor_item` is `3`. The corresponding output
  //! ``result`` in those threads will be
  //! ``{ [1,-2,-1,0], [0,0,0,0], [1,1,0,0], [0,1,-3,3], ... }``.
  //! @endrst
  //!
  //! @param[out] output
  //!   Calling thread's adjacent difference result
  //!
  //! @param[in] input
  //!   Calling thread's input items (may be aliased to `output`)
  //!
  //! @param[in] difference_op
  //!   Binary difference operator
  //!
  //! @param[in] tile_predecessor_item
  //!   @rst
  //!   *thread*\ :sub:`0` only item which is going to be subtracted from the first tile item
  //!   (*input*\ :sub:`0` from *thread*\ :sub:`0`).
  //!   @endrst
  template <int ITEMS_PER_THREAD, typename OutputT, typename DifferenceOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SubtractLeft(
    T (&input)[ITEMS_PER_THREAD],
    OutputT (&output)[ITEMS_PER_THREAD],
    DifferenceOpT difference_op,
    T tile_predecessor_item)
  {
    // Share last item
    temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
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
      output[0] = difference_op(input[0], temp_storage.last_items[linear_tid - 1]);
    }
  }

  //! @rst
  //! Subtracts the left element of each adjacent pair of elements partitioned across a CUDA thread block.
  //!
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates how to use BlockAdjacentDifference to compute the left difference between
  //! adjacent elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/block/block_adjacent_difference.cuh>
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
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!      // Specialize BlockAdjacentDifference for a 1D block of
  //!      // 128 threads of type int
  //!      using BlockAdjacentDifferenceT =
  //!         cub::BlockAdjacentDifference<int, 128>;
  //!
  //!      // Allocate shared memory for BlockAdjacentDifference
  //!      __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
  //!
  //!      // Obtain a segment of consecutive items that are blocked across threads
  //!      int thread_data[4];
  //!      ...
  //!      int valid_items = 9;
  //!
  //!      // Collectively compute adjacent_difference
  //!      BlockAdjacentDifferenceT(temp_storage).SubtractLeftPartialTile(
  //!          thread_data,
  //!          thread_data,
  //!          CustomDifference(),
  //!          valid_items);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }``.
  //! The corresponding output ``result`` in those threads will be
  //! ``{ [4,-2,-1,0], [0,0,0,0], [1,3,3,3], [3,4,1,4], ... }``.
  //! @endrst
  //!
  //! @param[out] output
  //!   Calling thread's adjacent difference result
  //!
  //! @param[in] input
  //!   Calling thread's input items (may be aliased to `output`)
  //!
  //! @param[in] difference_op
  //!   Binary difference operator
  //!
  //! @param[in] valid_items
  //!   Number of valid items in thread block
  template <int ITEMS_PER_THREAD, typename OutputType, typename DifferenceOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SubtractLeftPartialTile(
    T (&input)[ITEMS_PER_THREAD], OutputType (&output)[ITEMS_PER_THREAD], DifferenceOpT difference_op, int valid_items)
  {
    // Share last item
    temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    if ((linear_tid + 1) * ITEMS_PER_THREAD <= valid_items)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = ITEMS_PER_THREAD - 1; item > 0; item--)
      {
        output[item] = difference_op(input[item], input[item - 1]);
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
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
      output[0] = difference_op(input[0], temp_storage.last_items[linear_tid - 1]);
    }
  }

  //! @rst
  //! Subtracts the left element of each adjacent pair of elements partitioned across a CUDA thread block.
  //!
  //! - @rowmajor
  //! - @smemreuse
  //!
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates how to use BlockAdjacentDifference to compute the left difference between
  //! adjacent elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/block/block_adjacent_difference.cuh>
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
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!      // Specialize BlockAdjacentDifference for a 1D block of
  //!      // 128 threads of type int
  //!      using BlockAdjacentDifferenceT =
  //!         cub::BlockAdjacentDifference<int, 128>;
  //!
  //!      // Allocate shared memory for BlockAdjacentDifference
  //!      __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
  //!
  //!      // Obtain a segment of consecutive items that are blocked across threads
  //!      int thread_data[4];
  //!      ...
  //!      int valid_items = 9;
  //!      int tile_predecessor_item = 4;
  //!
  //!      // Collectively compute adjacent_difference
  //!      BlockAdjacentDifferenceT(temp_storage).SubtractLeftPartialTile(
  //!          thread_data,
  //!          thread_data,
  //!          CustomDifference(),
  //!          valid_items,
  //!          tile_predecessor_item);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4], ... }``.
  //! The corresponding output ``result`` in those threads will be
  //! ``{ [0,-2,-1,0], [0,0,0,0], [1,3,3,3], [3,4,1,4], ... }``.
  //! @endrst
  //!
  //! @param[out] output
  //!   Calling thread's adjacent difference result
  //!
  //! @param[in] input
  //!   Calling thread's input items (may be aliased to `output`)
  //!
  //! @param[in] difference_op
  //!   Binary difference operator
  //!
  //! @param[in] valid_items
  //!   Number of valid items in thread block
  //!
  //! @param[in] tile_predecessor_item
  //!   @rst
  //!   *thread*\ :sub:`0` only item which is going to be subtracted from the first tile item
  //!   (*input*\ :sub:`0` from *thread*\ :sub:`0`).
  //!   @endrst
  template <int ITEMS_PER_THREAD, typename OutputType, typename DifferenceOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SubtractLeftPartialTile(
    T (&input)[ITEMS_PER_THREAD],
    OutputType (&output)[ITEMS_PER_THREAD],
    DifferenceOpT difference_op,
    int valid_items,
    T tile_predecessor_item)
  {
    // Share last item
    temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    if ((linear_tid + 1) * ITEMS_PER_THREAD <= valid_items)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = ITEMS_PER_THREAD - 1; item > 0; item--)
      {
        output[item] = difference_op(input[item], input[item - 1]);
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
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
      output[0] = difference_op(input[0], tile_predecessor_item);
    }
    else
    {
      output[0] = difference_op(input[0], temp_storage.last_items[linear_tid - 1]);
    }
  }

  //! @} end member group
  //! @name Read right operations
  //! @{
  //!
  //! @rst
  //!
  //! Subtracts the right element of each adjacent pair of elements partitioned across a CUDA thread block.
  //!
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates how to use BlockAdjacentDifference to compute the right difference between
  //! adjacent elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/block/block_adjacent_difference.cuh>
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
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockAdjacentDifference for a 1D block of
  //!        // 128 threads of type int
  //!        using BlockAdjacentDifferenceT =
  //!           cub::BlockAdjacentDifference<int, 128>;
  //!
  //!        // Allocate shared memory for BlockAdjacentDifference
  //!        __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute adjacent_difference
  //!        BlockAdjacentDifferenceT(temp_storage).SubtractRight(
  //!            thread_data,
  //!            thread_data,
  //!            CustomDifference());
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ ...3], [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4] }``.
  //! The corresponding output ``result`` in those threads will be
  //! ``{ ...-1, [2,1,0,0], [0,0,0,-1], [-1,0,0,0], [-1,3,-3,4] }``.
  //! @endrst
  //!
  //! @param[out] output
  //!   Calling thread's adjacent difference result
  //!
  //! @param[in] input
  //!   Calling thread's input items (may be aliased to `output`)
  //!
  //! @param[in] difference_op
  //!   Binary difference operator
  template <int ITEMS_PER_THREAD, typename OutputT, typename DifferenceOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  SubtractRight(T (&input)[ITEMS_PER_THREAD], OutputT (&output)[ITEMS_PER_THREAD], DifferenceOpT difference_op)
  {
    // Share first item
    temp_storage.first_items[linear_tid] = input[0];

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
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
        difference_op(input[ITEMS_PER_THREAD - 1], temp_storage.first_items[linear_tid + 1]);
    }
  }

  //! @rst
  //! Subtracts the right element of each adjacent pair of elements partitioned across a CUDA thread block.
  //!
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates how to use BlockAdjacentDifference to compute the right difference between
  //! adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/block/block_adjacent_difference.cuh>
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
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockAdjacentDifference for a 1D block of
  //!        // 128 threads of type int
  //!        using BlockAdjacentDifferenceT =
  //!           cub::BlockAdjacentDifference<int, 128>;
  //!
  //!        // Allocate shared memory for BlockAdjacentDifference
  //!        __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // The first item in the next tile:
  //!        int tile_successor_item = ...;
  //!
  //!        // Collectively compute adjacent_difference
  //!        BlockAdjacentDifferenceT(temp_storage).SubtractRight(
  //!            thread_data,
  //!            thread_data,
  //!            CustomDifference(),
  //!            tile_successor_item);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ ...3], [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4] }``,
  //! and that ``tile_successor_item`` is ``3``. The corresponding output ``result``
  //! in those threads will be
  //! ``{ ...-1, [2,1,0,0], [0,0,0,-1], [-1,0,0,0], [-1,3,-3,1] }``.
  //! @endrst
  //!
  //! @param[out] output
  //!   Calling thread's adjacent difference result
  //!
  //! @param[in] input
  //!   Calling thread's input items (may be aliased to `output`)
  //!
  //! @param[in] difference_op
  //!   Binary difference operator
  //!
  //! @param[in] tile_successor_item
  //!   @rst
  //!   *thread*\ :sub:`BLOCK_THREADS` only item which is going to be subtracted from the last tile item
  //!   (*input*\ :sub:`ITEMS_PER_THREAD` from *thread*\ :sub:`BLOCK_THREADS`).
  //!   @endrst
  template <int ITEMS_PER_THREAD, typename OutputT, typename DifferenceOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SubtractRight(
    T (&input)[ITEMS_PER_THREAD],
    OutputT (&output)[ITEMS_PER_THREAD],
    DifferenceOpT difference_op,
    T tile_successor_item)
  {
    // Share first item
    temp_storage.first_items[linear_tid] = input[0];

    __syncthreads();

    // Set flag for last thread-item
    T successor_item = (linear_tid == BLOCK_THREADS - 1)
                       ? tile_successor_item // Last thread
                       : temp_storage.first_items[linear_tid + 1];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ITEMS_PER_THREAD - 1; item++)
    {
      output[item] = difference_op(input[item], input[item + 1]);
    }

    output[ITEMS_PER_THREAD - 1] = difference_op(input[ITEMS_PER_THREAD - 1], successor_item);
  }

  //! @rst
  //! Subtracts the right element of each adjacent pair in range of elements partitioned across a CUDA thread block.
  //!
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates how to use BlockAdjacentDifference to compute the right difference between
  //! adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/block/block_adjacent_difference.cuh>
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
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockAdjacentDifference for a 1D block of
  //!        // 128 threads of type int
  //!        using BlockAdjacentDifferenceT =
  //!           cub::BlockAdjacentDifference<int, 128>;
  //!
  //!        // Allocate shared memory for BlockAdjacentDifference
  //!        __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute adjacent_difference
  //!        BlockAdjacentDifferenceT(temp_storage).SubtractRightPartialTile(
  //!            thread_data,
  //!            thread_data,
  //!            CustomDifference(),
  //!            valid_items);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ ...3], [4,2,1,1], [1,1,1,1], [2,3,3,3], [3,4,1,4] }``.
  //! and that ``valid_items`` is ``507``. The corresponding output ``result`` in
  //! those threads will be
  //! ``{ ...-1, [2,1,0,0], [0,0,0,-1], [-1,0,3,3], [3,4,1,4] }``.
  //! @endrst
  //!
  //! @param[out] output
  //!   Calling thread's adjacent difference result
  //!
  //! @param[in] input
  //!   Calling thread's input items (may be aliased to `output`)
  //!
  //! @param[in] difference_op
  //!   Binary difference operator
  //!
  //! @param[in] valid_items
  //!   Number of valid items in thread block
  template <int ITEMS_PER_THREAD, typename OutputT, typename DifferenceOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SubtractRightPartialTile(
    T (&input)[ITEMS_PER_THREAD], OutputT (&output)[ITEMS_PER_THREAD], DifferenceOpT difference_op, int valid_items)
  {
    // Share first item
    temp_storage.first_items[linear_tid] = input[0];

    __syncthreads();

    if ((linear_tid + 1) * ITEMS_PER_THREAD < valid_items)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ITEMS_PER_THREAD - 1; item++)
      {
        output[item] = difference_op(input[item], input[item + 1]);
      }

      output[ITEMS_PER_THREAD - 1] =
        difference_op(input[ITEMS_PER_THREAD - 1], temp_storage.first_items[linear_tid + 1]);
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
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
};

CUB_NAMESPACE_END
