/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * The cub::BlockDiscontinuity class provides [<em>collective</em>](../index.html#sec0) methods for
 * flagging discontinuities within an ordered set of items partitioned across a CUDA thread block.
 */

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
//! The BlockDiscontinuity class provides :ref:`collective <collective-primitives>` methods for
//! flagging discontinuities within an ordered set of items partitioned across a CUDA thread
//! block.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - A set of "head flags" (or "tail flags") is often used to indicate corresponding items
//!   that differ from their predecessors (or successors). For example, head flags are convenient
//!   for demarcating disjoint data segments as part of a segmented scan or reduction.
//! - @blocked
//!
//! Performance Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - @granularity
//! - Incurs zero bank conflicts for most types
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockDiscontinuity}
//!
//! The code snippet below illustrates the head flagging of 512 integer items that
//! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
//! where each thread owns 4 consecutive items.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_discontinuity.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize BlockDiscontinuity for a 1D block of 128 threads of type int
//!        using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
//!
//!        // Allocate shared memory for BlockDiscontinuity
//!        __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
//!
//!        // Obtain a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        ...
//!
//!        // Collectively compute head flags for discontinuities in the segment
//!        int head_flags[4];
//!        BlockDiscontinuity(temp_storage).FlagHeads(head_flags, thread_data, cub::Inequality());
//!
//! Suppose the set of input ``thread_data`` across the block of threads is
//! ``{ [0,0,1,1], [1,1,1,1], [2,3,3,3], [3,4,4,4], ... }``.
//! The corresponding output ``head_flags`` in those threads will be
//! ``{ [1,0,1,0], [0,0,0,0], [1,1,0,0], [0,1,0,0], ... }``.
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``examples/block/example_block_reduce_dyn_smem.cu`` example illustrates usage of
//! dynamically shared memory with BlockReduce and how to re-purpose the same memory region.
//! This example can be easily adapted to the storage required by BlockDiscontinuity.
//! @endrst
//!
//! @tparam T
//!   The data type to be flagged.
//!
//! @tparam BLOCK_DIM_X
//!   The thread block length in threads along the X dimension
//!
//! @tparam BLOCK_DIM_Y
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1)
//!
//! @tparam BLOCK_DIM_Z
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1>
class BlockDiscontinuity
{
private:
  enum
  {
    /// The thread block size in threads
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
  };

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
    static _CCCL_DEVICE _CCCL_FORCEINLINE bool FlagT(FlagOp flag_op, const T& a, const T& b, int idx)
    {
      return flag_op(a, b, idx);
    }
  };

  /// Specialization for when FlagOp does not have a third index param
  template <typename FlagOp>
  struct ApplyOp<FlagOp, false>
  {
    // Apply flag operator
    static _CCCL_DEVICE _CCCL_FORCEINLINE bool FlagT(FlagOp flag_op, const T& a, const T& b, int /*idx*/)
    {
      return flag_op(a, b);
    }
  };

  /// Templated unrolling of item comparison (inductive case)
  struct Iterate
  {
    /**
     * @brief Head flags
     *
     * @param[out] flags
     *   Calling thread's discontinuity head_flags
     *
     * @param[in] input
     *   Calling thread's input items
     *
     * @param[out] preds
     *   Calling thread's predecessor items
     *
     * @param[in] flag_op
     *   Binary boolean flag predicate
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
     * @brief Tail flags
     *
     * @param[out] flags
     *   Calling thread's discontinuity head_flags
     *
     * @param[in] input
     *   Calling thread's input items
     *
     * @param[in] flag_op
     *   Binary boolean flag predicate
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

  /******************************************************************************
   * Thread fields
   ******************************************************************************/

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  unsigned int linear_tid;

public:
  /// @smemstorage{BlockDiscontinuity}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  /**
   * @brief Collective constructor using a private static allocation of shared memory as temporary
   *        storage.
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockDiscontinuity()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation as temporary storage.
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockDiscontinuity(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @} end member group
  //! @name Head flag operations
  //! @{

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

  /**
   * @param[out] head_flags
   *   Calling thread's discontinuity head_flags
   *
   * @param[in] input
   *   Calling thread's input items
   *
   * @param[out] preds
   *   Calling thread's predecessor items
   *
   * @param[in] flag_op
   *   Binary boolean flag predicate
   */
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void FlagHeads(
    FlagT (&head_flags)[ITEMS_PER_THREAD], T (&input)[ITEMS_PER_THREAD], T (&preds)[ITEMS_PER_THREAD], FlagOp flag_op)
  {
    // Share last item
    temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    if (linear_tid == 0)
    {
      // Set flag for first thread-item (preds[0] is undefined)
      head_flags[0] = 1;
    }
    else
    {
      preds[0]      = temp_storage.last_items[linear_tid - 1];
      head_flags[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);
    }

    // Set head_flags for remaining items
    Iterate::FlagHeads(linear_tid, head_flags, input, preds, flag_op);
  }

  /**
   * @param[out] head_flags
   *   Calling thread's discontinuity head_flags
   *
   * @param[in] input
   *   Calling thread's input items
   *
   * @param[out] preds
   *   Calling thread's predecessor items
   *
   * @param[in] flag_op
   *   Binary boolean flag predicate
   *
   * @param[in] tile_predecessor_item
   *   <b>[<em>thread</em><sub>0</sub> only]</b> Item with which to compare the first tile item
   *   (<tt>input<sub>0</sub></tt> from <em>thread</em><sub>0</sub>).
   */
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void FlagHeads(
    FlagT (&head_flags)[ITEMS_PER_THREAD],
    T (&input)[ITEMS_PER_THREAD],
    T (&preds)[ITEMS_PER_THREAD],
    FlagOp flag_op,
    T tile_predecessor_item)
  {
    // Share last item
    temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    // Set flag for first thread-item
    preds[0] = (linear_tid == 0) ? tile_predecessor_item : // First thread
                 temp_storage.last_items[linear_tid - 1];

    head_flags[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);

    // Set head_flags for remaining items
    Iterate::FlagHeads(linear_tid, head_flags, input, preds, flag_op);
  }

#endif // _CCCL_DOXYGEN_INVOKED

  //! @rst
  //! Sets head flags indicating discontinuities between items partitioned across the thread
  //! block, for which the first item has no reference and is always flagged.
  //!
  //! - The flag ``head_flags[i]`` is set for item ``input[i]`` when ``flag_op(previous-item, input[i])`` returns
  //!   ``true`` (where ``previous-item`` is either the preceding item in the same thread or the last item in
  //!   the previous thread).
  //! - For *thread*\ :sub:`0`, item ``input[0]`` is always flagged.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the head-flagging of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_discontinuity.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockDiscontinuity for a 1D block of 128 threads of type int
  //!        using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
  //!
  //!        // Allocate shared memory for BlockDiscontinuity
  //!        __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute head flags for discontinuities in the segment
  //!        int head_flags[4];
  //!        BlockDiscontinuity(temp_storage).FlagHeads(head_flags, thread_data, cub::Inequality());
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,0,1,1], [1,1,1,1], [2,3,3,3], [3,4,4,4], ... }``.
  //! The corresponding output ``head_flags`` in those threads will be
  //! ``{ [1,0,1,0], [0,0,0,0], [1,1,0,0], [0,1,0,0], ... }``.
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread
  //!
  //! @tparam FlagT
  //!   **[inferred]** The flag type (must be an integer type)
  //!
  //! @tparam FlagOp
  //!   **[inferred]** Binary predicate functor type having member
  //!   `T operator()(const T &a, const T &b)` or member
  //!   `T operator()(const T &a, const T &b, unsigned int b_index)`, and returning `true`
  //!   if a discontinuity exists between `a` and `b`, otherwise `false`.
  //!   `b_index` is the rank of b in the aggregate tile of data.
  //!
  //! @param[out] head_flags
  //!   Calling thread's discontinuity head_flags
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[in] flag_op
  //!   Binary boolean flag predicate
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  FlagHeads(FlagT (&head_flags)[ITEMS_PER_THREAD], T (&input)[ITEMS_PER_THREAD], FlagOp flag_op)
  {
    T preds[ITEMS_PER_THREAD];
    FlagHeads(head_flags, input, preds, flag_op);
  }

  //! @rst
  //! Sets head flags indicating discontinuities between items partitioned across the thread block.
  //!
  //! - The flag ``head_flags[i]`` is set for item ``input[i]`` when ``flag_op(previous-item, input[i])``
  //!   returns ``true`` (where ``previous-item`` is either the preceding item in the same thread or the last item
  //!   in the previous thread).
  //! - For *thread*\ :sub:`0`, item ``input[0]`` is compared against ``tile_predecessor_item``.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the head-flagging of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_discontinuity.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockDiscontinuity for a 1D block of 128 threads of type int
  //!        using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
  //!
  //!        // Allocate shared memory for BlockDiscontinuity
  //!        __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Have thread0 obtain the predecessor item for the entire tile
  //!        int tile_predecessor_item;
  //!        if (threadIdx.x == 0) tile_predecessor_item == ...
  //!
  //!        // Collectively compute head flags for discontinuities in the segment
  //!        int head_flags[4];
  //!        BlockDiscontinuity(temp_storage).FlagHeads(
  //!            head_flags, thread_data, cub::Inequality(), tile_predecessor_item);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,0,1,1], [1,1,1,1], [2,3,3,3], [3,4,4,4], ... }``,
  //! and that ``tile_predecessor_item`` is ``0``.  The corresponding output ``head_flags`` in those
  //! threads will be ``{ [0,0,1,0], [0,0,0,0], [1,1,0,0], [0,1,0,0], ... }``.
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam FlagT
  //!   **[inferred]** The flag type (must be an integer type)
  //!
  //! @tparam FlagOp
  //!   **[inferred]** Binary predicate functor type having member
  //!   `T operator()(const T &a, const T &b)` or member
  //!   `T operator()(const T &a, const T &b, unsigned int b_index)`,
  //!   and returning `true` if a discontinuity exists between `a` and `b`,
  //!   otherwise `false`.  `b_index` is the rank of b in the aggregate tile of data.
  //!
  //! @param[out] head_flags
  //!   Calling thread's discontinuity `head_flags`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[in] flag_op
  //!   Binary boolean flag predicate
  //!
  //! @param[in] tile_predecessor_item
  //!   @rst
  //!   *thread*\ :sub:`0` only item with which to compare the first tile item (``input[0]`` from *thread*\ :sub:`0`).
  //!   @endrst
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void FlagHeads(
    FlagT (&head_flags)[ITEMS_PER_THREAD], T (&input)[ITEMS_PER_THREAD], FlagOp flag_op, T tile_predecessor_item)
  {
    T preds[ITEMS_PER_THREAD];
    FlagHeads(head_flags, input, preds, flag_op, tile_predecessor_item);
  }

  //! @} end member group
  //! @name Tail flag operations
  //! @{

  //! @rst
  //! Sets tail flags indicating discontinuities between items partitioned across the thread
  //! block, for which the last item has no reference and is always flagged.
  //!
  //! - The flag ``tail_flags[i]`` is set for item ``input[i]`` when
  //!   ``flag_op(input[i], next-item)``
  //!   returns ``true`` (where `next-item` is either the next item
  //!   in the same thread or the first item in the next thread).
  //! - For *thread*\ :sub:`BLOCK_THREADS - 1`, item ``input[ITEMS_PER_THREAD - 1]`` is always flagged.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the tail-flagging of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_discontinuity.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockDiscontinuity for a 1D block of 128 threads of type int
  //!        using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
  //!
  //!        // Allocate shared memory for BlockDiscontinuity
  //!        __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute tail flags for discontinuities in the segment
  //!        int tail_flags[4];
  //!        BlockDiscontinuity(temp_storage).FlagTails(tail_flags, thread_data, cub::Inequality());
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,0,1,1], [1,1,1,1], [2,3,3,3], ..., [124,125,125,125] }``.
  //! The corresponding output ``tail_flags`` in those threads will be
  //! ``{ [0,1,0,0], [0,0,0,1], [1,0,0,...], ..., [1,0,0,1] }``.
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam FlagT
  //!   **[inferred]** The flag type (must be an integer type)
  //!
  //! @tparam FlagOp
  //!   **[inferred]** Binary predicate functor type having member
  //!   `T operator()(const T &a, const T &b)` or member
  //!   `T operator()(const T &a, const T &b, unsigned int b_index)`, and returning `true`
  //!   if a discontinuity exists between `a` and `b`, otherwise `false`. `b_index` is the
  //!   rank of `b` in the aggregate tile of data.
  //!
  //! @param[out] tail_flags
  //!   Calling thread's discontinuity tail_flags
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[in] flag_op
  //!   Binary boolean flag predicate
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  FlagTails(FlagT (&tail_flags)[ITEMS_PER_THREAD], T (&input)[ITEMS_PER_THREAD], FlagOp flag_op)
  {
    // Share first item
    temp_storage.first_items[linear_tid] = input[0];

    __syncthreads();

    // Set flag for last thread-item
    tail_flags[ITEMS_PER_THREAD - 1] =
      (linear_tid == BLOCK_THREADS - 1) ? 1 : // Last thread
        ApplyOp<FlagOp>::FlagT(
          flag_op,
          input[ITEMS_PER_THREAD - 1],
          temp_storage.first_items[linear_tid + 1],
          (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

    // Set tail_flags for remaining items
    Iterate::FlagTails(linear_tid, tail_flags, input, flag_op);
  }

  //! @rst
  //! Sets tail flags indicating discontinuities between items partitioned across the thread block.
  //!
  //! - The flag ``tail_flags[i]`` is set for item ``input[i]`` when ``flag_op(input[i], next-item)``
  //!   returns ``true`` (where ``next-item`` is either the next item in the same thread or the first item in
  //!   the next thread).
  //! - For *thread*\ :sub:`BLOCK_THREADS - 1`, item ``input[ITEMS_PER_THREAD - 1]`` is compared against
  //!   ``tile_successor_item``.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the tail-flagging of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_discontinuity.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockDiscontinuity for a 1D block of 128 threads of type int
  //!        using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
  //!
  //!        // Allocate shared memory for BlockDiscontinuity
  //!        __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Have thread127 obtain the successor item for the entire tile
  //!        int tile_successor_item;
  //!        if (threadIdx.x == 127) tile_successor_item == ...
  //!
  //!        // Collectively compute tail flags for discontinuities in the segment
  //!        int tail_flags[4];
  //!        BlockDiscontinuity(temp_storage).FlagTails(
  //!            tail_flags, thread_data, cub::Inequality(), tile_successor_item);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,0,1,1], [1,1,1,1], [2,3,3,3], ..., [124,125,125,125] }``
  //! and that ``tile_successor_item`` is ``125``.  The corresponding output ``tail_flags`` in those
  //! threads will be ``{ [0,1,0,0], [0,0,0,1], [1,0,0,...], ..., [1,0,0,0] }``.
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam FlagT
  //!   **[inferred]** The flag type (must be an integer type)
  //!
  //! @tparam FlagOp
  //!   **[inferred]** Binary predicate functor type having member
  //!   `T operator()(const T &a, const T &b)` or member
  //!   `T operator()(const T &a, const T &b, unsigned int b_index)`, and returning `true`
  //!   if a discontinuity exists between `a` and `b`, otherwise `false`. `b_index` is the
  //!   rank of `b` in the aggregate tile of data.
  //!
  //! @param[out] tail_flags
  //!   Calling thread's discontinuity tail_flags
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[in] flag_op
  //!   Binary boolean flag predicate
  //!
  //! @param[in] tile_successor_item
  //!   @rst
  //!   *thread*\ :sub:`BLOCK_THREADS - 1` only item with which to
  //!   compare the last tile item (``input[ITEMS_PER_THREAD - 1]`` from
  //!   *thread*\ :sub:`BLOCK_THREADS - 1`).
  //!   @endrst
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  FlagTails(FlagT (&tail_flags)[ITEMS_PER_THREAD], T (&input)[ITEMS_PER_THREAD], FlagOp flag_op, T tile_successor_item)
  {
    // Share first item
    temp_storage.first_items[linear_tid] = input[0];

    __syncthreads();

    // Set flag for last thread-item
    T successor_item = (linear_tid == BLOCK_THREADS - 1) ? tile_successor_item : // Last thread
                         temp_storage.first_items[linear_tid + 1];

    tail_flags[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::FlagT(
      flag_op, input[ITEMS_PER_THREAD - 1], successor_item, (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

    // Set tail_flags for remaining items
    Iterate::FlagTails(linear_tid, tail_flags, input, flag_op);
  }

  //! @} end member group
  //! @name Head & tail flag operations
  //! @{

  //! @rst
  //! Sets both head and tail flags indicating discontinuities between items partitioned across the thread block.
  //!
  //! - The flag ``head_flags[i]`` is set for item ``input[i]`` when ``flag_op(previous-item, input[i])`` returns
  //!   ``true`` (where ``previous-item`` is either the preceding item in the same thread or the last item in
  //!   the previous thread).
  //! - For *thread*\ :sub:`0`, item ``input[0]`` is always flagged.
  //! - The flag ``tail_flags[i]`` is set for item ``input[i]`` when ``flag_op(input[i], next-item)``
  //!   returns ``true`` (where next-item is either the next item in the same thread or the first item in
  //!   the next thread).
  //! - For *thread*\ :sub:`BLOCK_THREADS - 1`, item ``input[ITEMS_PER_THREAD - 1]`` is always flagged.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the head- and tail-flagging of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_discontinuity.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockDiscontinuity for a 1D block of 128 threads of type int
  //!        using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
  //!
  //!        // Allocate shared memory for BlockDiscontinuity
  //!        __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute head and flags for discontinuities in the segment
  //!        int head_flags[4];
  //!        int tail_flags[4];
  //!        BlockDiscontinuity(temp_storage).FlagHeadsAndTails(
  //!            head_flags, tail_flags, thread_data, cub::Inequality());
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,0,1,1], [1,1,1,1], [2,3,3,3], ..., [124,125,125,125] }``
  //! and that the tile_successor_item is ``125``.  The corresponding output ``head_flags``
  //! in those threads will be ``{ [1,0,1,0], [0,0,0,0], [1,1,0,0], [0,1,0,0], ... }``.
  //! and the corresponding output ``tail_flags`` in those threads will be
  //! ``{ [0,1,0,0], [0,0,0,1], [1,0,0,...], ..., [1,0,0,1] }``.
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam FlagT
  //!   **[inferred]** The flag type (must be an integer type)
  //!
  //! @tparam FlagOp
  //!   **[inferred]** Binary predicate functor type having member
  //!   `T operator()(const T &a, const T &b)` or member
  //!   `T operator()(const T &a, const T &b, unsigned int b_index)`, and returning `true`
  //!   if a discontinuity exists between `a` and `b`, otherwise `false`. `b_index` is the
  //!   rank of `b` in the aggregate tile of data.
  //!
  //! @param[out] head_flags
  //!   Calling thread's discontinuity head_flags
  //!
  //! @param[out] tail_flags
  //!   Calling thread's discontinuity tail_flags
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[in] flag_op
  //!   Binary boolean flag predicate
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void FlagHeadsAndTails(
    FlagT (&head_flags)[ITEMS_PER_THREAD],
    FlagT (&tail_flags)[ITEMS_PER_THREAD],
    T (&input)[ITEMS_PER_THREAD],
    FlagOp flag_op)
  {
    // Share first and last items
    temp_storage.first_items[linear_tid] = input[0];
    temp_storage.last_items[linear_tid]  = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    T preds[ITEMS_PER_THREAD];

    // Set flag for first thread-item
    if (linear_tid == 0)
    {
      head_flags[0] = 1;
    }
    else
    {
      preds[0]      = temp_storage.last_items[linear_tid - 1];
      head_flags[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);
    }

    // Set flag for last thread-item
    tail_flags[ITEMS_PER_THREAD - 1] =
      (linear_tid == BLOCK_THREADS - 1) ? 1 : // Last thread
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

  //! @rst
  //! Sets both head and tail flags indicating discontinuities between items partitioned across the thread block.
  //!
  //! - The flag ``head_flags[i]`` is set for item ``input[i]`` when
  //!   ``flag_op(previous-item, input[i])`` returns ``true`` (where ``previous-item`` is either the preceding item
  //!   in the same thread or the last item in the previous thread).
  //! - For *thread*\ :sub:`0`, item ``input[0]`` is always flagged.
  //! - The flag ``tail_flags[i]`` is set for item ``input[i]`` when ``flag_op(input[i], next-item)`` returns ``true``
  //!   (where ``next-item`` is either the next item in the same thread or the first item in the next thread).
  //! - For *thread*\ :sub:`BLOCK_THREADS - 1`, item ``input[ITEMS_PER_THREAD - 1]`` is compared
  //!   against ``tile_predecessor_item``.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the head- and tail-flagging of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_discontinuity.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockDiscontinuity for a 1D block of 128 threads of type int
  //!        using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
  //!
  //!        // Allocate shared memory for BlockDiscontinuity
  //!        __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Have thread127 obtain the successor item for the entire tile
  //!        int tile_successor_item;
  //!        if (threadIdx.x == 127) tile_successor_item == ...
  //!
  //!        // Collectively compute head and flags for discontinuities in the segment
  //!        int head_flags[4];
  //!        int tail_flags[4];
  //!        BlockDiscontinuity(temp_storage).FlagHeadsAndTails(
  //!            head_flags, tail_flags, tile_successor_item, thread_data, cub::Inequality());
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,0,1,1], [1,1,1,1], [2,3,3,3], ..., [124,125,125,125] }``
  //! and that the tile_successor_item is ``125``. The corresponding output ``head_flags``
  //! in those threads will be ``{ [1,0,1,0], [0,0,0,0], [1,1,0,0], [0,1,0,0], ... }``.
  //! and the corresponding output ``tail_flags`` in those threads will be
  //! ``{ [0,1,0,0], [0,0,0,1], [1,0,0,...], ..., [1,0,0,0] }``.
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam FlagT
  //!   **[inferred]** The flag type (must be an integer type)
  //!
  //! @tparam FlagOp
  //!   **[inferred]** Binary predicate functor type having member
  //!   `T operator()(const T &a, const T &b)` or member
  //!   `T operator()(const T &a, const T &b, unsigned int b_index)`, and returning `true`
  //!   if a discontinuity exists between `a` and `b`, otherwise `false`. `b_index` is the
  //!   rank of b in the aggregate tile of data.
  //!
  //! @param[out] head_flags
  //!   Calling thread's discontinuity head_flags
  //!
  //! @param[out] tail_flags
  //!   Calling thread's discontinuity tail_flags
  //!
  //! @param[in] tile_successor_item
  //!   @rst
  //!   *thread*\ :sub:`BLOCK_THREADS - 1` only item with which to compare
  //!   the last tile item (``input[ITEMS_PER_THREAD - 1]`` from
  //!   *thread*\ :sub:`BLOCK_THREADS - 1`).
  //!   @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[in] flag_op
  //!   Binary boolean flag predicate
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void FlagHeadsAndTails(
    FlagT (&head_flags)[ITEMS_PER_THREAD],
    FlagT (&tail_flags)[ITEMS_PER_THREAD],
    T tile_successor_item,
    T (&input)[ITEMS_PER_THREAD],
    FlagOp flag_op)
  {
    // Share first and last items
    temp_storage.first_items[linear_tid] = input[0];
    temp_storage.last_items[linear_tid]  = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    T preds[ITEMS_PER_THREAD];

    // Set flag for first thread-item
    if (linear_tid == 0)
    {
      head_flags[0] = 1;
    }
    else
    {
      preds[0]      = temp_storage.last_items[linear_tid - 1];
      head_flags[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);
    }

    // Set flag for last thread-item
    T successor_item = (linear_tid == BLOCK_THREADS - 1) ? tile_successor_item : // Last thread
                         temp_storage.first_items[linear_tid + 1];

    tail_flags[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::FlagT(
      flag_op, input[ITEMS_PER_THREAD - 1], successor_item, (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

    // Set head_flags for remaining items
    Iterate::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

    // Set tail_flags for remaining items
    Iterate::FlagTails(linear_tid, tail_flags, input, flag_op);
  }

  //! @rst
  //! Sets both head and tail flags indicating discontinuities between items partitioned across the thread block.
  //!
  //! - The flag ``head_flags[i]`` is set for item ``input[i]`` when ``flag_op(previous-item, input[i])``
  //!   returns ``true`` (where ``previous-item`` is either the preceding item in the same thread or the last item
  //!   in the previous thread).
  //! - For *thread*\ :sub:`0`, item ``input[0]`` is compared against ``tile_predecessor_item``.
  //! - The flag ``tail_flags[i]`` is set for item ``input[i]`` when
  //!   ``flag_op(input[i], next-item)`` returns ``true`` (where ``next-item`` is either the next item
  //!   in the same thread or the first item in the next thread).
  //! - For *thread*\ :sub:`BLOCK_THREADS - 1`, item
  //!   ``input[ITEMS_PER_THREAD - 1]`` is always flagged.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the head- and tail-flagging of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_discontinuity.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockDiscontinuity for a 1D block of 128 threads of type int
  //!        using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
  //!
  //!        // Allocate shared memory for BlockDiscontinuity
  //!        __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Have thread0 obtain the predecessor item for the entire tile
  //!        int tile_predecessor_item;
  //!        if (threadIdx.x == 0) tile_predecessor_item == ...
  //!
  //!        // Have thread127 obtain the successor item for the entire tile
  //!        int tile_successor_item;
  //!        if (threadIdx.x == 127) tile_successor_item == ...
  //!
  //!        // Collectively compute head and flags for discontinuities in the segment
  //!        int head_flags[4];
  //!        int tail_flags[4];
  //!        BlockDiscontinuity(temp_storage).FlagHeadsAndTails(
  //!            head_flags, tile_predecessor_item, tail_flags, tile_successor_item,
  //!            thread_data, cub::Inequality());
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,0,1,1], [1,1,1,1], [2,3,3,3], ..., [124,125,125,125] }``,
  //! that the ``tile_predecessor_item`` is ``0``, and that the ``tile_successor_item`` is ``125``.
  //! The corresponding output ``head_flags`` in those threads will be
  //! ``{ [0,0,1,0], [0,0,0,0], [1,1,0,0], [0,1,0,0], ... }``, and the corresponding output ``tail_flags``
  //! in those threads will be ``{ [0,1,0,0], [0,0,0,1], [1,0,0,...], ..., [1,0,0,1] }``.
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam FlagT
  //!   **[inferred]** The flag type (must be an integer type)
  //!
  //! @tparam FlagOp
  //!   **[inferred]** Binary predicate functor type having member
  //!   `T operator()(const T &a, const T &b)` or member
  //!   `T operator()(const T &a, const T &b, unsigned int b_index)`, and returning `true`
  //!   if a discontinuity exists between `a` and `b`, otherwise `false`. `b_index` is the rank
  //!   of b in the aggregate tile of data.
  //!
  //! @param[out] head_flags
  //!   Calling thread's discontinuity head_flags
  //!
  //! @param[in] tile_predecessor_item
  //!   @rst
  //!   *thread*\ :sub:`0` only item with which to compare the first tile item (``input[0]`` from *thread*\ :sub:`0`).
  //!   @endrst
  //!
  //! @param[out] tail_flags
  //!   Calling thread's discontinuity tail_flags
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[in] flag_op
  //!   Binary boolean flag predicate
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void FlagHeadsAndTails(
    FlagT (&head_flags)[ITEMS_PER_THREAD],
    T tile_predecessor_item,
    FlagT (&tail_flags)[ITEMS_PER_THREAD],
    T (&input)[ITEMS_PER_THREAD],
    FlagOp flag_op)
  {
    // Share first and last items
    temp_storage.first_items[linear_tid] = input[0];
    temp_storage.last_items[linear_tid]  = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    T preds[ITEMS_PER_THREAD];

    // Set flag for first thread-item
    preds[0] = (linear_tid == 0) ? tile_predecessor_item : // First thread
                 temp_storage.last_items[linear_tid - 1];

    head_flags[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);

    // Set flag for last thread-item
    tail_flags[ITEMS_PER_THREAD - 1] =
      (linear_tid == BLOCK_THREADS - 1) ? 1 : // Last thread
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

  //! @rst
  //! Sets both head and tail flags indicating discontinuities between items partitioned across the thread block.
  //!
  //! - The flag ``head_flags[i]`` is set for item ``input[i]`` when ``flag_op(previous-item, input[i])``
  //!   returns ``true`` (where ``previous-item`` is either the preceding item in the same thread or the last item in
  //!   the previous thread).
  //! - For *thread*\ :sub:`0`, item ``input[0]`` is compared against ``tile_predecessor_item``.
  //! - The flag ``tail_flags[i]`` is set for item ``input[i]`` when ``flag_op(input[i], next-item)``
  //!   returns ``true`` (where ``next-item`` is either the next item in the same thread or the first item in
  //!   the next thread).
  //! - For *thread*\ :sub:`BLOCK_THREADS - 1`, item ``input[ITEMS_PER_THREAD - 1]`` is compared
  //!   against ``tile_successor_item``.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the head- and tail-flagging of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_discontinuity.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockDiscontinuity for a 1D block of 128 threads of type int
  //!        using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
  //!
  //!        // Allocate shared memory for BlockDiscontinuity
  //!        __shared__ typename BlockDiscontinuity::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Have thread0 obtain the predecessor item for the entire tile
  //!        int tile_predecessor_item;
  //!        if (threadIdx.x == 0) tile_predecessor_item == ...
  //!
  //!        // Have thread127 obtain the successor item for the entire tile
  //!        int tile_successor_item;
  //!        if (threadIdx.x == 127) tile_successor_item == ...
  //!
  //!        // Collectively compute head and flags for discontinuities in the segment
  //!        int head_flags[4];
  //!        int tail_flags[4];
  //!        BlockDiscontinuity(temp_storage).FlagHeadsAndTails(
  //!            head_flags, tile_predecessor_item, tail_flags, tile_successor_item,
  //!            thread_data, cub::Inequality());
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,0,1,1], [1,1,1,1], [2,3,3,3], ..., [124,125,125,125] }``,
  //! that the ``tile_predecessor_item`` is ``0``, and that the
  //! ``tile_successor_item`` is ``125``. The corresponding output ``head_flags``
  //! in those threads will be ``{ [0,0,1,0], [0,0,0,0], [1,1,0,0], [0,1,0,0], ... }``.
  //! and the corresponding output ``tail_flags`` in those threads will be
  //! ``{ [0,1,0,0], [0,0,0,1], [1,0,0,...], ..., [1,0,0,0] }``.
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam FlagT
  //!   **[inferred]** The flag type (must be an integer type)
  //!
  //! @tparam FlagOp
  //!   **[inferred]** Binary predicate functor type having member
  //!   `T operator()(const T &a, const T &b)` or member
  //!   `T operator()(const T &a, const T &b, unsigned int b_index)`, and returning `true`
  //!   if a discontinuity exists between `a` and `b`, otherwise `false`. `b_index` is the rank
  //!   of `b` in the aggregate tile of data.
  //!
  //! @param[out] head_flags
  //!   Calling thread's discontinuity head_flags
  //!
  //! @param[in] tile_predecessor_item
  //!   @rst
  //!   *thread*\ :sub:`0` only item with which to compare the first tile item (``input[0]`` from *thread*\ :sub:`0`).
  //!   @endrst
  //!
  //! @param[out] tail_flags
  //!   Calling thread's discontinuity tail_flags
  //!
  //! @param[in] tile_successor_item
  //!   @rst
  //!   *thread*\ :sub:`BLOCK_THREADS - 1` only item with which to compare the last tile item
  //!   (``input[ITEMS_PER_THREAD - 1]`` from *thread*\ :sub:`BLOCK_THREADS - 1`).
  //!   @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[in] flag_op
  //!   Binary boolean flag predicate
  template <int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void FlagHeadsAndTails(
    FlagT (&head_flags)[ITEMS_PER_THREAD],
    T tile_predecessor_item,
    FlagT (&tail_flags)[ITEMS_PER_THREAD],
    T tile_successor_item,
    T (&input)[ITEMS_PER_THREAD],
    FlagOp flag_op)
  {
    // Share first and last items
    temp_storage.first_items[linear_tid] = input[0];
    temp_storage.last_items[linear_tid]  = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    T preds[ITEMS_PER_THREAD];

    // Set flag for first thread-item
    preds[0] = (linear_tid == 0) ? tile_predecessor_item : // First thread
                 temp_storage.last_items[linear_tid - 1];

    head_flags[0] = ApplyOp<FlagOp>::FlagT(flag_op, preds[0], input[0], linear_tid * ITEMS_PER_THREAD);

    // Set flag for last thread-item
    T successor_item = (linear_tid == BLOCK_THREADS - 1) ? tile_successor_item : // Last thread
                         temp_storage.first_items[linear_tid + 1];

    tail_flags[ITEMS_PER_THREAD - 1] = ApplyOp<FlagOp>::FlagT(
      flag_op, input[ITEMS_PER_THREAD - 1], successor_item, (linear_tid * ITEMS_PER_THREAD) + ITEMS_PER_THREAD);

    // Set head_flags for remaining items
    Iterate::FlagHeads(linear_tid, head_flags, input, preds, flag_op);

    // Set tail_flags for remaining items
    Iterate::FlagTails(linear_tid, tail_flags, input, flag_op);
  }

  //! @} end member group
};

CUB_NAMESPACE_END
