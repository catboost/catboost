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

//! @file
//! The cub::BlockReduce class provides :ref:`collective <collective-primitives>` methods for computing a parallel
//! reduction of items partitioned across a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/specializations/block_reduce_raking.cuh>
#include <cub/block/specializations/block_reduce_raking_commutative_only.cuh>
#include <cub/block/specializations/block_reduce_warp_reductions.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Algorithmic variants
 ******************************************************************************/

//! BlockReduceAlgorithm enumerates alternative algorithms for parallel reduction across a CUDA thread block.
enum BlockReduceAlgorithm
{

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! An efficient "raking" reduction algorithm that only supports commutative
  //! reduction operators (true for most operations, e.g., addition).
  //!
  //! Execution is comprised of three phases:
  //!   #. Upsweep sequential reduction in registers (if threads contribute more
  //!      than one input each). Threads in warps other than the first warp place
  //!      their partial reductions into shared memory.
  //!   #. Upsweep sequential reduction in shared memory. Threads within the first
  //!      warp continue to accumulate by raking across segments of shared partial reductions
  //!   #. A warp-synchronous Kogge-Stone style reduction within the raking warp.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - This variant performs less communication than BLOCK_REDUCE_RAKING_NON_COMMUTATIVE
  //!   and is preferable when the reduction operator is commutative. This variant
  //!   applies fewer reduction operators than BLOCK_REDUCE_WARP_REDUCTIONS, and can provide higher overall
  //!   throughput across the GPU when suitably occupied. However, turn-around latency may be
  //!   higher than to BLOCK_REDUCE_WARP_REDUCTIONS and thus less-desirable
  //!   when the GPU is under-occupied.
  //!
  //! @endrst
  BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! An efficient "raking" reduction algorithm that supports commutative
  //! (e.g., addition) and non-commutative (e.g., string concatenation) reduction
  //! operators. @blocked.
  //!
  //! Execution is comprised of three phases:
  //!   #. Upsweep sequential reduction in registers (if threads contribute more
  //!      than one input each). Each thread then places the partial reduction
  //!      of its item(s) into shared memory.
  //!   #. Upsweep sequential reduction in shared memory. Threads within a
  //!      single warp rake across segments of shared partial reductions.
  //!   #. A warp-synchronous Kogge-Stone style reduction within the raking warp.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - This variant performs more communication than BLOCK_REDUCE_RAKING
  //!   and is only preferable when the reduction operator is non-commutative. This variant
  //!   applies fewer reduction operators than BLOCK_REDUCE_WARP_REDUCTIONS, and can provide higher overall
  //!   throughput across the GPU when suitably occupied. However, turn-around latency may be
  //!   higher than to BLOCK_REDUCE_WARP_REDUCTIONS and thus less-desirable
  //!   when the GPU is under-occupied.
  //!
  //! @endrst
  BLOCK_REDUCE_RAKING,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A quick "tiled warp-reductions" reduction algorithm that supports commutative
  //! (e.g., addition) and non-commutative (e.g., string concatenation) reduction
  //! operators.
  //!
  //! Execution is comprised of four phases:
  //!   #. Upsweep sequential reduction in registers (if threads contribute more
  //!      than one input each). Each thread then places the partial reduction
  //!      of its item(s) into shared memory.
  //!   #. Compute a shallow, but inefficient warp-synchronous Kogge-Stone style
  //!      reduction within each warp.
  //!   #. A propagation phase where the warp reduction outputs in each warp are
  //!      updated with the aggregate from each preceding warp.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - This variant applies more reduction operators than BLOCK_REDUCE_RAKING
  //!   or BLOCK_REDUCE_RAKING_NON_COMMUTATIVE, which may result in lower overall
  //!   throughput across the GPU. However turn-around latency may be lower and
  //!   thus useful when the GPU is under-occupied.
  //!
  //! @endrst
  BLOCK_REDUCE_WARP_REDUCTIONS,
};

//! @rst
//! The BlockReduce class provides :ref:`collective <collective-primitives>` methods for computing a parallel reduction
//! of items partitioned across a CUDA thread block.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`_ (or *fold*) uses a binary combining
//!   operator to compute a single aggregate from a list of input elements.
//! - @rowmajor
//! - BlockReduce can be optionally specialized by algorithm to accommodate different latency/throughput
//!   workload profiles:
//!
//!   #. :cpp:enumerator:`cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY`:
//!      An efficient "raking" reduction algorithm that only supports commutative reduction operators.
//!   #. :cpp:enumerator:`cub::BLOCK_REDUCE_RAKING`:
//!      An efficient "raking" reduction algorithm that supports commutative and non-commutative reduction operators.
//!   #. :cpp:enumerator:`cub::BLOCK_REDUCE_WARP_REDUCTIONS`:
//!      A quick "tiled warp-reductions" reduction algorithm that supports commutative and non-commutative
//!      reduction operators.
//!
//! Performance Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - @granularity
//! - Very efficient (only one synchronization barrier).
//! - Incurs zero bank conflicts for most types
//! - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
//!   - Summation (vs. generic reduction)
//!   - ``BLOCK_THREADS`` is a multiple of the architecture's warp size
//!   - Every thread has a valid input (i.e., full vs. partial-tiles)
//! - See cub::BlockReduceAlgorithm for performance details regarding algorithmic alternatives
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockReduce}
//!
//! The code snippet below illustrates a sum reduction of 512 integer items that
//! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
//! where each thread owns 4 consecutive items.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_reduce.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize BlockReduce for a 1D block of 128 threads of type int
//!        using BlockReduce = cub::BlockReduce<int, 128>;
//!
//!        // Allocate shared memory for BlockReduce
//!        __shared__ typename BlockReduce::TempStorage temp_storage;
//!
//!        // Obtain a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        ...
//!
//!        // Compute the block-wide sum for thread0
//!        int aggregate = BlockReduce(temp_storage).Sum(thread_data);
//!    }
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of dynamically shared memory with
//! BlockReduce and how to re-purpose the same memory region.
//!
//! @endrst
//!
//! @tparam T
//!   Data type being reduced
//!
//! @tparam BLOCK_DIM_X
//!   The thread block length in threads along the X dimension
//!
//! @tparam ALGORITHM
//!   **[optional]** cub::BlockReduceAlgorithm enumerator specifying the underlying algorithm to use
//!   (default: cub::BLOCK_REDUCE_WARP_REDUCTIONS)
//!
//! @tparam BLOCK_DIM_Y
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1)
//!
//! @tparam BLOCK_DIM_Z
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
template <typename T,
          int BLOCK_DIM_X,
          BlockReduceAlgorithm ALGORITHM = BLOCK_REDUCE_WARP_REDUCTIONS,
          int BLOCK_DIM_Y                = 1,
          int BLOCK_DIM_Z                = 1>
class BlockReduce
{
private:
  /// Constants
  enum
  {
    /// The thread block size in threads
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
  };

  using WarpReductions        = detail::BlockReduceWarpReductions<T, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>;
  using RakingCommutativeOnly = detail::BlockReduceRakingCommutativeOnly<T, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>;
  using Raking                = detail::BlockReduceRaking<T, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>;

  /// Internal specialization type
  using InternalBlockReduce =
    ::cuda::std::_If<ALGORITHM == BLOCK_REDUCE_WARP_REDUCTIONS,
                     WarpReductions,
                     ::cuda::std::_If<ALGORITHM == BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                                      RakingCommutativeOnly,
                                      Raking>>; // BlockReduceRaking

  /// Shared memory storage layout type for BlockReduce
  using _TempStorage = typename InternalBlockReduce::TempStorage;

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  unsigned int linear_tid;

public:
  /// @smemstorage{BlockReduce}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockReduce()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation as temporary storage.
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockReduce(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @}  end member group
  //! @name Generic reductions
  //! @{

  //! @rst
  //! Computes a block-wide reduction for thread\ :sub:`0` using the specified binary reduction functor.
  //! Each thread contributes one input element.
  //!
  //! - The return value is undefined in threads other than thread\ :sub:`0`.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a max reduction of 128 integer items that
  //! are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_reduce.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockReduce for a 1D block of 128 threads of type int
  //!        using BlockReduce = cub::BlockReduce<int, 128>;
  //!
  //!        // Allocate shared memory for BlockReduce
  //!        __shared__ typename BlockReduce::TempStorage temp_storage;
  //!
  //!        // Each thread obtains an input item
  //!        int thread_data;
  //!        ...
  //!
  //!        // Compute the block-wide max for thread0
  //!        int aggregate = BlockReduce(temp_storage).Reduce(thread_data, cuda::maximum<>{});
  //!    }
  //!
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    return InternalBlockReduce(temp_storage).template Reduce<true>(input, BLOCK_THREADS, reduction_op);
  }

  //! @rst
  //! Computes a block-wide reduction for thread\ :sub:`0` using the specified binary reduction functor.
  //! Each thread contributes an array of consecutive input elements.
  //!
  //! - The return value is undefined in threads other than thread\ :sub:`0`.
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a max reduction of 512 integer items that are partitioned in a
  //! :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads where each thread owns
  //! 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_reduce.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockReduce for a 1D block of 128 threads of type int
  //!        using BlockReduce = cub::BlockReduce<int, 128>;
  //!
  //!        // Allocate shared memory for BlockReduce
  //!        __shared__ typename BlockReduce::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Compute the block-wide max for thread0
  //!        int aggregate = BlockReduce(temp_storage).Reduce(thread_data, cuda::maximum<>{});
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] inputs
  //!   Calling thread's input segment
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  template <int ITEMS_PER_THREAD, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T (&inputs)[ITEMS_PER_THREAD], ReductionOp reduction_op)
  {
    // Reduce partials
    T partial = cub::ThreadReduce(inputs, reduction_op);
    return Reduce(partial, reduction_op);
  }

  //! @rst
  //! Computes a block-wide reduction for thread\ :sub:`0` using the specified binary reduction functor.
  //! The first ``num_valid`` threads each contribute one input element.
  //!
  //! - The return value is undefined in threads other than thread<sub>0</sub>.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a max reduction of a partially-full tile of integer items
  //! that are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_reduce.cuh>
  //!
  //!    __global__ void ExampleKernel(int num_valid, ...)
  //!    {
  //!        // Specialize BlockReduce for a 1D block of 128 threads of type int
  //!        using BlockReduce = cub::BlockReduce<int, 128>;
  //!
  //!        // Allocate shared memory for BlockReduce
  //!        __shared__ typename BlockReduce::TempStorage temp_storage;
  //!
  //!        // Each thread obtains an input item
  //!        int thread_data;
  //!        if (threadIdx.x < num_valid) thread_data = ...
  //!
  //!        // Compute the block-wide max for thread0
  //!        int aggregate = BlockReduce(temp_storage).Reduce(thread_data, cuda::maximum<>{}, num_valid);
  //!    }
  //!
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] num_valid
  //!   Number of threads containing valid elements (may be less than BLOCK_THREADS)
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op, int num_valid)
  {
    // Determine if we skip bounds checking
    if (num_valid >= BLOCK_THREADS)
    {
      return InternalBlockReduce(temp_storage).template Reduce<true>(input, num_valid, reduction_op);
    }
    else
    {
      return InternalBlockReduce(temp_storage).template Reduce<false>(input, num_valid, reduction_op);
    }
  }

  //! @}  end member group
  //! @name Summation reductions
  //! @{

  //! @rst
  //! Computes a block-wide reduction for thread\ :sub:`0` using addition (+) as the reduction operator.
  //! Each thread contributes one input element.
  //!
  //! - The return value is undefined in threads other than thread\ :sub:`0`.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a sum reduction of 128 integer items that
  //! are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_reduce.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockReduce for a 1D block of 128 threads of type int
  //!        using BlockReduce = cub::BlockReduce<int, 128>;
  //!
  //!        // Allocate shared memory for BlockReduce
  //!        __shared__ typename BlockReduce::TempStorage temp_storage;
  //!
  //!        // Each thread obtains an input item
  //!        int thread_data;
  //!        ...
  //!
  //!        // Compute the block-wide sum for thread0
  //!        int aggregate = BlockReduce(temp_storage).Sum(thread_data);
  //!    }
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input)
  {
    return InternalBlockReduce(temp_storage).template Sum<true>(input, BLOCK_THREADS);
  }

  //! @rst
  //! Computes a block-wide reduction for thread<sub>0</sub> using addition (+) as the reduction operator.
  //! Each thread contributes an array of consecutive input elements.
  //!
  //! - The return value is undefined in threads other than thread\ :sub:`0`.
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a sum reduction of 512 integer items that are partitioned in a
  //! :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads where each thread owns
  //! 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_reduce.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockReduce for a 1D block of 128 threads of type int
  //!        using BlockReduce = cub::BlockReduce<int, 128>;
  //!
  //!        // Allocate shared memory for BlockReduce
  //!        __shared__ typename BlockReduce::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Compute the block-wide sum for thread0
  //!        int aggregate = BlockReduce(temp_storage).Sum(thread_data);
  //!    }
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @param[in] inputs
  //!   Calling thread's input segment
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T (&inputs)[ITEMS_PER_THREAD])
  {
    // Reduce partials
    T partial = cub::ThreadReduce(inputs, ::cuda::std::plus<>{});
    return Sum(partial);
  }

  //! @rst
  //! Computes a block-wide reduction for thread\ :sub:`0` using addition (+) as the reduction operator.
  //! The first ``num_valid`` threads each contribute one input element.
  //!
  //! - The return value is undefined in threads other than thread\ :sub:`0`.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a sum reduction of a partially-full tile of integer items
  //! that are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_reduce.cuh>
  //!
  //!    __global__ void ExampleKernel(int num_valid, ...)
  //!    {
  //!        // Specialize BlockReduce for a 1D block of 128 threads of type int
  //!        using BlockReduce = cub::BlockReduce<int, 128>;
  //!
  //!        // Allocate shared memory for BlockReduce
  //!        __shared__ typename BlockReduce::TempStorage temp_storage;
  //!
  //!        // Each thread obtains an input item (up to num_items)
  //!        int thread_data;
  //!        if (threadIdx.x < num_valid)
  //!            thread_data = ...
  //!
  //!        // Compute the block-wide sum for thread0
  //!        int aggregate = BlockReduce(temp_storage).Sum(thread_data, num_valid);
  //!    }
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input
  //!
  //! @param[in] num_valid
  //!   Number of threads containing valid elements (may be less than BLOCK_THREADS)
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input, int num_valid)
  {
    // Determine if we skip bounds checking
    if (num_valid >= BLOCK_THREADS)
    {
      return InternalBlockReduce(temp_storage).template Sum<true>(input, num_valid);
    }
    else
    {
      return InternalBlockReduce(temp_storage).template Sum<false>(input, num_valid);
    }
  }

  //! @}  end member group
};

CUB_NAMESPACE_END

#include <cub/backward.cuh>
