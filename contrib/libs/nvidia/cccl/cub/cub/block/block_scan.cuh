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
//! The cub::BlockScan class provides :ref:`collective <collective-primitives>` methods for computing a parallel prefix
//! sum/scan of items partitioned across a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/specializations/block_scan_raking.cuh>
#include <cub/block/specializations/block_scan_warp_scans.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Algorithmic variants
 ******************************************************************************/

//! @brief BlockScanAlgorithm enumerates alternative algorithms for cub::BlockScan to compute a
//!        parallel prefix scan across a CUDA thread block.
enum BlockScanAlgorithm
{

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! An efficient "raking reduce-then-scan" prefix scan algorithm. Execution is comprised of five phases:
  //!
  //! #. Upsweep sequential reduction in registers (if threads contribute more than one input each).
  //!    Each thread then places the partial reduction of its item(s) into shared memory.
  //! #. Upsweep sequential reduction in shared memory.
  //!    Threads within a single warp rake across segments of shared partial reductions.
  //! #. A warp-synchronous Kogge-Stone style exclusive scan within the raking warp.
  //! #. Downsweep sequential exclusive scan in shared memory.
  //!    Threads within a single warp rake across segments of shared partial reductions,
  //!    seeded with the warp-scan output.
  //! #. Downsweep sequential scan in registers (if threads contribute more than one input),
  //!    seeded with the raking scan output.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - Although this variant may suffer longer turnaround latencies when the
  //!   GPU is under-occupied, it can often provide higher overall throughput
  //!   across the GPU when suitably occupied.
  //!
  //! @endrst
  BLOCK_SCAN_RAKING,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! Similar to cub::BLOCK_SCAN_RAKING, but with fewer shared memory reads at the expense of higher
  //! register pressure. Raking threads preserve their "upsweep" segment of values in registers while performing
  //! warp-synchronous scan, allowing the "downsweep" not to re-read them from shared memory.
  //!
  //! @endrst
  BLOCK_SCAN_RAKING_MEMOIZE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A quick "tiled warpscans" prefix scan algorithm. Execution is comprised of four phases:
  //!   #. Upsweep sequential reduction in registers (if threads contribute more than one input each).
  //!      Each thread then places the partial reduction of its item(s) into shared memory.
  //!   #. Compute a shallow, but inefficient warp-synchronous Kogge-Stone style scan within each warp.
  //!   #. A propagation phase where the warp scan outputs in each warp are updated with the aggregate
  //!      from each preceding warp.
  //!   #. Downsweep sequential scan in registers (if threads contribute more than one input),
  //!      seeded with the raking scan output.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - Although this variant may suffer lower overall throughput across the
  //!   GPU because due to a heavy reliance on inefficient warpscans, it can
  //!   often provide lower turnaround latencies when the GPU is under-occupied.
  //!
  //! @endrst
  BLOCK_SCAN_WARP_SCANS,
};

//! @rst
//! The BlockScan class provides :ref:`collective <collective-primitives>` methods for computing a parallel prefix
//! sum/scan of items partitioned across a CUDA thread block.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - Given a list of input elements and a binary reduction operator, a
//!   `prefix scan <http://en.wikipedia.org/wiki/Prefix_sum>`_ produces an output list where each element is computed
//!   to be the reduction of the elements occurring earlier in the input list. *Prefix sum* connotes a prefix scan with
//!   the addition operator. The term *inclusive indicates* that the *i*\ :sup:`th` output reduction incorporates
//!   the *i*\ :sup:`th` input. The term *exclusive* indicates the *i*\ :sup:`th` input is not incorporated into
//!   the *i*\ :sup:`th` output reduction.
//! - @rowmajor
//! - BlockScan can be optionally specialized by algorithm to accommodate different workload profiles:
//!
//!   #. :cpp:enumerator:`cub::BLOCK_SCAN_RAKING`:
//!      An efficient (high throughput) "raking reduce-then-scan" prefix scan algorithm.
//!   #. :cpp:enumerator:`cub::BLOCK_SCAN_RAKING_MEMOIZE`:
//!      Similar to cub::BLOCK_SCAN_RAKING, but having higher throughput at the expense of additional
//!      register pressure for intermediate storage.
//!   #. :cpp:enumerator:`cub::BLOCK_SCAN_WARP_SCANS`:
//!      A quick (low latency) "tiled warpscans" prefix scan algorithm.
//!
//! Performance Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - @granularity
//! - Uses special instructions when applicable (e.g., warp ``SHFL``)
//! - Uses synchronization-free communication between warp lanes when applicable
//! - Invokes a minimal number of minimal block-wide synchronization barriers (only
//!   one or two depending on algorithm selection)
//! - Incurs zero bank conflicts for most types
//! - Computation is slightly more efficient (i.e., having lower instruction overhead) for:
//!
//!   - Prefix sum variants (vs. generic scan)
//!   - @blocksize
//!
//! - See cub::BlockScanAlgorithm for performance details regarding algorithmic alternatives
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockScan}
//!
//! The code snippet below illustrates an exclusive prefix sum of 512 integer items that
//! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
//! where each thread owns 4 consecutive items.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize BlockScan for a 1D block of 128 threads of type int
//!        using BlockScan = cub::BlockScan<int, 128>;
//!
//!        // Allocate shared memory for BlockScan
//!        __shared__ typename BlockScan::TempStorage temp_storage;
//!
//!        // Obtain a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        ...
//!
//!        // Collectively compute the block-wide exclusive prefix sum
//!        BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
//!
//! Suppose the set of input ``thread_data`` across the block of threads is
//! ``{[1,1,1,1], [1,1,1,1], ..., [1,1,1,1]}``.
//! The corresponding output ``thread_data`` in those threads will be
//! ``{[0,1,2,3], [4,5,6,7], ..., [508,509,510,511]}``.
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of dynamically shared memory with
//! BlockReduce and how to re-purpose the same memory region.
//! This example can be easily adapted to the storage required by BlockScan.
//!
//! @endrst
//!
//! @tparam T
//!   Data type being scanned
//!
//! @tparam BLOCK_DIM_X
//!   The thread block length in threads along the X dimension
//!
//! @tparam ALGORITHM
//!   **[optional]** cub::BlockScanAlgorithm enumerator specifying the underlying algorithm to use
//!   (default: cub::BLOCK_SCAN_RAKING)
//!
//! @tparam BLOCK_DIM_Y
//!   **[optional]** The thread block length in threads along the Y dimension
//!   (default: 1)
//!
//! @tparam BLOCK_DIM_Z
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
template <typename T,
          int BLOCK_DIM_X,
          BlockScanAlgorithm ALGORITHM = BLOCK_SCAN_RAKING,
          int BLOCK_DIM_Y              = 1,
          int BLOCK_DIM_Z              = 1>
class BlockScan
{
private:
  /// Constants
  enum
  {
    /// The thread block size in threads
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
  };

  /**
   * Ensure the template parameterization meets the requirements of the
   * specified algorithm. Currently, the BLOCK_SCAN_WARP_SCANS policy
   * cannot be used with thread block sizes not a multiple of the
   * architectural warp size.
   */
  static constexpr BlockScanAlgorithm SAFE_ALGORITHM =
    ((ALGORITHM == BLOCK_SCAN_WARP_SCANS) && (BLOCK_THREADS % detail::warp_threads != 0))
      ? BLOCK_SCAN_RAKING
      : ALGORITHM;

  using WarpScans = detail::BlockScanWarpScans<T, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>;
  using Raking =
    detail::BlockScanRaking<T, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, (SAFE_ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE)>;

  /// Define the delegate type for the desired algorithm
  using InternalBlockScan = ::cuda::std::_If<SAFE_ALGORITHM == BLOCK_SCAN_WARP_SCANS, WarpScans, Raking>;

  /// Shared memory storage layout type for BlockScan
  using _TempStorage = typename InternalBlockScan::TempStorage;

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  unsigned int linear_tid;

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

public:
  /// @smemstorage{BlockScan}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockScan()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation as temporary storage.
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockScan(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @}  end member group
  //! @name Exclusive prefix sum operations
  //! @{

  //! @rst
  //! Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes one input element. The value of 0 is applied as the initial value, and is assigned
  //! to ``output`` in *thread*\ :sub:`0`.
  //!
  //! - @identityzero
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an exclusive prefix sum of 128 integer items that
  //! are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>  // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain input item for each thread
  //!        int thread_data;
  //!        ...
  //!
  //!        // Collectively compute the block-wide exclusive prefix sum
  //!        BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is ``1, 1, ..., 1``.
  //! The corresponding output ``thread_data`` in those threads will be ``0, 1, ..., 127``.
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveSum(T input, T& output)
  {
    T initial_value{};

    ExclusiveScan(input, output, initial_value, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes one input element.
  //! The value of 0 is applied as the initial value, and is assigned to ``output`` in *thread*\ :sub:`0`.
  //! Also provides every thread with the block-wide ``block_aggregate`` of all inputs.
  //!
  //! - @identityzero
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an exclusive prefix sum of 128 integer items that
  //! are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain input item for each thread
  //!        int thread_data;
  //!        ...
  //!
  //!        // Collectively compute the block-wide exclusive prefix sum
  //!        int block_aggregate;
  //!        BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is ``1, 1, ..., 1``.
  //! The corresponding output ``thread_data`` in those threads will be ``0, 1, ..., 127``.
  //! Furthermore the value ``128`` will be stored in ``block_aggregate`` for all threads.
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[out] block_aggregate
  //!   block-wide aggregate reduction of input items
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveSum(T input, T& output, T& block_aggregate)
  {
    T initial_value{};

    ExclusiveScan(input, output, initial_value, ::cuda::std::plus<>{}, block_aggregate);
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes one input element.  Instead of using 0 as the block-wide prefix, the call-back functor
  //! ``block_prefix_callback_op`` is invoked by the first warp in the block, and the value returned by
  //! *lane*\ :sub:`0` in that warp is used as the "seed" value that logically prefixes the thread block's
  //! scan inputs.
  //!
  //! - @identityzero
  //! - The ``block_prefix_callback_op`` functor must implement a member function
  //!   ``T operator()(T block_aggregate)``. The functor will be invoked by the first warp of threads in the block,
  //!   however only the return value from *lane*\ :sub:`0` is applied as the block-wide prefix. Can be stateful.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a single thread block that progressively
  //! computes an exclusive prefix sum over multiple "tiles" of input using a
  //! prefix functor to maintain a running total between block-wide scans.  Each tile consists
  //! of 128 integer items that are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    // A stateful callback functor that maintains a running prefix to be applied
  //!    // during consecutive scan operations.
  //!    struct BlockPrefixCallbackOp
  //!    {
  //!        // Running prefix
  //!        int running_total;
  //!
  //!        // Constructor
  //!        __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  //!
  //!        // Callback operator to be entered by the first warp of threads in the block.
  //!        // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  //!        __device__ int operator()(int block_aggregate)
  //!        {
  //!            int old_prefix = running_total;
  //!            running_total += block_aggregate;
  //!            return old_prefix;
  //!        }
  //!    };
  //!
  //!    __global__ void ExampleKernel(int *d_data, int num_items, ...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Initialize running total
  //!        BlockPrefixCallbackOp prefix_op(0);
  //!
  //!        // Have the block iterate over segments of items
  //!        for (int block_offset = 0; block_offset < num_items; block_offset += 128)
  //!        {
  //!            // Load a segment of consecutive items that are blocked across threads
  //!            int thread_data = d_data[block_offset + threadIdx.x];
  //!
  //!            // Collectively compute the block-wide exclusive prefix sum
  //!            BlockScan(temp_storage).ExclusiveSum(
  //!                thread_data, thread_data, prefix_op);
  //!            __syncthreads();
  //!
  //!            // Store scanned items to output segment
  //!            d_data[block_offset + threadIdx.x] = thread_data;
  //!        }
  //!
  //! Suppose the input ``d_data`` is ``1, 1, 1, 1, 1, 1, 1, 1, ...``.
  //! The corresponding output for the first segment will be ``0, 1, ..., 127``.
  //! The output for the second segment will be ``128, 129, ..., 255``.
  //!
  //! @endrst
  //!
  //! @tparam BlockPrefixCallbackOp
  //!   **[inferred]** Call-back functor type having member `T operator()(T block_aggregate)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[in,out] block_prefix_callback_op
  //!   @rst
  //!   *warp*\ :sub:`0` only call-back functor for specifying a block-wide prefix to be applied to
  //!   the logical input sequence.
  //!   @endrst
  template <typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveSum(T input, T& output, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    ExclusiveScan(input, output, ::cuda::std::plus<>{}, block_prefix_callback_op);
  }

  //! @} end member group
  //! @name Exclusive prefix sum operations (multiple data per thread)
  //! @{

  //! @rst
  //! Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes an array of consecutive input elements.
  //! The value of 0 is applied as the initial value, and is assigned to ``output[0]`` in *thread*\ :sub:`0`.
  //!
  //! - @identityzero
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an exclusive prefix sum of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute the block-wide exclusive prefix sum
  //!        BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [1,1,1,1], [1,1,1,1], ..., [1,1,1,1] }``.
  //! The corresponding output ``thread_data`` in those threads will be
  //! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveSum(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD])
  {
    T initial_value{};

    ExclusiveScan(input, output, initial_value, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes an array of consecutive input elements.
  //! The value of 0 is applied as the initial value, and is assigned to ``output[0]`` in *thread*\ :sub:`0`.
  //! Also provides every thread with the block-wide ``block_aggregate`` of all inputs.
  //!
  //! - @identityzero
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an exclusive prefix sum of 512 integer items that are partitioned in
  //! a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads where each thread owns
  //! 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute the block-wide exclusive prefix sum
  //!        int block_aggregate;
  //!        BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [1,1,1,1], [1,1,1,1], ..., [1,1,1,1] }``.
  //! The corresponding output ``thread_data`` in those threads will be
  //! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
  //! Furthermore the value ``512`` will be stored in ``block_aggregate`` for all threads.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[out] block_aggregate
  //!   block-wide aggregate reduction of input items
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveSum(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], T& block_aggregate)
  {
    // Reduce consecutive thread items in registers
    T initial_value{};

    ExclusiveScan(input, output, initial_value, ::cuda::std::plus<>{}, block_aggregate);
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes an array of consecutive input elements.
  //! Instead of using 0 as the block-wide prefix, the call-back functor ``block_prefix_callback_op`` is invoked by
  //! the first warp in the block, and the value returned by *lane*\ :sub:`0` in that warp is used as the "seed"
  //! value that logically prefixes the thread block's scan inputs.
  //!
  //! - @identityzero
  //! - The ``block_prefix_callback_op`` functor must implement a member function ``T operator()(T block_aggregate)``.
  //!   The functor will be invoked by the first warp of threads in the block, however only the return value from
  //!   *lane*\ :sub:`0` is applied as the block-wide prefix. Can be stateful.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a single thread block that progressively
  //! computes an exclusive prefix sum over multiple "tiles" of input using a
  //! prefix functor to maintain a running total between block-wide scans.  Each tile consists
  //! of 512 integer items that are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>`
  //! across 128 threads where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    // A stateful callback functor that maintains a running prefix to be applied
  //!    // during consecutive scan operations.
  //!    struct BlockPrefixCallbackOp
  //!    {
  //!        // Running prefix
  //!        int running_total;
  //!
  //!        // Constructor
  //!        __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  //!
  //!        // Callback operator to be entered by the first warp of threads in the block.
  //!        // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  //!        __device__ int operator()(int block_aggregate)
  //!        {
  //!            int old_prefix = running_total;
  //!            running_total += block_aggregate;
  //!            return old_prefix;
  //!        }
  //!    };
  //!
  //!    __global__ void ExampleKernel(int *d_data, int num_items, ...)
  //!    {
  //!        // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
  //!        using BlockLoad  = cub::BlockLoad<int*, 128, 4, BLOCK_LOAD_TRANSPOSE>;
  //!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE>;
  //!        using BlockScan  = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  //!        __shared__ union {
  //!            typename BlockLoad::TempStorage     load;
  //!            typename BlockScan::TempStorage     scan;
  //!            typename BlockStore::TempStorage    store;
  //!        } temp_storage;
  //!
  //!        // Initialize running total
  //!        BlockPrefixCallbackOp prefix_op(0);
  //!
  //!        // Have the block iterate over segments of items
  //!        for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
  //!        {
  //!            // Load a segment of consecutive items that are blocked across threads
  //!            int thread_data[4];
  //!            BlockLoad(temp_storage.load).Load(d_data + block_offset, thread_data);
  //!            __syncthreads();
  //!
  //!            // Collectively compute the block-wide exclusive prefix sum
  //!            int block_aggregate;
  //!            BlockScan(temp_storage.scan).ExclusiveSum(
  //!                thread_data, thread_data, prefix_op);
  //!            __syncthreads();
  //!
  //!            // Store scanned items to output segment
  //!            BlockStore(temp_storage.store).Store(d_data + block_offset, thread_data);
  //!            __syncthreads();
  //!        }
  //!
  //! Suppose the input ``d_data`` is ``1, 1, 1, 1, 1, 1, 1, 1, ...``.
  //! The corresponding output for the first segment will be ``0, 1, 2, 3, ..., 510, 511``.
  //! The output for the second segment will be ``512, 513, 514, 515, ..., 1022, 1023``.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam BlockPrefixCallbackOp
  //!   **[inferred]** Call-back functor type having member
  //!   `T operator()(T block_aggregate)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in,out] block_prefix_callback_op
  //!   @rst
  //!   *warp*\ :sub:`0` only call-back functor for specifying a block-wide prefix to be applied to
  //!   the logical input sequence.
  //!   @endrst
  template <int ITEMS_PER_THREAD, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveSum(
    T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    ExclusiveScan(input, output, ::cuda::std::plus<>{}, block_prefix_callback_op);
  }

  //! @} end member group // Exclusive prefix sums (multiple data per thread)
  //! @name Exclusive prefix scan operations
  //! @{

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes one input element.
  //!
  //! - Supports non-commutative scan operators.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an exclusive prefix max scan of 128 integer items that
  //! are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain input item for each thread
  //!        int thread_data;
  //!        ...
  //!
  //!        // Collectively compute the block-wide exclusive prefix max scan
  //!        BlockScan(temp_storage).ExclusiveScan(thread_data, thread_data, INT_MIN, cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is ``0, -1, 2, -3, ..., 126, -127``.
  //! The corresponding output ``thread_data`` in those threads will be ``INT_MIN, 0, 0, 2, ..., 124, 126``.
  //!
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[in] initial_value
  //!   @rst
  //!   Initial value to seed the exclusive scan (and is assigned to `output[0]` in *thread*\ :sub:`0`)
  //!   @endrst
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& output, T initial_value, ScanOp scan_op)
  {
    InternalBlockScan(temp_storage).ExclusiveScan(input, output, initial_value, scan_op);
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes one input element.
  //! Also provides every thread with the block-wide ``block_aggregate`` of all inputs.
  //!
  //! - Supports non-commutative scan operators.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an exclusive prefix max scan of 128 integer items that
  //! are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain input item for each thread
  //!        int thread_data;
  //!        ...
  //!
  //!        // Collectively compute the block-wide exclusive prefix max scan
  //!        int block_aggregate;
  //!        BlockScan(temp_storage).ExclusiveScan(
  //!            thread_data, thread_data, INT_MIN, cuda::maximum<>{}, block_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is ``0, -1, 2, -3, ..., 126, -127``.
  //! The corresponding output ``thread_data`` in those threads will be ``INT_MIN, 0, 0, 2, ..., 124, 126``.
  //! Furthermore the value ``126`` will be stored in ``block_aggregate`` for all threads.
  //!
  //! .. note::
  //!
  //!    ``initial_value`` is not applied to the block-wide aggregate.
  //!
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member ``T operator()(const T &a, const T &b)``
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to ``input``)
  //!
  //! @param[in] initial_value
  //!   @rst
  //!   Initial value to seed the exclusive scan (and is assigned to ``output[0]`` in *thread*\ :sub:`0`). It is not
  //!   taken into account for ``block_aggregate``.
  //!
  //!   @endrst
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[out] block_aggregate
  //!   block-wide aggregate reduction of input items
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T input, T& output, T initial_value, ScanOp scan_op, T& block_aggregate)
  {
    InternalBlockScan(temp_storage).ExclusiveScan(input, output, initial_value, scan_op, block_aggregate);
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes one input element. The call-back functor ``block_prefix_callback_op`` is invoked by
  //! the first warp in the block, and the value returned by *lane*\ :sub:`0` in that warp is used as
  //! the "seed" value that logically prefixes the thread block's scan inputs.
  //!
  //! - The ``block_prefix_callback_op`` functor must implement a member function ``T operator()(T block_aggregate)``.
  //!   The functor will be invoked by the first warp of threads in the block, however only the return value from
  //!   *lane*\ :sub:`0` is applied as the block-wide prefix. Can be stateful.
  //! - Supports non-commutative scan operators.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a single thread block that progressively
  //! computes an exclusive prefix max scan over multiple "tiles" of input using a
  //! prefix functor to maintain a running total between block-wide scans.
  //! Each tile consists of 128 integer items that are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    // A stateful callback functor that maintains a running prefix to be applied
  //!    // during consecutive scan operations.
  //!    struct BlockPrefixCallbackOp
  //!    {
  //!        // Running prefix
  //!        int running_total;
  //!
  //!        // Constructor
  //!        __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  //!
  //!        // Callback operator to be entered by the first warp of threads in the block.
  //!        // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  //!        __device__ int operator()(int block_aggregate)
  //!        {
  //!            int old_prefix = running_total;
  //!            running_total = (block_aggregate > old_prefix) ? block_aggregate : old_prefix;
  //!            return old_prefix;
  //!        }
  //!    };
  //!
  //!    __global__ void ExampleKernel(int *d_data, int num_items, ...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Initialize running total
  //!        BlockPrefixCallbackOp prefix_op(INT_MIN);
  //!
  //!        // Have the block iterate over segments of items
  //!        for (int block_offset = 0; block_offset < num_items; block_offset += 128)
  //!        {
  //!            // Load a segment of consecutive items that are blocked across threads
  //!            int thread_data = d_data[block_offset + threadIdx.x];
  //!
  //!            // Collectively compute the block-wide exclusive prefix max scan
  //!            BlockScan(temp_storage).ExclusiveScan(
  //!                thread_data, thread_data, INT_MIN, cuda::maximum<>{}, prefix_op);
  //!            __syncthreads();
  //!
  //!            // Store scanned items to output segment
  //!            d_data[block_offset + threadIdx.x] = thread_data;
  //!        }
  //!
  //! Suppose the input ``d_data`` is ``0, -1, 2, -3, 4, -5, ...``.
  //! The corresponding output for the first segment will be ``INT_MIN, 0, 0, 2, ..., 124, 126``.
  //! The output for the second segment will be ``126, 128, 128, 130, ..., 252, 254``.
  //!
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam BlockPrefixCallbackOp
  //!   **[inferred]** Call-back functor type having member `T operator()(T block_aggregate)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in,out] block_prefix_callback_op
  //!   @rst
  //!   *warp*\ :sub:`0` only call-back functor for specifying a block-wide prefix to be applied to
  //!   the logical input sequence.
  //!   @endrst
  template <typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T input, T& output, ScanOp scan_op, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    InternalBlockScan(temp_storage).ExclusiveScan(input, output, scan_op, block_prefix_callback_op);
  }

  //! @} end member group // Inclusive prefix sums
  //! @name Exclusive prefix scan operations (multiple data per thread)
  //! @{

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements.
  //!
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an exclusive prefix max scan of 512 integer
  //! items that are partitioned in a [<em>blocked arrangement</em>](../index.html#sec5sec3)
  //! across 128 threads where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute the block-wide exclusive prefix max scan
  //!        BlockScan(temp_storage).ExclusiveScan(thread_data, thread_data, INT_MIN, cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,-1,2,-3], [4,-5,6,-7], ..., [508,-509,510,-511] }``.
  //! The corresponding output ``thread_data`` in those threads will be
  //! ``{ [INT_MIN,0,0,2], [2,4,4,6], ..., [506,508,508,510] }``.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] initial_value
  //!   @rst
  //!   Initial value to seed the exclusive scan (and is assigned to `output[0]` in *thread*\ :sub:`0`)
  //!   @endrst
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  template <int ITEMS_PER_THREAD, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], T initial_value, ScanOp scan_op)
  {
    // Reduce consecutive thread items in registers
    T thread_prefix = cub::ThreadReduce(input, scan_op);

    // Exclusive thread block-scan
    ExclusiveScan(thread_prefix, thread_prefix, initial_value, scan_op);

    // Exclusive scan in registers with prefix as seed
    detail::ThreadScanExclusive(input, output, scan_op, thread_prefix);
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements.
  //! Also provides every thread with the block-wide ``block_aggregate`` of all inputs.
  //!
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an exclusive prefix max scan of 512 integer items that are partitioned in
  //! a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads where each thread owns
  //! 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute the block-wide exclusive prefix max scan
  //!        int block_aggregate;
  //!        BlockScan(temp_storage).ExclusiveScan(
  //!            thread_data, thread_data, INT_MIN, cuda::maximum<>{}, block_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,-1,2,-3], [4,-5,6,-7], ..., [508,-509,510,-511] }``.
  //! The corresponding output ``thread_data`` in those threads will be
  //! ``{ [INT_MIN,0,0,2], [2,4,4,6], ..., [506,508,508,510] }``.
  //! Furthermore the value ``510`` will be stored in ``block_aggregate`` for all threads.
  //!
  //! .. note::
  //!
  //!    ``initial_value`` is not applied to the block-wide aggregate.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] initial_value
  //!   @rst
  //!   Initial value to seed the exclusive scan (and is assigned to `output[0]` in *thread*\ :sub:`0`). It is not taken
  //!   into account for ``block_aggregate``.
  //!   @endrst
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[out] block_aggregate
  //!   block-wide aggregate reduction of input items
  template <int ITEMS_PER_THREAD, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(
    T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], T initial_value, ScanOp scan_op, T& block_aggregate)
  {
    // Reduce consecutive thread items in registers
    T thread_prefix = cub::ThreadReduce(input, scan_op);

    // Exclusive thread block-scan
    ExclusiveScan(thread_prefix, thread_prefix, initial_value, scan_op, block_aggregate);

    // Exclusive scan in registers with prefix as seed
    detail::ThreadScanExclusive(input, output, scan_op, thread_prefix);
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements.
  //! The call-back functor ``block_prefix_callback_op`` is invoked by the first warp in the block, and the value
  //! returned by *lane*\ :sub:`0` in that warp is used as the "seed" value that logically prefixes the thread
  //! block's scan inputs.
  //!
  //! - The ``block_prefix_callback_op`` functor must implement a member function
  //!   ``T operator()(T block_aggregate)``. The functor will be invoked by the
  //!   first warp of threads in the block, however only the return value from
  //!   *lane*\ :sub:`0` is applied as the block-wide prefix. Can be stateful.
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a single thread block that progressively
  //! computes an exclusive prefix max scan over multiple "tiles" of input using a
  //! prefix functor to maintain a running total between block-wide scans. Each tile consists
  //! of 128 integer items that are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    // A stateful callback functor that maintains a running prefix to be applied
  //!    // during consecutive scan operations.
  //!    struct BlockPrefixCallbackOp
  //!    {
  //!        // Running prefix
  //!        int running_total;
  //!
  //!        // Constructor
  //!        __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  //!
  //!        // Callback operator to be entered by the first warp of threads in the block.
  //!        // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  //!        __device__ int operator()(int block_aggregate)
  //!        {
  //!            int old_prefix = running_total;
  //!            running_total = (block_aggregate > old_prefix) ? block_aggregate : old_prefix;
  //!            return old_prefix;
  //!        }
  //!    };
  //!
  //!    __global__ void ExampleKernel(int *d_data, int num_items, ...)
  //!    {
  //!        // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
  //!        using BlockLoad = cub::BlockLoad<int*, 128, 4, BLOCK_LOAD_TRANSPOSE>  ;
  //!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE> ;
  //!        using BlockScan = cub::BlockScan<int, 128>                            ;
  //!
  //!        // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  //!        __shared__ union {
  //!            typename BlockLoad::TempStorage     load;
  //!            typename BlockScan::TempStorage     scan;
  //!            typename BlockStore::TempStorage    store;
  //!        } temp_storage;
  //!
  //!        // Initialize running total
  //!        BlockPrefixCallbackOp prefix_op(0);
  //!
  //!        // Have the block iterate over segments of items
  //!        for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
  //!        {
  //!            // Load a segment of consecutive items that are blocked across threads
  //!            int thread_data[4];
  //!            BlockLoad(temp_storage.load).Load(d_data + block_offset, thread_data);
  //!            __syncthreads();
  //!
  //!            // Collectively compute the block-wide exclusive prefix max scan
  //!            BlockScan(temp_storage.scan).ExclusiveScan(
  //!                thread_data, thread_data, INT_MIN, cuda::maximum<>{}, prefix_op);
  //!            __syncthreads();
  //!
  //!            // Store scanned items to output segment
  //!            BlockStore(temp_storage.store).Store(d_data + block_offset, thread_data);
  //!            __syncthreads();
  //!        }
  //!
  //! Suppose the input ``d_data`` is ``0, -1, 2, -3, 4, -5, ...``.
  //! The corresponding output for the first segment will be
  //! ``INT_MIN, 0, 0, 2, 2, 4, ..., 508, 510``.
  //! The output for the second segment will be
  //! ``510, 512, 512, 514, 514, 516, ..., 1020, 1022``.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam BlockPrefixCallbackOp
  //!   **[inferred]** Call-back functor type having member `T operator()(T block_aggregate)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in,out] block_prefix_callback_op
  //!   @rst
  //!   *warp*\ :sub:`0` only call-back functor for specifying a block-wide prefix to be applied to
  //!   the logical input sequence.
  //!   @endrst
  template <int ITEMS_PER_THREAD, typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(
    T (&input)[ITEMS_PER_THREAD],
    T (&output)[ITEMS_PER_THREAD],
    ScanOp scan_op,
    BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    // Reduce consecutive thread items in registers
    T thread_prefix = cub::ThreadReduce(input, scan_op);

    // Exclusive thread block-scan
    ExclusiveScan(thread_prefix, thread_prefix, scan_op, block_prefix_callback_op);

    // Exclusive scan in registers with prefix as seed
    detail::ThreadScanExclusive(input, output, scan_op, thread_prefix);
  }

  //! @}  end member group
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document no-initial-value scans

  //! @name Exclusive prefix scan operations (no initial value, single datum per thread)
  //! @{

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes one input element.
  //! With no initial value, the output computed for *thread*\ :sub:`0` is undefined.
  //!
  //! - Supports non-commutative scan operators.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& output, ScanOp scan_op)
  {
    InternalBlockScan(temp_storage).ExclusiveScan(input, output, scan_op);
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes one input element. Also provides every thread with the block-wide
  //! ``block_aggregate`` of all inputs. With no initial value, the output computed for
  //! *thread*\ :sub:`0` is undefined.
  //!
  //! - Supports non-commutative scan operators.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[out] block_aggregate
  //!   block-wide aggregate reduction of input items
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& output, ScanOp scan_op, T& block_aggregate)
  {
    InternalBlockScan(temp_storage).ExclusiveScan(input, output, scan_op, block_aggregate);
  }

  //! @} end member group // Exclusive prefix scans (no initial value, single datum per thread)
  //! @name Exclusive prefix scan operations (no initial value, multiple data per thread)
  //! @{

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements. With no initial value, the
  //! output computed for *thread*\ :sub:`0` is undefined.
  //!
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  template <int ITEMS_PER_THREAD, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], ScanOp scan_op)
  {
    // Reduce consecutive thread items in registers
    T thread_partial = cub::ThreadReduce(input, scan_op);

    // Exclusive thread block-scan
    ExclusiveScan(thread_partial, thread_partial, scan_op);

    // Exclusive scan in registers with prefix
    detail::ThreadScanExclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
  }

  //! @rst
  //! Computes an exclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements. Also provides every thread
  //! with the block-wide ``block_aggregate`` of all inputs.
  //! With no initial value, the output computed for *thread*\ :sub:`0` is undefined.
  //!
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[out] block_aggregate
  //!   block-wide aggregate reduction of input items
  template <int ITEMS_PER_THREAD, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], ScanOp scan_op, T& block_aggregate)
  {
    // Reduce consecutive thread items in registers
    T thread_partial = cub::ThreadReduce(input, scan_op);

    // Exclusive thread block-scan
    ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate);

    // Exclusive scan in registers with prefix
    detail::ThreadScanExclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
  }

  //! @} end member group // Exclusive prefix scans (no initial value, multiple data per thread)
#endif // _CCCL_DOXYGEN_INVOKED  // Do not document no-initial-value scans

  //! @name Inclusive prefix sum operations
  //! @{

  //! @rst
  //! Computes an inclusive block-wide prefix scan using addition (+)
  //! as the scan operator. Each thread contributes one input element.
  //!
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix sum of 128 integer items that
  //! are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain input item for each thread
  //!        int thread_data;
  //!        ...
  //!
  //!        // Collectively compute the block-wide inclusive prefix sum
  //!        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is ``1, 1, ..., 1``.
  //! The corresponding output ``thread_data`` in those threads will be ``1, 2, ..., 128``.
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveSum(T input, T& output)
  {
    InclusiveScan(input, output, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes one input element.
  //! Also provides every thread with the block-wide ``block_aggregate`` of all inputs.
  //!
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix sum of 128 integer items that
  //! are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain input item for each thread
  //!        int thread_data;
  //!        ...
  //!
  //!        // Collectively compute the block-wide inclusive prefix sum
  //!        int block_aggregate;
  //!        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is ``1, 1, ..., 1``.
  //! The corresponding output ``thread_data`` in those threads will be ``1, 2, ..., 128``.
  //! Furthermore the value ``128`` will be stored in ``block_aggregate`` for all threads.
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[out] block_aggregate
  //!   block-wide aggregate reduction of input items
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveSum(T input, T& output, T& block_aggregate)
  {
    InclusiveScan(input, output, ::cuda::std::plus<>{}, block_aggregate);
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes one input element. Instead of using 0 as the block-wide prefix, the call-back functor
  //! ``block_prefix_callback_op`` is invoked by the first warp in the block, and the value returned by
  //! *lane*\ :sub:`0` in that warp is used as the "seed" value that logically prefixes the thread block's
  //! scan inputs.
  //!
  //! - The ``block_prefix_callback_op`` functor must implement a member function
  //!   ``T operator()(T block_aggregate)``. The functor will be invoked by the first warp of threads in the block,
  //!   however only the return value from *lane*\ :sub:`0` is applied as the block-wide prefix. Can be stateful.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a single thread block that progressively
  //! computes an inclusive prefix sum over multiple "tiles" of input using a
  //! prefix functor to maintain a running total between block-wide scans.
  //! Each tile consists of 128 integer items that are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    // A stateful callback functor that maintains a running prefix to be applied
  //!    // during consecutive scan operations.
  //!    struct BlockPrefixCallbackOp
  //!    {
  //!        // Running prefix
  //!        int running_total;
  //!
  //!        // Constructor
  //!        __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  //!
  //!        // Callback operator to be entered by the first warp of threads in the block.
  //!        // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  //!        __device__ int operator()(int block_aggregate)
  //!        {
  //!            int old_prefix = running_total;
  //!            running_total += block_aggregate;
  //!            return old_prefix;
  //!        }
  //!    };
  //!
  //!    __global__ void ExampleKernel(int *d_data, int num_items, ...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Initialize running total
  //!        BlockPrefixCallbackOp prefix_op(0);
  //!
  //!        // Have the block iterate over segments of items
  //!        for (int block_offset = 0; block_offset < num_items; block_offset += 128)
  //!        {
  //!            // Load a segment of consecutive items that are blocked across threads
  //!            int thread_data = d_data[block_offset + threadIdx.x];
  //!
  //!            // Collectively compute the block-wide inclusive prefix sum
  //!            BlockScan(temp_storage).InclusiveSum(
  //!                thread_data, thread_data, prefix_op);
  //!            __syncthreads();
  //!
  //!            // Store scanned items to output segment
  //!            d_data[block_offset + threadIdx.x] = thread_data;
  //!        }
  //!
  //! Suppose the input ``d_data`` is ``1, 1, 1, 1, 1, 1, 1, 1, ...``.
  //! The corresponding output for the first segment will be ``1, 2, ..., 128``.
  //! The output for the second segment will be ``129, 130, ..., 256``.
  //!
  //! @endrst
  //!
  //! @tparam BlockPrefixCallbackOp
  //!   **[inferred]** Call-back functor type having member `T operator()(T block_aggregate)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[in,out] block_prefix_callback_op
  //!   @rst
  //!   *warp*\ :sub:`0` only call-back functor for specifying a block-wide prefix to be applied
  //!   to the logical input sequence.
  //!   @endrst
  template <typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveSum(T input, T& output, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    InclusiveScan(input, output, ::cuda::std::plus<>{}, block_prefix_callback_op);
  }

  //! @}  end member group
  //! @name Inclusive prefix sum operations (multiple data per thread)
  //! @{

  //! @rst
  //! Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes an array of consecutive input elements.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix sum of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute the block-wide inclusive prefix sum
  //!        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [1,1,1,1], [1,1,1,1], ..., [1,1,1,1] }``. The corresponding output
  //! ``thread_data`` in those threads will be ``{ [1,2,3,4], [5,6,7,8], ..., [509,510,511,512] }``.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveSum(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD])
  {
    if (ITEMS_PER_THREAD == 1)
    {
      InclusiveSum(input[0], output[0]);
    }
    else
    {
      // Reduce consecutive thread items in registers
      ::cuda::std::plus<> scan_op;
      T thread_prefix = cub::ThreadReduce(input, scan_op);

      // Exclusive thread block-scan
      ExclusiveSum(thread_prefix, thread_prefix);

      // Inclusive scan in registers with prefix as seed
      detail::ThreadScanInclusive(input, output, scan_op, thread_prefix, (linear_tid != 0));
    }
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes an array of consecutive input elements.
  //! Also provides every thread with the block-wide ``block_aggregate`` of all inputs.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix sum of 512 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute the block-wide inclusive prefix sum
  //!        int block_aggregate;
  //!        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [1,1,1,1], [1,1,1,1], ..., [1,1,1,1] }``. The
  //! corresponding output ``thread_data`` in those threads will be
  //! ``{ [1,2,3,4], [5,6,7,8], ..., [509,510,511,512] }``.
  //! Furthermore the value ``512`` will be stored in ``block_aggregate`` for all threads.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[out] block_aggregate
  //!   block-wide aggregate reduction of input items
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveSum(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], T& block_aggregate)
  {
    if (ITEMS_PER_THREAD == 1)
    {
      InclusiveSum(input[0], output[0], block_aggregate);
    }
    else
    {
      // Reduce consecutive thread items in registers
      ::cuda::std::plus<> scan_op;
      T thread_prefix = cub::ThreadReduce(input, scan_op);

      // Exclusive thread block-scan
      ExclusiveSum(thread_prefix, thread_prefix, block_aggregate);

      // Inclusive scan in registers with prefix as seed
      detail::ThreadScanInclusive(input, output, scan_op, thread_prefix, (linear_tid != 0));
    }
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.
  //! Each thread contributes an array of consecutive input elements.
  //! Instead of using 0 as the block-wide prefix, the call-back functor ``block_prefix_callback_op`` is invoked by
  //! the first warp in the block, and the value returned by *lane*\ :sub:`0` in that warp is used as the "seed"
  //! value that logically prefixes the thread block's scan inputs.
  //!
  //! - The ``block_prefix_callback_op`` functor must implement a member function
  //!   ``T operator()(T block_aggregate)``. The functor will be invoked by the first warp of threads in the block,
  //!   however only the return value from *lane*\ :sub:`0` is applied as the block-wide prefix. Can be stateful.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a single thread block that progressively
  //! computes an inclusive prefix sum over multiple "tiles" of input using a
  //! prefix functor to maintain a running total between block-wide scans.  Each tile consists
  //! of 512 integer items that are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>`
  //! across 128 threads where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    // A stateful callback functor that maintains a running prefix to be applied
  //!    // during consecutive scan operations.
  //!    struct BlockPrefixCallbackOp
  //!    {
  //!        // Running prefix
  //!        int running_total;
  //!
  //!        // Constructor
  //!        __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  //!
  //!        // Callback operator to be entered by the first warp of threads in the block.
  //!        // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  //!        __device__ int operator()(int block_aggregate)
  //!        {
  //!            int old_prefix = running_total;
  //!            running_total += block_aggregate;
  //!            return old_prefix;
  //!        }
  //!    };
  //!
  //!    __global__ void ExampleKernel(int *d_data, int num_items, ...)
  //!    {
  //!        // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
  //!        using BlockLoad = cub::BlockLoad<int*, 128, 4, BLOCK_LOAD_TRANSPOSE>  ;
  //!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE> ;
  //!        using BlockScan = cub::BlockScan<int, 128>                            ;
  //!
  //!        // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  //!        __shared__ union {
  //!            typename BlockLoad::TempStorage     load;
  //!            typename BlockScan::TempStorage     scan;
  //!            typename BlockStore::TempStorage    store;
  //!        } temp_storage;
  //!
  //!        // Initialize running total
  //!        BlockPrefixCallbackOp prefix_op(0);
  //!
  //!        // Have the block iterate over segments of items
  //!        for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
  //!        {
  //!            // Load a segment of consecutive items that are blocked across threads
  //!            int thread_data[4];
  //!            BlockLoad(temp_storage.load).Load(d_data + block_offset, thread_data);
  //!            __syncthreads();
  //!
  //!            // Collectively compute the block-wide inclusive prefix sum
  //!            BlockScan(temp_storage.scan).IncluisveSum(
  //!                thread_data, thread_data, prefix_op);
  //!            __syncthreads();
  //!
  //!            // Store scanned items to output segment
  //!            BlockStore(temp_storage.store).Store(d_data + block_offset, thread_data);
  //!            __syncthreads();
  //!        }
  //!
  //! Suppose the input ``d_data`` is ``1, 1, 1, 1, 1, 1, 1, 1, ...``.
  //! The corresponding output for the first segment will be
  //! ``1, 2, 3, 4, ..., 511, 512``. The output for the second segment will be
  //! ``513, 514, 515, 516, ..., 1023, 1024``.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam BlockPrefixCallbackOp
  //!   **[inferred]** Call-back functor type having member `T operator()(T block_aggregate)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in,out] block_prefix_callback_op
  //!   @rst
  //!   *warp*\ :sub:`0` only call-back functor for specifying a block-wide prefix to be applied to the
  //!   logical input sequence.
  //!   @endrst
  template <int ITEMS_PER_THREAD, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveSum(
    T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    if (ITEMS_PER_THREAD == 1)
    {
      InclusiveSum(input[0], output[0], block_prefix_callback_op);
    }
    else
    {
      // Reduce consecutive thread items in registers
      ::cuda::std::plus<> scan_op;
      T thread_prefix = cub::ThreadReduce(input, scan_op);

      // Exclusive thread block-scan
      ExclusiveSum(thread_prefix, thread_prefix, block_prefix_callback_op);

      // Inclusive scan in registers with prefix as seed
      detail::ThreadScanInclusive(input, output, scan_op, thread_prefix);
    }
  }

  //! @}  end member group
  //! @name Inclusive prefix scan operations
  //! @{

  //! @rst
  //! Computes an inclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes one input element.
  //!
  //! - Supports non-commutative scan operators.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix max scan of 128 integer items that
  //! are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain input item for each thread
  //!        int thread_data;
  //!        ...
  //!
  //!        // Collectively compute the block-wide inclusive prefix max scan
  //!        BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``0, -1, 2, -3, ..., 126, -127``. The corresponding output ``thread_data``
  //! in those threads will be ``0, 0, 2, 2, ..., 126, 126``.
  //!
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& output, ScanOp scan_op)
  {
    InternalBlockScan(temp_storage).InclusiveScan(input, output, scan_op);
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes one input element. Also provides every thread with the block-wide
  //! ``block_aggregate`` of all inputs.
  //!
  //! - Supports non-commutative scan operators.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix max scan of 128
  //! integer items that are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain input item for each thread
  //!        int thread_data;
  //!        ...
  //!
  //!        // Collectively compute the block-wide inclusive prefix max scan
  //!        int block_aggregate;
  //!        BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, cuda::maximum<>{}, block_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``0, -1, 2, -3, ..., 126, -127``. The corresponding output ``thread_data``
  //! in those threads will be ``0, 0, 2, 2, ..., 126, 126``. Furthermore the value
  //! ``126`` will be stored in ``block_aggregate`` for all threads.
  //!
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[out] block_aggregate
  //!   Block-wide aggregate reduction of input items
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& output, ScanOp scan_op, T& block_aggregate)
  {
    InternalBlockScan(temp_storage).InclusiveScan(input, output, scan_op, block_aggregate);
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes one input element. The call-back functor ``block_prefix_callback_op``
  //! is invoked by the first warp in the block, and the value returned by *lane*\ :sub:`0` in that warp is used as
  //! the "seed" value that logically prefixes the thread block's scan inputs.
  //!
  //! - The ``block_prefix_callback_op`` functor must implement a member function
  //!   ``T operator()(T block_aggregate)``. The functor's input parameter
  //!   The functor will be invoked by the first warp of threads in the block,
  //!   however only the return value from *lane*\ :sub:`0` is applied
  //!   as the block-wide prefix. Can be stateful.
  //! - Supports non-commutative scan operators.
  //! - @rowmajor
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a single thread block that progressively
  //! computes an inclusive prefix max scan over multiple "tiles" of input using a
  //! prefix functor to maintain a running total between block-wide scans.  Each tile consists
  //! of 128 integer items that are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    // A stateful callback functor that maintains a running prefix to be applied
  //!    // during consecutive scan operations.
  //!    struct BlockPrefixCallbackOp
  //!    {
  //!        // Running prefix
  //!        int running_total;
  //!
  //!        // Constructor
  //!        __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  //!
  //!        // Callback operator to be entered by the first warp of threads in the block.
  //!        // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  //!        __device__ int operator()(int block_aggregate)
  //!        {
  //!            int old_prefix = running_total;
  //!            running_total = (block_aggregate > old_prefix) ? block_aggregate : old_prefix;
  //!            return old_prefix;
  //!        }
  //!    };
  //!
  //!    __global__ void ExampleKernel(int *d_data, int num_items, ...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Initialize running total
  //!        BlockPrefixCallbackOp prefix_op(INT_MIN);
  //!
  //!        // Have the block iterate over segments of items
  //!        for (int block_offset = 0; block_offset < num_items; block_offset += 128)
  //!        {
  //!            // Load a segment of consecutive items that are blocked across threads
  //!            int thread_data = d_data[block_offset + threadIdx.x];
  //!
  //!            // Collectively compute the block-wide inclusive prefix max scan
  //!            BlockScan(temp_storage).InclusiveScan(
  //!                thread_data, thread_data, cuda::maximum<>{}, prefix_op);
  //!            __syncthreads();
  //!
  //!            // Store scanned items to output segment
  //!            d_data[block_offset + threadIdx.x] = thread_data;
  //!        }
  //!
  //! Suppose the input ``d_data`` is ``0, -1, 2, -3, 4, -5, ...``.
  //! The corresponding output for the first segment will be
  //! ``0, 0, 2, 2, ..., 126, 126``. The output for the second segment
  //! will be ``128, 128, 130, 130, ..., 254, 254``.
  //!
  //! @endrst
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam BlockPrefixCallbackOp
  //!   **[inferred]** Call-back functor type having member `T operator()(T block_aggregate)`
  //!
  //! @param[in] input
  //!   Calling thread's input item
  //!
  //! @param[out] output
  //!   Calling thread's output item (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in,out] block_prefix_callback_op
  //!   @rst
  //!   *warp*\ :sub:`0` only call-back functor for specifying a block-wide prefix to be applied to
  //!   the logical input sequence.
  //!   @endrst
  template <typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScan(T input, T& output, ScanOp scan_op, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    InternalBlockScan(temp_storage).InclusiveScan(input, output, scan_op, block_prefix_callback_op);
  }

  //! @}  end member group
  //! @name Inclusive prefix scan operations (multiple data per thread)
  //! @{

  //! @rst
  //! Computes an inclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements.
  //!
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix max scan of 512 integer items that
  //! are partitioned in a [<em>blocked arrangement</em>](../index.html#sec5sec3) across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute the block-wide inclusive prefix max scan
  //!        BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, cuda::maximum<>{});
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,-1,2,-3], [4,-5,6,-7], ..., [508,-509,510,-511] }``.
  //! The corresponding output ``thread_data`` in those threads will be
  //! ``{ [0,0,2,2], [4,4,6,6], ..., [508,508,510,510] }``.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  template <int ITEMS_PER_THREAD, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScan(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], ScanOp scan_op)
  {
    if (ITEMS_PER_THREAD == 1)
    {
      InclusiveScan(input[0], output[0], scan_op);
    }
    else
    {
      // Reduce consecutive thread items in registers
      T thread_prefix = cub::ThreadReduce(input, scan_op);

      // Exclusive thread block-scan
      ExclusiveScan(thread_prefix, thread_prefix, scan_op);

      // Inclusive scan in registers with prefix as seed (first thread does not seed)
      detail::ThreadScanInclusive(input, output, scan_op, thread_prefix, (linear_tid != 0));
    }
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements.
  //!
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix max scan of 128 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 64 threads
  //! where each thread owns 2 consecutive items.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_block_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-scan-array-init-value
  //!     :end-before: example-end inclusive-scan-array-init-value
  //!
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] initial_value
  //!   Initial value to seed the inclusive scan (uniform across block)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  template <int ITEMS_PER_THREAD, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScan(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], T initial_value, ScanOp scan_op)
  {
    // Reduce consecutive thread items in registers
    T thread_prefix = cub::ThreadReduce(input, scan_op);

    // Exclusive thread block-scan
    ExclusiveScan(thread_prefix, thread_prefix, initial_value, scan_op);

    // Exclusive scan in registers with prefix as seed
    detail::ThreadScanInclusive(input, output, scan_op, thread_prefix);
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements. Also provides every thread
  //! with the block-wide ``block_aggregate`` of all inputs.
  //!
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix max scan of 512 integer items that
  //! are partitioned in a [<em>blocked arrangement</em>](../index.html#sec5sec3) across 128 threads
  //! where each thread owns 4 consecutive items.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize BlockScan for a 1D block of 128 threads of type int
  //!        using BlockScan = cub::BlockScan<int, 128>;
  //!
  //!        // Allocate shared memory for BlockScan
  //!        __shared__ typename BlockScan::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Collectively compute the block-wide inclusive prefix max scan
  //!        int block_aggregate;
  //!        BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, cuda::maximum<>{}, block_aggregate);
  //!
  //! Suppose the set of input ``thread_data`` across the block of threads is
  //! ``{ [0,-1,2,-3], [4,-5,6,-7], ..., [508,-509,510,-511] }``.
  //! The corresponding output ``thread_data`` in those threads will be
  //! ``{ [0,0,2,2], [4,4,6,6], ..., [508,508,510,510] }``.
  //! Furthermore the value ``510`` will be stored in ``block_aggregate`` for all threads.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[out] block_aggregate
  //!   Block-wide aggregate reduction of input items
  template <int ITEMS_PER_THREAD, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScan(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], ScanOp scan_op, T& block_aggregate)
  {
    if (ITEMS_PER_THREAD == 1)
    {
      InclusiveScan(input[0], output[0], scan_op, block_aggregate);
    }
    else
    {
      // Reduce consecutive thread items in registers
      T thread_prefix = cub::ThreadReduce(input, scan_op);

      // Exclusive thread block-scan (with no initial value)
      ExclusiveScan(thread_prefix, thread_prefix, scan_op, block_aggregate);

      // Inclusive scan in registers with prefix as seed (first thread does not seed)
      detail::ThreadScanInclusive(input, output, scan_op, thread_prefix, (linear_tid != 0));
    }
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements. Also provides every thread
  //! with the block-wide ``block_aggregate`` of all inputs.
  //!
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates an inclusive prefix max scan of 128 integer items that
  //! are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 64 threads
  //! where each thread owns 2 consecutive items.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_block_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-scan-array-aggregate-init-value
  //!     :end-before: example-end inclusive-scan-array-aggregate-init-value
  //!
  //! The value ``126`` will be stored in ``block_aggregate`` for all threads.
  //!
  //! .. note::
  //!
  //!    ``initial_value`` is not applied to the block-wide aggregate.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] initial_value
  //!   Initial value to seed the inclusive scan (uniform across block). It is not taken
  //!   into account for ``block_aggregate``.
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[out] block_aggregate
  //!   Block-wide aggregate reduction of input items
  template <int ITEMS_PER_THREAD, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(
    T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], T initial_value, ScanOp scan_op, T& block_aggregate)
  {
    // Reduce consecutive thread items in registers
    T thread_prefix = cub::ThreadReduce(input, scan_op);

    // Exclusive thread block-scan
    ExclusiveScan(thread_prefix, thread_prefix, initial_value, scan_op, block_aggregate);

    // Exclusive scan in registers with prefix as seed
    detail::ThreadScanInclusive(input, output, scan_op, thread_prefix);
  }

  //! @rst
  //! Computes an inclusive block-wide prefix scan using the specified binary ``scan_op`` functor.
  //! Each thread contributes an array of consecutive input elements.
  //! The call-back functor ``block_prefix_callback_op`` is invoked by the first warp in the block,
  //! and the value returned by *lane*\ :sub:`0` in that warp is used as the "seed" value that logically prefixes the
  //! thread block's scan inputs.
  //!
  //! - The ``block_prefix_callback_op`` functor must implement a member function ``T operator()(T block_aggregate)``.
  //!   The functor will be invoked by the first warp of threads in the block, however only the return value
  //!   from *lane*\ :sub:`0` is applied as the block-wide prefix. Can be stateful.
  //! - Supports non-commutative scan operators.
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a single thread block that progressively
  //! computes an inclusive prefix max scan over multiple "tiles" of input using a
  //! prefix functor to maintain a running total between block-wide scans.  Each tile consists
  //! of 128 integer items that are partitioned across 128 threads.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
  //!
  //!    // A stateful callback functor that maintains a running prefix to be applied
  //!    // during consecutive scan operations.
  //!    struct BlockPrefixCallbackOp
  //!    {
  //!        // Running prefix
  //!        int running_total;
  //!
  //!        // Constructor
  //!        __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
  //!
  //!        // Callback operator to be entered by the first warp of threads in the block.
  //!        // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  //!        __device__ int operator()(int block_aggregate)
  //!        {
  //!            int old_prefix = running_total;
  //!            running_total = (block_aggregate > old_prefix) ? block_aggregate : old_prefix;
  //!            return old_prefix;
  //!        }
  //!    };
  //!
  //!    __global__ void ExampleKernel(int *d_data, int num_items, ...)
  //!    {
  //!        // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
  //!        using BlockLoad = cub::BlockLoad<int*, 128, 4, BLOCK_LOAD_TRANSPOSE>  ;
  //!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE> ;
  //!        using BlockScan = cub::BlockScan<int, 128>                            ;
  //!
  //!        // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  //!        __shared__ union {
  //!            typename BlockLoad::TempStorage     load;
  //!            typename BlockScan::TempStorage     scan;
  //!            typename BlockStore::TempStorage    store;
  //!        } temp_storage;
  //!
  //!        // Initialize running total
  //!        BlockPrefixCallbackOp prefix_op(0);
  //!
  //!        // Have the block iterate over segments of items
  //!        for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
  //!        {
  //!            // Load a segment of consecutive items that are blocked across threads
  //!            int thread_data[4];
  //!            BlockLoad(temp_storage.load).Load(d_data + block_offset, thread_data);
  //!            __syncthreads();
  //!
  //!            // Collectively compute the block-wide inclusive prefix max scan
  //!            BlockScan(temp_storage.scan).InclusiveScan(
  //!                thread_data, thread_data, cuda::maximum<>{}, prefix_op);
  //!            __syncthreads();
  //!
  //!            // Store scanned items to output segment
  //!            BlockStore(temp_storage.store).Store(d_data + block_offset, thread_data);
  //!            __syncthreads();
  //!        }
  //!
  //! Suppose the input ``d_data`` is ``0, -1, 2, -3, 4, -5, ...``.
  //! The corresponding output for the first segment will be
  //! ``0, 0, 2, 2, 4, 4, ..., 510, 510``. The output for the second
  //! segment will be ``512, 512, 514, 514, 516, 516, ..., 1022, 1022``.
  //!
  //! @endrst
  //!
  //! @tparam ITEMS_PER_THREAD
  //!   **[inferred]** The number of consecutive items partitioned onto each thread.
  //!
  //! @tparam ScanOp
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam BlockPrefixCallbackOp
  //!   **[inferred]** Call-back functor type having member `T operator()(T block_aggregate)`
  //!
  //! @param[in] input
  //!   Calling thread's input items
  //!
  //! @param[out] output
  //!   Calling thread's output items (may be aliased to `input`)
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in,out] block_prefix_callback_op
  //!   @rst
  //!   *warp*\ :sub:`0` only call-back functor for specifying a block-wide prefix to be applied to
  //!   the logical input sequence.
  //!   @endrst
  template <int ITEMS_PER_THREAD, typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(
    T (&input)[ITEMS_PER_THREAD],
    T (&output)[ITEMS_PER_THREAD],
    ScanOp scan_op,
    BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    if (ITEMS_PER_THREAD == 1)
    {
      InclusiveScan(input[0], output[0], scan_op, block_prefix_callback_op);
    }
    else
    {
      // Reduce consecutive thread items in registers
      T thread_prefix = cub::ThreadReduce(input, scan_op);

      // Exclusive thread block-scan
      ExclusiveScan(thread_prefix, thread_prefix, scan_op, block_prefix_callback_op);

      // Inclusive scan in registers with prefix as seed
      detail::ThreadScanInclusive(input, output, scan_op, thread_prefix);
    }
  }

  //! @}  end member group
};

CUB_NAMESPACE_END
