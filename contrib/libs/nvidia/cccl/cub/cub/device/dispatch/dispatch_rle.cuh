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
 *   cub::DeviceRle provides device-wide, parallel operations for run-length-encoding sequences of
 *   data items residing within device-accessible memory.
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

#include <cub/agent/agent_rle.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_run_length_encode.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/__algorithm_>

#include <nv/target>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::rle
{

template <typename PrecedingKeyItT, typename RunLengthT, typename GlobalOffsetT>
struct streaming_context
{
  bool first_partition;
  bool last_partition;
  // Global offset of the current partition to compute and write out the correct absolute offsets to the user
  GlobalOffsetT current_base_offset;

  // We use a double-buffer to track the aggregated run-length of the last run of the previous partition
  RunLengthT* preceding_length;
  RunLengthT* length_out;

  // We use a double-buffer to track the number of runs of previous partition
  GlobalOffsetT* d_num_previous_uniques_in;
  GlobalOffsetT* d_num_accumulated_uniques_out;

  _CCCL_DEVICE _CCCL_FORCEINLINE GlobalOffsetT num_accumulated_uniques_out() const
  {
    return first_partition ? GlobalOffsetT{0} : *d_num_previous_uniques_in;
  };

  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE RunLengthT prefix() const
  {
    return *preceding_length;
  }

  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE void write_prefix(RunLengthT prefix) const
  {
    *length_out = prefix;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE GlobalOffsetT* previous_uniques_ptr() const
  {
    return d_num_previous_uniques_in;
  }

  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE GlobalOffsetT num_uniques() const
  {
    return num_accumulated_uniques_out();
  }

  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE GlobalOffsetT base_offset() const
  {
    return current_base_offset;
  }

  template <typename NumUniquesT>
  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE GlobalOffsetT add_num_uniques(NumUniquesT num_uniques) const
  {
    GlobalOffsetT total_uniques = num_accumulated_uniques_out() + static_cast<GlobalOffsetT>(num_uniques);

    // Otherwise, just write out the number of unique items in this partition
    *d_num_accumulated_uniques_out = total_uniques;

    return total_uniques;
  }
};

/**
 * Select kernel entry point (multi-block)
 *
 * Performs functor-based selection if SelectOp functor type != NullType
 * Otherwise performs flag-based selection if FlagIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 *
 * @tparam AgentRlePolicyT
 *   Parameterized AgentRlePolicyT tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OffsetsOutputIteratorT
 *   Random-access output iterator type for writing run-offset values @iterator
 *
 * @tparam LengthsOutputIteratorT
 *   Random-access output iterator type for writing run-length values @iterator
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording the number of runs encountered @iterator
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOpT
 *   T equality operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param d_in
 *   Pointer to input sequence of data items
 *
 * @param d_offsets_out
 *   Pointer to output sequence of run-offsets
 *
 * @param d_lengths_out
 *   Pointer to output sequence of run-lengths
 *
 * @param d_num_runs_out
 *   Pointer to total number of runs (i.e., length of `d_offsets_out`)
 *
 * @param tile_status
 *   Tile status interface
 *
 * @param equality_op
 *   Equality operator for input items
 *
 * @param num_items
 *   Total number of input items (i.e., length of `d_in`)
 *
 * @param num_tiles
 *   Total number of tiles for the entire problem
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OffsetsOutputIteratorT,
          typename LengthsOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename OffsetT,
          typename GlobalOffsetT,
          typename StreamingContextT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::RleSweepPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRleSweepKernel(
    InputIteratorT d_in,
    OffsetsOutputIteratorT d_offsets_out,
    LengthsOutputIteratorT d_lengths_out,
    NumRunsOutputIteratorT d_num_runs_out,
    ScanTileStateT tile_status,
    EqualityOpT equality_op,
    OffsetT num_items,
    int num_tiles,
    _CCCL_GRID_CONSTANT const StreamingContextT streaming_context)
{
  using AgentRlePolicyT = typename ChainedPolicyT::ActivePolicy::RleSweepPolicyT;

  // Thread block type for selecting data from input tiles
  using AgentRleT =
    AgentRle<AgentRlePolicyT,
             InputIteratorT,
             OffsetsOutputIteratorT,
             LengthsOutputIteratorT,
             EqualityOpT,
             OffsetT,
             GlobalOffsetT,
             StreamingContextT>;

  // Shared memory for AgentRle
  __shared__ typename AgentRleT::TempStorage temp_storage;

  // Process tiles
  AgentRleT(temp_storage, d_in, d_offsets_out, d_lengths_out, equality_op, num_items, streaming_context)
    .ConsumeRange(num_tiles, tile_status, d_num_runs_out);
}
} // namespace detail::rle

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceRle
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OffsetsOutputIteratorT
 *   Random-access output iterator type for writing run-offset values @iterator
 *
 * @tparam LengthsOutputIteratorT
 *   Random-access output iterator type for writing run-length values @iterator
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording the number of runs encountered @iterator
 *
 * @tparam EqualityOpT
 *   T equality operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam PolicyHub
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <typename InputIteratorT,
          typename OffsetsOutputIteratorT,
          typename LengthsOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename EqualityOpT,
          typename OffsetT,
          typename PolicyHub =
            detail::rle::non_trivial_runs::policy_hub<cub::detail::non_void_value_t<LengthsOutputIteratorT, OffsetT>,
                                                      cub::detail::it_value_t<InputIteratorT>>>
struct DeviceRleDispatch
{
  /******************************************************************************
   * Types and constants
   ******************************************************************************/
  // Offsets to index items within one partition (i.e., a single kernel invocation)
  using local_offset_t = _CUDA_VSTD::int32_t;

  // If the number of items provided by the user may exceed the maximum number of items processed by a single kernel
  // invocation, we may require multiple kernel invocations
  static constexpr bool use_streaming_invocation = _CUDA_VSTD::numeric_limits<OffsetT>::max()
                                                 > _CUDA_VSTD::numeric_limits<local_offset_t>::max();

  // Offsets to index any item within the entire input (large enough to cover num_items)
  using global_offset_t = OffsetT;

  // The lengths output value type
  using length_t = cub::detail::non_void_value_t<LengthsOutputIteratorT, global_offset_t>;

  // Type used to provide context about the current partition during a streaming invocation
  using streaming_context_t =
    ::cuda::std::conditional_t<use_streaming_invocation,
                               detail::rle::streaming_context<InputIteratorT, length_t, global_offset_t>,
                               NullType>;

  static constexpr int init_kernel_threads = 128;

  // Tile status descriptor interface type
  using ScanTileStateT = ReduceByKeyScanTileState<length_t, local_offset_t>;

  void* d_temp_storage;
  size_t& temp_storage_bytes;
  InputIteratorT d_in;
  OffsetsOutputIteratorT d_offsets_out;
  LengthsOutputIteratorT d_lengths_out;
  NumRunsOutputIteratorT d_num_runs_out;
  EqualityOpT equality_op;
  global_offset_t num_items;
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DeviceRleDispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OffsetsOutputIteratorT d_offsets_out,
    LengthsOutputIteratorT d_lengths_out,
    NumRunsOutputIteratorT d_num_runs_out,
    EqualityOpT equality_op,
    global_offset_t num_items,
    cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_offsets_out(d_offsets_out)
      , d_lengths_out(d_lengths_out)
      , d_num_runs_out(d_num_runs_out)
      , equality_op(equality_op)
      , num_items(num_items)
      , stream(stream)
  {}

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/

  /**
   * Internal dispatch routine for computing a device-wide run-length-encode using the
   * specified kernel functions.
   *
   * @tparam DeviceScanInitKernelPtr
   *   Function type of cub::DeviceScanInitKernel
   *
   * @tparam DeviceRleSweepKernelPtr
   *   Function type of cub::DeviceRleSweepKernelPtr
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to the input sequence of data items
   *
   * @param d_offsets_out
   *   Pointer to the output sequence of run-offsets
   *
   * @param d_lengths_out
   *   Pointer to the output sequence of run-lengths
   *
   * @param d_num_runs_out
   *   Pointer to the total number of runs encountered (i.e., length of `d_offsets_out`)
   *
   * @param equality_op
   *   Equality operator for input items
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   *
   * @param ptx_version
   *   PTX version of dispatch kernels
   *
   * @param device_scan_init_kernel
   *   Kernel function pointer to parameterization of cub::DeviceScanInitKernel
   *
   * @param device_rle_sweep_kernel
   *   Kernel function pointer to parameterization of cub::DeviceRleSweepKernel
   */
  template <typename ActivePolicyT, typename DeviceScanInitKernelPtr, typename DeviceRleSweepKernelPtr>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(DeviceScanInitKernelPtr device_scan_init_kernel, DeviceRleSweepKernelPtr device_rle_sweep_kernel)
  {
    cudaError error = cudaSuccess;

    constexpr int block_threads    = ActivePolicyT::RleSweepPolicyT::BLOCK_THREADS;
    constexpr int items_per_thread = ActivePolicyT::RleSweepPolicyT::ITEMS_PER_THREAD;
    constexpr auto tile_size       = static_cast<global_offset_t>(block_threads * items_per_thread);

    // The upper bound of for the number of items that a single kernel invocation will ever process
    auto capped_num_items_per_invocation = num_items;
    if constexpr (use_streaming_invocation)
    {
      capped_num_items_per_invocation = static_cast<global_offset_t>(_CUDA_VSTD::numeric_limits<local_offset_t>::max());
      // Make sure that the number of items is a multiple of tile size
      capped_num_items_per_invocation -= (capped_num_items_per_invocation % tile_size);
    }

    // Across invocations, the maximum number of items that a single kernel invocation will ever process
    const auto max_num_items_per_invocation =
      use_streaming_invocation ? _CUDA_VSTD::min(capped_num_items_per_invocation, num_items) : num_items;

    // Number of invocations required to "iterate" over the total input (at least one iteration to process zero items)
    auto const num_partitions =
      (capped_num_items_per_invocation == 0)
        ? global_offset_t{1}
        : ::cuda::ceil_div(num_items, capped_num_items_per_invocation);

    // Number of input tiles
    int max_num_tiles = static_cast<int>(::cuda::ceil_div(max_num_items_per_invocation, tile_size));

    // Specify temporary storage allocation requirements
    size_t allocation_sizes[3];
    error = CubDebug(ScanTileStateT::AllocationSize(max_num_tiles, allocation_sizes[0]));
    if (cudaSuccess != error)
    {
      return error;
    }
    allocation_sizes[1] = num_partitions > 1 ? sizeof(global_offset_t) * 2 : size_t{0};
    allocation_sizes[2] = num_partitions > 1 ? sizeof(local_offset_t) * 2 : size_t{0};

    // Compute allocation pointers into the single storage blob (or compute the necessary size of
    // the blob)
    void* allocations[3] = {};

    error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
    if (error != cudaSuccess)
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage allocation
      return error;
    }

    // Iterate over the partitions until all input is processed
    for (global_offset_t partition_idx = 0; partition_idx < num_partitions; partition_idx++)
    {
      global_offset_t current_partition_offset = partition_idx * capped_num_items_per_invocation;
      global_offset_t current_num_items =
        (partition_idx + 1 == num_partitions)
          ? (num_items - current_partition_offset)
          : capped_num_items_per_invocation;

      // Construct the tile status interface
      const auto num_current_tiles = static_cast<int>(::cuda::ceil_div(current_num_items, tile_size));

      // Construct the tile status interface
      ScanTileStateT tile_status;
      error = CubDebug(tile_status.Init(num_current_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        return error;
      }

      // Log init_kernel configuration
      int init_grid_size = _CUDA_VSTD::max(1, ::cuda::ceil_div(num_current_tiles, init_kernel_threads));

#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking device_scan_init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              init_kernel_threads,
              (long long) stream);
#endif // CUB_DEBUG_LOG

      // Invoke device_scan_init_kernel to initialize tile descriptors and queue descriptors
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(init_grid_size, init_kernel_threads, 0, stream)
        .doit(device_scan_init_kernel, tile_status, num_current_tiles, d_num_runs_out);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        return error;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }

      // Return if empty problem: note, we're initializing d_num_runs_out to 0 in device_scan_init_kernel above
      if (num_items <= 1)
      {
        return error;
      }

// Log device_rle_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking device_rle_sweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per "
              "thread\n",
              num_current_tiles,
              block_threads,
              (long long) stream,
              items_per_thread);
#endif // CUB_DEBUG_LOG

      // Invoke device_rle_sweep_kernel
      if constexpr (use_streaming_invocation)
      {
        auto tmp_num_uniques = static_cast<global_offset_t*>(allocations[1]);
        auto tmp_prefix      = static_cast<length_t*>(allocations[2]);

        const bool is_first_partition = (partition_idx == 0);
        const bool is_last_partition  = (partition_idx + 1 == num_partitions);
        const int buffer_selector     = partition_idx % 2;

        streaming_context_t streaming_context{
          is_first_partition,
          is_last_partition,
          current_partition_offset,
          &tmp_prefix[buffer_selector],
          &tmp_prefix[buffer_selector ^ 0x01],
          &tmp_num_uniques[buffer_selector],
          &tmp_num_uniques[buffer_selector ^ 0x01]};

        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(num_current_tiles, block_threads, 0, stream)
          .doit(device_rle_sweep_kernel,
                d_in + current_partition_offset,
                d_offsets_out,
                d_lengths_out,
                d_num_runs_out,
                tile_status,
                equality_op,
                static_cast<local_offset_t>(current_num_items),
                num_current_tiles,
                streaming_context);
      }
      else
      {
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(num_current_tiles, block_threads, 0, stream)
          .doit(device_rle_sweep_kernel,
                d_in + current_partition_offset,
                d_offsets_out,
                d_lengths_out,
                d_num_runs_out,
                tile_status,
                equality_op,
                static_cast<local_offset_t>(current_num_items),
                num_current_tiles,
                NullType{});
      }

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        return error;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }
    return cudaSuccess;
  }

  template <class ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    return Invoke<ActivePolicyT>(
      detail::scan::DeviceCompactInitKernel<ScanTileStateT, NumRunsOutputIteratorT>,
      detail::rle::DeviceRleSweepKernel<
        typename PolicyHub::MaxPolicy,
        InputIteratorT,
        OffsetsOutputIteratorT,
        LengthsOutputIteratorT,
        NumRunsOutputIteratorT,
        ScanTileStateT,
        EqualityOpT,
        local_offset_t,
        global_offset_t,
        streaming_context_t>);
  }

  /**
   * Internal dispatch routine
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to input sequence of data items
   *
   * @param d_offsets_out
   *   Pointer to output sequence of run-offsets
   *
   * @param d_lengths_out
   *   Pointer to output sequence of run-lengths
   *
   * @param d_num_runs_out
   *   Pointer to total number of runs (i.e., length of `d_offsets_out`)
   *
   * @param equality_op
   *   Equality operator for input items
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OffsetsOutputIteratorT d_offsets_out,
    LengthsOutputIteratorT d_lengths_out,
    NumRunsOutputIteratorT d_num_runs_out,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream)
  {
    cudaError error = cudaSuccess;

    // Get PTX version
    int ptx_version = 0;
    error           = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    DeviceRleDispatch dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_offsets_out,
      d_lengths_out,
      d_num_runs_out,
      equality_op,
      num_items,
      stream);

    // Dispatch
    error = CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
    if (cudaSuccess != error)
    {
      return error;
    }
    return cudaSuccess;
  }
};

CUB_NAMESPACE_END
