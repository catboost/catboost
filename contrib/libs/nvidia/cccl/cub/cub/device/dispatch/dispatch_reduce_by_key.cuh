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
 * @file cub::DeviceReduceByKey provides device-wide, parallel operations for
 *       reducing segments of values residing within device-accessible memory.
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

#include <cub/agent/agent_reduce_by_key.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce_by_key.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::reduce
{

template <typename PrecedingKeyItT, typename AccumT, typename GlobalOffsetT>
struct streaming_context
{
  bool first_partition;
  bool last_partition;
  PrecedingKeyItT preceding_key_it;

  // We use a double-buffer to track the aggregate of the last run of the previous partition
  AccumT* preceding_prefix;
  AccumT* prefix_out;

  // We use a double-buffer to track the number of runs of previous partition
  GlobalOffsetT* d_num_previous_uniques_in;
  GlobalOffsetT* d_num_accumulated_uniques_out;

  _CCCL_DEVICE _CCCL_FORCEINLINE GlobalOffsetT num_accumulated_uniques_out() const
  {
    return first_partition ? GlobalOffsetT{0} : *d_num_previous_uniques_in;
  };

  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE bool is_first_partition() const
  {
    return first_partition;
  }
  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE bool is_last_partition() const
  {
    return last_partition;
  }

  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE auto predecessor_key() const
  {
    return *preceding_key_it;
  }

  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE AccumT prefix() const
  {
    return *preceding_prefix;
  }

  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE void write_prefix(AccumT prefix) const
  {
    *prefix_out = prefix;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE GlobalOffsetT* previous_uniques_ptr() const
  {
    return d_num_previous_uniques_in;
  }

  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE GlobalOffsetT num_uniques() const
  {
    return num_accumulated_uniques_out();
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
 * @brief Multi-block reduce-by-key sweep kernel entry point
 *
 * @tparam AgentReduceByKeyPolicyT
 *   Parameterized AgentReduceByKeyPolicyT tuning policy type
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam UniqueOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam AggregatesOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording number of segments encountered
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam EqualityOpT
 *   KeyT equality operator type
 *
 * @tparam ReductionOpT
 *   ValueT reduction operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param d_keys_in
 *   Pointer to the input sequence of keys
 *
 * @param d_unique_out
 *   Pointer to the output sequence of unique keys (one key per run)
 *
 * @param d_values_in
 *   Pointer to the input sequence of corresponding values
 *
 * @param d_aggregates_out
 *   Pointer to the output sequence of value aggregates (one aggregate per run)
 *
 * @param d_num_runs_out
 *   Pointer to total number of runs encountered
 *   (i.e., the length of d_unique_out)
 *
 * @param tile_state
 *   Tile status interface
 *
 * @param start_tile
 *   The starting tile for the current grid
 *
 * @param equality_op
 *   KeyT equality operator
 *
 * @param reduction_op
 *   ValueT reduction operator
 *
 * @param num_items
 *   Total number of items to select from
 */
template <typename ChainedPolicyT,
          typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename ScanTileStateT,
          typename EqualityOpT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT,
          typename StreamingContextT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReduceByKeyPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceReduceByKeyKernel(
    KeysInputIteratorT d_keys_in,
    UniqueOutputIteratorT d_unique_out,
    ValuesInputIteratorT d_values_in,
    AggregatesOutputIteratorT d_aggregates_out,
    NumRunsOutputIteratorT d_num_runs_out,
    ScanTileStateT tile_state,
    int start_tile,
    EqualityOpT equality_op,
    ReductionOpT reduction_op,
    OffsetT num_items,
    _CCCL_GRID_CONSTANT const StreamingContextT streaming_context)
{
  using AgentReduceByKeyPolicyT = typename ChainedPolicyT::ActivePolicy::ReduceByKeyPolicyT;

  // Thread block type for reducing tiles of value segments
  using AgentReduceByKeyT = AgentReduceByKey<
    AgentReduceByKeyPolicyT,
    KeysInputIteratorT,
    UniqueOutputIteratorT,
    ValuesInputIteratorT,
    AggregatesOutputIteratorT,
    NumRunsOutputIteratorT,
    EqualityOpT,
    ReductionOpT,
    OffsetT,
    AccumT,
    StreamingContextT>;

  // Shared memory for AgentReduceByKey
  __shared__ typename AgentReduceByKeyT::TempStorage temp_storage;

  // Process tiles
  AgentReduceByKeyT(
    temp_storage,
    d_keys_in,
    d_unique_out,
    d_values_in,
    d_aggregates_out,
    d_num_runs_out,
    equality_op,
    reduction_op,
    streaming_context)
    .ConsumeRange(num_items, tile_state, start_tile);
}

} // namespace detail::reduce

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        DeviceReduceByKey
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam UniqueOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam AggregatesOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam NumRunsOutputIteratorT
 *   Output iterator type for recording number of segments encountered
 *
 * @tparam EqualityOpT
 *   KeyT equality operator type
 *
 * @tparam ReductionOpT
 *   ValueT reduction operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam PolicyHub
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename EqualityOpT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT    = ::cuda::std::__accumulator_t<ReductionOpT,
                                                            cub::detail::it_value_t<ValuesInputIteratorT>,
                                                            cub::detail::it_value_t<ValuesInputIteratorT>>,
          typename PolicyHub = detail::reduce_by_key::policy_hub<
            ReductionOpT,
            AccumT,
            cub::detail::non_void_value_t<UniqueOutputIteratorT, cub::detail::it_value_t<KeysInputIteratorT>>>>
struct DispatchReduceByKey
{
  //-------------------------------------------------------------------------
  // Types and constants
  //-------------------------------------------------------------------------

  // The input values type
  using ValueInputT = cub::detail::it_value_t<ValuesInputIteratorT>;

  static constexpr int INIT_KERNEL_THREADS = 128;

  // Tile status descriptor interface type
  using ScanTileStateT = ReduceByKeyScanTileState<AccumT, OffsetT>;

  void* d_temp_storage;
  size_t& temp_storage_bytes;
  KeysInputIteratorT d_keys_in;
  UniqueOutputIteratorT d_unique_out;
  ValuesInputIteratorT d_values_in;
  AggregatesOutputIteratorT d_aggregates_out;
  NumRunsOutputIteratorT d_num_runs_out;
  EqualityOpT equality_op;
  ReductionOpT reduction_op;
  OffsetT num_items;
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchReduceByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    UniqueOutputIteratorT d_unique_out,
    ValuesInputIteratorT d_values_in,
    AggregatesOutputIteratorT d_aggregates_out,
    NumRunsOutputIteratorT d_num_runs_out,
    EqualityOpT equality_op,
    ReductionOpT reduction_op,
    OffsetT num_items,
    cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_unique_out(d_unique_out)
      , d_values_in(d_values_in)
      , d_aggregates_out(d_aggregates_out)
      , d_num_runs_out(d_num_runs_out)
      , equality_op(equality_op)
      , reduction_op(reduction_op)
      , num_items(num_items)
      , stream(stream)
  {}

  //---------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------

  template <typename ActivePolicyT, typename ScanInitKernelT, typename ReduceByKeyKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  Invoke(ScanInitKernelT init_kernel, ReduceByKeyKernelT reduce_by_key_kernel)
  {
    using AgentReduceByKeyPolicyT  = typename ActivePolicyT::ReduceByKeyPolicyT;
    constexpr int block_threads    = AgentReduceByKeyPolicyT::BLOCK_THREADS;
    constexpr int items_per_thread = AgentReduceByKeyPolicyT::ITEMS_PER_THREAD;

    cudaError error = cudaSuccess;
    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Number of input tiles
      int tile_size = block_threads * items_per_thread;
      int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[1];
      error = CubDebug(ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break; // bytes needed for tile status descriptors
      }

      // Compute allocation pointers into the single storage blob (or compute
      // the necessary size of the blob)
      void* allocations[1] = {};

      error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_state;
      error = CubDebug(tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Log init_kernel configuration
      int init_grid_size = _CUDA_VSTD::max(1, ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS));

#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);
#endif // CUB_DEBUG_LOG

      // Invoke init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
        .doit(init_kernel, tile_state, num_tiles, d_num_runs_out);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Return if empty problem: note, we're initializing d_num_runs_out to 0 in init_kernel above
      if (num_items == 0)
      {
        break;
      }

      // Get SM occupancy for reduce_by_key_kernel
      int reduce_by_key_sm_occupancy;
      error = CubDebug(MaxSmOccupancy(reduce_by_key_sm_occupancy, reduce_by_key_kernel, block_threads));

      if (cudaSuccess != error)
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      error = CubDebug(cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Run grids in epochs (in case number of tiles exceeds max x-dimension
      int scan_grid_size = _CUDA_VSTD::min(num_tiles, max_dim_x);
      for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
      {
// Log reduce_by_key_kernel configuration
#ifdef CUB_DEBUG_LOG
        _CubLog("Invoking %d reduce_by_key_kernel<<<%d, %d, 0, %lld>>>(), %d "
                "items per thread, %d SM occupancy\n",
                start_tile,
                scan_grid_size,
                block_threads,
                (long long) stream,
                items_per_thread,
                reduce_by_key_sm_occupancy);
#endif // CUB_DEBUG_LOG

        // Invoke reduce_by_key_kernel
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(scan_grid_size, block_threads, 0, stream)
          .doit(reduce_by_key_kernel,
                d_keys_in,
                d_unique_out,
                d_values_in,
                d_aggregates_out,
                d_num_runs_out,
                tile_state,
                start_tile,
                equality_op,
                reduction_op,
                num_items,
                NullType{});

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }

        // Sync the stream if specified to flush runtime errors
        error = CubDebug(detail::DebugSyncStream(stream));
        if (cudaSuccess != error)
        {
          break;
        }
      }
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    return Invoke<ActivePolicyT>(
      detail::scan::DeviceCompactInitKernel<ScanTileStateT, NumRunsOutputIteratorT>,
      detail::reduce::DeviceReduceByKeyKernel<
        typename PolicyHub::MaxPolicy,
        KeysInputIteratorT,
        UniqueOutputIteratorT,
        ValuesInputIteratorT,
        AggregatesOutputIteratorT,
        NumRunsOutputIteratorT,
        ScanTileStateT,
        EqualityOpT,
        ReductionOpT,
        OffsetT,
        AccumT,
        NullType>);
  }

  /**
   * Internal dispatch routine
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input sequence of keys
   *
   * @param[out] d_unique_out
   *   Pointer to the output sequence of unique keys (one key per run)
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of corresponding values
   *
   * @param[out] d_aggregates_out
   *   Pointer to the output sequence of value aggregates
   *   (one aggregate per run)
   *
   * @param[out] d_num_runs_out
   *   Pointer to total number of runs encountered
   *   (i.e., the length of d_unique_out)
   *
   * @param[in] equality_op
   *   KeyT equality operator
   *
   * @param[in] reduction_op
   *   ValueT reduction operator
   *
   * @param[in] num_items
   *   Total number of items to select from
   *
   * @param[in] stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    UniqueOutputIteratorT d_unique_out,
    ValuesInputIteratorT d_values_in,
    AggregatesOutputIteratorT d_aggregates_out,
    NumRunsOutputIteratorT d_num_runs_out,
    EqualityOpT equality_op,
    ReductionOpT reduction_op,
    OffsetT num_items,
    cudaStream_t stream)
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      DispatchReduceByKey dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_unique_out,
        d_values_in,
        d_aggregates_out,
        d_num_runs_out,
        equality_op,
        reduction_op,
        num_items,
        stream);

      // Dispatch
      error = CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

CUB_NAMESPACE_END
