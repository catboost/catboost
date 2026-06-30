/******************************************************************************
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

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_three_way_partition.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_three_way_partition.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/__algorithm_>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::three_way_partition
{
// Offset type used to instantiate the stream three-way-partition-kernel and agent to index the items within one
// partition
using per_partition_offset_t = ::cuda::std::int32_t;

template <typename TotalNumItemsT>
class streaming_context_t
{
private:
  bool first_partition = true;
  bool last_partition  = false;
  TotalNumItemsT total_previous_num_items{};

  // We use a double-buffer for keeping track of the number of previously selected items
  TotalNumItemsT* d_num_selected_in  = nullptr;
  TotalNumItemsT* d_num_selected_out = nullptr;

public:
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  streaming_context_t(TotalNumItemsT* d_num_selected_in, TotalNumItemsT* d_num_selected_out, bool is_last_partition)
      : last_partition(is_last_partition)
      , d_num_selected_in(d_num_selected_in)
      , d_num_selected_out(d_num_selected_out)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(TotalNumItemsT num_items, bool next_partition_is_the_last)
  {
    ::cuda::std::swap(d_num_selected_in, d_num_selected_out);
    first_partition = false;
    last_partition  = next_partition_is_the_last;
    total_previous_num_items += num_items;
  };

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT input_offset() const
  {
    return first_partition ? TotalNumItemsT{0} : total_previous_num_items;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_selected_first() const
  {
    return first_partition ? TotalNumItemsT{0} : d_num_selected_in[0];
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_selected_second() const
  {
    return first_partition ? TotalNumItemsT{0} : d_num_selected_in[1];
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_rejected() const
  {
    return first_partition ? TotalNumItemsT{0} : d_num_selected_in[2];
    ;
  };

  template <typename NumSelectedIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void update_num_selected(
    NumSelectedIteratorT user_num_selected_out_it,
    TotalNumItemsT num_selected_first,
    TotalNumItemsT num_selected_second,
    TotalNumItemsT num_items_in_partition) const
  {
    if (last_partition)
    {
      user_num_selected_out_it[0] = num_previously_selected_first() + num_selected_first;
      user_num_selected_out_it[1] = num_previously_selected_second() + num_selected_second;
    }
    else
    {
      d_num_selected_out[0] = num_previously_selected_first() + num_selected_first;
      d_num_selected_out[1] = num_previously_selected_second() + num_selected_second;
      d_num_selected_out[2] =
        num_previously_rejected() + (num_items_in_partition - num_selected_second - num_selected_first);
    }
  }
};

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename OffsetT,
          typename StreamingContextT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ThreeWayPartitionPolicy::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceThreeWayPartitionKernel(
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    ScanTileStateT tile_status,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    int num_tiles,
    _CCCL_GRID_CONSTANT const StreamingContextT streaming_context)
{
  using AgentThreeWayPartitionPolicyT = typename ChainedPolicyT::ActivePolicy::ThreeWayPartitionPolicy;

  // Thread block type for selecting data from input tiles
  using AgentThreeWayPartitionT = AgentThreeWayPartition<
    AgentThreeWayPartitionPolicyT,
    InputIteratorT,
    FirstOutputIteratorT,
    SecondOutputIteratorT,
    UnselectedOutputIteratorT,
    SelectFirstPartOp,
    SelectSecondPartOp,
    OffsetT,
    StreamingContextT>;

  // Shared memory for AgentThreeWayPartition
  __shared__ typename AgentThreeWayPartitionT::TempStorage temp_storage;

  // Process tiles
  AgentThreeWayPartitionT(
    temp_storage,
    d_in,
    d_first_part_out,
    d_second_part_out,
    d_unselected_out,
    select_first_part_op,
    select_second_part_op,
    num_items,
    streaming_context)
    .ConsumeRange(num_tiles, tile_status, d_num_selected_out);
}

/**
 * @brief Initialization kernel for tile status initialization (multi-block)
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @param[in] tile_state_1
 *   Tile status interface
 *
 * @param[in] tile_state_2
 *   Tile status interface
 *
 * @param[in] num_tiles
 *   Number of tiles
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected
 *   (i.e., length of @p d_selected_out)
 */
template <typename ScanTileStateT, typename NumSelectedIteratorT>
CUB_DETAIL_KERNEL_ATTRIBUTES void
DeviceThreeWayPartitionInitKernel(ScanTileStateT tile_state, int num_tiles, NumSelectedIteratorT d_num_selected_out)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);

  // Initialize d_num_selected_out
  if (blockIdx.x == 0)
  {
    if (threadIdx.x < 2)
    {
      d_num_selected_out[threadIdx.x] = 0;
    }
  }
}
} // namespace detail::three_way_partition

/******************************************************************************
 * Dispatch
 ******************************************************************************/

template <typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename OffsetT,
          typename PolicyHub = detail::three_way_partition::
            policy_hub<cub::detail::it_value_t<InputIteratorT>, detail::three_way_partition::per_partition_offset_t>>
struct DispatchThreeWayPartitionIf
{
  /*****************************************************************************
   * Types and constants
   ****************************************************************************/

  // Offset type used to instantiate the three-way partition-kernel and agent to index the items within one partition
  using per_partition_offset_t = detail::three_way_partition::per_partition_offset_t;

  // Type used to provide streaming information about each partition's context
  static constexpr per_partition_offset_t partition_size = ::cuda::std::numeric_limits<per_partition_offset_t>::max();

  using streaming_context_t = detail::three_way_partition::streaming_context_t<OffsetT>;

  using AccumPackHelperT = detail::three_way_partition::accumulator_pack_t<per_partition_offset_t>;
  using AccumPackT       = typename AccumPackHelperT::pack_t;
  using ScanTileStateT   = cub::ScanTileState<AccumPackT>;

  static constexpr int INIT_KERNEL_THREADS = 256;

  void* d_temp_storage;
  size_t& temp_storage_bytes;
  InputIteratorT d_in;
  FirstOutputIteratorT d_first_part_out;
  SecondOutputIteratorT d_second_part_out;
  UnselectedOutputIteratorT d_unselected_out;
  NumSelectedIteratorT d_num_selected_out;
  SelectFirstPartOp select_first_part_op;
  SelectSecondPartOp select_second_part_op;
  OffsetT num_items;
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchThreeWayPartitionIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_first_part_out(d_first_part_out)
      , d_second_part_out(d_second_part_out)
      , d_unselected_out(d_unselected_out)
      , d_num_selected_out(d_num_selected_out)
      , select_first_part_op(select_first_part_op)
      , select_second_part_op(select_second_part_op)
      , num_items(num_items)
      , stream(stream)
  {}

  /*****************************************************************************
   * Dispatch entrypoints
   ****************************************************************************/

  template <typename ActivePolicyT, typename ScanInitKernelPtrT, typename SelectIfKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(ScanInitKernelPtrT three_way_partition_init_kernel, SelectIfKernelPtrT three_way_partition_kernel)
  {
    cudaError error = cudaSuccess;

    constexpr int block_threads    = ActivePolicyT::ThreeWayPartitionPolicy::BLOCK_THREADS;
    constexpr int items_per_thread = ActivePolicyT::ThreeWayPartitionPolicy::ITEMS_PER_THREAD;
    constexpr int tile_size        = block_threads * items_per_thread;

    // The maximum number of items for which we will ever invoke the kernel (i.e. largest partition size)
    auto const max_partition_size = static_cast<OffsetT>(
      (::cuda::std::min) (static_cast<uint64_t>(num_items), static_cast<uint64_t>(partition_size)));

    // The number of partitions required to "iterate" over the total input
    auto const num_partitions =
      (max_partition_size == 0) ? OffsetT{1} : ::cuda::ceil_div(num_items, max_partition_size);

    // The maximum number of tiles for which we will ever invoke the kernel
    auto const max_num_tiles_per_invocation = static_cast<OffsetT>(::cuda::ceil_div(max_partition_size, tile_size));

    // For streaming invocations, we need two sets (for double-buffering) of three counters each
    constexpr ::cuda::std::size_t num_counters_per_pass  = 3;
    constexpr ::cuda::std::size_t num_streaming_counters = 2 * num_counters_per_pass;
    ::cuda::std::size_t streaming_selection_storage_bytes =
      (num_partitions > 1) ? num_streaming_counters * sizeof(OffsetT) : ::cuda::std::size_t{0};

    // Specify temporary storage allocation requirements
    size_t allocation_sizes[2] = {0ULL, streaming_selection_storage_bytes};

    error =
      CubDebug(ScanTileStateT::AllocationSize(static_cast<int>(max_num_tiles_per_invocation), allocation_sizes[0]));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
    void* allocations[2] = {};

    error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
    if (cudaSuccess != error)
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      return cudaSuccess;
    }

    // Initialize the streaming context with the temporary storage for double-buffering the previously selected items
    // and the total number (across all partitions) of items
    OffsetT* tmp_num_selected_out = static_cast<OffsetT*>(allocations[1]);
    streaming_context_t streaming_context{
      tmp_num_selected_out, (tmp_num_selected_out + num_counters_per_pass), (num_partitions <= 1)};

    // Iterate over the partitions until all input is processed
    for (OffsetT partition_idx = 0; partition_idx < num_partitions; partition_idx++)
    {
      OffsetT current_partition_offset = partition_idx * max_partition_size;
      OffsetT current_num_items =
        (partition_idx + 1 == num_partitions) ? (num_items - current_partition_offset) : max_partition_size;

      // Construct the tile status interface
      const auto current_num_tiles = static_cast<int>(::cuda::ceil_div(current_num_items, tile_size));

      // Construct the tile status interface
      ScanTileStateT tile_status;
      error = CubDebug(tile_status.Init(current_num_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        return error;
      }

      // Log three_way_partition_init_kernel configuration
      int init_grid_size = _CUDA_VSTD::max(1, ::cuda::ceil_div(current_num_tiles, INIT_KERNEL_THREADS));

#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking three_way_partition_init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              INIT_KERNEL_THREADS,
              reinterpret_cast<long long>(stream));
#endif // CUB_DEBUG_LOG

      // Invoke three_way_partition_init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
        .doit(three_way_partition_init_kernel, tile_status, current_num_tiles, d_num_selected_out);

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

      // No more items to process (note, we do not want to return early for num_items==0, because we need to make sure
      // that `three_way_partition_init_kernel` has written '0' to d_num_selected_out)
      if (current_num_items == 0)
      {
        return cudaSuccess;
      }

// Log select_if_kernel configuration
#ifdef CUB_DEBUG_LOG
      {
        // Get SM occupancy for select_if_kernel
        int range_select_sm_occupancy;
        error = CubDebug(MaxSmOccupancy(range_select_sm_occupancy, // out
                                        three_way_partition_kernel,
                                        block_threads));
        if (cudaSuccess != error)
        {
          return error;
        }

        _CubLog("Invoking three_way_partition_kernel<<<%d, %d, 0, %lld>>>(), %d "
                "items per thread, %d SM occupancy\n",
                current_num_tiles,
                block_threads,
                reinterpret_cast<long long>(stream),
                items_per_thread,
                range_select_sm_occupancy);
      }
#endif // CUB_DEBUG_LOG

      // Invoke select_if_kernel
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(current_num_tiles, block_threads, 0, stream)
        .doit(three_way_partition_kernel,
              d_in,
              d_first_part_out,
              d_second_part_out,
              d_unselected_out,
              d_num_selected_out,
              tile_status,
              select_first_part_op,
              select_second_part_op,
              static_cast<per_partition_offset_t>(current_num_items),
              current_num_tiles,
              streaming_context);

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

      // Prepare streaming context for next partition (swap double buffers, advance number of processed items, etc.)
      streaming_context.advance(current_num_items, (partition_idx + OffsetT{2} == num_partitions));
    }

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MaxPolicyT = typename PolicyHub::MaxPolicy;
    return Invoke<ActivePolicyT>(
      detail::three_way_partition::DeviceThreeWayPartitionInitKernel<ScanTileStateT, NumSelectedIteratorT>,
      detail::three_way_partition::DeviceThreeWayPartitionKernel<
        MaxPolicyT,
        InputIteratorT,
        FirstOutputIteratorT,
        SecondOutputIteratorT,
        UnselectedOutputIteratorT,
        NumSelectedIteratorT,
        ScanTileStateT,
        SelectFirstPartOp,
        SelectSecondPartOp,
        per_partition_offset_t,
        streaming_context_t>);
  }

  /**
   * Internal dispatch routine
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename PolicyHub::MaxPolicy;

    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(cub::PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      DispatchThreeWayPartitionIf dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        select_first_part_op,
        select_second_part_op,
        num_items,
        stream);

      // Dispatch
      error = CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

CUB_NAMESPACE_END
