
// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce_by_key.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/iterator/offset_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <iostream>
#endif // !_CCCL_COMPILER(NVRTC)

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{

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
struct DispatchStreamingReduceByKey
{
  //-------------------------------------------------------------------------
  // Types and constants
  //-------------------------------------------------------------------------
  // Offsets to index items within one partition (i.e., a single kernel invocation)
  using local_offset_t = _CUDA_VSTD::int32_t;

  // If the number of items provided by the user may exceed the maximum number of items processed by a single kernel
  // invocation, we may require multiple kernel invocations
  static constexpr bool use_streaming_invocation = _CUDA_VSTD::numeric_limits<OffsetT>::max()
                                                 > _CUDA_VSTD::numeric_limits<local_offset_t>::max();

  // Offsets to index any item within the entire input (large enough to cover num_items)
  using global_offset_t = OffsetT;

  // Type used to provide context about the current partition during a streaming invocation
  using streaming_context_t =
    ::cuda::std::conditional_t<use_streaming_invocation,
                               detail::reduce::streaming_context<KeysInputIteratorT, AccumT, global_offset_t>,
                               NullType>;

  // The input values type
  using ValueInputT = cub::detail::it_value_t<ValuesInputIteratorT>;

  static constexpr int init_kernel_threads = 128;

  // Tile status descriptor interface type
  using ScanTileStateT = ReduceByKeyScanTileState<AccumT, local_offset_t>;

  void* d_temp_storage;
  size_t& temp_storage_bytes;
  KeysInputIteratorT d_keys_in;
  UniqueOutputIteratorT d_unique_out;
  ValuesInputIteratorT d_values_in;
  AggregatesOutputIteratorT d_aggregates_out;
  NumRunsOutputIteratorT d_num_runs_out;
  EqualityOpT equality_op;
  ReductionOpT reduction_op;
  global_offset_t num_items;
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchStreamingReduceByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    UniqueOutputIteratorT d_unique_out,
    ValuesInputIteratorT d_values_in,
    AggregatesOutputIteratorT d_aggregates_out,
    NumRunsOutputIteratorT d_num_runs_out,
    EqualityOpT equality_op,
    ReductionOpT reduction_op,
    global_offset_t num_items,
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

    // The upper bound of for the number of items that a single kernel invocation will ever process
    auto capped_num_items_per_invocation = num_items;
    if constexpr (use_streaming_invocation)
    {
      capped_num_items_per_invocation = static_cast<global_offset_t>(_CUDA_VSTD::numeric_limits<local_offset_t>::max());
      // Make sure that the number of items is a multiple of tile size
      capped_num_items_per_invocation -= (capped_num_items_per_invocation % (block_threads * items_per_thread));
    }

    // Across invocations, the maximum number of items that a single kernel invocation will ever process
    const auto max_num_items_per_invocation =
      use_streaming_invocation ? _CUDA_VSTD::min(capped_num_items_per_invocation, num_items) : num_items;

    // Number of invocations required to "iterate" over the total input (at least one iteration to process zero items)
    auto const num_partitions =
      (num_items == 0) ? global_offset_t{1} : ::cuda::ceil_div(num_items, capped_num_items_per_invocation);

    cudaError error = cudaSuccess;

    // Number of input tiles
    const auto tile_size = static_cast<global_offset_t>(block_threads * items_per_thread);
    int max_num_tiles    = static_cast<int>(::cuda::ceil_div(max_num_items_per_invocation, tile_size));

    // Specify temporary storage allocation requirements
    size_t allocation_sizes[3];
    error = CubDebug(ScanTileStateT::AllocationSize(max_num_tiles, allocation_sizes[0]));
    if (cudaSuccess != error)
    {
      return error;
    }
    allocation_sizes[1] = num_partitions > 1 ? sizeof(global_offset_t) * 2 : size_t{0};
    allocation_sizes[2] = num_partitions > 1 ? sizeof(AccumT) * 2 : size_t{0};

    // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
    void* allocations[3] = {};

    error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
    if (cudaSuccess != error)
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

      streaming_context_t streaming_context{};
      if constexpr (use_streaming_invocation)
      {
        auto tmp_num_uniques = static_cast<global_offset_t*>(allocations[1]);
        auto tmp_prefix      = static_cast<AccumT*>(allocations[2]);

        const bool is_first_partition = (partition_idx == 0);
        const bool is_last_partition  = (partition_idx + 1 == num_partitions);
        const int buffer_selector     = partition_idx % 2;

        streaming_context = streaming_context_t{
          is_first_partition,
          is_last_partition,
          is_first_partition ? d_keys_in : d_keys_in + current_partition_offset - 1,
          &tmp_prefix[buffer_selector],
          &tmp_prefix[buffer_selector ^ 0x01],
          &tmp_num_uniques[buffer_selector],
          &tmp_num_uniques[buffer_selector ^ 0x01]};
      }

      // Construct the tile status interface
      const auto num_current_tiles = static_cast<int>(::cuda::ceil_div(current_num_items, tile_size));

      // Construct the tile status interface
      ScanTileStateT tile_state;
      error = CubDebug(tile_state.Init(num_current_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        return error;
      }

      // Log init_kernel configuration
      int init_grid_size = _CUDA_VSTD::max(1, ::cuda::ceil_div(num_current_tiles, init_kernel_threads));

#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, init_kernel_threads, (long long) stream);
#endif // CUB_DEBUG_LOG

      // Invoke init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(init_grid_size, init_kernel_threads, 0, stream)
        .doit(init_kernel, tile_state, num_current_tiles, d_num_runs_out);

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

      // For empty problems we can skip the reduce_by_key_kernel
      if (num_items == 0)
      {
        return error;
      }

// Log reduce_by_key_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking reduce_by_key_kernel<<<%d, %d, 0, %lld>>>(), %d "
              "items per thread\n",
              num_current_tiles,
              block_threads,
              (long long) stream,
              items_per_thread);
#endif // CUB_DEBUG_LOG

      // Invoke reduce_by_key_kernel
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(num_current_tiles, block_threads, 0, stream)
        .doit(reduce_by_key_kernel,
              d_keys_in + current_partition_offset,
              d_unique_out,
              d_values_in + current_partition_offset,
              d_aggregates_out,
              d_num_runs_out,
              tile_state,
              0,
              equality_op,
              reduction_op,
              static_cast<local_offset_t>(current_num_items),
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
    }

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
        local_offset_t,
        AccumT,
        streaming_context_t>);
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
    global_offset_t num_items,
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

      DispatchStreamingReduceByKey dispatch(
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

} // namespace detail::reduce

CUB_NAMESPACE_END
