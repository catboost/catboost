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
 *   cub::DeviceSelect provides device-wide, parallel operations for selecting items from sequences
 *   of data items residing within device-accessible memory.
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

#include <cub/agent/agent_select_if.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_select_if.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/__algorithm_>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::select
{
// Offset type used to instantiate the stream compaction-kernel and agent to index the items within one partition
using per_partition_offset_t = ::cuda::std::int32_t;

template <typename TotalNumItemsT, bool IsStreamingInvocation>
class streaming_context_t
{
private:
  bool first_partition = true;
  bool last_partition  = false;
  TotalNumItemsT total_num_items{};
  TotalNumItemsT total_previous_num_items{};

  // We use a double-buffer for keeping track of the number of previously selected items
  TotalNumItemsT* d_num_selected_in  = nullptr;
  TotalNumItemsT* d_num_selected_out = nullptr;

public:
  using total_num_items_t = TotalNumItemsT;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE streaming_context_t(
    TotalNumItemsT* d_num_selected_in,
    TotalNumItemsT* d_num_selected_out,
    TotalNumItemsT total_num_items,
    bool is_last_partition)
      : last_partition(is_last_partition)
      , total_num_items(total_num_items)
      , d_num_selected_in(d_num_selected_in)
      , d_num_selected_out(d_num_selected_out)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(TotalNumItemsT num_items, bool next_partition_is_the_last)
  {
    using ::cuda::std::swap;
    swap(d_num_selected_in, d_num_selected_out);
    first_partition = false;
    last_partition  = next_partition_is_the_last;
    total_previous_num_items += num_items;
  };

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT input_offset() const
  {
    return first_partition ? TotalNumItemsT{0} : total_previous_num_items;
  };

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT is_first_partition() const
  {
    return first_partition;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_selected() const
  {
    return first_partition ? TotalNumItemsT{0} : *d_num_selected_in;
  };

  _CCCL_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_rejected() const
  {
    return first_partition ? TotalNumItemsT{0} : (total_previous_num_items - num_previously_selected());
  };

  template <typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_total_items(OffsetT) const
  {
    return total_num_items;
  }

  template <typename NumSelectedIteratorT, typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void
  update_num_selected(NumSelectedIteratorT user_num_selected_out_it, OffsetT num_selections) const
  {
    if (last_partition)
    {
      *user_num_selected_out_it = num_previously_selected() + static_cast<TotalNumItemsT>(num_selections);
    }
    else
    {
      *d_num_selected_out = num_previously_selected() + static_cast<TotalNumItemsT>(num_selections);
    }
  }
};

template <typename TotalNumItemsT>
class streaming_context_t<TotalNumItemsT, false>
{
public:
  using total_num_items_t = TotalNumItemsT;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE streaming_context_t(TotalNumItemsT*, TotalNumItemsT*, TotalNumItemsT, bool) {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(TotalNumItemsT, bool) {};

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT input_offset() const
  {
    return TotalNumItemsT{0};
  };

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT is_first_partition() const
  {
    return true;
  };

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_selected() const
  {
    return TotalNumItemsT{0};
  };

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_previously_rejected() const
  {
    return TotalNumItemsT{0};
  };

  template <typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE TotalNumItemsT num_total_items(OffsetT num_partition_items) const
  {
    return num_partition_items;
  }

  template <typename NumSelectedIteratorT, typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void
  update_num_selected(NumSelectedIteratorT user_num_selected_out_it, OffsetT num_selections) const
  {
    *user_num_selected_out_it = num_selections;
  }
};

/**
 * @brief Wrapper that partially specializes the `AgentSelectIf` on the non-type name parameter `KeepRejects`.
 */
template <SelectImpl SelectionOpt>
struct agent_select_if_wrapper_t
{
  // Using an explicit list of template parameters forwarded to AgentSelectIf, since MSVC complains about a template
  // argument following a parameter pack expansion like `AgentSelectIf<Ts..., KeepRejects, MayAlias>`
  template <typename AgentSelectIfPolicyT,
            typename InputIteratorT,
            typename FlagsInputIteratorT,
            typename SelectedOutputIteratorT,
            typename SelectOpT,
            typename EqualityOpT,
            typename OffsetT,
            typename StreamingContextT>
  struct agent_t
      : public AgentSelectIf<AgentSelectIfPolicyT,
                             InputIteratorT,
                             FlagsInputIteratorT,
                             SelectedOutputIteratorT,
                             SelectOpT,
                             EqualityOpT,
                             OffsetT,
                             StreamingContextT,
                             SelectionOpt>
  {
    using AgentSelectIf<AgentSelectIfPolicyT,
                        InputIteratorT,
                        FlagsInputIteratorT,
                        SelectedOutputIteratorT,
                        SelectOpT,
                        EqualityOpT,
                        OffsetT,
                        StreamingContextT,
                        SelectionOpt>::AgentSelectIf;
  };
};

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Select kernel entry point (multi-block)
 *
 * Performs functor-based selection if SelectOpT functor type != NullType
 * Otherwise performs flag-based selection if FlagsInputIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items
 *
 * @tparam FlagsInputIteratorT
 *   Random-access input iterator type for reading selection flags (NullType* if a selection functor
 *   or discontinuity flagging is to be used for selection)
 *
 * @tparam SelectedOutputIteratorT
 *   Random-access output iterator type for writing selected items
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam SelectOpT
 *   Selection operator type (NullType if selection flags or discontinuity flagging is
 *   to be used for selection)
 *
 * @tparam EqualityOpT
 *   Equality operator type (NullType if selection functor or selection flags is
 *   to be used for selection)
 *
 * @tparam OffsetT
 *   Signed integer type for offsets within a partition
 *
 * @tparam StreamingContextT
 *   Type providing the context information for the current partition, with the following member functions:
 *    input_offset() -> base offset for the input (and flags) iterator
 *    num_previously_selected() -> base offset for the output iterator for selected items
 *    num_previously_rejected() -> base offset for the output iterator for rejected items (partition only)
 *    num_total_items() -> total number of items across all partitions (partition only)
 *    update_num_selected(d_num_sel_out, num_selected) -> invoked by last CTA with number of selected
 *
 * @tparam KeepRejects
 *   Whether or not we push rejected items to the back of the output
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[in] d_flags
 *   Pointer to the input sequence of selection flags (if applicable)
 *
 * @param[out] d_selected_out
 *   Pointer to the output sequence of selected data items
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected (i.e., length of \p d_selected_out)
 *
 * @param[in] tile_status
 *   Tile status interface
 *
 * @param[in] select_op
 *   Selection operator
 *
 * @param[in] equality_op
 *   Equality operator
 *
 * @param[in] num_items
 *   Total number of input items (i.e., length of \p d_in)
 *
 * @param[in] num_tiles
 *   Total number of tiles for the entire problem
 *
 * @param[in] streaming_context
 *   The context information for the current partition
 *
 * @param[in] vsmem
 *   Memory to support virtual shared memory
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename FlagsInputIteratorT,
          typename SelectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename SelectOpT,
          typename EqualityOpT,
          typename OffsetT,
          typename StreamingContextT,
          SelectImpl SelectionOpt>
__launch_bounds__(int(
  vsmem_helper_default_fallback_policy_t<
    typename ChainedPolicyT::ActivePolicy::SelectIfPolicyT,
    agent_select_if_wrapper_t<SelectionOpt>::template agent_t,
    InputIteratorT,
    FlagsInputIteratorT,
    SelectedOutputIteratorT,
    SelectOpT,
    EqualityOpT,
    OffsetT,
    StreamingContextT>::agent_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSelectSweepKernel(
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    ScanTileStateT tile_status,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    int num_tiles,
    _CCCL_GRID_CONSTANT const StreamingContextT streaming_context,
    vsmem_t vsmem)
{
  using VsmemHelperT = vsmem_helper_default_fallback_policy_t<
    typename ChainedPolicyT::ActivePolicy::SelectIfPolicyT,
    agent_select_if_wrapper_t<SelectionOpt>::template agent_t,
    InputIteratorT,
    FlagsInputIteratorT,
    SelectedOutputIteratorT,
    SelectOpT,
    EqualityOpT,
    OffsetT,
    StreamingContextT>;

  using AgentSelectIfPolicyT = typename VsmemHelperT::agent_policy_t;

  // Thread block type for selecting data from input tiles
  using AgentSelectIfT = typename VsmemHelperT::agent_t;

  // Static shared memory allocation
  __shared__ typename VsmemHelperT::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename AgentSelectIfT::TempStorage& temp_storage = VsmemHelperT::get_temp_storage(static_temp_storage, vsmem);

  // Process tiles
  AgentSelectIfT(temp_storage, d_in, d_flags, d_selected_out, select_op, equality_op, num_items, streaming_context)
    .ConsumeRange(num_tiles, tile_status, d_num_selected_out);

  // If applicable, hints to discard modified cache lines for vsmem
  VsmemHelperT::discard_temp_storage(temp_storage);
}
} // namespace detail::select

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSelect and DevicePartition
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items
 *
 * @tparam FlagsInputIteratorT
 *   Random-access input iterator type for reading selection flags (NullType* if a selection functor or discontinuity
 *   flagging is used for selection)
 *
 * @tparam SelectedOutputIteratorT
 *   Random-access output iterator type for writing selected items
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @tparam SelectOpT
 *   Selection operator type (NullType if selection flags or discontinuity flagging is used for selection)
 *
 * @tparam EqualityOpT
 *   Equality operator type (NullType if selection functor or selection flags are used for selection)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam SelectionOpt
 *   SelectImpl indicating whether to partition, just selection or selection where the memory for the input and
 *   output may alias each other.
 */
template <
  typename InputIteratorT,
  typename FlagsInputIteratorT,
  typename SelectedOutputIteratorT,
  typename NumSelectedIteratorT,
  typename SelectOpT,
  typename EqualityOpT,
  typename OffsetT,
  SelectImpl SelectionOpt,
  typename PolicyHub = detail::select::policy_hub<
    detail::it_value_t<InputIteratorT>,
    detail::it_value_t<FlagsInputIteratorT>,
    // if/flagged/unique only have a single code path for different offset types, partition has different code paths
    ::cuda::std::conditional_t<SelectionOpt == SelectImpl::Partition, OffsetT, detail::select::per_partition_offset_t>,
    detail::select::is_partition_distinct_output_t<SelectedOutputIteratorT>::value,
    SelectionOpt>>
struct DispatchSelectIf
{
  /******************************************************************************
   * Types and constants
   ******************************************************************************/

  // Offset type used to instantiate the stream compaction-kernel and agent to index the items within one partition
  using per_partition_offset_t = detail::select::per_partition_offset_t;

  // Offset type large enough to represent any index within the input and output iterators
  using num_total_items_t = OffsetT;

  // Type used to provide streaming information about each partition's context
  static constexpr per_partition_offset_t partition_size = ::cuda::std::numeric_limits<per_partition_offset_t>::max();

  // If the values representable by OffsetT exceed the partition_size, we use a kernel template specialization that
  // supports streaming (i.e., splitting the input into partitions of up to partition_size number of items)
  static constexpr bool may_require_streaming =
    (static_cast<::cuda::std::uint64_t>(partition_size)
     < static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<OffsetT>::max()));

  using streaming_context_t = detail::select::streaming_context_t<num_total_items_t, may_require_streaming>;

  using ScanTileStateT = ScanTileState<per_partition_offset_t>;

  static constexpr int INIT_KERNEL_THREADS = 128;

  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to `temp_storage_bytes`
  /// and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the input sequence of selection flags (if applicable)
  FlagsInputIteratorT d_flags;

  /// Pointer to the output sequence of selected data items
  SelectedOutputIteratorT d_selected_out;

  /// Pointer to the total number of items selected (i.e., length of `d_selected_out`)
  NumSelectedIteratorT d_num_selected_out;

  /// Selection operator
  SelectOpT select_op;

  /// Equality operator
  EqualityOpT equality_op;

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  /**
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When `nullptr`, the required allocation size is written to `temp_storage_bytes`
   *   and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to the input sequence of data items
   *
   * @param d_flags
   *   Pointer to the input sequence of selection flags (if applicable)
   *
   * @param d_selected_out
   *   Pointer to the output sequence of selected data items
   *
   * @param d_num_selected_out
   *  Pointer to the total number of items selected (i.e., length of `d_selected_out`)
   *
   * @param select_op
   *   Selection operator
   *
   * @param equality_op
   *   Equality operator
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchSelectIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_flags(d_flags)
      , d_selected_out(d_selected_out)
      , d_num_selected_out(d_num_selected_out)
      , select_op(select_op)
      , equality_op(equality_op)
      , num_items(num_items)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/

  /**
   * Internal dispatch routine for computing a device-wide selection using the
   * specified kernel functions.
   */
  template <typename ActivePolicyT, typename ScanInitKernelPtrT, typename SelectIfKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(ScanInitKernelPtrT scan_init_kernel, SelectIfKernelPtrT select_if_kernel)
  {
    using Policy = typename ActivePolicyT::SelectIfPolicyT;

    using VsmemHelperT = cub::detail::vsmem_helper_default_fallback_policy_t<
      Policy,
      detail::select::agent_select_if_wrapper_t<SelectionOpt>::template agent_t,
      InputIteratorT,
      FlagsInputIteratorT,
      SelectedOutputIteratorT,
      SelectOpT,
      EqualityOpT,
      per_partition_offset_t,
      streaming_context_t>;
    cudaError error = cudaSuccess;

    constexpr auto block_threads    = VsmemHelperT::agent_policy_t::BLOCK_THREADS;
    constexpr auto items_per_thread = VsmemHelperT::agent_policy_t::ITEMS_PER_THREAD;
    constexpr auto tile_size        = static_cast<OffsetT>(block_threads * items_per_thread);

    // The maximum number of items for which we will ever invoke the kernel (i.e. largest partition size)
    // The extra check of may_require_streaming ensures that OffsetT is larger than per_partition_offset_t to avoid
    // truncation of partition_size
    auto const max_partition_size =
      (may_require_streaming && num_items > static_cast<OffsetT>(partition_size))
        ? static_cast<OffsetT>(partition_size)
        : num_items;

    // The number of partitions required to "iterate" over the total input (ternary to avoid div-by-zero)
    auto const num_partitions =
      (max_partition_size == 0) ? static_cast<OffsetT>(1) : ::cuda::ceil_div(num_items, max_partition_size);

    // The maximum number of tiles for which we will ever invoke the kernel
    auto const max_num_tiles_per_invocation = static_cast<OffsetT>(::cuda::ceil_div(max_partition_size, tile_size));

    // The amount of virtual shared memory to allocate
    const auto vsmem_size = max_num_tiles_per_invocation * VsmemHelperT::vsmem_per_block;

    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Specify temporary storage allocation requirements
      ::cuda::std::size_t streaming_selection_storage_bytes =
        (num_partitions > 1) ? 2 * sizeof(num_total_items_t) : ::cuda::std::size_t{0};
      ::cuda::std::size_t allocation_sizes[3] = {0ULL, vsmem_size, streaming_selection_storage_bytes};

      // Bytes needed for tile status descriptors
      error =
        CubDebug(ScanTileStateT::AllocationSize(static_cast<int>(max_num_tiles_per_invocation), allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
      void* allocations[3] = {};

      error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      // Initialize the streaming context with the temporary storage for double-buffering the previously selected items
      // and the total number (across all partitions) of items
      num_total_items_t* tmp_num_selected_out = reinterpret_cast<num_total_items_t*>(allocations[2]);
      streaming_context_t streaming_context{
        tmp_num_selected_out, (tmp_num_selected_out + 1), num_items, (num_partitions <= 1)};

      // Iterate over the partitions until all input is processed
      for (OffsetT partition_idx = 0; partition_idx < num_partitions; partition_idx++)
      {
        OffsetT current_partition_offset = partition_idx * max_partition_size;
        OffsetT current_num_items =
          (partition_idx + 1 == num_partitions) ? (num_items - current_partition_offset) : max_partition_size;

        // Construct the tile status interface
        const auto current_num_tiles = static_cast<int>(::cuda::ceil_div(current_num_items, tile_size));
        ScanTileStateT tile_status;
        error = CubDebug(tile_status.Init(current_num_tiles, allocations[0], allocation_sizes[0]));
        if (cudaSuccess != error)
        {
          return error;
        }

        // Log scan_init_kernel configuration
        int init_grid_size = _CUDA_VSTD::max(1, ::cuda::ceil_div(current_num_tiles, INIT_KERNEL_THREADS));

#ifdef CUB_DEBUG_LOG
        _CubLog("Invoking scan_init_kernel<<<%d, %d, 0, %lld>>>()\n",
                init_grid_size,
                INIT_KERNEL_THREADS,
                (long long) stream);
#endif

        // Invoke scan_init_kernel to initialize tile descriptors
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
          .doit(scan_init_kernel, tile_status, current_num_tiles, d_num_selected_out);

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
        // that `scan_init_kernel` has written '0' to d_num_selected_out)
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
                                          select_if_kernel,
                                          block_threads));
          if (cudaSuccess != error)
          {
            return error;
          }

          _CubLog("Invoking select_if_kernel<<<%d, %d, 0, "
                  "%lld>>>(), %d items per thread, %d SM occupancy\n",
                  current_num_tiles,
                  block_threads,
                  (long long) stream,
                  items_per_thread,
                  range_select_sm_occupancy);
        }
#endif

        // Invoke select_if_kernel
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(current_num_tiles, block_threads, 0, stream)
          .doit(select_if_kernel,
                d_in,
                d_flags,
                d_selected_out,
                d_num_selected_out,
                tile_status,
                select_op,
                equality_op,
                static_cast<per_partition_offset_t>(current_num_items),
                current_num_tiles,
                streaming_context,
                cub::detail::vsmem_t{allocations[1]});

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
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    return Invoke<ActivePolicyT>(
      detail::scan::DeviceCompactInitKernel<ScanTileStateT, NumSelectedIteratorT>,
      detail::select::DeviceSelectSweepKernel<
        typename PolicyHub::MaxPolicy,
        InputIteratorT,
        FlagsInputIteratorT,
        SelectedOutputIteratorT,
        NumSelectedIteratorT,
        ScanTileStateT,
        SelectOpT,
        EqualityOpT,
        per_partition_offset_t,
        streaming_context_t,
        SelectionOpt>);
  }

  /**
   * Internal dispatch routine
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When `nullptr`, the required allocation size is written to `temp_storage_bytes`
   *   and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to the input sequence of data items
   *
   * @param d_flags
   *   Pointer to the input sequence of selection flags (if applicable)
   *
   * @param d_selected_out
   *   Pointer to the output sequence of selected data items
   *
   * @param d_num_selected_out
   *  Pointer to the total number of items selected (i.e., length of `d_selected_out`)
   *
   * @param select_op
   *   Selection operator
   *
   * @param equality_op
   *   Equality operator
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream)
  {
    int ptx_version = 0;
    if (cudaError_t error = CubDebug(PtxVersion(ptx_version)))
    {
      return error;
    }

    DispatchSelectIf dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_flags,
      d_selected_out,
      d_num_selected_out,
      select_op,
      equality_op,
      num_items,
      stream,
      ptx_version);

    return CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
  }
};

CUB_NAMESPACE_END
