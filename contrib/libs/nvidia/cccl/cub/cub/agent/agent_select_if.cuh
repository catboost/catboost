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
 * \file
 * cub::AgentSelectIf implements a stateful abstraction of CUDA thread blocks for participating in device-wide select.
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

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentSelectIf
 *
 * @tparam _BLOCK_THREADS
 *   Threads per thread block
 *
 * @tparam _ITEMS_PER_THREAD
 *   Items per thread (per tile of input)
 *
 * @tparam _LOAD_ALGORITHM
 *   The BlockLoad algorithm to use
 *
 * @tparam _LOAD_MODIFIER
 *   Cache load modifier for reading input elements
 *
 * @tparam _SCAN_ALGORITHM
 *   The BlockScan algorithm to use
 *
 * @tparam DelayConstructorT
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD,
          BlockLoadAlgorithm _LOAD_ALGORITHM,
          CacheLoadModifier _LOAD_MODIFIER,
          BlockScanAlgorithm _SCAN_ALGORITHM,
          typename DelayConstructorT = detail::fixed_delay_constructor_t<350, 450>>
struct AgentSelectIfPolicy
{
  enum
  {
    /// Threads per thread block
    BLOCK_THREADS = _BLOCK_THREADS,

    /// Items per thread (per tile of input)
    ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
  };

  /// The BlockLoad algorithm to use
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;

  /// Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;

  /// The BlockScan algorithm to use
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;

  struct detail
  {
    using delay_constructor_t = DelayConstructorT;
  };
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

namespace detail
{
namespace select
{

template <typename EqualityOpT>
struct guarded_inequality_op
{
  EqualityOpT op;
  int num_remaining;

  template <typename T,
            ::cuda::std::enable_if_t<::cuda::std::__is_callable_v<EqualityOpT&, const T&, const T&>, int> = 0>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(const T& a, const T& b, int idx) noexcept(
    ::cuda::std::__is_nothrow_callable_v<EqualityOpT&, const T&, const T&>)
  {
    if (idx < num_remaining)
    {
      return !op(a, b); // In bounds
    }

    // Flag out-of-bounds items as selected (as they are discounted for in the agent implementation)
    return true;
  }

  template <typename T,
            ::cuda::std::enable_if_t<::cuda::std::__is_callable_v<const EqualityOpT&, const T&, const T&>, int> = 0>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(const T& a, const T& b, int idx) const
    noexcept(::cuda::std::__is_nothrow_callable_v<const EqualityOpT&, const T&, const T&>)
  {
    if (idx < num_remaining)
    {
      return !op(a, b); // In bounds
    }

    // Flag out-of-bounds items as selected (as they are discounted for in the agent implementation)
    return true;
  }
};

template <typename SelectedOutputItT, typename RejectedOutputItT>
struct partition_distinct_output_t
{
  using selected_iterator_t = SelectedOutputItT;
  using rejected_iterator_t = RejectedOutputItT;

  selected_iterator_t selected_it;
  rejected_iterator_t rejected_it;
};

template <typename OutputIterator>
struct is_partition_distinct_output_t : ::cuda::std::false_type
{};

template <typename SelectedOutputItT, typename RejectedOutputItT>
struct is_partition_distinct_output_t<partition_distinct_output_t<SelectedOutputItT, RejectedOutputItT>>
    : ::cuda::std::true_type
{};

/**
 * @brief AgentSelectIf implements a stateful abstraction of CUDA thread blocks for participating in
 * device-wide selection
 *
 * Performs functor-based selection if SelectOpT functor type != NullType
 * Otherwise performs flag-based selection if FlagsInputIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 *
 * @tparam AgentSelectIfPolicyT
 *   Parameterized AgentSelectIfPolicy tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for selection items
 *
 * @tparam FlagsInputIteratorT
 *   Random-access input iterator type for selections (NullType* if a selection functor or
 *   discontinuity flagging is to be used for selection)
 *
 * @tparam OutputIteratorWrapperT
 *   Either a random-access iterator or an instance of the `partition_distinct_output_t` template.
 *
 * @tparam SelectOpT
 *   Selection operator type (NullType if selections or discontinuity flagging is to be used for
 * selection)
 *
 * @tparam EqualityOpT
 *   Equality operator type (NullType if selection functor or selections is to be used for
 * selection)
 *
 * @tparam OffsetT
 *   Signed integer type for offsets within a partition
 *
 * @tparam StreamingContextT
 *   Type providing the context information for the current partition, with the following member functions:
 *    input_offset() -> base offset for the input (and flags) iterator
 *    is_first_partition() -> [Select::Unique-only] whether this is the first partition
 *    num_previously_selected() -> base offset for the output iterator for selected items
 *    num_previously_rejected() -> base offset for the output iterator for rejected items (partition only)
 *    num_total_items() -> total number of items across all partitions (partition only)
 *    update_num_selected(d_num_sel_out, num_selected) -> invoked by last CTA with number of selected
 *
 * @tparam SelectImpl SelectionOpt
 *   SelectImpl indicating whether to partition, just selection or selection where the memory for the input and
 *   output may alias each other.
 */
template <typename AgentSelectIfPolicyT,
          typename InputIteratorT,
          typename FlagsInputIteratorT,
          typename OutputIteratorWrapperT,
          typename SelectOpT,
          typename EqualityOpT,
          typename OffsetT,
          typename StreamingContextT,
          SelectImpl SelectionOpt>
struct AgentSelectIf
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------
  using ScanTileStateT = ScanTileState<OffsetT>;

  // Indicates whether the BlockLoad algorithm uses shared memory to load or exchange the data
  static constexpr bool loads_via_smem =
    !(AgentSelectIfPolicyT::LOAD_ALGORITHM == BLOCK_LOAD_DIRECT
      || AgentSelectIfPolicyT::LOAD_ALGORITHM == BLOCK_LOAD_STRIPED
      || AgentSelectIfPolicyT::LOAD_ALGORITHM == BLOCK_LOAD_VECTORIZE);

  // If this may be an *in-place* stream compaction, we need to ensure that all of a tile's items have been loaded
  // before signalling a subsequent thread block's partial or inclusive state, hence we need a store release when
  // updating a tile state. Similarly, we need to make sure that the load of previous tile states precede writing of
  // the stream-compacted items and, hence, we need a load acquire when reading those tile states.
  static constexpr MemoryOrder memory_order =
    ((SelectionOpt == SelectImpl::SelectPotentiallyInPlace) && (!loads_via_smem))
      ? MemoryOrder::acquire_release
      : MemoryOrder::relaxed;

  // If we need to enforce memory order for in-place stream compaction, wrap the default decoupled look-back tile
  // state in a helper class that enforces memory order on reads and writes
  using MemoryOrderedTileStateT = tile_state_with_memory_order<ScanTileStateT, memory_order>;

  // The input value type
  using InputT = it_value_t<InputIteratorT>;

  // The flag value type
  using FlagT = it_value_t<FlagsInputIteratorT>;

  // Constants
  enum
  {
    USE_SELECT_OP,
    USE_SELECT_FLAGS,
    USE_DISCONTINUITY,
    USE_STENCIL_WITH_OP
  };

  static constexpr ::cuda::std::int32_t BLOCK_THREADS    = AgentSelectIfPolicyT::BLOCK_THREADS;
  static constexpr ::cuda::std::int32_t ITEMS_PER_THREAD = AgentSelectIfPolicyT::ITEMS_PER_THREAD;
  static constexpr ::cuda::std::int32_t TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr bool TWO_PHASE_SCATTER                = (ITEMS_PER_THREAD > 1);

  static constexpr bool has_select_op       = (!::cuda::std::is_same_v<SelectOpT, NullType>);
  static constexpr bool has_flags_it        = (!::cuda::std::is_same_v<FlagT, NullType>);
  static constexpr bool use_stencil_with_op = has_select_op && has_flags_it;
  static constexpr auto SELECT_METHOD =
    use_stencil_with_op ? USE_STENCIL_WITH_OP
    : has_select_op     ? USE_SELECT_OP
    : has_flags_it      ? USE_SELECT_FLAGS
                        : USE_DISCONTINUITY;

  // Cache-modified Input iterator wrapper type (for applying cache modifier) for items
  // Wrap the native input pointer with CacheModifiedValuesInputIterator
  // or directly use the supplied input iterator type
  using WrappedInputIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<InputIteratorT>,
                     CacheModifiedInputIterator<AgentSelectIfPolicyT::LOAD_MODIFIER, InputT, OffsetT>,
                     InputIteratorT>;

  // Cache-modified Input iterator wrapper type (for applying cache modifier) for values
  // Wrap the native input pointer with CacheModifiedValuesInputIterator
  // or directly use the supplied input iterator type
  using WrappedFlagsInputIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<FlagsInputIteratorT>,
                     CacheModifiedInputIterator<AgentSelectIfPolicyT::LOAD_MODIFIER, FlagT, OffsetT>,
                     FlagsInputIteratorT>;

  // Parameterized BlockLoad type for input data
  using BlockLoadT = BlockLoad<InputT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentSelectIfPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockLoad type for flags
  using BlockLoadFlags = BlockLoad<FlagT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentSelectIfPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockDiscontinuity type for items
  using BlockDiscontinuityT = BlockDiscontinuity<InputT, BLOCK_THREADS>;

  // Parameterized BlockScan type
  using BlockScanT = BlockScan<OffsetT, BLOCK_THREADS, AgentSelectIfPolicyT::SCAN_ALGORITHM>;

  // Callback type for obtaining tile prefix during block scan
  using DelayConstructorT = typename AgentSelectIfPolicyT::detail::delay_constructor_t;
  using TilePrefixCallbackOpT =
    TilePrefixCallbackOp<OffsetT, ::cuda::std::plus<>, MemoryOrderedTileStateT, DelayConstructorT>;

  // Item exchange type
  using ItemExchangeT = InputT[TILE_ITEMS];

  // Shared memory type for this thread block
  union _TempStorage
  {
    struct ScanStorage
    {
      // Smem needed for tile scanning
      typename BlockScanT::TempStorage scan;

      // Smem needed for cooperative prefix callback
      typename TilePrefixCallbackOpT::TempStorage prefix;

      // Smem needed for discontinuity detection
      typename BlockDiscontinuityT::TempStorage discontinuity;
    } scan_storage;

    // Smem needed for loading items
    typename BlockLoadT::TempStorage load_items;

    // Smem needed for loading values
    typename BlockLoadFlags::TempStorage load_flags;

    // Smem needed for compacting items (allows non POD items in this union)
    Uninitialized<ItemExchangeT> raw_exchange;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage& temp_storage; ///< Reference to temp_storage
  WrappedInputIteratorT d_in; ///< Input items
  OutputIteratorWrapperT d_selected_out; ///< Output iterator for the selected items
  WrappedFlagsInputIteratorT d_flags_in; ///< Input selection flags (if applicable)
  EqualityOpT equality_op; ///< T equality operator
  SelectOpT select_op; ///< Selection operator
  OffsetT num_items; ///< Total number of input items

  // Note: This is a const reference because we have seen double-digit percentage perf regressions otherwise
  const StreamingContextT& streaming_context; ///< Context for the current partition

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  /**
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param d_in
   *   Input data
   *
   * @param d_flags_in
   *   Input selection flags (if applicable)
   *
   * @param d_selected_out
   *   Output data
   *
   * @param select_op
   *   Selection operator
   *
   * @param equality_op
   *   Equality operator
   *
   * @param num_items
   *   Total number of input items
   *
   * @param streaming_context
   *   Context for the current partition
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentSelectIf(
    TempStorage& temp_storage,
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags_in,
    OutputIteratorWrapperT d_selected_out,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    const StreamingContextT& streaming_context)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_selected_out(d_selected_out)
      , d_flags_in(d_flags_in)
      , equality_op(equality_op)
      , select_op(select_op)
      , num_items(num_items)
      , streaming_context(streaming_context)
  {}

  //---------------------------------------------------------------------
  // Utility methods for initializing the selections
  //---------------------------------------------------------------------

  /**
   * Initialize selections (specialized for selection operator)
   */
  template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitializeSelections(
    OffsetT /*tile_offset*/,
    OffsetT num_tile_items,
    InputT (&items)[ITEMS_PER_THREAD],
    OffsetT (&selection_flags)[ITEMS_PER_THREAD],
    constant_t<USE_SELECT_OP> /*select_method*/)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      selection_flags[ITEM] = 1;

      if (!IS_LAST_TILE || (static_cast<OffsetT>(threadIdx.x * ITEMS_PER_THREAD + ITEM) < num_tile_items))
      {
        selection_flags[ITEM] = static_cast<bool>(select_op(items[ITEM]));
      }
    }
  }

  /**
   * Initialize selections (specialized for selection_op applied to d_flags_in)
   */
  template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitializeSelections(
    OffsetT tile_offset,
    OffsetT num_tile_items,
    InputT (& /*items*/)[ITEMS_PER_THREAD],
    OffsetT (&selection_flags)[ITEMS_PER_THREAD],
    constant_t<USE_STENCIL_WITH_OP> /*select_method*/)
  {
    __syncthreads();

    FlagT flags[ITEMS_PER_THREAD];
    if (IS_LAST_TILE)
    {
      // Initialize the out-of-bounds flags
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        selection_flags[ITEM] = true;
      }
      // Guarded loads
      BlockLoadFlags(temp_storage.load_flags)
        .Load((d_flags_in + streaming_context.input_offset()) + tile_offset, flags, num_tile_items);
    }
    else
    {
      BlockLoadFlags(temp_storage.load_flags).Load((d_flags_in + streaming_context.input_offset()) + tile_offset, flags);
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Set selection_flags for out-of-bounds items
      if ((!IS_LAST_TILE) || (static_cast<OffsetT>(threadIdx.x * ITEMS_PER_THREAD + ITEM) < num_tile_items))
      {
        selection_flags[ITEM] = static_cast<bool>(select_op(flags[ITEM]));
      }
    }
  }

  /**
   * Initialize selections (specialized for valid flags)
   */
  template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitializeSelections(
    OffsetT tile_offset,
    OffsetT num_tile_items,
    InputT (& /*items*/)[ITEMS_PER_THREAD],
    OffsetT (&selection_flags)[ITEMS_PER_THREAD],
    constant_t<USE_SELECT_FLAGS> /*select_method*/)
  {
    __syncthreads();

    FlagT flags[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Out-of-bounds items are selection_flags
      BlockLoadFlags(temp_storage.load_flags)
        .Load((d_flags_in + streaming_context.input_offset()) + tile_offset, flags, num_tile_items, 1);
    }
    else
    {
      BlockLoadFlags(temp_storage.load_flags).Load((d_flags_in + streaming_context.input_offset()) + tile_offset, flags);
    }

    // Convert flag type to selection_flags type
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      selection_flags[ITEM] = static_cast<bool>(flags[ITEM]);
    }
  }

  /**
   * Initialize selections (specialized for discontinuity detection)
   */
  template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitializeSelections(
    OffsetT tile_offset,
    OffsetT num_tile_items,
    InputT (&items)[ITEMS_PER_THREAD],
    OffsetT (&selection_flags)[ITEMS_PER_THREAD],
    constant_t<USE_DISCONTINUITY> /*select_method*/)
  {
    // We previously invoked the equality operator on out-of-bounds items
    // While fixing that issue there were some performance regressions that we had to work around
    // To avoid invoking equality operator on unexpected values we are doing on of two things:
    // (1) for primitive types AND ::cuda::std::equal_to: we compare all items, including out-of-bounds items and later
    // correct the flags of the out-of-bounds items
    // (2) otherwise, we are guarding against invoking the equality operator on out-of-bounds items
    static constexpr bool use_flag_fixup_code_path =
      ::cuda::std::is_arithmetic_v<InputT>
      && (::cuda::std::is_same_v<EqualityOpT, ::cuda::std::equal_to<>>
          || ::cuda::std::is_same_v<EqualityOpT, ::cuda::std::equal_to<InputT>>);

    if (IS_FIRST_TILE && streaming_context.is_first_partition())
    {
      __syncthreads();

      if constexpr (IS_LAST_TILE && !use_flag_fixup_code_path)
      {
        // Use custom flag operator to additionally flag the first out-of-bounds item
        guarded_inequality_op<EqualityOpT> flag_op{equality_op, num_tile_items};

        // Set head selection_flags.  First tile sets the first flag for the first item
        BlockDiscontinuityT(temp_storage.scan_storage.discontinuity).FlagHeads(selection_flags, items, flag_op);
      }
      else
      {
        // Set head selection_flags.  First tile sets the first flag for the first item
        BlockDiscontinuityT(temp_storage.scan_storage.discontinuity)
          .FlagHeads(selection_flags, items, InequalityWrapper<EqualityOpT>{equality_op});
      }
    }
    else
    {
      InputT tile_predecessor;
      if (threadIdx.x == 0)
      {
        tile_predecessor = d_in[tile_offset + streaming_context.input_offset() - 1];
      }

      __syncthreads();

      if constexpr (IS_LAST_TILE && !use_flag_fixup_code_path)
      {
        // Use custom flag operator to additionally flag the first out-of-bounds item
        guarded_inequality_op<EqualityOpT> flag_op{equality_op, num_tile_items};

        // Set head selection_flags.  First tile sets the first flag for the first item
        BlockDiscontinuityT(temp_storage.scan_storage.discontinuity)
          .FlagHeads(selection_flags, items, flag_op, tile_predecessor);
      }
      else
      {
        // Set head selection_flags.  First tile sets the first flag for the first item
        BlockDiscontinuityT(temp_storage.scan_storage.discontinuity)
          .FlagHeads(selection_flags, items, InequalityWrapper<EqualityOpT>{equality_op}, tile_predecessor);
      }
    }

    // For primitive types with default equality operator, we need to fix up the flags for the out-of-bounds items
    if constexpr (use_flag_fixup_code_path)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        // Set selection_flags for out-of-bounds items
        if ((IS_LAST_TILE) && (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM >= num_tile_items))
        {
          selection_flags[ITEM] = 1;
        }
      }
    }
  }

  //---------------------------------------------------------------------
  // Scatter utility methods
  //---------------------------------------------------------------------

  /**
   * Scatter flagged items to output offsets (specialized for direct scattering).
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterSelectedDirect(
    InputT (&items)[ITEMS_PER_THREAD],
    OffsetT (&selection_flags)[ITEMS_PER_THREAD],
    OffsetT (&selection_indices)[ITEMS_PER_THREAD],
    OffsetT num_selections)
  {
    // Scatter flagged items
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (selection_flags[ITEM])
      {
        if ((!IS_LAST_TILE) || selection_indices[ITEM] < num_selections)
        {
          *((d_selected_out + streaming_context.num_previously_selected()) + selection_indices[ITEM]) = items[ITEM];
        }
      }
    }
  }

  /**
   * @brief Scatter flagged items to output offsets (specialized for two-phase scattering)
   *
   * @param num_tile_items
   *   Number of valid items in this tile
   *
   * @param num_tile_selections
   *   Number of selections in this tile
   *
   * @param num_selections_prefix
   *   Total number of selections prior to this tile
   *
   * @param num_rejected_prefix
   *   Total number of rejections prior to this tile
   *
   * @param is_keep_rejects
   *   Marker type indicating whether to keep rejected items in the second partition
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterSelectedTwoPhase(
    InputT (&items)[ITEMS_PER_THREAD],
    OffsetT (&selection_flags)[ITEMS_PER_THREAD],
    OffsetT (&selection_indices)[ITEMS_PER_THREAD],
    int num_tile_selections,
    OffsetT num_selections_prefix)
  {
    __syncthreads();

    // Compact and scatter items
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int local_scatter_offset = selection_indices[ITEM] - num_selections_prefix;
      if (selection_flags[ITEM])
      {
        temp_storage.raw_exchange.Alias()[local_scatter_offset] = items[ITEM];
      }
    }

    __syncthreads();

    for (int item = threadIdx.x; item < num_tile_selections; item += BLOCK_THREADS)
    {
      *((d_selected_out + streaming_context.num_previously_selected()) + (num_selections_prefix + item)) =
        temp_storage.raw_exchange.Alias()[item];
    }
  }

  /**
   * @brief Scatter flagged items. Specialized for selection algorithm that simply discards rejected items
   *
   * @param num_tile_items
   *   Number of valid items in this tile
   *
   * @param num_tile_selections
   *   Number of selections in this tile
   *
   * @param num_selections_prefix
   *   Total number of selections prior to this tile
   *
   * @param num_rejected_prefix
   *   Total number of rejections prior to this tile
   *
   * @param num_selections
   *   Total number of selections including this tile
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Scatter(
    InputT (&items)[ITEMS_PER_THREAD],
    OffsetT (&selection_flags)[ITEMS_PER_THREAD],
    OffsetT (&selection_indices)[ITEMS_PER_THREAD],
    int num_tile_items,
    int num_tile_selections,
    OffsetT num_selections_prefix,
    OffsetT num_rejected_prefix,
    OffsetT num_selections,
    ::cuda::std::false_type /*is_keep_rejects*/)
  {
    // Do a two-phase scatter if two-phase is enabled and the average number of selection_flags items per thread is
    // greater than one
    if (TWO_PHASE_SCATTER && (num_tile_selections > BLOCK_THREADS))
    {
      ScatterSelectedTwoPhase<IS_LAST_TILE>(
        items, selection_flags, selection_indices, num_tile_selections, num_selections_prefix);
    }
    else
    {
      ScatterSelectedDirect<IS_LAST_TILE>(items, selection_flags, selection_indices, num_selections);
    }
  }

  /**
   * @brief Scatter flagged items. Specialized for partitioning algorithm that writes rejected items to a second
   * partition.
   *
   * @param num_tile_items
   *   Number of valid items in this tile
   *
   * @param num_tile_selections
   *   Number of selections in this tile
   *
   * @param num_selections_prefix
   *   Total number of selections prior to this tile
   *
   * @param num_rejected_prefix
   *   Total number of rejections prior to this tile
   *
   * @param is_keep_rejects
   *   Marker type indicating whether to keep rejected items in the second partition
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Scatter(
    InputT (&items)[ITEMS_PER_THREAD],
    OffsetT (&selection_flags)[ITEMS_PER_THREAD],
    OffsetT (&selection_indices)[ITEMS_PER_THREAD],
    int num_tile_items,
    int num_tile_selections,
    OffsetT num_selections_prefix,
    OffsetT num_rejected_prefix,
    OffsetT num_selections,
    ::cuda::std::true_type /*is_keep_rejects*/)
  {
    __syncthreads();

    int tile_num_rejections = num_tile_items - num_tile_selections;

    // Scatter items to shared memory (rejections first)
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int item_idx            = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
      int local_selection_idx = selection_indices[ITEM] - num_selections_prefix;
      int local_rejection_idx = item_idx - local_selection_idx;
      int local_scatter_offset =
        (selection_flags[ITEM]) ? tile_num_rejections + local_selection_idx : local_rejection_idx;

      temp_storage.raw_exchange.Alias()[local_scatter_offset] = items[ITEM];
    }

    // Ensure all threads finished scattering to shared memory
    __syncthreads();

    // Gather items from shared memory and scatter to global
    ScatterPartitionsToGlobal<IS_LAST_TILE>(
      num_tile_items, tile_num_rejections, num_selections_prefix, num_rejected_prefix, d_selected_out);
  }

  /**
   * @brief Second phase of scattering partitioned items to global memory. Specialized for partitioning to two
   * distinct partitions.
   */
  template <bool IS_LAST_TILE, typename SelectedItT, typename RejectedItT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterPartitionsToGlobal(
    int num_tile_items,
    int tile_num_rejections,
    OffsetT num_selections_prefix,
    OffsetT num_rejected_prefix,
    partition_distinct_output_t<SelectedItT, RejectedItT> partitioned_out_wrapper)
  {
    auto selected_out_it = partitioned_out_wrapper.selected_it + streaming_context.num_previously_selected();
    auto rejected_out_it = partitioned_out_wrapper.rejected_it + streaming_context.num_previously_rejected();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int item_idx      = (ITEM * BLOCK_THREADS) + threadIdx.x;
      int rejection_idx = item_idx;
      int selection_idx = item_idx - tile_num_rejections;
      OffsetT scatter_offset =
        (item_idx < tile_num_rejections) ? num_rejected_prefix + rejection_idx : num_selections_prefix + selection_idx;

      InputT item = temp_storage.raw_exchange.Alias()[item_idx];

      if (!IS_LAST_TILE || (item_idx < num_tile_items))
      {
        if (item_idx >= tile_num_rejections)
        {
          selected_out_it[scatter_offset] = item;
        }
        else
        {
          rejected_out_it[scatter_offset] = item;
        }
      }
    }
  }

  /**
   * @brief Second phase of scattering partitioned items to global memory. Specialized for partitioning to a single
   * iterator, where selected items are written in order from the beginning of the iterator and rejected items are
   * writtem from the iterators end backwards.
   */
  template <bool IS_LAST_TILE, typename PartitionedOutputItT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterPartitionsToGlobal(
    int num_tile_items,
    int tile_num_rejections,
    OffsetT num_selections_prefix,
    OffsetT num_rejected_prefix,
    PartitionedOutputItT partitioned_out_it)
  {
    using total_offset_t = typename StreamingContextT::total_num_items_t;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int item_idx      = (ITEM * BLOCK_THREADS) + threadIdx.x;
      int rejection_idx = item_idx;
      int selection_idx = item_idx - tile_num_rejections;
      total_offset_t scatter_offset =
        (item_idx < tile_num_rejections)
          ? (streaming_context.num_total_items(num_items) - streaming_context.num_previously_rejected()
             - static_cast<total_offset_t>(num_rejected_prefix) - static_cast<total_offset_t>(rejection_idx)
             - total_offset_t{1})
          : (streaming_context.num_previously_selected() + static_cast<total_offset_t>(num_selections_prefix)
             + static_cast<total_offset_t>(selection_idx));

      InputT item = temp_storage.raw_exchange.Alias()[item_idx];
      if (!IS_LAST_TILE || (item_idx < num_tile_items))
      {
        partitioned_out_it[scatter_offset] = item;
      }
    }
  }

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  /**
   * @brief Process first tile of input (dynamic chained scan).
   *
   * @param num_tile_items
   *   Number of input items comprising this tile
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param tile_state_wrapper
   *   A global tile state descriptor wrapped in a MemoryOrderedTileStateT that ensures consistent memory order across
   *   all tile status updates and loads
   *
   * @return The running count of selections (including this tile)
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT
  ConsumeFirstTile(int num_tile_items, OffsetT tile_offset, MemoryOrderedTileStateT& tile_state_wrapper)
  {
    InputT items[ITEMS_PER_THREAD];
    OffsetT selection_flags[ITEMS_PER_THREAD];
    OffsetT selection_indices[ITEMS_PER_THREAD];

    // Load items
    if (IS_LAST_TILE)
    {
      BlockLoadT(temp_storage.load_items)
        .Load((d_in + streaming_context.input_offset()) + tile_offset, items, num_tile_items);
    }
    else
    {
      BlockLoadT(temp_storage.load_items).Load((d_in + streaming_context.input_offset()) + tile_offset, items);
    }

    // Initialize selection_flags
    InitializeSelections<true, IS_LAST_TILE>(
      tile_offset, num_tile_items, items, selection_flags, constant_v<SELECT_METHOD>);

    // Ensure temporary storage used during block load can be reused
    // Also, in case of in-place stream compaction, this is needed to order the loads of
    // *all threads of this thread block* before the st.release of the thread writing this thread block's tile state
    __syncthreads();

    // Exclusive scan of selection_flags
    OffsetT num_tile_selections;
    BlockScanT(temp_storage.scan_storage.scan).ExclusiveSum(selection_flags, selection_indices, num_tile_selections);

    if (threadIdx.x == 0)
    {
      // Update tile status if this is not the last tile
      if (!IS_LAST_TILE)
      {
        tile_state_wrapper.SetInclusive(0, num_tile_selections);
      }
    }

    // Discount any out-of-bounds selections
    if (IS_LAST_TILE)
    {
      num_tile_selections -= (TILE_ITEMS - num_tile_items);
    }

    // Scatter flagged items
    Scatter<IS_LAST_TILE>(
      items,
      selection_flags,
      selection_indices,
      num_tile_items,
      num_tile_selections,
      0,
      0,
      num_tile_selections,
      bool_constant_v < SelectionOpt == SelectImpl::Partition >);

    return num_tile_selections;
  }

  /**
   * @brief Process subsequent tile of input (dynamic chained scan).
   *
   * @param num_tile_items
   *   Number of input items comprising this tile
   *
   * @param tile_idx
   *   Tile index
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param tile_state_wrapper
   *   A global tile state descriptor wrapped in a MemoryOrderedTileStateT that ensures consistent memory order across
   *   all tile status updates and loads
   *
   * @return The running count of selections (including this tile)
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT ConsumeSubsequentTile(
    int num_tile_items, int tile_idx, OffsetT tile_offset, MemoryOrderedTileStateT& tile_state_wrapper)
  {
    InputT items[ITEMS_PER_THREAD];
    OffsetT selection_flags[ITEMS_PER_THREAD];
    OffsetT selection_indices[ITEMS_PER_THREAD];

    // Load items
    if (IS_LAST_TILE)
    {
      BlockLoadT(temp_storage.load_items)
        .Load((d_in + streaming_context.input_offset()) + tile_offset, items, num_tile_items);
    }
    else
    {
      BlockLoadT(temp_storage.load_items).Load((d_in + streaming_context.input_offset()) + tile_offset, items);
    }

    // Initialize selection_flags
    InitializeSelections<false, IS_LAST_TILE>(
      tile_offset, num_tile_items, items, selection_flags, constant_v<SELECT_METHOD>);

    // Ensure temporary storage used during block load can be reused
    // Also, in case of in-place stream compaction, this is needed to order the loads of
    // *all threads of this thread block* before the st.release of the thread writing this thread block's tile state
    __syncthreads();

    // Exclusive scan of values and selection_flags
    TilePrefixCallbackOpT prefix_op(
      tile_state_wrapper, temp_storage.scan_storage.prefix, ::cuda::std::plus<>{}, tile_idx);
    BlockScanT(temp_storage.scan_storage.scan).ExclusiveSum(selection_flags, selection_indices, prefix_op);

    OffsetT num_tile_selections   = prefix_op.GetBlockAggregate();
    OffsetT num_selections        = prefix_op.GetInclusivePrefix();
    OffsetT num_selections_prefix = prefix_op.GetExclusivePrefix();
    OffsetT num_rejected_prefix   = tile_offset - num_selections_prefix;

    // Discount any out-of-bounds selections
    if (IS_LAST_TILE)
    {
      int num_discount = TILE_ITEMS - num_tile_items;
      num_selections -= num_discount;
      num_tile_selections -= num_discount;
    }

    // note (only applies to in-place stream compaction): We can avoid having to introduce explicit memory order between
    // the look-back (i.e., loading previous tiles' states) and scattering items (which means, potentially overwriting
    // previous tiles' input items, in case of in-place compaction), because this is implicitly ensured through
    // execution dependency: The scatter stage requires the offset from the prefix-sum and it can only know the
    // prefix-sum after having read that from the decoupled look-back. Scatter flagged items
    Scatter<IS_LAST_TILE>(
      items,
      selection_flags,
      selection_indices,
      num_tile_items,
      num_tile_selections,
      num_selections_prefix,
      num_rejected_prefix,
      num_selections,
      bool_constant_v < SelectionOpt == SelectImpl::Partition >);

    return num_selections;
  }

  /**
   * @brief Process a tile of input
   *
   * @param num_tile_items
   *   Number of input items comprising this tile
   *
   * @param tile_idx
   *   Tile index
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param tile_state_wrapper
   *   A global tile state descriptor wrapped in a MemoryOrderedTileStateT that ensures consistent memory order across
   *   all tile status updates and loads
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT
  ConsumeTile(int num_tile_items, int tile_idx, OffsetT tile_offset, MemoryOrderedTileStateT& tile_state_wrapper)
  {
    OffsetT num_selections;
    if (tile_idx == 0)
    {
      num_selections = ConsumeFirstTile<IS_LAST_TILE>(num_tile_items, tile_offset, tile_state_wrapper);
    }
    else
    {
      num_selections = ConsumeSubsequentTile<IS_LAST_TILE>(num_tile_items, tile_idx, tile_offset, tile_state_wrapper);
    }

    return num_selections;
  }

  /**
   * @brief Scan tiles of items as part of a dynamic chained scan
   *
   * @param num_tiles
   *   Total number of input tiles
   *
   * @param tile_state
   *   Global tile state descriptor
   *
   * @param d_num_selected_out
   *   Output total number selection_flags
   *
   * @tparam NumSelectedIteratorT
   *   Output iterator type for recording number of items selection_flags
   */
  template <typename NumSelectedIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeRange(int num_tiles, ScanTileStateT& tile_state, NumSelectedIteratorT d_num_selected_out)
  {
    // Ensure consistent memory order across all tile status updates and loads
    auto tile_state_wrapper = MemoryOrderedTileStateT{tile_state};

    // Blocks are launched in increasing order, so just assign one tile per block
    // TODO (elstehle): replacing this term with just `blockIdx.x` degrades perf for partition. Once we get to re-tune
    // the algorithm, we want to replace this term with `blockIdx.x`
    int tile_idx{};
    if constexpr (SELECT_METHOD != USE_DISCONTINUITY)
    {
      tile_idx = (blockIdx.x * gridDim.y) + blockIdx.y; // Current tile index
    }
    else
    {
      tile_idx = blockIdx.x; // Current tile index
    }
    OffsetT tile_offset = static_cast<OffsetT>(tile_idx) * static_cast<OffsetT>(TILE_ITEMS);

    if (tile_idx < num_tiles - 1)
    {
      // Not the last tile (full)
      ConsumeTile<false>(TILE_ITEMS, tile_idx, tile_offset, tile_state_wrapper);
    }
    else
    {
      // The last tile (possibly partially-full)
      OffsetT num_remaining  = num_items - tile_offset;
      OffsetT num_selections = ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state_wrapper);

      if (threadIdx.x == 0)
      {
        // Update the number of selected items with this partition's selections
        streaming_context.update_num_selected(d_num_selected_out, num_selections);
      }
    }
  }
};

} // namespace select
} // namespace detail

CUB_NAMESPACE_END
