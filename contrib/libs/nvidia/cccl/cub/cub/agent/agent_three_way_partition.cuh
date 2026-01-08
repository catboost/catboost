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

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD,
          BlockLoadAlgorithm _LOAD_ALGORITHM,
          CacheLoadModifier _LOAD_MODIFIER,
          BlockScanAlgorithm _SCAN_ALGORITHM,
          class DelayConstructorT = detail::fixed_delay_constructor_t<350, 450>>
struct AgentThreeWayPartitionPolicy
{
  static constexpr int BLOCK_THREADS                 = _BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD              = _ITEMS_PER_THREAD;
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
  static constexpr CacheLoadModifier LOAD_MODIFIER   = _LOAD_MODIFIER;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;

  struct detail
  {
    using delay_constructor_t = DelayConstructorT;
  };
};

namespace detail
{

namespace three_way_partition
{

template <class OffsetT>
struct pair_pack_t
{
  OffsetT x, y;

  _CCCL_DEVICE pair_pack_t<OffsetT> operator+(const pair_pack_t<OffsetT>& other) const
  {
    return {x + other.x, y + other.y};
  }
};

template <class OffsetT, class = void>
struct accumulator_pack_base_t
{
  using pack_t = pair_pack_t<OffsetT>;

  _CCCL_DEVICE static pack_t pack(OffsetT f, OffsetT s)
  {
    return {f, s};
  }
  _CCCL_DEVICE static OffsetT first(pack_t packed)
  {
    return packed.x;
  }
  _CCCL_DEVICE static OffsetT second(pack_t packed)
  {
    return packed.y;
  }
};

template <class OffsetT>
struct accumulator_pack_base_t<OffsetT, ::cuda::std::enable_if_t<sizeof(OffsetT) == 4>>
{
  using pack_t = uint64_t;

  _CCCL_DEVICE static pack_t pack(OffsetT f, OffsetT s)
  {
    return (static_cast<pack_t>(f) << 32) | static_cast<pack_t>(s);
  }

  _CCCL_DEVICE static OffsetT first(pack_t packed)
  {
    return static_cast<OffsetT>(packed >> 32);
  }

  _CCCL_DEVICE static OffsetT second(pack_t packed)
  {
    return static_cast<OffsetT>(packed & 0xFFFFFFFF);
  }
};

template <class OffsetT>
struct accumulator_pack_t : accumulator_pack_base_t<OffsetT>
{
  using base = accumulator_pack_base_t<OffsetT>;
  using typename base::pack_t;

  _CCCL_DEVICE static void subtract(pack_t& packed, OffsetT val)
  {
    packed = base::pack(base::first(packed) - val, base::second(packed) - val);
  }

  _CCCL_DEVICE static OffsetT sum(pack_t& packed)
  {
    return base::first(packed) + base::second(packed);
  }

  _CCCL_DEVICE static pack_t zero()
  {
    return {};
  }
};

/**
 * \brief Implements a device-wide three-way partitioning
 *
 * Splits input data into three parts based on the selection functors. If the
 * first functor selects an item, the algorithm places it in the first part.
 * Otherwise, if the second functor selects an item, the algorithm places it in
 * the second part. If both functors don't select an item, the algorithm places
 * it into the unselected part.
 */
template <typename PolicyT,
          typename InputIteratorT,
          typename FirstOutputIteratorT,
          typename SecondOutputIteratorT,
          typename UnselectedOutputIteratorT,
          typename SelectFirstPartOp,
          typename SelectSecondPartOp,
          typename OffsetT,
          typename StreamingContextT>
struct AgentThreeWayPartition
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input value type
  using InputT = it_value_t<InputIteratorT>;

  using AccumPackHelperT = accumulator_pack_t<OffsetT>;
  using AccumPackT       = typename AccumPackHelperT::pack_t;

  // Tile status descriptor interface type
  using ScanTileStateT = cub::ScanTileState<AccumPackT>;

  // Constants
  static constexpr int BLOCK_THREADS    = PolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = PolicyT::ITEMS_PER_THREAD;
  static constexpr int TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD;

  using WrappedInputIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<InputIteratorT>,
                     cub::CacheModifiedInputIterator<PolicyT::LOAD_MODIFIER, InputT, OffsetT>,
                     InputIteratorT>;

  // Parameterized BlockLoad type for input data
  using BlockLoadT = cub::BlockLoad<InputT, BLOCK_THREADS, ITEMS_PER_THREAD, PolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockScan type
  using BlockScanT = cub::BlockScan<AccumPackT, BLOCK_THREADS, PolicyT::SCAN_ALGORITHM>;

  // Callback type for obtaining tile prefix during block scan
  using DelayConstructorT = typename PolicyT::detail::delay_constructor_t;
  using TilePrefixCallbackOpT =
    cub::TilePrefixCallbackOp<AccumPackT, ::cuda::std::plus<>, ScanTileStateT, DelayConstructorT>;

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
    } scan_storage;

    // Smem needed for loading items
    typename BlockLoadT::TempStorage load_items;

    // Smem needed for compacting items (allows non POD items in this union)
    cub::Uninitialized<ItemExchangeT> raw_exchange;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : cub::Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage& temp_storage; ///< Reference to temp_storage
  WrappedInputIteratorT d_in; ///< Input items
  FirstOutputIteratorT d_first_part_out;
  SecondOutputIteratorT d_second_part_out;
  UnselectedOutputIteratorT d_unselected_out;
  SelectFirstPartOp select_first_part_op;
  SelectSecondPartOp select_second_part_op;
  OffsetT num_items; ///< Total number of input items

  // Note: This is a const reference because we have seen double-digit percentage perf regressions otherwise
  const StreamingContextT& streaming_context; ///< Context for the current partition

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  // Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentThreeWayPartition(
    TempStorage& temp_storage,
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    OffsetT num_items,
    const StreamingContextT& streaming_context)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_first_part_out(d_first_part_out)
      , d_second_part_out(d_second_part_out)
      , d_unselected_out(d_unselected_out)
      , select_first_part_op(select_first_part_op)
      , select_second_part_op(select_second_part_op)
      , num_items(num_items)
      , streaming_context(streaming_context)
  {}

  //---------------------------------------------------------------------
  // Utility methods for initializing the selections
  //---------------------------------------------------------------------

  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Initialize(
    OffsetT num_tile_items, InputT (&items)[ITEMS_PER_THREAD], AccumPackT (&items_selection_flags)[ITEMS_PER_THREAD])
  {
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      items_selection_flags[ITEM] = AccumPackHelperT::pack(1, 1);

      if (!IS_LAST_TILE || (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
      {
        OffsetT first_item_selected = select_first_part_op(items[ITEM]);
        items_selection_flags[ITEM] =
          AccumPackHelperT::pack(first_item_selected, first_item_selected ? 0 : select_second_part_op(items[ITEM]));
      }
    }
  }

  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Scatter(
    InputT (&items)[ITEMS_PER_THREAD],
    AccumPackT (&items_selection_flags)[ITEMS_PER_THREAD],
    AccumPackT (&items_selection_indices)[ITEMS_PER_THREAD],
    int num_tile_items,
    AccumPackT num_tile_selected,
    AccumPackT num_tile_selected_prefix,
    OffsetT num_rejected_prefix)
  {
    __syncthreads();

    const OffsetT num_first_selections_prefix  = AccumPackHelperT::first(num_tile_selected_prefix);
    const OffsetT num_second_selections_prefix = AccumPackHelperT::second(num_tile_selected_prefix);

    const int first_item_end  = AccumPackHelperT::first(num_tile_selected);
    const int second_item_end = first_item_end + AccumPackHelperT::second(num_tile_selected);

    // Scatter items to shared memory (rejections first)
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int item_idx = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;

      const OffsetT first_items_selection_indices  = AccumPackHelperT::first(items_selection_indices[ITEM]);
      const OffsetT second_items_selection_indices = AccumPackHelperT::second(items_selection_indices[ITEM]);

      if (!IS_LAST_TILE || (item_idx < num_tile_items))
      {
        int local_scatter_offset = 0;

        if (AccumPackHelperT::first(items_selection_flags[ITEM]))
        {
          local_scatter_offset = first_items_selection_indices - num_first_selections_prefix;
        }
        else if (AccumPackHelperT::second(items_selection_flags[ITEM]))
        {
          local_scatter_offset = first_item_end + second_items_selection_indices - num_second_selections_prefix;
        }
        else
        {
          // Medium item
          int local_selection_idx = (first_items_selection_indices - num_first_selections_prefix)
                                  + (second_items_selection_indices - num_second_selections_prefix);
          local_scatter_offset = second_item_end + item_idx - local_selection_idx;
        }

        temp_storage.raw_exchange.Alias()[local_scatter_offset] = items[ITEM];
      }
    }

    __syncthreads();

    // Gather items from shared memory and scatter to global
    auto first_base =
      d_first_part_out + (streaming_context.num_previously_selected_first() + num_first_selections_prefix);
    auto second_base =
      d_second_part_out + (streaming_context.num_previously_selected_second() + num_second_selections_prefix);
    auto unselected_base = d_unselected_out + (streaming_context.num_previously_rejected() + num_rejected_prefix);
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int item_idx = (ITEM * BLOCK_THREADS) + threadIdx.x;

      if (!IS_LAST_TILE || (item_idx < num_tile_items))
      {
        InputT item = temp_storage.raw_exchange.Alias()[item_idx];

        if (item_idx < first_item_end)
        {
          first_base[item_idx] = item;
        }
        else if (item_idx < second_item_end)
        {
          second_base[item_idx - first_item_end] = item;
        }
        else
        {
          int rejection_idx              = item_idx - second_item_end;
          unselected_base[rejection_idx] = item;
        }
      }
    }
  }

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  /**
   * Process first tile of input (dynamic chained scan).
   * Returns the running count of selections (including this tile)
   *
   * @param num_tile_items Number of input items comprising this tile
   * @param tile_offset Tile offset
   * @param first_tile_state Global tile state descriptor
   * @param second_tile_state Global tile state descriptor
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeFirstTile(int num_tile_items, OffsetT tile_offset, ScanTileStateT& tile_state, AccumPackT& num_items_selected)
  {
    InputT items[ITEMS_PER_THREAD];

    AccumPackT items_selection_flags[ITEMS_PER_THREAD];
    AccumPackT items_selection_indices[ITEMS_PER_THREAD];

    // Load items
    if (IS_LAST_TILE)
    {
      BlockLoadT(temp_storage.load_items)
        .Load(d_in + streaming_context.input_offset() + tile_offset, items, num_tile_items);
    }
    else
    {
      BlockLoadT(temp_storage.load_items).Load(d_in + streaming_context.input_offset() + tile_offset, items);
    }

    // Initialize selection_flags
    Initialize<IS_LAST_TILE>(num_tile_items, items, items_selection_flags);
    __syncthreads();

    // Exclusive scan of selection_flags
    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveSum(items_selection_flags, items_selection_indices, num_items_selected);

    if (threadIdx.x == 0)
    {
      // Update tile status if this is not the last tile
      if (!IS_LAST_TILE)
      {
        tile_state.SetInclusive(0, num_items_selected);
      }
    }

    // Discount any out-of-bounds selections
    if (IS_LAST_TILE)
    {
      AccumPackHelperT::subtract(num_items_selected, TILE_ITEMS - num_tile_items);
    }

    // Scatter flagged items
    Scatter<IS_LAST_TILE>(
      items,
      items_selection_flags,
      items_selection_indices,
      num_tile_items,
      num_items_selected,
      // all the prefixes equal to 0 because it's the first tile
      AccumPackHelperT::zero(),
      0);
  }

  /**
   * Process subsequent tile of input (dynamic chained scan).
   * Returns the running count of selections (including this tile)
   *
   * @param num_tile_items Number of input items comprising this tile
   * @param tile_idx Tile index
   * @param tile_offset Tile offset
   * @param first_tile_state Global tile state descriptor
   * @param second_tile_state Global tile state descriptor
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeSubsequentTile(
    int num_tile_items, int tile_idx, OffsetT tile_offset, ScanTileStateT& tile_state, AccumPackT& num_items_selected)
  {
    InputT items[ITEMS_PER_THREAD];

    AccumPackT items_selected_flags[ITEMS_PER_THREAD];
    AccumPackT items_selected_indices[ITEMS_PER_THREAD];

    // Load items
    if (IS_LAST_TILE)
    {
      BlockLoadT(temp_storage.load_items)
        .Load(d_in + streaming_context.input_offset() + tile_offset, items, num_tile_items);
    }
    else
    {
      BlockLoadT(temp_storage.load_items).Load(d_in + streaming_context.input_offset() + tile_offset, items);
    }

    // Initialize selection_flags
    Initialize<IS_LAST_TILE>(num_tile_items, items, items_selected_flags);
    __syncthreads();

    // Exclusive scan of values and selection_flags
    TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.scan_storage.prefix, ::cuda::std::plus<>{}, tile_idx);

    BlockScanT(temp_storage.scan_storage.scan).ExclusiveSum(items_selected_flags, items_selected_indices, prefix_op);

    num_items_selected                    = prefix_op.GetInclusivePrefix();
    AccumPackT num_items_in_tile_selected = prefix_op.GetBlockAggregate();
    AccumPackT num_items_selected_prefix  = prefix_op.GetExclusivePrefix();

    __syncthreads();

    OffsetT num_rejected_prefix = (tile_idx * TILE_ITEMS) - AccumPackHelperT::sum(num_items_selected_prefix);

    // Discount any out-of-bounds selections. There are exactly
    // TILE_ITEMS - num_tile_items elements like that because we
    // marked them as selected in Initialize method.
    if (IS_LAST_TILE)
    {
      const int num_discount = TILE_ITEMS - num_tile_items;

      AccumPackHelperT::subtract(num_items_selected, num_discount);
      AccumPackHelperT::subtract(num_items_in_tile_selected, num_discount);
    }

    // Scatter flagged items
    Scatter<IS_LAST_TILE>(
      items,
      items_selected_flags,
      items_selected_indices,
      num_tile_items,
      num_items_in_tile_selected,
      num_items_selected_prefix,
      num_rejected_prefix);
  }

  /**
   * Process a tile of input
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeTile(int num_tile_items, int tile_idx, OffsetT tile_offset, ScanTileStateT& tile_state, AccumPackT& accum)
  {
    if (tile_idx == 0)
    {
      ConsumeFirstTile<IS_LAST_TILE>(num_tile_items, tile_offset, tile_state, accum);
    }
    else
    {
      ConsumeSubsequentTile<IS_LAST_TILE>(num_tile_items, tile_idx, tile_offset, tile_state, accum);
    }
  }

  /**
   * Scan tiles of items as part of a dynamic chained scan
   *
   * @tparam NumSelectedIteratorT
   *   Output iterator type for recording number of items selection_flags
   *
   * @param num_tiles
   *   Total number of input tiles
   *
   * @param first_tile_state
   *   Global tile state descriptor
   *
   * @param second_tile_state
   *   Global tile state descriptor
   *
   * @param d_num_selected_out
   *   Output total number selection_flags
   */
  template <typename NumSelectedIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeRange(int num_tiles, ScanTileStateT& tile_state, NumSelectedIteratorT d_num_selected_out)
  {
    // Blocks are launched in increasing order, so just assign one tile per block
    // Current tile index
    const int tile_idx = blockIdx.x;

    // Global offset for the current tile
    const OffsetT tile_offset = tile_idx * TILE_ITEMS;

    AccumPackT accum;

    if (tile_idx < num_tiles - 1)
    {
      // Not the last tile (full)
      ConsumeTile<false>(TILE_ITEMS, tile_idx, tile_offset, tile_state, accum);
    }
    else
    {
      // The last tile (possibly partially-full)
      const OffsetT num_remaining = num_items - tile_offset;

      ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state, accum);

      if (threadIdx.x == 0)
      {
        // Update the number of selected items with this partition's selections
        streaming_context.update_num_selected(
          d_num_selected_out, AccumPackHelperT::first(accum), AccumPackHelperT::second(accum), num_items);
      }
    }
  }
};

} // namespace three_way_partition
} // namespace detail

CUB_NAMESPACE_END
