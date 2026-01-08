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
#pragma clang system_header


#include <iterator>
#include <type_traits>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/config.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>


CUB_NAMESPACE_BEGIN


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD,
          BlockLoadAlgorithm _LOAD_ALGORITHM,
          CacheLoadModifier _LOAD_MODIFIER,
          BlockScanAlgorithm _SCAN_ALGORITHM>
struct AgentThreeWayPartitionPolicy
{
  constexpr static int BLOCK_THREADS                 = _BLOCK_THREADS;
  constexpr static int ITEMS_PER_THREAD              = _ITEMS_PER_THREAD;
  constexpr static BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
  constexpr static CacheLoadModifier LOAD_MODIFIER   = _LOAD_MODIFIER;
  constexpr static BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
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
          typename OffsetT>
struct AgentThreeWayPartition
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input value type
  using InputT = cub::detail::value_t<InputIteratorT>;

  // Tile status descriptor interface type
  using ScanTileStateT = cub::ScanTileState<OffsetT>;

  // Constants
  constexpr static int BLOCK_THREADS = PolicyT::BLOCK_THREADS;
  constexpr static int ITEMS_PER_THREAD = PolicyT::ITEMS_PER_THREAD;
  constexpr static int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

  using WrappedInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<InputIteratorT>::value,
    cub::CacheModifiedInputIterator<PolicyT::LOAD_MODIFIER, InputT, OffsetT>,
    InputIteratorT>;

  // Parameterized BlockLoad type for input data
  using BlockLoadT = cub::BlockLoad<InputT,
                                    BLOCK_THREADS,
                                    ITEMS_PER_THREAD,
                                    PolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockScan type
  using BlockScanT =
    cub::BlockScan<OffsetT, BLOCK_THREADS, PolicyT::SCAN_ALGORITHM>;

  // Callback type for obtaining tile prefix during block scan
  using TilePrefixCallbackOpT =
    cub::TilePrefixCallbackOp<OffsetT, cub::Sum, ScanTileStateT>;

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
  struct TempStorage : cub::Uninitialized<_TempStorage> {};


  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage&                        temp_storage;       ///< Reference to temp_storage
  WrappedInputIteratorT                d_in;               ///< Input items
  FirstOutputIteratorT                 d_first_part_out;
  SecondOutputIteratorT                d_second_part_out;
  UnselectedOutputIteratorT            d_unselected_out;
  SelectFirstPartOp                    select_first_part_op;
  SelectSecondPartOp                   select_second_part_op;
  OffsetT                              num_items;          ///< Total number of input items


  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  // Constructor
  __device__ __forceinline__
  AgentThreeWayPartition(TempStorage &temp_storage,
                         InputIteratorT d_in,
                         FirstOutputIteratorT d_first_part_out,
                         SecondOutputIteratorT d_second_part_out,
                         UnselectedOutputIteratorT d_unselected_out,
                         SelectFirstPartOp select_first_part_op,
                         SelectSecondPartOp select_second_part_op,
                         OffsetT num_items)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_first_part_out(d_first_part_out)
      , d_second_part_out(d_second_part_out)
      , d_unselected_out(d_unselected_out)
      , select_first_part_op(select_first_part_op)
      , select_second_part_op(select_second_part_op)
      , num_items(num_items)
  {}

  //---------------------------------------------------------------------
  // Utility methods for initializing the selections
  //---------------------------------------------------------------------

  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void
  Initialize(OffsetT num_tile_items,
             InputT (&items)[ITEMS_PER_THREAD],
             OffsetT (&first_items_selection_flags)[ITEMS_PER_THREAD],
             OffsetT (&second_items_selection_flags)[ITEMS_PER_THREAD])
  {
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      first_items_selection_flags[ITEM]  = 1;
      second_items_selection_flags[ITEM] = 1;

      if (!IS_LAST_TILE ||
          (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
      {
        first_items_selection_flags[ITEM] = select_first_part_op(items[ITEM]);
        second_items_selection_flags[ITEM] =
          first_items_selection_flags[ITEM]
            ? 0
            : select_second_part_op(items[ITEM]);
      }
    }
  }

  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void Scatter(
    InputT          (&items)[ITEMS_PER_THREAD],
    OffsetT         (&first_items_selection_flags)[ITEMS_PER_THREAD],
    OffsetT         (&first_items_selection_indices)[ITEMS_PER_THREAD],
    OffsetT         (&second_items_selection_flags)[ITEMS_PER_THREAD],
    OffsetT         (&second_items_selection_indices)[ITEMS_PER_THREAD],
    int             num_tile_items,
    int             num_first_tile_selections,
    int             num_second_tile_selections,
    OffsetT         num_first_selections_prefix,
    OffsetT         num_second_selections_prefix,
    OffsetT         num_rejected_prefix)
  {
    CTA_SYNC();

    int first_item_end = num_first_tile_selections;
    int second_item_end = first_item_end + num_second_tile_selections;

    // Scatter items to shared memory (rejections first)
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int item_idx = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;

      if (!IS_LAST_TILE || (item_idx < num_tile_items))
      {
        int local_scatter_offset = 0;

        if (first_items_selection_flags[ITEM])
        {
          local_scatter_offset = first_items_selection_indices[ITEM]
                               - num_first_selections_prefix;
        }
        else if (second_items_selection_flags[ITEM])
        {
          local_scatter_offset = first_item_end +
                                 second_items_selection_indices[ITEM] -
                                 num_second_selections_prefix;
        }
        else
        {
          // Medium item
          int local_selection_idx =
           (first_items_selection_indices[ITEM] - num_first_selections_prefix)
         + (second_items_selection_indices[ITEM] - num_second_selections_prefix);
          local_scatter_offset = second_item_end + item_idx - local_selection_idx;
        }

        temp_storage.raw_exchange.Alias()[local_scatter_offset] = items[ITEM];
      }
    }

    CTA_SYNC();

    // Gather items from shared memory and scatter to global
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int item_idx = (ITEM * BLOCK_THREADS) + threadIdx.x;

      if (!IS_LAST_TILE || (item_idx < num_tile_items))
      {
        InputT item = temp_storage.raw_exchange.Alias()[item_idx];

        if (item_idx < first_item_end)
        {
          d_first_part_out[num_first_selections_prefix + item_idx] = item;
        }
        else if (item_idx < second_item_end)
        {
          d_second_part_out[num_second_selections_prefix + item_idx - first_item_end] = item;
        }
        else
        {
          int rejection_idx = item_idx - second_item_end;
          d_unselected_out[num_rejected_prefix + rejection_idx] = item;
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
  __device__ __forceinline__ void
  ConsumeFirstTile(int num_tile_items,
                   OffsetT tile_offset,
                   ScanTileStateT &first_tile_state,
                   ScanTileStateT &second_tile_state,
                   OffsetT &first_items,
                   OffsetT &second_items)
  {
    InputT items[ITEMS_PER_THREAD];

    OffsetT first_items_selection_flags[ITEMS_PER_THREAD];
    OffsetT first_items_selection_indices[ITEMS_PER_THREAD];

    OffsetT second_items_selection_flags[ITEMS_PER_THREAD];
    OffsetT second_items_selection_indices[ITEMS_PER_THREAD];

    // Load items
    if (IS_LAST_TILE)
    {
      BlockLoadT(temp_storage.load_items)
        .Load(d_in + tile_offset, items, num_tile_items);
    }
    else
    {
      BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items);
    }

    // Initialize selection_flags
    Initialize<IS_LAST_TILE>(
      num_tile_items,
      items,
      first_items_selection_flags,
      second_items_selection_flags);

    CTA_SYNC();

    // Exclusive scan of selection_flags
    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveSum(first_items_selection_flags,
                    first_items_selection_indices,
                    first_items);

    if (threadIdx.x == 0)
    {
      // Update tile status if this is not the last tile
      if (!IS_LAST_TILE)
      {
        first_tile_state.SetInclusive(0, first_items);
      }
    }

    CTA_SYNC();

    // Exclusive scan of selection_flags
    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveSum(second_items_selection_flags,
                    second_items_selection_indices,
                    second_items);

    if (threadIdx.x == 0)
    {
      // Update tile status if this is not the last tile
      if (!IS_LAST_TILE)
      {
        second_tile_state.SetInclusive(0, second_items);
      }
    }

    // Discount any out-of-bounds selections
    if (IS_LAST_TILE)
    {
      first_items -= (TILE_ITEMS - num_tile_items);
      second_items -= (TILE_ITEMS - num_tile_items);
    }

    // Scatter flagged items
    Scatter<IS_LAST_TILE>(
      items,
      first_items_selection_flags,
      first_items_selection_indices,
      second_items_selection_flags,
      second_items_selection_indices,
      num_tile_items,
      first_items,
      second_items,
      // all the prefixes equal to 0 because it's the first tile
      0, 0, 0);
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
  __device__ __forceinline__ void
  ConsumeSubsequentTile(int num_tile_items,
                        int tile_idx,
                        OffsetT tile_offset,
                        ScanTileStateT &first_tile_state,
                        ScanTileStateT &second_tile_state,
                        OffsetT &num_first_items_selections,
                        OffsetT &num_second_items_selections)
  {
    InputT items[ITEMS_PER_THREAD];

    OffsetT first_items_selection_flags[ITEMS_PER_THREAD];
    OffsetT first_items_selection_indices[ITEMS_PER_THREAD];

    OffsetT second_items_selection_flags[ITEMS_PER_THREAD];
    OffsetT second_items_selection_indices[ITEMS_PER_THREAD];

    // Load items
    if (IS_LAST_TILE)
    {
      BlockLoadT(temp_storage.load_items).Load(
        d_in + tile_offset, items, num_tile_items);
    }
    else
    {
      BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items);
    }

    // Initialize selection_flags
    Initialize<IS_LAST_TILE>(
      num_tile_items,
      items,
      first_items_selection_flags,
      second_items_selection_flags);

    CTA_SYNC();

    // Exclusive scan of values and selection_flags
    TilePrefixCallbackOpT first_prefix_op(first_tile_state,
                                          temp_storage.scan_storage.prefix,
                                          cub::Sum(),
                                          tile_idx);

    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveSum(first_items_selection_flags,
                    first_items_selection_indices,
                    first_prefix_op);

    num_first_items_selections                  = first_prefix_op.GetInclusivePrefix();
    OffsetT num_first_items_in_tile_selections  = first_prefix_op.GetBlockAggregate();
    OffsetT num_first_items_selections_prefix   = first_prefix_op.GetExclusivePrefix();

    CTA_SYNC();

    TilePrefixCallbackOpT second_prefix_op(second_tile_state,
                                           temp_storage.scan_storage.prefix,
                                           cub::Sum(),
                                           tile_idx);
    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveSum(second_items_selection_flags,
                    second_items_selection_indices,
                    second_prefix_op);

    num_second_items_selections                  = second_prefix_op.GetInclusivePrefix();
    OffsetT num_second_items_in_tile_selections  = second_prefix_op.GetBlockAggregate();
    OffsetT num_second_items_selections_prefix   = second_prefix_op.GetExclusivePrefix();

    OffsetT num_rejected_prefix = (tile_idx * TILE_ITEMS)
                                  - num_first_items_selections_prefix
                                  - num_second_items_selections_prefix;

    // Discount any out-of-bounds selections. There are exactly
    // TILE_ITEMS - num_tile_items elements like that because we
    // marked them as selected in Initialize method.
    if (IS_LAST_TILE)
    {
      const int num_discount = TILE_ITEMS - num_tile_items;

      num_first_items_selections          -= num_discount;
      num_first_items_in_tile_selections  -= num_discount;
      num_second_items_selections         -= num_discount;
      num_second_items_in_tile_selections -= num_discount;
    }

    // Scatter flagged items
    Scatter<IS_LAST_TILE>(
      items,
      first_items_selection_flags,
      first_items_selection_indices,
      second_items_selection_flags,
      second_items_selection_indices,
      num_tile_items,
      num_first_items_in_tile_selections,
      num_second_items_in_tile_selections,
      num_first_items_selections_prefix,
      num_second_items_selections_prefix,
      num_rejected_prefix);
  }


  /**
   * Process a tile of input
   */
  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void ConsumeTile(
    int                 num_tile_items,
    int                 tile_idx,
    OffsetT             tile_offset,
    ScanTileStateT&     first_tile_state,
    ScanTileStateT&     second_tile_state,
    OffsetT&            first_items,
    OffsetT&            second_items)
  {
    if (tile_idx == 0)
    {
      ConsumeFirstTile<IS_LAST_TILE>(num_tile_items,
                                     tile_offset,
                                     first_tile_state,
                                     second_tile_state,
                                     first_items,
                                     second_items);
    }
    else
    {
      ConsumeSubsequentTile<IS_LAST_TILE>(num_tile_items,
                                          tile_idx,
                                          tile_offset,
                                          first_tile_state,
                                          second_tile_state,
                                          first_items,
                                          second_items);
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
  __device__ __forceinline__ void
  ConsumeRange(int num_tiles,
               ScanTileStateT &first_tile_state,
               ScanTileStateT &second_tile_state,
               NumSelectedIteratorT d_num_selected_out)
  {
    // Blocks are launched in increasing order, so just assign one tile per block
    // Current tile index
    int tile_idx = static_cast<int>((blockIdx.x * gridDim.y) + blockIdx.y);

    // Global offset for the current tile
    OffsetT tile_offset = tile_idx * TILE_ITEMS;

    OffsetT num_first_selections;
    OffsetT num_second_selections;

    if (tile_idx < num_tiles - 1)
    {
      // Not the last tile (full)
      ConsumeTile<false>(TILE_ITEMS,
                         tile_idx,
                         tile_offset,
                         first_tile_state,
                         second_tile_state,
                         num_first_selections,
                         num_second_selections);
    }
    else
    {
      // The last tile (possibly partially-full)
      OffsetT num_remaining = num_items - tile_offset;

      ConsumeTile<true>(num_remaining,
                        tile_idx,
                        tile_offset,
                        first_tile_state,
                        second_tile_state,
                        num_first_selections,
                        num_second_selections);

      if (threadIdx.x == 0)
      {
        // Output the total number of items selection_flags
        d_num_selected_out[0] = num_first_selections;
        d_num_selected_out[1] = num_second_selections;
      }
    }
  }

};


CUB_NAMESPACE_END
