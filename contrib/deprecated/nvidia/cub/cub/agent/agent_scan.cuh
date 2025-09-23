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
 * @file cub::AgentScan implements a stateful abstraction of CUDA thread blocks 
 *       for participating in device-wide prefix scan .
 */

#pragma once
#pragma clang system_header


#include <iterator>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/config.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * @brief Parameterizable tuning policy type for AgentScan
 *
 * @tparam NOMINAL_BLOCK_THREADS_4B
 *   Threads per thread block
 *
 * @tparam NOMINAL_ITEMS_PER_THREAD_4B
 *   Items per thread (per tile of input)
 *
 * @tparam ComputeT
 *   Dominant compute type
 *
 * @tparam _LOAD_ALGORITHM
 *   The BlockLoad algorithm to use
 *
 * @tparam _LOAD_MODIFIER
 *   Cache load modifier for reading input elements
 *
 * @tparam _STORE_ALGORITHM
 *   The BlockStore algorithm to use
 *
 * @tparam _SCAN_ALGORITHM
 *   The BlockScan algorithm to use
 *
 */
template <int NOMINAL_BLOCK_THREADS_4B,
          int NOMINAL_ITEMS_PER_THREAD_4B,
          typename ComputeT,
          BlockLoadAlgorithm _LOAD_ALGORITHM,
          CacheLoadModifier _LOAD_MODIFIER,
          BlockStoreAlgorithm _STORE_ALGORITHM,
          BlockScanAlgorithm _SCAN_ALGORITHM,
          typename ScalingType = MemBoundScaling<NOMINAL_BLOCK_THREADS_4B,
                                                 NOMINAL_ITEMS_PER_THREAD_4B,
                                                 ComputeT>>
struct AgentScanPolicy : ScalingType
{
  static const BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
  static const CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
  static const BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
  static const BlockScanAlgorithm SCAN_ALGORITHM   = _SCAN_ALGORITHM;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * @brief AgentScan implements a stateful abstraction of CUDA thread blocks for
 *        participating in device-wide prefix scan.
 * @tparam AgentScanPolicyT
 *   Parameterized AgentScanPolicyT tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam ScanOpT
 *   Scan functor type
 *
 * @tparam InitValueT
 *   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 */
template <typename AgentScanPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT>
struct AgentScan
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input value type
  using InputT = cub::detail::value_t<InputIteratorT>;

  // Tile status descriptor interface type
  using ScanTileStateT = ScanTileState<AccumT>;

  // Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using WrappedInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<InputIteratorT>::value,
    CacheModifiedInputIterator<AgentScanPolicyT::LOAD_MODIFIER, InputT, OffsetT>,
    InputIteratorT>;

  // Constants
  enum
  {
    // Inclusive scan if no init_value type is provided
    IS_INCLUSIVE     = std::is_same<InitValueT, NullType>::value,
    BLOCK_THREADS    = AgentScanPolicyT::BLOCK_THREADS,
    ITEMS_PER_THREAD = AgentScanPolicyT::ITEMS_PER_THREAD,
    TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD,
  };

  // Parameterized BlockLoad type
  typedef BlockLoad<AccumT,
                    AgentScanPolicyT::BLOCK_THREADS,
                    AgentScanPolicyT::ITEMS_PER_THREAD,
                    AgentScanPolicyT::LOAD_ALGORITHM>
    BlockLoadT;

  // Parameterized BlockStore type
  typedef BlockStore<AccumT,
                     AgentScanPolicyT::BLOCK_THREADS,
                     AgentScanPolicyT::ITEMS_PER_THREAD,
                     AgentScanPolicyT::STORE_ALGORITHM>
    BlockStoreT;

  // Parameterized BlockScan type
  typedef BlockScan<AccumT,
                    AgentScanPolicyT::BLOCK_THREADS,
                    AgentScanPolicyT::SCAN_ALGORITHM>
    BlockScanT;

  // Callback type for obtaining tile prefix during block scan
  typedef TilePrefixCallbackOp<AccumT, ScanOpT, ScanTileStateT>
    TilePrefixCallbackOpT;

  // Stateful BlockScan prefix callback type for managing a running total while
  // scanning consecutive tiles
  typedef BlockScanRunningPrefixOp<AccumT, ScanOpT> RunningPrefixCallbackOp;

  // Shared memory type for this thread block
  union _TempStorage
  {
    // Smem needed for tile loading
    typename BlockLoadT::TempStorage load;

    // Smem needed for tile storing
    typename BlockStoreT::TempStorage store;

    struct ScanStorage
    {
      // Smem needed for cooperative prefix callback
      typename TilePrefixCallbackOpT::TempStorage prefix;

      // Smem needed for tile scanning
      typename BlockScanT::TempStorage scan;
    } scan_storage;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage &temp_storage; ///< Reference to temp_storage
  WrappedInputIteratorT d_in; ///< Input data
  OutputIteratorT d_out;      ///< Output data
  ScanOpT scan_op;            ///< Binary scan operator
  InitValueT init_value;      ///< The init_value element for ScanOpT

  //---------------------------------------------------------------------
  // Block scan utility methods
  //---------------------------------------------------------------------

  /**
   * Exclusive scan specialization (first tile)
   */
  __device__ __forceinline__ void ScanTile(AccumT (&items)[ITEMS_PER_THREAD],
                                           AccumT init_value,
                                           ScanOpT scan_op,
                                           AccumT &block_aggregate,
                                           Int2Type<false> /*is_inclusive*/)
  {
    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveScan(items, items, init_value, scan_op, block_aggregate);
    block_aggregate = scan_op(init_value, block_aggregate);
  }

  /**
   * Inclusive scan specialization (first tile)
   */
  __device__ __forceinline__ void ScanTile(AccumT (&items)[ITEMS_PER_THREAD],
                                           InitValueT /*init_value*/,
                                           ScanOpT scan_op,
                                           AccumT &block_aggregate,
                                           Int2Type<true> /*is_inclusive*/)
  {
    BlockScanT(temp_storage.scan_storage.scan)
      .InclusiveScan(items, items, scan_op, block_aggregate);
  }

  /**
   * Exclusive scan specialization (subsequent tiles)
   */
  template <typename PrefixCallback>
  __device__ __forceinline__ void ScanTile(AccumT (&items)[ITEMS_PER_THREAD],
                                           ScanOpT scan_op,
                                           PrefixCallback &prefix_op,
                                           Int2Type<false> /*is_inclusive*/)
  {
    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveScan(items, items, scan_op, prefix_op);
  }

  /**
   * Inclusive scan specialization (subsequent tiles)
   */
  template <typename PrefixCallback>
  __device__ __forceinline__ void ScanTile(AccumT (&items)[ITEMS_PER_THREAD],
                                           ScanOpT scan_op,
                                           PrefixCallback &prefix_op,
                                           Int2Type<true> /*is_inclusive*/)
  {
    BlockScanT(temp_storage.scan_storage.scan)
      .InclusiveScan(items, items, scan_op, prefix_op);
  }

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
   * @param d_out
   *   Output data
   *
   * @param scan_op
   *   Binary scan operator
   *
   * @param init_value
   *   Initial value to seed the exclusive scan
   */
  __device__ __forceinline__ AgentScan(TempStorage &temp_storage,
                                       InputIteratorT d_in,
                                       OutputIteratorT d_out,
                                       ScanOpT scan_op,
                                       InitValueT init_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , init_value(init_value)
  {}

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  /**
   * Process a tile of input (dynamic chained scan)
   * @tparam IS_LAST_TILE
   *   Whether the current tile is the last tile
   *
   * @param num_remaining
   *   Number of global input items remaining (including this tile)
   *
   * @param tile_idx
   *   Tile index
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param tile_state
   *   Global tile state descriptor
   */
  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void ConsumeTile(OffsetT num_remaining,
                                              int tile_idx,
                                              OffsetT tile_offset,
                                              ScanTileStateT &tile_state)
  {
    // Load items
    AccumT items[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element because collectives are
      // not suffix guarded.
      BlockLoadT(temp_storage.load)
        .Load(d_in + tile_offset, items, num_remaining, *(d_in + tile_offset));
    }
    else
    {
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
    }

    CTA_SYNC();

    // Perform tile scan
    if (tile_idx == 0)
    {
      // Scan first tile
      AccumT block_aggregate;
      ScanTile(items,
               init_value,
               scan_op,
               block_aggregate,
               Int2Type<IS_INCLUSIVE>());

      if ((!IS_LAST_TILE) && (threadIdx.x == 0))
      {
        tile_state.SetInclusive(0, block_aggregate);
      }
    }
    else
    {
      // Scan non-first tile
      TilePrefixCallbackOpT prefix_op(tile_state,
                                      temp_storage.scan_storage.prefix,
                                      scan_op,
                                      tile_idx);
      ScanTile(items, scan_op, prefix_op, Int2Type<IS_INCLUSIVE>());
    }

    CTA_SYNC();

    // Store items
    if (IS_LAST_TILE)
    {
      BlockStoreT(temp_storage.store)
        .Store(d_out + tile_offset, items, num_remaining);
    }
    else
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
    }
  }

  /**
   * @brief Scan tiles of items as part of a dynamic chained scan
   *
   * @param num_items
   *   Total number of input items
   *
   * @param tile_state
   *   Global tile state descriptor
   *
   * @param start_tile
   *   The starting tile for the current grid
   */
  __device__ __forceinline__ void ConsumeRange(OffsetT num_items,
                                               ScanTileStateT &tile_state,
                                               int start_tile)
  {
    // Blocks are launched in increasing order, so just assign one tile per
    // block

    // Current tile index
    int tile_idx = start_tile + blockIdx.x;

    // Global offset for the current tile
    OffsetT tile_offset = OffsetT(TILE_ITEMS) * tile_idx;

    // Remaining items (including this tile)
    OffsetT num_remaining = num_items - tile_offset;

    if (num_remaining > TILE_ITEMS)
    {
      // Not last tile
      ConsumeTile<false>(num_remaining, tile_idx, tile_offset, tile_state);
    }
    else if (num_remaining > 0)
    {
      // Last tile
      ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
    }
  }

  //---------------------------------------------------------------------------
  // Scan an sequence of consecutive tiles (independent of other thread blocks)
  //---------------------------------------------------------------------------

  /**
   * @brief Process a tile of input
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param prefix_op
   *   Running prefix operator
   *
   * @param valid_items
   *   Number of valid items in the tile
   */
  template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
  __device__ __forceinline__ void ConsumeTile(OffsetT tile_offset,
                                              RunningPrefixCallbackOp &prefix_op,
                                              int valid_items = TILE_ITEMS)
  {
    // Load items
    AccumT items[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element because collectives are
      // not suffix guarded.
      BlockLoadT(temp_storage.load)
        .Load(d_in + tile_offset, items, valid_items, *(d_in + tile_offset));
    }
    else
    {
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
    }

    CTA_SYNC();

    // Block scan
    if (IS_FIRST_TILE)
    {
      AccumT block_aggregate;
      ScanTile(items,
               init_value,
               scan_op,
               block_aggregate,
               Int2Type<IS_INCLUSIVE>());
      prefix_op.running_total = block_aggregate;
    }
    else
    {
      ScanTile(items, scan_op, prefix_op, Int2Type<IS_INCLUSIVE>());
    }

    CTA_SYNC();

    // Store items
    if (IS_LAST_TILE)
    {
      BlockStoreT(temp_storage.store)
        .Store(d_out + tile_offset, items, valid_items);
    }
    else
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
    }
  }

  /**
   * @brief Scan a consecutive share of input tiles
   *
   * @param[in] range_offset
   *   Threadblock begin offset (inclusive)
   *
   * @param[in] range_end
   *   Threadblock end offset (exclusive)
   */
  __device__ __forceinline__ void ConsumeRange(OffsetT range_offset,
                                               OffsetT range_end)
  {
    BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(scan_op);

    if (range_offset + TILE_ITEMS <= range_end)
    {
      // Consume first tile of input (full)
      ConsumeTile<true, true>(range_offset, prefix_op);
      range_offset += TILE_ITEMS;

      // Consume subsequent full tiles of input
      while (range_offset + TILE_ITEMS <= range_end)
      {
        ConsumeTile<false, true>(range_offset, prefix_op);
        range_offset += TILE_ITEMS;
      }

      // Consume a partially-full tile
      if (range_offset < range_end)
      {
        int valid_items = range_end - range_offset;
        ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
      }
    }
    else
    {
      // Consume the first tile of input (partially-full)
      int valid_items = range_end - range_offset;
      ConsumeTile<true, false>(range_offset, prefix_op, valid_items);
    }
  }

  /**
   * @brief Scan a consecutive share of input tiles, seeded with the
   *        specified prefix value
   * @param[in] range_offset
   *   Threadblock begin offset (inclusive)
   *
   * @param[in] range_end
   *   Threadblock end offset (exclusive)
   *
   * @param[in] prefix
   *   The prefix to apply to the scan segment
   */
  __device__ __forceinline__ void ConsumeRange(OffsetT range_offset,
                                               OffsetT range_end,
                                               AccumT prefix)
  {
    BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(prefix, scan_op);

    // Consume full tiles of input
    while (range_offset + TILE_ITEMS <= range_end)
    {
      ConsumeTile<true, false>(range_offset, prefix_op);
      range_offset += TILE_ITEMS;
    }

    // Consume a partially-full tile
    if (range_offset < range_end)
    {
      int valid_items = range_end - range_offset;
      ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
    }
  }
};

CUB_NAMESPACE_END
