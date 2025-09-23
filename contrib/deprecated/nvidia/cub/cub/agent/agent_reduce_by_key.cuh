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
 * @file cub::AgentReduceByKey implements a stateful abstraction of CUDA thread
 *       blocks for participating in device-wide reduce-value-by-key.
 */

#pragma once
#pragma clang system_header


#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/config.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>

#include <iterator>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * @brief Parameterizable tuning policy type for AgentReduceByKey
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
 */
template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD,
          BlockLoadAlgorithm _LOAD_ALGORITHM,
          CacheLoadModifier _LOAD_MODIFIER,
          BlockScanAlgorithm _SCAN_ALGORITHM>
struct AgentReduceByKeyPolicy
{
  ///< Threads per thread block
  static constexpr int BLOCK_THREADS = _BLOCK_THREADS;

  ///< Items per thread (per tile of input)
  static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;

  ///< The BlockLoad algorithm to use
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;

  ///< Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;

  ///< The BlockScan algorithm to use
  static constexpr const BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * @brief AgentReduceByKey implements a stateful abstraction of CUDA thread
 *        blocks for participating in device-wide reduce-value-by-key
 *
 * @tparam AgentReduceByKeyPolicyT
 *   Parameterized AgentReduceByKeyPolicy tuning policy type
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
 *   Output iterator type for recording number of items selected
 *
 * @tparam EqualityOpT
 *   KeyT equality operator type
 *
 * @tparam ReductionOpT
 *   ValueT reduction operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <typename AgentReduceByKeyPolicyT,
          typename KeysInputIteratorT,
          typename UniqueOutputIteratorT,
          typename ValuesInputIteratorT,
          typename AggregatesOutputIteratorT,
          typename NumRunsOutputIteratorT,
          typename EqualityOpT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT>
struct AgentReduceByKey
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input keys type
  using KeyInputT = cub::detail::value_t<KeysInputIteratorT>;

  // The output keys type
  using KeyOutputT = cub::detail::non_void_value_t<UniqueOutputIteratorT, KeyInputT>;

  // The input values type
  using ValueInputT = cub::detail::value_t<ValuesInputIteratorT>;

  // Tuple type for scanning (pairs accumulated segment-value with
  // segment-index)
  using OffsetValuePairT = KeyValuePair<OffsetT, AccumT>;

  // Tuple type for pairing keys and values
  using KeyValuePairT = KeyValuePair<KeyOutputT, AccumT>;

  // Tile status descriptor interface type
  using ScanTileStateT = ReduceByKeyScanTileState<AccumT, OffsetT>;

  // Guarded inequality functor
  template <typename _EqualityOpT>
  struct GuardedInequalityWrapper
  {
    /// Wrapped equality operator
    _EqualityOpT op;

    /// Items remaining
    int num_remaining;

    /// Constructor
    __host__ __device__ __forceinline__ GuardedInequalityWrapper(_EqualityOpT op, int num_remaining)
        : op(op)
        , num_remaining(num_remaining)
    {}

    /// Boolean inequality operator, returns <tt>(a != b)</tt>
    template <typename T>
    __host__ __device__ __forceinline__ bool operator()(const T &a, const T &b, int idx) const
    {
      if (idx < num_remaining)
      {
        return !op(a, b); // In bounds
      }

      // Return true if first out-of-bounds item, false otherwise
      return (idx == num_remaining);
    }
  };

  // Constants
  static constexpr int BLOCK_THREADS     = AgentReduceByKeyPolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD  = AgentReduceByKeyPolicyT::ITEMS_PER_THREAD;
  static constexpr int TILE_ITEMS        = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr int TWO_PHASE_SCATTER = (ITEMS_PER_THREAD > 1);

  // Whether or not the scan operation has a zero-valued identity value (true
  // if we're performing addition on a primitive type)
  static constexpr int HAS_IDENTITY_ZERO = (std::is_same<ReductionOpT, cub::Sum>::value) &&
                                           (Traits<AccumT>::PRIMITIVE);

  // Cache-modified Input iterator wrapper type (for applying cache modifier)
  // for keys Wrap the native input pointer with
  // CacheModifiedValuesInputIterator or directly use the supplied input
  // iterator type
  using WrappedKeysInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<KeysInputIteratorT>::value,
    CacheModifiedInputIterator<AgentReduceByKeyPolicyT::LOAD_MODIFIER, KeyInputT, OffsetT>,
    KeysInputIteratorT>;

  // Cache-modified Input iterator wrapper type (for applying cache modifier)
  // for values Wrap the native input pointer with
  // CacheModifiedValuesInputIterator or directly use the supplied input
  // iterator type
  using WrappedValuesInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<ValuesInputIteratorT>::value,
    CacheModifiedInputIterator<AgentReduceByKeyPolicyT::LOAD_MODIFIER, ValueInputT, OffsetT>,
    ValuesInputIteratorT>;

  // Cache-modified Input iterator wrapper type (for applying cache modifier)
  // for fixup values Wrap the native input pointer with
  // CacheModifiedValuesInputIterator or directly use the supplied input
  // iterator type
  using WrappedFixupInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<AggregatesOutputIteratorT>::value,
    CacheModifiedInputIterator<AgentReduceByKeyPolicyT::LOAD_MODIFIER, ValueInputT, OffsetT>,
    AggregatesOutputIteratorT>;

  // Reduce-value-by-segment scan operator
  using ReduceBySegmentOpT = ReduceBySegmentOp<ReductionOpT>;

  // Parameterized BlockLoad type for keys
  using BlockLoadKeysT =
    BlockLoad<KeyOutputT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentReduceByKeyPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockLoad type for values
  using BlockLoadValuesT =
    BlockLoad<AccumT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentReduceByKeyPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockDiscontinuity type for keys
  using BlockDiscontinuityKeys = BlockDiscontinuity<KeyOutputT, BLOCK_THREADS>;

  // Parameterized BlockScan type
  using BlockScanT =
    BlockScan<OffsetValuePairT, BLOCK_THREADS, AgentReduceByKeyPolicyT::SCAN_ALGORITHM>;

  // Callback type for obtaining tile prefix during block scan
  using TilePrefixCallbackOpT =
    TilePrefixCallbackOp<OffsetValuePairT, ReduceBySegmentOpT, ScanTileStateT>;

  // Key and value exchange types
  typedef KeyOutputT KeyExchangeT[TILE_ITEMS + 1];
  typedef AccumT ValueExchangeT[TILE_ITEMS + 1];

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
      typename BlockDiscontinuityKeys::TempStorage discontinuity;
    } scan_storage;

    // Smem needed for loading keys
    typename BlockLoadKeysT::TempStorage load_keys;

    // Smem needed for loading values
    typename BlockLoadValuesT::TempStorage load_values;

    // Smem needed for compacting key value pairs(allows non POD items in this
    // union)
    Uninitialized<KeyValuePairT[TILE_ITEMS + 1]> raw_exchange;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  /// Reference to temp_storage
  _TempStorage &temp_storage;

  /// Input keys
  WrappedKeysInputIteratorT d_keys_in;

  /// Unique output keys
  UniqueOutputIteratorT d_unique_out;

  /// Input values
  WrappedValuesInputIteratorT d_values_in;

  /// Output value aggregates
  AggregatesOutputIteratorT d_aggregates_out;

  /// Output pointer for total number of segments identified
  NumRunsOutputIteratorT d_num_runs_out;

  /// KeyT equality operator
  EqualityOpT equality_op;

  /// Reduction operator
  ReductionOpT reduction_op;

  /// Reduce-by-segment scan operator
  ReduceBySegmentOpT scan_op;

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  /**
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param d_keys_in
   *   Input keys
   *
   * @param d_unique_out
   *   Unique output keys
   *
   * @param d_values_in
   *   Input values
   *
   * @param d_aggregates_out
   *   Output value aggregates
   *
   * @param d_num_runs_out
   *   Output pointer for total number of segments identified
   *
   * @param equality_op
   *   KeyT equality operator
   *
   * @param reduction_op
   *   ValueT reduction operator
   */
  __device__ __forceinline__ AgentReduceByKey(TempStorage &temp_storage,
                                              KeysInputIteratorT d_keys_in,
                                              UniqueOutputIteratorT d_unique_out,
                                              ValuesInputIteratorT d_values_in,
                                              AggregatesOutputIteratorT d_aggregates_out,
                                              NumRunsOutputIteratorT d_num_runs_out,
                                              EqualityOpT equality_op,
                                              ReductionOpT reduction_op)
      : temp_storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_unique_out(d_unique_out)
      , d_values_in(d_values_in)
      , d_aggregates_out(d_aggregates_out)
      , d_num_runs_out(d_num_runs_out)
      , equality_op(equality_op)
      , reduction_op(reduction_op)
      , scan_op(reduction_op)
  {}

  //---------------------------------------------------------------------
  // Scatter utility methods
  //---------------------------------------------------------------------

  /**
   * Directly scatter flagged items to output offsets
   */
  __device__ __forceinline__ void ScatterDirect(KeyValuePairT (&scatter_items)[ITEMS_PER_THREAD],
                                                OffsetT (&segment_flags)[ITEMS_PER_THREAD],
                                                OffsetT (&segment_indices)[ITEMS_PER_THREAD])
  {
// Scatter flagged keys and values
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (segment_flags[ITEM])
      {
        d_unique_out[segment_indices[ITEM]]     = scatter_items[ITEM].key;
        d_aggregates_out[segment_indices[ITEM]] = scatter_items[ITEM].value;
      }
    }
  }

  /**
   * 2-phase scatter flagged items to output offsets
   *
   * The exclusive scan causes each head flag to be paired with the previous
   * value aggregate: the scatter offsets must be decremented for value
   * aggregates
   */
  __device__ __forceinline__ void ScatterTwoPhase(KeyValuePairT (&scatter_items)[ITEMS_PER_THREAD],
                                                  OffsetT (&segment_flags)[ITEMS_PER_THREAD],
                                                  OffsetT (&segment_indices)[ITEMS_PER_THREAD],
                                                  OffsetT num_tile_segments,
                                                  OffsetT num_tile_segments_prefix)
  {
    CTA_SYNC();

// Compact and scatter pairs
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (segment_flags[ITEM])
      {
        temp_storage.raw_exchange.Alias()[segment_indices[ITEM] - num_tile_segments_prefix] =
          scatter_items[ITEM];
      }
    }

    CTA_SYNC();

    for (int item = threadIdx.x; item < num_tile_segments; item += BLOCK_THREADS)
    {
      KeyValuePairT pair                                = temp_storage.raw_exchange.Alias()[item];
      d_unique_out[num_tile_segments_prefix + item]     = pair.key;
      d_aggregates_out[num_tile_segments_prefix + item] = pair.value;
    }
  }

  /**
   * Scatter flagged items
   */
  __device__ __forceinline__ void Scatter(KeyValuePairT (&scatter_items)[ITEMS_PER_THREAD],
                                          OffsetT (&segment_flags)[ITEMS_PER_THREAD],
                                          OffsetT (&segment_indices)[ITEMS_PER_THREAD],
                                          OffsetT num_tile_segments,
                                          OffsetT num_tile_segments_prefix)
  {
    // Do a one-phase scatter if (a) two-phase is disabled or (b) the average
    // number of selected items per thread is less than one
    if (TWO_PHASE_SCATTER && (num_tile_segments > BLOCK_THREADS))
    {
      ScatterTwoPhase(scatter_items,
                      segment_flags,
                      segment_indices,
                      num_tile_segments,
                      num_tile_segments_prefix);
    }
    else
    {
      ScatterDirect(scatter_items, segment_flags, segment_indices);
    }
  }

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  /**
   * @brief Process a tile of input (dynamic chained scan)
   *
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
  __device__ __forceinline__ void
  ConsumeTile(OffsetT num_remaining, int tile_idx, OffsetT tile_offset, ScanTileStateT &tile_state)
  {
    // Tile keys
    KeyOutputT keys[ITEMS_PER_THREAD];

    // Tile keys shuffled up
    KeyOutputT prev_keys[ITEMS_PER_THREAD];

    // Tile values
    AccumT values[ITEMS_PER_THREAD];

    // Segment head flags
    OffsetT head_flags[ITEMS_PER_THREAD];

    // Segment indices
    OffsetT segment_indices[ITEMS_PER_THREAD];

    // Zipped values and segment flags|indices
    OffsetValuePairT scan_items[ITEMS_PER_THREAD];

    // Zipped key value pairs for scattering
    KeyValuePairT scatter_items[ITEMS_PER_THREAD];

    // Load keys
    if (IS_LAST_TILE)
    {
      BlockLoadKeysT(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys, num_remaining);
    }
    else
    {
      BlockLoadKeysT(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys);
    }

    // Load tile predecessor key in first thread
    KeyOutputT tile_predecessor;
    if (threadIdx.x == 0)
    {
      // if (tile_idx == 0)
      //   first tile gets repeat of first item (thus first item will not
      //   be flagged as a head)
      // else
      //   Subsequent tiles get last key from previous tile
      tile_predecessor = (tile_idx == 0) ? keys[0] : d_keys_in[tile_offset - 1];
    }

    CTA_SYNC();

    // Load values
    if (IS_LAST_TILE)
    {
      BlockLoadValuesT(temp_storage.load_values)
        .Load(d_values_in + tile_offset, values, num_remaining);
    }
    else
    {
      BlockLoadValuesT(temp_storage.load_values).Load(d_values_in + tile_offset, values);
    }

    CTA_SYNC();

    // Initialize head-flags and shuffle up the previous keys
    if (IS_LAST_TILE)
    {
      // Use custom flag operator to additionally flag the first out-of-bounds
      // item
      GuardedInequalityWrapper<EqualityOpT> flag_op(equality_op, num_remaining);
      BlockDiscontinuityKeys(temp_storage.scan_storage.discontinuity)
        .FlagHeads(head_flags, keys, prev_keys, flag_op, tile_predecessor);
    }
    else
    {
      InequalityWrapper<EqualityOpT> flag_op(equality_op);
      BlockDiscontinuityKeys(temp_storage.scan_storage.discontinuity)
        .FlagHeads(head_flags, keys, prev_keys, flag_op, tile_predecessor);
    }

    // Reset head-flag on the very first item to make sure we don't start a new run for data where
    // (key[0] == key[0]) is false (e.g., when key[0] is NaN)
    if (threadIdx.x == 0 && tile_idx == 0)
    {
      head_flags[0] = 0;
    }

    // Zip values and head flags
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      scan_items[ITEM].value = values[ITEM];
      scan_items[ITEM].key   = head_flags[ITEM];
    }

    // Perform exclusive tile scan
    // Inclusive block-wide scan aggregate
    OffsetValuePairT block_aggregate;

    // Number of segments prior to this tile
    OffsetT num_segments_prefix;

    // The tile prefix folded with block_aggregate
    OffsetValuePairT total_aggregate;

    if (tile_idx == 0)
    {
      // Scan first tile
      BlockScanT(temp_storage.scan_storage.scan)
        .ExclusiveScan(scan_items, scan_items, scan_op, block_aggregate);
      num_segments_prefix = 0;
      total_aggregate     = block_aggregate;

      // Update tile status if there are successor tiles
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
      BlockScanT(temp_storage.scan_storage.scan)
        .ExclusiveScan(scan_items, scan_items, scan_op, prefix_op);

      block_aggregate     = prefix_op.GetBlockAggregate();
      num_segments_prefix = prefix_op.GetExclusivePrefix().key;
      total_aggregate     = prefix_op.GetInclusivePrefix();
    }

// Rezip scatter items and segment indices
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      scatter_items[ITEM].key   = prev_keys[ITEM];
      scatter_items[ITEM].value = scan_items[ITEM].value;
      segment_indices[ITEM]     = scan_items[ITEM].key;
    }

    // At this point, each flagged segment head has:
    //  - The key for the previous segment
    //  - The reduced value from the previous segment
    //  - The segment index for the reduced value

    // Scatter flagged keys and values
    OffsetT num_tile_segments = block_aggregate.key;
    Scatter(scatter_items, head_flags, segment_indices, num_tile_segments, num_segments_prefix);

    // Last thread in last tile will output final count (and last pair, if
    // necessary)
    if ((IS_LAST_TILE) && (threadIdx.x == BLOCK_THREADS - 1))
    {
      OffsetT num_segments = num_segments_prefix + num_tile_segments;

      // If the last tile is a whole tile, output the final_value
      if (num_remaining == TILE_ITEMS)
      {
        d_unique_out[num_segments]     = keys[ITEMS_PER_THREAD - 1];
        d_aggregates_out[num_segments] = total_aggregate.value;
        num_segments++;
      }

      // Output the total number of items selected
      *d_num_runs_out = num_segments;
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
};

CUB_NAMESPACE_END
