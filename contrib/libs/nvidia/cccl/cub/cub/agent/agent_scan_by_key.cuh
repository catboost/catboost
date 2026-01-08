/******************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * @file AgentScanByKey implements a stateful abstraction of CUDA thread blocks
 *       for participating in device-wide prefix scan by key.
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
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentScanByKey
 *
 * @tparam DelayConstructorT
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD                = 1,
          BlockLoadAlgorithm _LOAD_ALGORITHM   = BLOCK_LOAD_DIRECT,
          CacheLoadModifier _LOAD_MODIFIER     = LOAD_DEFAULT,
          BlockScanAlgorithm _SCAN_ALGORITHM   = BLOCK_SCAN_WARP_SCANS,
          BlockStoreAlgorithm _STORE_ALGORITHM = BLOCK_STORE_DIRECT,
          typename DelayConstructorT           = detail::fixed_delay_constructor_t<350, 450>>
struct AgentScanByKeyPolicy
{
  static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;

  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
  static constexpr CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM   = _SCAN_ALGORITHM;
  static constexpr BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;

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
namespace scan_by_key
{

/**
 * @brief AgentScanByKey implements a stateful abstraction of CUDA thread
 *        blocks for participating in device-wide prefix scan by key.
 *
 * @tparam AgentScanByKeyPolicyT
 *   Parameterized AgentScanPolicyT tuning policy type
 *
 * @tparam KeysInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam ValuesOutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam EqualityOp
 *   Equality functor type
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
 * @tparam AccumT
 *   The type of intermediate accumulator (according to P2322R6)
 */
template <typename AgentScanByKeyPolicyT,
          typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename EqualityOp,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT>
struct AgentScanByKey
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  using KeyT               = it_value_t<KeysInputIteratorT>;
  using InputT             = it_value_t<ValuesInputIteratorT>;
  using FlagValuePairT     = KeyValuePair<int, AccumT>;
  using ReduceBySegmentOpT = ScanBySegmentOp<ScanOpT>;

  using ScanTileStateT = ReduceByKeyScanTileState<AccumT, int>;

  // Constants
  // Inclusive scan if no init_value type is provided
  static constexpr int IS_INCLUSIVE     = ::cuda::std::is_same_v<InitValueT, NullType>;
  static constexpr int BLOCK_THREADS    = AgentScanByKeyPolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = AgentScanByKeyPolicyT::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

  using WrappedKeysInputIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<KeysInputIteratorT>,
                     CacheModifiedInputIterator<AgentScanByKeyPolicyT::LOAD_MODIFIER, KeyT, OffsetT>,
                     KeysInputIteratorT>;

  using WrappedValuesInputIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<ValuesInputIteratorT>,
                     CacheModifiedInputIterator<AgentScanByKeyPolicyT::LOAD_MODIFIER, InputT, OffsetT>,
                     ValuesInputIteratorT>;

  using BlockLoadKeysT = BlockLoad<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentScanByKeyPolicyT::LOAD_ALGORITHM>;

  using BlockLoadValuesT = BlockLoad<AccumT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentScanByKeyPolicyT::LOAD_ALGORITHM>;

  using BlockStoreValuesT = BlockStore<AccumT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentScanByKeyPolicyT::STORE_ALGORITHM>;

  using BlockDiscontinuityKeysT = BlockDiscontinuity<KeyT, BLOCK_THREADS, 1, 1>;

  using DelayConstructorT = typename AgentScanByKeyPolicyT::detail::delay_constructor_t;
  using TilePrefixCallbackT =
    TilePrefixCallbackOp<FlagValuePairT, ReduceBySegmentOpT, ScanTileStateT, DelayConstructorT>;

  using BlockScanT = BlockScan<FlagValuePairT, BLOCK_THREADS, AgentScanByKeyPolicyT::SCAN_ALGORITHM, 1, 1>;

  union TempStorage_
  {
    struct ScanStorage
    {
      typename BlockScanT::TempStorage scan;
      typename TilePrefixCallbackT::TempStorage prefix;
      typename BlockDiscontinuityKeysT::TempStorage discontinuity;
    } scan_storage;

    typename BlockLoadKeysT::TempStorage load_keys;
    typename BlockLoadValuesT::TempStorage load_values;
    typename BlockStoreValuesT::TempStorage store_values;
  };

  struct TempStorage : cub::Uninitialized<TempStorage_>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  TempStorage_& storage;
  WrappedKeysInputIteratorT d_keys_in;
  KeyT* d_keys_prev_in;
  WrappedValuesInputIteratorT d_values_in;
  ValuesOutputIteratorT d_values_out;
  InequalityWrapper<EqualityOp> inequality_op;
  ScanOpT scan_op;
  ReduceBySegmentOpT pair_scan_op;
  InitValueT init_value;

  //---------------------------------------------------------------------
  // Block scan utility methods (first tile)
  //---------------------------------------------------------------------

  // Exclusive scan specialization
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScanTile(FlagValuePairT (&scan_items)[ITEMS_PER_THREAD],
           FlagValuePairT& tile_aggregate,
           ::cuda::std::false_type /* is_inclusive */)
  {
    BlockScanT(storage.scan_storage.scan).ExclusiveScan(scan_items, scan_items, pair_scan_op, tile_aggregate);
  }

  // Inclusive scan specialization
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScanTile(FlagValuePairT (&scan_items)[ITEMS_PER_THREAD],
           FlagValuePairT& tile_aggregate,
           ::cuda::std::true_type /* is_inclusive */)
  {
    BlockScanT(storage.scan_storage.scan).InclusiveScan(scan_items, scan_items, pair_scan_op, tile_aggregate);
  }

  //---------------------------------------------------------------------
  // Block scan utility methods (subsequent tiles)
  //---------------------------------------------------------------------

  // Exclusive scan specialization (with prefix from predecessors)
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanTile(
    FlagValuePairT (&scan_items)[ITEMS_PER_THREAD],
    FlagValuePairT& tile_aggregate,
    TilePrefixCallbackT& prefix_op,
    ::cuda::std::false_type /* is_inclusive */)
  {
    BlockScanT(storage.scan_storage.scan).ExclusiveScan(scan_items, scan_items, pair_scan_op, prefix_op);
    tile_aggregate = prefix_op.GetBlockAggregate();
  }

  // Inclusive scan specialization (with prefix from predecessors)
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanTile(
    FlagValuePairT (&scan_items)[ITEMS_PER_THREAD],
    FlagValuePairT& tile_aggregate,
    TilePrefixCallbackT& prefix_op,
    ::cuda::std::true_type /* is_inclusive */)
  {
    BlockScanT(storage.scan_storage.scan).InclusiveScan(scan_items, scan_items, pair_scan_op, prefix_op);
    tile_aggregate = prefix_op.GetBlockAggregate();
  }

  //---------------------------------------------------------------------
  // Zip utility methods
  //---------------------------------------------------------------------

  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ZipValuesAndFlags(
    OffsetT num_remaining,
    AccumT (&values)[ITEMS_PER_THREAD],
    OffsetT (&segment_flags)[ITEMS_PER_THREAD],
    FlagValuePairT (&scan_items)[ITEMS_PER_THREAD])
  {
    // Zip values and segment_flags
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Set segment_flags for first out-of-bounds item, zero for others
      if (IS_LAST_TILE && OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM == num_remaining)
      {
        segment_flags[ITEM] = 1;
      }

      scan_items[ITEM].value = values[ITEM];
      scan_items[ITEM].key   = segment_flags[ITEM];
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  UnzipValues(AccumT (&values)[ITEMS_PER_THREAD], FlagValuePairT (&scan_items)[ITEMS_PER_THREAD])
  {
    // Unzip values and segment_flags
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      values[ITEM] = scan_items[ITEM].value;
    }
  }

  template <bool IsNull = ::cuda::std::is_same_v<InitValueT, NullType>, ::cuda::std::enable_if_t<!IsNull, int> = 0>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  AddInitToScan(AccumT (&items)[ITEMS_PER_THREAD], OffsetT (&flags)[ITEMS_PER_THREAD])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      items[ITEM] = flags[ITEM] ? init_value : scan_op(init_value, items[ITEM]);
    }
  }

  template <bool IsNull = ::cuda::std::is_same_v<InitValueT, NullType>, ::cuda::std::enable_if_t<IsNull, int> = 0>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  AddInitToScan(AccumT (& /*items*/)[ITEMS_PER_THREAD], OffsetT (& /*flags*/)[ITEMS_PER_THREAD])
  {}

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  // Process a tile of input (dynamic chained scan)
  //
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeTile(OffsetT /*num_items*/, OffsetT num_remaining, int tile_idx, OffsetT tile_base, ScanTileStateT& tile_state)
  {
    // Load items
    KeyT keys[ITEMS_PER_THREAD];
    AccumT values[ITEMS_PER_THREAD];
    OffsetT segment_flags[ITEMS_PER_THREAD];
    FlagValuePairT scan_items[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element
      // because collectives are not suffix guarded
      BlockLoadKeysT(storage.load_keys).Load(d_keys_in + tile_base, keys, num_remaining, *(d_keys_in + tile_base));
    }
    else
    {
      BlockLoadKeysT(storage.load_keys).Load(d_keys_in + tile_base, keys);
    }

    __syncthreads();

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element
      // because collectives are not suffix guarded
      BlockLoadValuesT(storage.load_values)
        .Load(d_values_in + tile_base, values, num_remaining, *(d_values_in + tile_base));
    }
    else
    {
      BlockLoadValuesT(storage.load_values).Load(d_values_in + tile_base, values);
    }

    __syncthreads();

    // first tile
    if (tile_idx == 0)
    {
      BlockDiscontinuityKeysT(storage.scan_storage.discontinuity).FlagHeads(segment_flags, keys, inequality_op);

      // Zip values and segment_flags
      ZipValuesAndFlags<IS_LAST_TILE>(num_remaining, values, segment_flags, scan_items);

      // Exclusive scan of values and segment_flags
      FlagValuePairT tile_aggregate;
      ScanTile(scan_items, tile_aggregate, bool_constant_v<IS_INCLUSIVE>);

      if (threadIdx.x == 0)
      {
        if (!IS_LAST_TILE)
        {
          tile_state.SetInclusive(0, tile_aggregate);
        }

        scan_items[0].key = 0;
      }
    }
    else
    {
      KeyT tile_pred_key = (threadIdx.x == 0) ? d_keys_prev_in[tile_idx] : KeyT();

      BlockDiscontinuityKeysT(storage.scan_storage.discontinuity)
        .FlagHeads(segment_flags, keys, inequality_op, tile_pred_key);

      // Zip values and segment_flags
      ZipValuesAndFlags<IS_LAST_TILE>(num_remaining, values, segment_flags, scan_items);

      FlagValuePairT tile_aggregate;
      TilePrefixCallbackT prefix_op(tile_state, storage.scan_storage.prefix, pair_scan_op, tile_idx);
      ScanTile(scan_items, tile_aggregate, prefix_op, bool_constant_v<IS_INCLUSIVE>);
    }

    __syncthreads();

    UnzipValues(values, scan_items);

    AddInitToScan(values, segment_flags);

    // Store items
    if (IS_LAST_TILE)
    {
      BlockStoreValuesT(storage.store_values).Store(d_values_out + tile_base, values, num_remaining);
    }
    else
    {
      BlockStoreValuesT(storage.store_values).Store(d_values_out + tile_base, values);
    }
  }

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  // Dequeue and scan tiles of items as part of a dynamic chained scan
  // with Init functor
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentScanByKey(
    TempStorage& storage,
    KeysInputIteratorT d_keys_in,
    KeyT* d_keys_prev_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    EqualityOp equality_op,
    ScanOpT scan_op,
    InitValueT init_value)
      : storage(storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_prev_in(d_keys_prev_in)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , inequality_op(equality_op)
      , scan_op(scan_op)
      , pair_scan_op(scan_op)
      , init_value(init_value)
  {}

  /**
   * Scan tiles of items as part of a dynamic chained scan
   *
   * @param num_items
   *   Total number of input items
   *
   * @param tile_state
   *   Global tile state descriptor
   *
   * start_tile
   *   The starting tile for the current grid
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(OffsetT num_items, ScanTileStateT& tile_state, int start_tile)
  {
    int tile_idx          = blockIdx.x;
    OffsetT tile_base     = OffsetT(ITEMS_PER_TILE) * tile_idx;
    OffsetT num_remaining = num_items - tile_base;

    if (num_remaining > ITEMS_PER_TILE)
    {
      // Not the last tile (full)
      ConsumeTile<false>(num_items, num_remaining, tile_idx, tile_base, tile_state);
    }
    else if (num_remaining > 0)
    {
      // The last tile (possibly partially-full)
      ConsumeTile<true>(num_items, num_remaining, tile_idx, tile_base, tile_state);
    }
  }
};

} // namespace scan_by_key
} // namespace detail

CUB_NAMESPACE_END
