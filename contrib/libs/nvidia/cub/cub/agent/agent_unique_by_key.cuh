/******************************************************************************
 * Copyright (c) NVIDIA CORPORATION.  All rights reserved.
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
 * cub::AgentUniqueByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide unique-by-key.
 */

#pragma once

#include <iterator>
#include <type_traits>

#include "../thread/thread_operators.cuh"
#include "../block/block_load.cuh"
#include "../block/block_scan.cuh"
#include "../agent/single_pass_scan_operators.cuh"
#include "../block/block_discontinuity.cuh"

CUB_NAMESPACE_BEGIN


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentUniqueByKey
 */
template <int                     _BLOCK_THREADS,
          int                     _ITEMS_PER_THREAD = 1,
          cub::BlockLoadAlgorithm _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
          cub::CacheLoadModifier  _LOAD_MODIFIER    = cub::LOAD_LDG,
          cub::BlockScanAlgorithm _SCAN_ALGORITHM   = cub::BLOCK_SCAN_WARP_SCANS>
struct AgentUniqueByKeyPolicy
{
    enum
    {
        BLOCK_THREADS    = _BLOCK_THREADS,
        ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
    };
    static const cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
    static const cub::CacheLoadModifier  LOAD_MODIFIER  = _LOAD_MODIFIER;
    static const cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/


/**
 * \brief AgentUniqueByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide unique-by-key
 */
template <
    typename AgentUniqueByKeyPolicyT,           ///< Parameterized AgentUniqueByKeyPolicy tuning policy type
    typename KeyInputIteratorT,                 ///< Random-access input iterator type for keys
    typename ValueInputIteratorT,               ///< Random-access input iterator type for values
    typename KeyOutputIteratorT,                ///< Random-access output iterator type for keys
    typename ValueOutputIteratorT,              ///< Random-access output iterator type for values
    typename EqualityOpT,                       ///< Equality operator type
    typename OffsetT>                           ///< Signed integer type for global offsets
struct AgentUniqueByKey
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // The input key and value type
    using KeyT = typename std::iterator_traits<KeyInputIteratorT>::value_type;
    using ValueT = typename std::iterator_traits<ValueInputIteratorT>::value_type;

    // Tile status descriptor interface type
    using ScanTileStateT = ScanTileState<OffsetT>;

    // Constants
    enum
    {
        BLOCK_THREADS           = AgentUniqueByKeyPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentUniqueByKeyPolicyT::ITEMS_PER_THREAD,
        ITEMS_PER_TILE          = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    // Cache-modified Input iterator wrapper type (for applying cache modifier) for keys
    using WrappedKeyInputIteratorT = typename std::conditional<std::is_pointer<KeyInputIteratorT>::value,
            CacheModifiedInputIterator<AgentUniqueByKeyPolicyT::LOAD_MODIFIER, KeyT, OffsetT>,     // Wrap the native input pointer with CacheModifiedValuesInputIterator
            KeyInputIteratorT>::type;                                                              // Directly use the supplied input iterator type

    // Cache-modified Input iterator wrapper type (for applying cache modifier) for values
    using WrappedValueInputIteratorT = typename std::conditional<std::is_pointer<ValueInputIteratorT>::value,
            CacheModifiedInputIterator<AgentUniqueByKeyPolicyT::LOAD_MODIFIER, ValueT, OffsetT>,   // Wrap the native input pointer with CacheModifiedValuesInputIterator
            ValueInputIteratorT>::type;                                                            // Directly use the supplied input iterator type

    // Parameterized BlockLoad type for input data
    using BlockLoadKeys = BlockLoad<
            KeyT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            AgentUniqueByKeyPolicyT::LOAD_ALGORITHM>;

    // Parameterized BlockLoad type for flags
    using BlockLoadValues = BlockLoad<
            ValueT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            AgentUniqueByKeyPolicyT::LOAD_ALGORITHM>;

    // Parameterized BlockDiscontinuity type for items
    using BlockDiscontinuityKeys = cub::BlockDiscontinuity<KeyT, BLOCK_THREADS>;

    // Parameterized BlockScan type
    using BlockScanT = cub::BlockScan<OffsetT, BLOCK_THREADS, AgentUniqueByKeyPolicyT::SCAN_ALGORITHM>;

    // Parameterized BlockDiscontinuity type for items
    using TilePrefixCallback = cub::TilePrefixCallbackOp<OffsetT, cub::Sum, ScanTileStateT>;

    // Key exchange type
    using KeyExchangeT = KeyT[ITEMS_PER_TILE];

    // Value exchange type
    using ValueExchangeT = ValueT[ITEMS_PER_TILE];

    // Shared memory type for this thread block
    union _TempStorage
    {
        struct ScanStorage
        {
            typename BlockScanT::TempStorage             scan;
            typename TilePrefixCallback::TempStorage     prefix;
            typename BlockDiscontinuityKeys::TempStorage discontinuity;
        } scan_storage;

        // Smem needed for loading keys
        typename BlockLoadKeys::TempStorage   load_keys;

        // Smem needed for loading values
        typename BlockLoadValues::TempStorage load_values;

        // Smem needed for compacting items (allows non POD items in this union)
        Uninitialized<KeyExchangeT>   shared_keys;
        Uninitialized<ValueExchangeT> shared_values;
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage&                       temp_storage;
    WrappedKeyInputIteratorT            d_keys_in;
    WrappedValueInputIteratorT          d_values_in;
    KeyOutputIteratorT                  d_keys_out;
    ValueOutputIteratorT                d_values_out;
    cub::InequalityWrapper<EqualityOpT> inequality_op;
    OffsetT                             num_items;


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    AgentUniqueByKey(
        TempStorage                  &temp_storage_,
        WrappedKeyInputIteratorT     d_keys_in_,
        WrappedValueInputIteratorT   d_values_in_,
        KeyOutputIteratorT           d_keys_out_,
        ValueOutputIteratorT         d_values_out_,
        EqualityOpT                  equality_op_,
        OffsetT                      num_items_)
    : 
        temp_storage(temp_storage_.Alias()),
        d_keys_in(d_keys_in_),
        d_values_in(d_values_in_),
        d_keys_out(d_keys_out_),
        d_values_out(d_values_out_),
        inequality_op(equality_op_),
        num_items(num_items_)
    {}



    //---------------------------------------------------------------------
    // Utility functions
    //---------------------------------------------------------------------

    struct KeyTagT {};
    struct ValueTagT {};

    __device__ __forceinline__
    KeyExchangeT &GetShared(KeyTagT)
    {
        return temp_storage.shared_keys.Alias();
    }
    __device__ __forceinline__
    ValueExchangeT &GetShared(ValueTagT)
    {
        return temp_storage.shared_values.Alias();
    }


    //---------------------------------------------------------------------
    // Scatter utility methods
    //---------------------------------------------------------------------
    template <typename Tag,
              typename OutputIt,
              typename T>
    __device__ __forceinline__ void Scatter(
        Tag      tag,
        OutputIt items_out,
        T (&items)[ITEMS_PER_THREAD],
        OffsetT (&selection_flags)[ITEMS_PER_THREAD],
        OffsetT (&selection_indices)[ITEMS_PER_THREAD],
        int  /*num_tile_items*/,
        int  num_tile_selections,
        OffsetT num_selections_prefix,
        OffsetT /*num_selections*/)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            int local_scatter_offset = selection_indices[ITEM] -
                                       num_selections_prefix;
            if (selection_flags[ITEM])
            {
                GetShared(tag)[local_scatter_offset] = items[ITEM];
            }
        }

        CTA_SYNC();

        for (int item = threadIdx.x;
             item < num_tile_selections;
             item += BLOCK_THREADS)
        {
            items_out[num_selections_prefix + item] = GetShared(tag)[item];
        }

        CTA_SYNC();
    }


    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------


    /**
     * Process first tile of input (dynamic chained scan).  Returns the running count of selections (including this tile)
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ OffsetT ConsumeFirstTile(
        int                 num_tile_items,     ///< Number of input items comprising this tile
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        KeyT        keys[ITEMS_PER_THREAD];
        OffsetT     selection_flags[ITEMS_PER_THREAD];
        OffsetT     selection_idx[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
            // Fill last elements with the first element
            // because collectives are not suffix guarded
            BlockLoadKeys(temp_storage.load_keys)
                .Load(d_keys_in + tile_offset,
                      keys,
                      num_tile_items,
                      *(d_keys_in + tile_offset));
        }
        else
        {
            BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys);
        }


        CTA_SYNC();

        ValueT values[ITEMS_PER_THREAD];
        if (IS_LAST_TILE)
        {
            // Fill last elements with the first element
            // because collectives are not suffix guarded
            BlockLoadValues(temp_storage.load_values)
                .Load(d_values_in + tile_offset,
                      values,
                      num_tile_items,
                      *(d_values_in + tile_offset));
        }
        else
        {
            BlockLoadValues(temp_storage.load_values)
                .Load(d_values_in + tile_offset, values);
        }

        CTA_SYNC();

        BlockDiscontinuityKeys(temp_storage.scan_storage.discontinuity)
             .FlagHeads(selection_flags, keys, inequality_op);
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Set selection_flags for out-of-bounds items
            if ((IS_LAST_TILE) && (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM >= num_tile_items))
                selection_flags[ITEM] = 1;
        }

        CTA_SYNC();


        OffsetT num_tile_selections   = 0;
        OffsetT num_selections        = 0;
        OffsetT num_selections_prefix = 0;

        BlockScanT(temp_storage.scan_storage.scan)
             .ExclusiveSum(selection_flags,
                           selection_idx,
                           num_tile_selections);

        if (threadIdx.x == 0)
        {
            // Update tile status if this is not the last tile
            if (!IS_LAST_TILE)
                tile_state.SetInclusive(0, num_tile_selections);
        }

        // Do not count any out-of-bounds selections
        if (IS_LAST_TILE)
        {
            int num_discount = ITEMS_PER_TILE - num_tile_items;
            num_tile_selections -= num_discount;
        }
        num_selections = num_tile_selections;

        CTA_SYNC();

        Scatter(KeyTagT(),
                d_keys_out,
                keys,
                selection_flags,
                selection_idx,
                num_tile_items,
                num_tile_selections,
                num_selections_prefix,
                num_selections);

        CTA_SYNC();

        Scatter(ValueTagT(),
                d_values_out,
                values,
                selection_flags,
                selection_idx,
                num_tile_items,
                num_tile_selections,
                num_selections_prefix,
                num_selections);

        return num_selections;
    }

    /**
     * Process subsequent tile of input (dynamic chained scan).  Returns the running count of selections (including this tile)
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ OffsetT ConsumeSubsequentTile(
        int                 num_tile_items,     ///< Number of input items comprising this tile
        int                 tile_idx,           ///< Tile index
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        KeyT        keys[ITEMS_PER_THREAD];
        OffsetT     selection_flags[ITEMS_PER_THREAD];
        OffsetT     selection_idx[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
            // Fill last elements with the first element
            // because collectives are not suffix guarded
            BlockLoadKeys(temp_storage.load_keys)
                .Load(d_keys_in + tile_offset,
                      keys,
                      num_tile_items,
                      *(d_keys_in + tile_offset));
        }
        else
        {
            BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys);
        }


        CTA_SYNC();

        ValueT values[ITEMS_PER_THREAD];
        if (IS_LAST_TILE)
        {
            // Fill last elements with the first element
            // because collectives are not suffix guarded
            BlockLoadValues(temp_storage.load_values)
                .Load(d_values_in + tile_offset,
                      values,
                      num_tile_items,
                      *(d_values_in + tile_offset));
        }
        else
        {
            BlockLoadValues(temp_storage.load_values)
                .Load(d_values_in + tile_offset, values);
        }

        CTA_SYNC();

        KeyT tile_predecessor = d_keys_in[tile_offset - 1];
        BlockDiscontinuityKeys(temp_storage.scan_storage.discontinuity)
            .FlagHeads(selection_flags, keys, inequality_op, tile_predecessor);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Set selection_flags for out-of-bounds items
            if ((IS_LAST_TILE) && (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM >= num_tile_items))
                selection_flags[ITEM] = 1;
        }

        CTA_SYNC();


        OffsetT num_tile_selections   = 0;
        OffsetT num_selections        = 0;
        OffsetT num_selections_prefix = 0;

        TilePrefixCallback prefix_cb(tile_state,
                                     temp_storage.scan_storage.prefix,
                                     cub::Sum(),
                                     tile_idx);
        BlockScanT(temp_storage.scan_storage.scan)
            .ExclusiveSum(selection_flags,
                          selection_idx,
                          prefix_cb);

        num_selections        = prefix_cb.GetInclusivePrefix();
        num_tile_selections   = prefix_cb.GetBlockAggregate();
        num_selections_prefix = prefix_cb.GetExclusivePrefix();

        if (IS_LAST_TILE)
        {
            int num_discount = ITEMS_PER_TILE - num_tile_items;
            num_tile_selections -= num_discount;
            num_selections -= num_discount;
        }

        CTA_SYNC();

        Scatter(KeyTagT(),
                d_keys_out,
                keys,
                selection_flags,
                selection_idx,
                num_tile_items,
                num_tile_selections,
                num_selections_prefix,
                num_selections);

        CTA_SYNC();

        Scatter(ValueTagT(),
                d_values_out,
                values,
                selection_flags,
                selection_idx,
                num_tile_items,
                num_tile_selections,
                num_selections_prefix,
                num_selections);

        return num_selections;
    }


    /**
     * Process a tile of input
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ OffsetT ConsumeTile(
        int                 num_tile_items,     ///< Number of input items comprising this tile
        int                 tile_idx,           ///< Tile index
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        OffsetT num_selections;
        if (tile_idx == 0)
        {
            num_selections = ConsumeFirstTile<IS_LAST_TILE>(num_tile_items, tile_offset, tile_state);
        }
        else
        {
            num_selections = ConsumeSubsequentTile<IS_LAST_TILE>(num_tile_items, tile_idx, tile_offset, tile_state);
        }

        return num_selections;
    }

    /**
     * Scan tiles of items as part of a dynamic chained scan
     */
    template <typename NumSelectedIteratorT>        ///< Output iterator type for recording number of items selection_flags
    __device__ __forceinline__ void ConsumeRange(
        int                     num_tiles,          ///< Total number of input tiles
        ScanTileStateT&         tile_state,         ///< Global tile state descriptor
        NumSelectedIteratorT    d_num_selected_out) ///< Output total number selection_flags
    {
        // Blocks are launched in increasing order, so just assign one tile per block
        int     tile_idx        = (blockIdx.x * gridDim.y) + blockIdx.y;    // Current tile index
        OffsetT tile_offset     = tile_idx * ITEMS_PER_TILE;                // Global offset for the current tile

        if (tile_idx < num_tiles - 1)
        {
            ConsumeTile<false>(ITEMS_PER_TILE,
                               tile_idx,
                               tile_offset,
                               tile_state);
        }
        else
        {
            int  num_remaining  = static_cast<int>(num_items - tile_offset);
            OffsetT num_selections = ConsumeTile<true>(num_remaining,
                                                       tile_idx,                                    
                                                       tile_offset,
                                                       tile_state);
            if (threadIdx.x == 0)                                                               
            {
                *d_num_selected_out = num_selections;
            }
        }
    }
};



CUB_NAMESPACE_END
