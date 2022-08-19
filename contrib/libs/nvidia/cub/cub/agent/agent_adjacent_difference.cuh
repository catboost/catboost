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

#include "../config.cuh"
#include "../util_type.cuh"
#include "../util_namespace.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_adjacent_difference.cuh"

#include <thrust/system/cuda/detail/core/util.h>


CUB_NAMESPACE_BEGIN


template <
  int                      _BLOCK_THREADS,
  int                      _ITEMS_PER_THREAD = 1,
  cub::BlockLoadAlgorithm  _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
  cub::CacheLoadModifier   _LOAD_MODIFIER    = cub::LOAD_LDG,
  cub::BlockStoreAlgorithm _STORE_ALGORITHM  = cub::BLOCK_STORE_DIRECT>
struct AgentAdjacentDifferencePolicy
{
  static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
};

template <typename Policy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename DifferenceOpT,
          typename OffsetT,
          typename InputT,
          typename OutputT,
          bool MayAlias,
          bool ReadLeft>
struct AgentDifference
{
  using LoadIt = typename THRUST_NS_QUALIFIER::cuda_cub::core::LoadIterator<Policy, InputIteratorT>::type;

  using BlockLoad = typename cub::BlockLoadType<Policy, LoadIt>::type;
  using BlockStore = typename cub::BlockStoreType<Policy, OutputIteratorT, OutputT>::type;

  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<InputT, Policy::BLOCK_THREADS>;

  union _TempStorage
  {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
    typename BlockAdjacentDifferenceT::TempStorage adjacent_difference;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage> {};

  static constexpr int BLOCK_THREADS = Policy::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE = Policy::ITEMS_PER_TILE;
  static constexpr int SHARED_MEMORY_SIZE = static_cast<int>(sizeof(TempStorage));

  _TempStorage &temp_storage;
  InputIteratorT input_it;
  LoadIt load_it;
  InputT *first_tile_previous;
  OutputIteratorT result;
  DifferenceOpT difference_op;
  OffsetT num_items;

  __device__ __forceinline__ AgentDifference(TempStorage &temp_storage,
                                             InputIteratorT input_it,
                                             InputT *first_tile_previous,
                                             OutputIteratorT result,
                                             DifferenceOpT difference_op,
                                             OffsetT num_items)
      : temp_storage(temp_storage.Alias())
      , input_it(input_it)
      , load_it(
          THRUST_NS_QUALIFIER::cuda_cub::core::make_load_iterator(Policy(),
                                                                  input_it))
      , first_tile_previous(first_tile_previous)
      , result(result)
      , difference_op(difference_op)
      , num_items(num_items)
  {}

  template <bool IS_LAST_TILE,
            bool IS_FIRST_TILE>
  __device__ __forceinline__ void consume_tile_impl(int num_remaining,
                                                    int tile_idx,
                                                    OffsetT tile_base)
  {
    InputT input[ITEMS_PER_THREAD];
    OutputT output[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Fill last elements with the first element
      // because collectives are not suffix guarded
      BlockLoad(temp_storage.load)
        .Load(load_it + tile_base, input, num_remaining, *(load_it + tile_base));
    }
    else
    {
      BlockLoad(temp_storage.load).Load(load_it + tile_base, input);
    }

    CTA_SYNC();

    if (ReadLeft)
    {
      if (IS_FIRST_TILE)
      {
        if (IS_LAST_TILE)
        {
          BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
            .SubtractLeftPartialTile(input,
                                     output,
                                     difference_op,
                                     num_remaining);
        }
        else
        {
          BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
            .SubtractLeft(input, output, difference_op);
        }
      }
      else
      {
        InputT tile_prev_input = MayAlias 
                               ? first_tile_previous[tile_idx]
                               : *(input_it + tile_base - 1);

        if (IS_LAST_TILE)
        {
          BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
            .SubtractLeftPartialTile(input,
                                     output,
                                     difference_op,
                                     num_remaining,
                                     tile_prev_input);
        }
        else
        {
          BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
            .SubtractLeft(input, output, difference_op, tile_prev_input);
        }
      }
    }
    else
    {
      if (IS_LAST_TILE)
      {
        BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
          .SubtractRightPartialTile(input, output, difference_op, num_remaining);
      }
      else
      {
        InputT tile_next_input = MayAlias
                               ? first_tile_previous[tile_idx]
                               : *(input_it + tile_base + ITEMS_PER_TILE);

        BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
          .SubtractRight(input, output, difference_op, tile_next_input);
      }
    }

    CTA_SYNC();

    if (IS_LAST_TILE)
    {
      BlockStore(temp_storage.store)
        .Store(result + tile_base, output, num_remaining);
    }
    else
    {
      BlockStore(temp_storage.store).Store(result + tile_base, output);
    }
  }

  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void consume_tile(int num_remaining,
                                               int tile_idx,
                                               OffsetT tile_base)
  {
    if (tile_idx == 0)
    {
      consume_tile_impl<IS_LAST_TILE, true>(num_remaining,
                                            tile_idx,
                                            tile_base);
    }
    else
    {
      consume_tile_impl<IS_LAST_TILE, false>(num_remaining,
                                             tile_idx,
                                             tile_base);
    }
  }

  __device__ __forceinline__ void Process(int tile_idx,
                                          OffsetT tile_base)
  {
    OffsetT num_remaining = num_items - tile_base;

    if (num_remaining > ITEMS_PER_TILE) // not a last tile
    {
      consume_tile<false>(num_remaining, tile_idx, tile_base);
    }
    else
    {
      consume_tile<true>(num_remaining, tile_idx, tile_base);
    }
  }
};

template <typename InputIteratorT,
          typename InputT,
          typename OffsetT,
          bool ReadLeft>
struct AgentDifferenceInit
{
  static constexpr int BLOCK_THREADS = 128;

  static __device__ __forceinline__ void Process(int tile_idx,
                                                 InputIteratorT first,
                                                 InputT *result,
                                                 OffsetT num_tiles,
                                                 int items_per_tile)
  {
    OffsetT tile_base  = static_cast<OffsetT>(tile_idx) * items_per_tile;

    if (tile_base > 0 && tile_idx < num_tiles)
    {
      if (ReadLeft)
      {
        result[tile_idx] = first[tile_base - 1];
      }
      else
      {
        result[tile_idx - 1] = first[tile_base];
      }
    }
  }
};


CUB_NAMESPACE_END
