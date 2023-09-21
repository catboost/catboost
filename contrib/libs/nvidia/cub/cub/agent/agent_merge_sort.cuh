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


#include "../config.cuh"
#include "../util_type.cuh"
#include "../util_namespace.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_merge_sort.cuh"

#include <thrust/system/cuda/detail/core/util.h>

CUB_NAMESPACE_BEGIN


template <
  int                      _BLOCK_THREADS,
  int                      _ITEMS_PER_THREAD = 1,
  cub::BlockLoadAlgorithm  _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
  cub::CacheLoadModifier   _LOAD_MODIFIER    = cub::LOAD_LDG,
  cub::BlockStoreAlgorithm _STORE_ALGORITHM  = cub::BLOCK_STORE_DIRECT>
struct AgentMergeSortPolicy
{
  static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
};

/// \brief This agent is responsible for the initial in-tile sorting.
template <typename Policy,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyIteratorT,
          typename ValueIteratorT,
          typename OffsetT,
          typename CompareOpT,
          typename KeyT,
          typename ValueT>
struct AgentBlockSort
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  static constexpr bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

  using BlockMergeSortT =
    BlockMergeSort<KeyT, Policy::BLOCK_THREADS, Policy::ITEMS_PER_THREAD, ValueT>;

  using KeysLoadIt  = typename THRUST_NS_QUALIFIER::cuda_cub::core::LoadIterator<Policy, KeyInputIteratorT>::type;
  using ItemsLoadIt = typename THRUST_NS_QUALIFIER::cuda_cub::core::LoadIterator<Policy, ValueInputIteratorT>::type;

  using BlockLoadKeys  = typename cub::BlockLoadType<Policy, KeysLoadIt>::type;
  using BlockLoadItems = typename cub::BlockLoadType<Policy, ItemsLoadIt>::type;

  using BlockStoreKeysIt   = typename cub::BlockStoreType<Policy, KeyIteratorT>::type;
  using BlockStoreItemsIt  = typename cub::BlockStoreType<Policy, ValueIteratorT>::type;
  using BlockStoreKeysRaw  = typename cub::BlockStoreType<Policy, KeyT *>::type;
  using BlockStoreItemsRaw = typename cub::BlockStoreType<Policy, ValueT *>::type;

  union _TempStorage
  {
    typename BlockLoadKeys::TempStorage load_keys;
    typename BlockLoadItems::TempStorage load_items;
    typename BlockStoreKeysIt::TempStorage store_keys_it;
    typename BlockStoreItemsIt::TempStorage store_items_it;
    typename BlockStoreKeysRaw::TempStorage store_keys_raw;
    typename BlockStoreItemsRaw::TempStorage store_items_raw;
    typename BlockMergeSortT::TempStorage block_merge;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage> {};

  static constexpr int BLOCK_THREADS = Policy::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE = Policy::ITEMS_PER_TILE;
  static constexpr int SHARED_MEMORY_SIZE =
    static_cast<int>(sizeof(TempStorage));

  //---------------------------------------------------------------------
  // Per thread data
  //---------------------------------------------------------------------

  bool ping;
  _TempStorage &storage;
  KeysLoadIt keys_in;
  ItemsLoadIt items_in;
  OffsetT keys_count;
  KeyIteratorT keys_out_it;
  ValueIteratorT items_out_it;
  KeyT *keys_out_raw;
  ValueT *items_out_raw;
  CompareOpT compare_op;

  __device__ __forceinline__ AgentBlockSort(bool ping_,
                                            TempStorage &storage_,
                                            KeysLoadIt keys_in_,
                                            ItemsLoadIt items_in_,
                                            OffsetT keys_count_,
                                            KeyIteratorT keys_out_it_,
                                            ValueIteratorT items_out_it_,
                                            KeyT *keys_out_raw_,
                                            ValueT *items_out_raw_,
                                            CompareOpT compare_op_)
      : ping(ping_)
      , storage(storage_.Alias())
      , keys_in(keys_in_)
      , items_in(items_in_)
      , keys_count(keys_count_)
      , keys_out_it(keys_out_it_)
      , items_out_it(items_out_it_)
      , keys_out_raw(keys_out_raw_)
      , items_out_raw(items_out_raw_)
      , compare_op(compare_op_)
  {
  }

  __device__ __forceinline__ void Process()
  {
    auto tile_idx     = static_cast<OffsetT>(blockIdx.x);
    auto num_tiles    = static_cast<OffsetT>(gridDim.x);
    auto tile_base    = tile_idx * ITEMS_PER_TILE;
    int items_in_tile = (cub::min)(keys_count - tile_base, int{ITEMS_PER_TILE});

    if (tile_idx < num_tiles - 1)
    {
      consume_tile<false>(tile_base, ITEMS_PER_TILE);
    }
    else
    {
      consume_tile<true>(tile_base, items_in_tile);
    }
  }

  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void consume_tile(OffsetT tile_base,
                                               int num_remaining)
  {
    ValueT items_local[ITEMS_PER_THREAD];
    if (!KEYS_ONLY)
    {
      if (IS_LAST_TILE)
      {
        BlockLoadItems(storage.load_items)
          .Load(items_in + tile_base,
                items_local,
                num_remaining,
                *(items_in + tile_base));
      }
      else
      {
        BlockLoadItems(storage.load_items).Load(items_in + tile_base, items_local);
      }

      CTA_SYNC();
    }

    KeyT keys_local[ITEMS_PER_THREAD];
    if (IS_LAST_TILE)
    {
      BlockLoadKeys(storage.load_keys)
        .Load(keys_in + tile_base,
              keys_local,
              num_remaining,
              *(keys_in + tile_base));
    }
    else
    {
      BlockLoadKeys(storage.load_keys)
        .Load(keys_in + tile_base, keys_local);
    }

    CTA_SYNC();

    if (IS_LAST_TILE)
    {
      BlockMergeSortT(storage.block_merge)
        .Sort(keys_local, items_local, compare_op, num_remaining, keys_local[0]);
    }
    else
    {
      BlockMergeSortT(storage.block_merge).Sort(keys_local, items_local, compare_op);
    }

    CTA_SYNC();

    if (ping)
    {
      if (IS_LAST_TILE)
      {
        BlockStoreKeysIt(storage.store_keys_it)
          .Store(keys_out_it + tile_base, keys_local, num_remaining);
      }
      else
      {
        BlockStoreKeysIt(storage.store_keys_it)
          .Store(keys_out_it + tile_base, keys_local);
      }

      if (!KEYS_ONLY)
      {
        CTA_SYNC();

        if (IS_LAST_TILE)
        {
          BlockStoreItemsIt(storage.store_items_it)
            .Store(items_out_it + tile_base, items_local, num_remaining);
        }
        else
        {
          BlockStoreItemsIt(storage.store_items_it)
            .Store(items_out_it + tile_base, items_local);
        }
      }
    }
    else
    {
      if (IS_LAST_TILE)
      {
        BlockStoreKeysRaw(storage.store_keys_raw)
          .Store(keys_out_raw + tile_base, keys_local, num_remaining);
      }
      else
      {
        BlockStoreKeysRaw(storage.store_keys_raw)
          .Store(keys_out_raw + tile_base, keys_local);
      }

      if (!KEYS_ONLY)
      {
        CTA_SYNC();

        if (IS_LAST_TILE)
        {
          BlockStoreItemsRaw(storage.store_items_raw)
            .Store(items_out_raw + tile_base, items_local, num_remaining);
        }
        else
        {
          BlockStoreItemsRaw(storage.store_items_raw)
            .Store(items_out_raw + tile_base, items_local);
        }
      }
    }
  }
};

/**
 * \brief This agent is responsible for partitioning a merge path into equal segments
 *
 * There are two sorted arrays to be merged into one array. If the first array
 * is partitioned between parallel workers by slicing it into ranges of equal
 * size, there could be a significant workload imbalance. The imbalance is
 * caused by the fact that the distribution of elements from the second array
 * is unknown beforehand. Instead, the MergePath is partitioned between workers.
 * This approach guarantees an equal amount of work being assigned to each worker.
 *
 * This approach is outlined in the paper:
 * Odeh et al, "Merge Path - Parallel Merging Made Simple"
 * doi:10.1109/IPDPSW.2012.202
 */
template <
  typename KeyIteratorT,
  typename OffsetT,
  typename CompareOpT,
  typename KeyT>
struct AgentPartition
{
  bool ping;
  KeyIteratorT keys_ping;
  KeyT *keys_pong;
  OffsetT keys_count;
  OffsetT partition_idx;
  OffsetT *merge_partitions;
  CompareOpT compare_op;
  OffsetT target_merged_tiles_number;
  int items_per_tile;

  __device__ __forceinline__ AgentPartition(bool ping,
                                            KeyIteratorT keys_ping,
                                            KeyT *keys_pong,
                                            OffsetT keys_count,
                                            OffsetT partition_idx,
                                            OffsetT *merge_partitions,
                                            CompareOpT compare_op,
                                            OffsetT target_merged_tiles_number,
                                            int items_per_tile)
      : ping(ping)
      , keys_ping(keys_ping)
      , keys_pong(keys_pong)
      , keys_count(keys_count)
      , partition_idx(partition_idx)
      , merge_partitions(merge_partitions)
      , compare_op(compare_op)
      , target_merged_tiles_number(target_merged_tiles_number)
      , items_per_tile(items_per_tile)
  {}

  __device__ __forceinline__ void Process()
  {
    OffsetT merged_tiles_number = target_merged_tiles_number / 2;

    // target_merged_tiles_number is a power of two.
    OffsetT mask = target_merged_tiles_number - 1;

    // The first tile number in the tiles group being merged, equal to:
    // target_merged_tiles_number * (partition_idx / target_merged_tiles_number)
    OffsetT list  = ~mask & partition_idx;
    OffsetT start = items_per_tile * list;
    OffsetT size  = items_per_tile * merged_tiles_number;

    // Tile number within the tile group being merged, equal to:
    // partition_idx / target_merged_tiles_number
    OffsetT local_tile_idx = mask & partition_idx;

    OffsetT keys1_beg = (cub::min)(keys_count, start);
    OffsetT keys1_end = (cub::min)(keys_count, start + size);
    OffsetT keys2_beg = keys1_end;
    OffsetT keys2_end = (cub::min)(keys_count, keys2_beg + size);

    OffsetT partition_at = (cub::min)(keys2_end - keys1_beg,
                                      items_per_tile * local_tile_idx);

    OffsetT partition_diag = ping ? MergePath<KeyT>(keys_ping + keys1_beg,
                                                    keys_ping + keys2_beg,
                                                    keys1_end - keys1_beg,
                                                    keys2_end - keys2_beg,
                                                    partition_at,
                                                    compare_op)
                                  : MergePath<KeyT>(keys_pong + keys1_beg,
                                                    keys_pong + keys2_beg,
                                                    keys1_end - keys1_beg,
                                                    keys2_end - keys2_beg,
                                                    partition_at,
                                                    compare_op);

    merge_partitions[partition_idx] = keys1_beg + partition_diag;
  }
};

/// \brief The agent is responsible for merging N consecutive sorted arrays into N/2 sorted arrays.
template <
  typename Policy,
  typename KeyIteratorT,
  typename ValueIteratorT,
  typename OffsetT,
  typename CompareOpT,
  typename KeyT,
  typename ValueT>
struct AgentMerge
{

  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------
  using KeysLoadPingIt  = typename THRUST_NS_QUALIFIER::cuda_cub::core::LoadIterator<Policy, KeyIteratorT>::type;
  using ItemsLoadPingIt = typename THRUST_NS_QUALIFIER::cuda_cub::core::LoadIterator<Policy, ValueIteratorT>::type;
  using KeysLoadPongIt  = typename THRUST_NS_QUALIFIER::cuda_cub::core::LoadIterator<Policy, KeyT *>::type;
  using ItemsLoadPongIt = typename THRUST_NS_QUALIFIER::cuda_cub::core::LoadIterator<Policy, ValueT *>::type;

  using KeysOutputPongIt  = KeyIteratorT;
  using ItemsOutputPongIt = ValueIteratorT;
  using KeysOutputPingIt  = KeyT*;
  using ItemsOutputPingIt = ValueT*;

  using BlockStoreKeysPong  = typename BlockStoreType<Policy, KeysOutputPongIt>::type;
  using BlockStoreItemsPong = typename BlockStoreType<Policy, ItemsOutputPongIt>::type;
  using BlockStoreKeysPing  = typename BlockStoreType<Policy, KeysOutputPingIt>::type;
  using BlockStoreItemsPing = typename BlockStoreType<Policy, ItemsOutputPingIt>::type;

  /// Parameterized BlockReduce primitive

  union _TempStorage
  {
    typename BlockStoreKeysPing::TempStorage  store_keys_ping;
    typename BlockStoreItemsPing::TempStorage store_items_ping;
    typename BlockStoreKeysPong::TempStorage  store_keys_pong;
    typename BlockStoreItemsPong::TempStorage store_items_pong;

    KeyT keys_shared[Policy::ITEMS_PER_TILE + 1];
    ValueT items_shared[Policy::ITEMS_PER_TILE + 1];
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage> {};

  static constexpr bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;
  static constexpr int BLOCK_THREADS = Policy::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE = Policy::ITEMS_PER_TILE;
  static constexpr int SHARED_MEMORY_SIZE =
    static_cast<int>(sizeof(TempStorage));

  //---------------------------------------------------------------------
  // Per thread data
  //---------------------------------------------------------------------

  bool            ping;
  _TempStorage&   storage;

  KeysLoadPingIt  keys_in_ping;
  ItemsLoadPingIt items_in_ping;
  KeysLoadPongIt  keys_in_pong;
  ItemsLoadPongIt items_in_pong;

  OffsetT keys_count;

  KeysOutputPongIt  keys_out_pong;
  ItemsOutputPongIt items_out_pong;
  KeysOutputPingIt  keys_out_ping;
  ItemsOutputPingIt items_out_ping;

  CompareOpT compare_op;
  OffsetT *merge_partitions;
  OffsetT target_merged_tiles_number;

  //---------------------------------------------------------------------
  // Utility functions
  //---------------------------------------------------------------------

  /**
   * \brief Concatenates up to ITEMS_PER_THREAD elements from input{1,2} into output array
   *
   * Reads data in a coalesced fashion [BLOCK_THREADS * item + tid] and
   * stores the result in output[item].
   */
  template <bool IS_FULL_TILE, class T, class It1, class It2>
  __device__ __forceinline__ void
  gmem_to_reg(T (&output)[ITEMS_PER_THREAD],
              It1 input1,
              It2 input2,
              int count1,
              int count2)
  {
    if (IS_FULL_TILE)
    {
#pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item)
      {
        int idx = BLOCK_THREADS * item + threadIdx.x;
        output[item] = (idx < count1) ? input1[idx] : input2[idx - count1];
      }
    }
    else
    {
#pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item)
      {
        int idx = BLOCK_THREADS * item + threadIdx.x;
        if (idx < count1 + count2)
        {
          output[item] = (idx < count1) ? input1[idx] : input2[idx - count1];
        }
      }
    }
  }

  /// \brief Stores data in a coalesced fashion in[item] -> out[BLOCK_THREADS * item + tid]
  template <class T, class It>
  __device__ __forceinline__ void
  reg_to_shared(It output,
                T (&input)[ITEMS_PER_THREAD])
  {
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
      int idx = BLOCK_THREADS * item + threadIdx.x;
      output[idx] = input[item];
    }
  }

  template <bool IS_FULL_TILE>
  __device__ __forceinline__ void
  consume_tile(int tid, OffsetT tile_idx, OffsetT tile_base, int count)
  {
    OffsetT partition_beg = merge_partitions[tile_idx + 0];
    OffsetT partition_end = merge_partitions[tile_idx + 1];

    // target_merged_tiles_number is a power of two.
    OffsetT merged_tiles_number = target_merged_tiles_number / 2;

    OffsetT mask  = target_merged_tiles_number - 1;

    // The first tile number in the tiles group being merged, equal to:
    // target_merged_tiles_number * (tile_idx / target_merged_tiles_number)
    OffsetT list  = ~mask & tile_idx;
    OffsetT start = ITEMS_PER_TILE * list;
    OffsetT size  = ITEMS_PER_TILE * merged_tiles_number;

    OffsetT diag = ITEMS_PER_TILE * tile_idx - start;

    OffsetT keys1_beg = partition_beg;
    OffsetT keys1_end = partition_end;
    OffsetT keys2_beg = (cub::min)(keys_count, 2 * start + size + diag - partition_beg);
    OffsetT keys2_end = (cub::min)(keys_count, 2 * start + size + diag + ITEMS_PER_TILE - partition_end);

    // Check if it's the last tile in the tile group being merged
    if (mask == (mask & tile_idx))
    {
      keys1_end = (cub::min)(keys_count, start + size);
      keys2_end = (cub::min)(keys_count, start + size * 2);
    }

    // number of keys per tile
    //
    int num_keys1 = static_cast<int>(keys1_end - keys1_beg);
    int num_keys2 = static_cast<int>(keys2_end - keys2_beg);

    // load keys1 & keys2
    KeyT keys_local[ITEMS_PER_THREAD];
    if (ping)
    {
      gmem_to_reg<IS_FULL_TILE>(keys_local,
                                keys_in_ping + keys1_beg,
                                keys_in_ping + keys2_beg,
                                num_keys1,
                                num_keys2);
    }
    else
    {
      gmem_to_reg<IS_FULL_TILE>(keys_local,
                                keys_in_pong + keys1_beg,
                                keys_in_pong + keys2_beg,
                                num_keys1,
                                num_keys2);
    }
    reg_to_shared(&storage.keys_shared[0], keys_local);

    // preload items into registers already
    //
    ValueT items_local[ITEMS_PER_THREAD];
    if (!KEYS_ONLY)
    {
      if (ping)
      {
        gmem_to_reg<IS_FULL_TILE>(items_local,
                                  items_in_ping + keys1_beg,
                                  items_in_ping + keys2_beg,
                                  num_keys1,
                                  num_keys2);
      }
      else
      {
        gmem_to_reg<IS_FULL_TILE>(items_local,
                                  items_in_pong + keys1_beg,
                                  items_in_pong + keys2_beg,
                                  num_keys1,
                                  num_keys2);
      }
    }

    CTA_SYNC();

    // use binary search in shared memory
    // to find merge path for each of thread
    // we can use int type here, because the number of
    // items in shared memory is limited
    //
    int diag0_local = (cub::min)(num_keys1 + num_keys2, ITEMS_PER_THREAD * tid);

    int keys1_beg_local = MergePath<KeyT>(&storage.keys_shared[0],
                                          &storage.keys_shared[num_keys1],
                                          num_keys1,
                                          num_keys2,
                                          diag0_local,
                                          compare_op);
    int keys1_end_local = num_keys1;
    int keys2_beg_local = diag0_local - keys1_beg_local;
    int keys2_end_local = num_keys2;

    int num_keys1_local = keys1_end_local - keys1_beg_local;
    int num_keys2_local = keys2_end_local - keys2_beg_local;

    // perform serial merge
    //
    int indices[ITEMS_PER_THREAD];

    SerialMerge(&storage.keys_shared[0],
                keys1_beg_local,
                keys2_beg_local + num_keys1,
                num_keys1_local,
                num_keys2_local,
                keys_local,
                indices,
                compare_op);

    CTA_SYNC();

    // write keys
    //
    if (ping)
    {
      if (IS_FULL_TILE)
      {
        BlockStoreKeysPing(storage.store_keys_ping)
          .Store(keys_out_ping + tile_base, keys_local);
      }
      else
      {
        BlockStoreKeysPing(storage.store_keys_ping)
          .Store(keys_out_ping + tile_base, keys_local, num_keys1 + num_keys2);
      }
    }
    else
    {
      if (IS_FULL_TILE)
      {
        BlockStoreKeysPong(storage.store_keys_pong)
          .Store(keys_out_pong + tile_base, keys_local);
      }
      else
      {
        BlockStoreKeysPong(storage.store_keys_pong)
          .Store(keys_out_pong + tile_base, keys_local, num_keys1 + num_keys2);
      }
    }

    // if items are provided, merge them
    if (!KEYS_ONLY)
    {
      CTA_SYNC();

      reg_to_shared(&storage.items_shared[0], items_local);

      CTA_SYNC();

      // gather items from shared mem
      //
#pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item)
      {
        items_local[item] = storage.items_shared[indices[item]];
      }

      CTA_SYNC();

      // write from reg to gmem
      //
      if (ping)
      {
        if (IS_FULL_TILE)
        {
          BlockStoreItemsPing(storage.store_items_ping)
            .Store(items_out_ping + tile_base, items_local);
        }
        else
        {
          BlockStoreItemsPing(storage.store_items_ping)
            .Store(items_out_ping + tile_base, items_local, count);
        }
      }
      else
      {
        if (IS_FULL_TILE)
        {
          BlockStoreItemsPong(storage.store_items_pong)
            .Store(items_out_pong + tile_base, items_local);
        }
        else
        {
          BlockStoreItemsPong(storage.store_items_pong)
            .Store(items_out_pong + tile_base, items_local, count);
        }
      }
    }
  }

  __device__ __forceinline__ AgentMerge(bool ping_,
                                        TempStorage &storage_,
                                        KeysLoadPingIt keys_in_ping_,
                                        ItemsLoadPingIt items_in_ping_,
                                        KeysLoadPongIt keys_in_pong_,
                                        ItemsLoadPongIt items_in_pong_,
                                        OffsetT keys_count_,
                                        KeysOutputPingIt keys_out_ping_,
                                        ItemsOutputPingIt items_out_ping_,
                                        KeysOutputPongIt keys_out_pong_,
                                        ItemsOutputPongIt items_out_pong_,
                                        CompareOpT compare_op_,
                                        OffsetT *merge_partitions_,
                                        OffsetT target_merged_tiles_number_)
      : ping(ping_)
      , storage(storage_.Alias())
      , keys_in_ping(keys_in_ping_)
      , items_in_ping(items_in_ping_)
      , keys_in_pong(keys_in_pong_)
      , items_in_pong(items_in_pong_)
      , keys_count(keys_count_)
      , keys_out_pong(keys_out_pong_)
      , items_out_pong(items_out_pong_)
      , keys_out_ping(keys_out_ping_)
      , items_out_ping(items_out_ping_)
      , compare_op(compare_op_)
      , merge_partitions(merge_partitions_)
      , target_merged_tiles_number(target_merged_tiles_number_)
  {}

  __device__ __forceinline__ void Process()
  {
    int tile_idx      = static_cast<int>(blockIdx.x);
    int num_tiles     = static_cast<int>(gridDim.x);
    OffsetT tile_base = OffsetT(tile_idx) * ITEMS_PER_TILE;
    int tid           = static_cast<int>(threadIdx.x);
    int items_in_tile = static_cast<int>(
      (cub::min)(static_cast<OffsetT>(ITEMS_PER_TILE), keys_count - tile_base));

    if (tile_idx < num_tiles - 1)
    {
      consume_tile<true>(tid, tile_idx, tile_base, ITEMS_PER_TILE);
    }
    else
    {
      consume_tile<false>(tid, tile_idx, tile_base, items_in_tile);
    }
  }
};


CUB_NAMESPACE_END
