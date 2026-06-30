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

#include <cub/block/block_load.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/load_iterator.h>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cccl/cuda_capabilities.h>

CUB_NAMESPACE_BEGIN

template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD                     = 1,
          cub::BlockLoadAlgorithm _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
          cub::CacheLoadModifier _LOAD_MODIFIER     = cub::LOAD_LDG,
          cub::BlockStoreAlgorithm _STORE_ALGORITHM = cub::BLOCK_STORE_DIRECT>
struct AgentMergeSortPolicy
{
  static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
};

namespace detail
{
namespace merge_sort
{

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

  static constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  using BlockMergeSortT = BlockMergeSort<KeyT, Policy::BLOCK_THREADS, Policy::ITEMS_PER_THREAD, ValueT>;

  using KeysLoadIt =
    typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, KeyInputIteratorT>::type;
  using ItemsLoadIt =
    typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, ValueInputIteratorT>::type;

  using BlockLoadKeys  = typename cub::BlockLoadType<Policy, KeysLoadIt>::type;
  using BlockLoadItems = typename cub::BlockLoadType<Policy, ItemsLoadIt>::type;

  using BlockStoreKeysIt   = typename cub::BlockStoreType<Policy, KeyIteratorT>::type;
  using BlockStoreItemsIt  = typename cub::BlockStoreType<Policy, ValueIteratorT>::type;
  using BlockStoreKeysRaw  = typename cub::BlockStoreType<Policy, KeyT*>::type;
  using BlockStoreItemsRaw = typename cub::BlockStoreType<Policy, ValueT*>::type;

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
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  static constexpr int BLOCK_THREADS    = Policy::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE   = Policy::ITEMS_PER_TILE;

  //---------------------------------------------------------------------
  // Per thread data
  //---------------------------------------------------------------------

  bool ping;
  _TempStorage& storage;
  KeysLoadIt keys_in;
  ItemsLoadIt items_in;
  OffsetT keys_count;
  KeyIteratorT keys_out_it;
  ValueIteratorT items_out_it;
  KeyT* keys_out_raw;
  ValueT* items_out_raw;
  CompareOpT compare_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentBlockSort(
    bool ping_,
    TempStorage& storage_,
    KeysLoadIt keys_in_,
    ItemsLoadIt items_in_,
    OffsetT keys_count_,
    KeyIteratorT keys_out_it_,
    ValueIteratorT items_out_it_,
    KeyT* keys_out_raw_,
    ValueT* items_out_raw_,
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
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process()
  {
    const auto tile_idx     = static_cast<OffsetT>(blockIdx.x);
    const auto num_tiles    = static_cast<OffsetT>(gridDim.x);
    const auto tile_base    = tile_idx * ITEMS_PER_TILE;
    const int items_in_tile = (::cuda::std::min) (static_cast<int>(keys_count - tile_base), int{ITEMS_PER_TILE});

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
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_tile(OffsetT tile_base, int num_remaining)
  {
    ValueT items_local[ITEMS_PER_THREAD];

    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    if constexpr (!KEYS_ONLY)
    {
      if constexpr (IS_LAST_TILE)
      {
        BlockLoadItems(storage.load_items)
          .Load(items_in + tile_base, items_local, num_remaining, *(items_in + tile_base));
      }
      else
      {
        BlockLoadItems(storage.load_items).Load(items_in + tile_base, items_local);
      }

      __syncthreads();
    }

    KeyT keys_local[ITEMS_PER_THREAD];
    if constexpr (IS_LAST_TILE)
    {
      BlockLoadKeys(storage.load_keys).Load(keys_in + tile_base, keys_local, num_remaining, *(keys_in + tile_base));
    }
    else
    {
      BlockLoadKeys(storage.load_keys).Load(keys_in + tile_base, keys_local);
    }

    __syncthreads();
    _CCCL_PDL_TRIGGER_NEXT_LAUNCH();

    if constexpr (IS_LAST_TILE)
    {
      BlockMergeSortT(storage.block_merge).Sort(keys_local, items_local, compare_op, num_remaining, keys_local[0]);
    }
    else
    {
      BlockMergeSortT(storage.block_merge).Sort(keys_local, items_local, compare_op);
    }

    __syncthreads();

    if (ping)
    {
      if constexpr (IS_LAST_TILE)
      {
        BlockStoreKeysIt(storage.store_keys_it).Store(keys_out_it + tile_base, keys_local, num_remaining);
      }
      else
      {
        BlockStoreKeysIt(storage.store_keys_it).Store(keys_out_it + tile_base, keys_local);
      }

      if constexpr (!KEYS_ONLY)
      {
        __syncthreads();

        if constexpr (IS_LAST_TILE)
        {
          BlockStoreItemsIt(storage.store_items_it).Store(items_out_it + tile_base, items_local, num_remaining);
        }
        else
        {
          BlockStoreItemsIt(storage.store_items_it).Store(items_out_it + tile_base, items_local);
        }
      }
    }
    else
    {
      if constexpr (IS_LAST_TILE)
      {
        BlockStoreKeysRaw(storage.store_keys_raw).Store(keys_out_raw + tile_base, keys_local, num_remaining);
      }
      else
      {
        BlockStoreKeysRaw(storage.store_keys_raw).Store(keys_out_raw + tile_base, keys_local);
      }

      if constexpr (!KEYS_ONLY)
      {
        __syncthreads();

        if constexpr (IS_LAST_TILE)
        {
          BlockStoreItemsRaw(storage.store_items_raw).Store(items_out_raw + tile_base, items_local, num_remaining);
        }
        else
        {
          BlockStoreItemsRaw(storage.store_items_raw).Store(items_out_raw + tile_base, items_local);
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
template <typename KeyIteratorT, typename OffsetT, typename CompareOpT, typename KeyT>
struct AgentPartition
{
  bool ping;
  KeyIteratorT keys_ping;
  KeyT* keys_pong;
  OffsetT keys_count;
  OffsetT partition_idx;
  OffsetT* merge_partitions;
  CompareOpT compare_op;
  OffsetT target_merged_tiles_number;
  int items_per_tile;
  OffsetT num_partitions;

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process()
  {
    const OffsetT merged_tiles_number = target_merged_tiles_number / 2;

    // target_merged_tiles_number is a power of two.
    const OffsetT mask = target_merged_tiles_number - 1;

    // The first tile number in the tiles group being merged, equal to:
    // target_merged_tiles_number * (partition_idx / target_merged_tiles_number)
    const OffsetT list  = ~mask & partition_idx;
    const OffsetT start = items_per_tile * list;
    const OffsetT size  = items_per_tile * merged_tiles_number;

    // Tile number within the tile group being merged, equal to:
    // partition_idx / target_merged_tiles_number
    const OffsetT local_tile_idx = mask & partition_idx;

    const OffsetT keys1_beg = (::cuda::std::min) (keys_count, start);
    const OffsetT keys1_end = (::cuda::std::min) (keys_count, detail::safe_add_bound_to_max(start, size));
    const OffsetT keys2_beg = keys1_end;
    const OffsetT keys2_end = (::cuda::std::min) (keys_count, detail::safe_add_bound_to_max(keys2_beg, size));

    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    // The last partition (which is one-past-the-last-tile) is only to mark the end of keys1_end for the merge stage
    if (partition_idx + 1 == num_partitions)
    {
      merge_partitions[partition_idx] = keys1_end;
    }
    else
    {
      const OffsetT partition_at = (::cuda::std::min) (keys2_end - keys1_beg, items_per_tile * local_tile_idx);

      OffsetT partition_diag =
        ping
          ? MergePath(keys_ping + keys1_beg,
                      keys_ping + keys2_beg,
                      keys1_end - keys1_beg,
                      keys2_end - keys2_beg,
                      partition_at,
                      compare_op)
          : MergePath(keys_pong + keys1_beg,
                      keys_pong + keys2_beg,
                      keys1_end - keys1_beg,
                      keys2_end - keys2_beg,
                      partition_at,
                      compare_op);

      merge_partitions[partition_idx] = keys1_beg + partition_diag;
    }

    // TODO(bgruber): looking at SASS triggering the next launch here just generates a lot of noise and the PRE-EXIT
    // just ends of right before EXIT anyway. So let's omit it.
    // _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
  }
};

/**
 * \brief Concatenates up to ITEMS_PER_THREAD elements from input{1,2} into output array
 *
 * Reads data in a coalesced fashion [BLOCK_THREADS * item + tid] and
 * stores the result in output[item].
 */
template <int BLOCK_THREADS, bool IS_FULL_TILE, int ITEMS_PER_THREAD, class T, class It1, class It2>
_CCCL_DEVICE _CCCL_FORCEINLINE void
gmem_to_reg(T (&output)[ITEMS_PER_THREAD], It1 input1, It2 input2, int count1, int count2)
{
  if constexpr (IS_FULL_TILE)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
      const int idx = BLOCK_THREADS * item + threadIdx.x;
      // It1 and It2 could have different value types. Convert after load.
      output[item] = (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
    }
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ITEMS_PER_THREAD; ++item)
    {
      const int idx = BLOCK_THREADS * item + threadIdx.x;
      if (idx < count1 + count2)
      {
        output[item] = (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
      }
    }
  }
}

/// \brief Stores data in a coalesced fashion in[item] -> out[BLOCK_THREADS * item + tid]
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, class T, class It>
_CCCL_DEVICE _CCCL_FORCEINLINE void reg_to_shared(It output, T (&input)[ITEMS_PER_THREAD])
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ITEMS_PER_THREAD; ++item)
  {
    const int idx = BLOCK_THREADS * item + threadIdx.x;
    output[idx]   = input[item];
  }
}

/// \brief The agent is responsible for merging N consecutive sorted arrays into N/2 sorted arrays.
template <typename Policy,
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
  using KeysLoadPingIt = typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, KeyIteratorT>::type;
  using ItemsLoadPingIt =
    typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, ValueIteratorT>::type;
  using KeysLoadPongIt  = typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, KeyT*>::type;
  using ItemsLoadPongIt = typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, ValueT*>::type;

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
    typename BlockStoreKeysPing::TempStorage store_keys_ping;
    typename BlockStoreItemsPing::TempStorage store_items_ping;
    typename BlockStoreKeysPong::TempStorage store_keys_pong;
    typename BlockStoreItemsPong::TempStorage store_items_pong;

    KeyT keys_shared[Policy::ITEMS_PER_TILE + 1];
    ValueT items_shared[Policy::ITEMS_PER_TILE + 1];
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  static constexpr bool KEYS_ONLY       = ::cuda::std::is_same_v<ValueT, NullType>;
  static constexpr int BLOCK_THREADS    = Policy::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE   = Policy::ITEMS_PER_TILE;

  //---------------------------------------------------------------------
  // Per thread data
  //---------------------------------------------------------------------

  bool ping;
  _TempStorage& storage;

  KeysLoadPingIt keys_in_ping;
  ItemsLoadPingIt items_in_ping;
  KeysLoadPongIt keys_in_pong;
  ItemsLoadPongIt items_in_pong;

  OffsetT keys_count;

  KeysOutputPongIt keys_out_pong;
  ItemsOutputPongIt items_out_pong;
  KeysOutputPingIt keys_out_ping;
  ItemsOutputPingIt items_out_ping;

  CompareOpT compare_op;
  OffsetT* merge_partitions;
  OffsetT target_merged_tiles_number;

  //---------------------------------------------------------------------
  // Utility functions
  //---------------------------------------------------------------------

  template <bool IS_FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_tile(int tid, OffsetT tile_idx, OffsetT tile_base, int count)
  {
    _CCCL_PDL_GRID_DEPENDENCY_SYNC();

    const OffsetT partition_beg = merge_partitions[tile_idx + 0];
    const OffsetT partition_end = merge_partitions[tile_idx + 1];

    // target_merged_tiles_number is a power of two.
    const OffsetT merged_tiles_number = target_merged_tiles_number / 2;

    const OffsetT mask = target_merged_tiles_number - 1;

    // The first tile number in the tiles group being merged, equal to:
    // target_merged_tiles_number * (tile_idx / target_merged_tiles_number)
    const OffsetT list  = ~mask & tile_idx;
    const OffsetT start = ITEMS_PER_TILE * list;
    const OffsetT size  = ITEMS_PER_TILE * merged_tiles_number;

    const OffsetT diag = ITEMS_PER_TILE * tile_idx - start;

    const OffsetT keys1_beg = partition_beg - start;
    OffsetT keys1_end       = partition_end - start;

    const OffsetT keys_end_dist_from_start = keys_count - start;
    const OffsetT max_keys2                = (keys_end_dist_from_start > size) ? (keys_end_dist_from_start - size) : 0;

    // We have the following invariants:
    // diag >= keys1_beg, because diag is the distance of the total merge path so far (keys1 + keys2)
    // diag+ITEMS_PER_TILE >= keys1_end, because diag+ITEMS_PER_TILE is the distance of the merge path for the next tile
    // and keys1_end is key1's component of that path
    const OffsetT keys2_beg = (::cuda::std::min) (max_keys2, diag - keys1_beg);
    OffsetT keys2_end =
      (::cuda::std::min) (max_keys2,
                          detail::safe_add_bound_to_max(diag, static_cast<OffsetT>(ITEMS_PER_TILE)) - keys1_end);

    // Check if it's the last tile in the tile group being merged
    if (mask == (mask & tile_idx))
    {
      keys1_end = (::cuda::std::min) (keys_count - start, size);
      keys2_end = (::cuda::std::min) (max_keys2, size);
    }

    // number of keys per tile
    const int num_keys1 = static_cast<int>(keys1_end - keys1_beg);
    const int num_keys2 = static_cast<int>(keys2_end - keys2_beg);

    // load keys1 & keys2
    KeyT keys_local[ITEMS_PER_THREAD];
    if (ping)
    {
      gmem_to_reg<BLOCK_THREADS, IS_FULL_TILE>(
        keys_local, keys_in_ping + start + keys1_beg, keys_in_ping + start + size + keys2_beg, num_keys1, num_keys2);
    }
    else
    {
      gmem_to_reg<BLOCK_THREADS, IS_FULL_TILE>(
        keys_local, keys_in_pong + start + keys1_beg, keys_in_pong + start + size + keys2_beg, num_keys1, num_keys2);
    }
    reg_to_shared<BLOCK_THREADS>(&storage.keys_shared[0], keys_local);

    // preload items into registers already
    //
    [[maybe_unused]] ValueT items_local[ITEMS_PER_THREAD];
    if constexpr (!KEYS_ONLY)
    {
      if (ping)
      {
        gmem_to_reg<BLOCK_THREADS, IS_FULL_TILE>(
          items_local,
          items_in_ping + start + keys1_beg,
          items_in_ping + start + size + keys2_beg,
          num_keys1,
          num_keys2);
      }
      else
      {
        gmem_to_reg<BLOCK_THREADS, IS_FULL_TILE>(
          items_local,
          items_in_pong + start + keys1_beg,
          items_in_pong + start + size + keys2_beg,
          num_keys1,
          num_keys2);
      }
    }

    __syncthreads();
    _CCCL_PDL_TRIGGER_NEXT_LAUNCH();

    // use binary search in shared memory
    // to find merge path for each of thread
    // we can use int type here, because the number of
    // items in shared memory is limited
    //
    const int diag0_local = (::cuda::std::min) (num_keys1 + num_keys2, ITEMS_PER_THREAD * tid);

    const int keys1_beg_local = MergePath(
      &storage.keys_shared[0], &storage.keys_shared[num_keys1], num_keys1, num_keys2, diag0_local, compare_op);
    const int keys1_end_local = num_keys1;
    const int keys2_beg_local = diag0_local - keys1_beg_local;
    const int keys2_end_local = num_keys2;

    const int num_keys1_local = keys1_end_local - keys1_beg_local;
    const int num_keys2_local = keys2_end_local - keys2_beg_local;

    // perform serial merge
    //
    int indices[ITEMS_PER_THREAD];

    SerialMerge(
      &storage.keys_shared[0],
      keys1_beg_local,
      keys2_beg_local + num_keys1,
      num_keys1_local,
      num_keys2_local,
      keys_local,
      indices,
      compare_op);

    __syncthreads();

    // write keys
    if (ping)
    {
      if constexpr (IS_FULL_TILE)
      {
        BlockStoreKeysPing(storage.store_keys_ping).Store(keys_out_ping + tile_base, keys_local);
      }
      else
      {
        BlockStoreKeysPing(storage.store_keys_ping).Store(keys_out_ping + tile_base, keys_local, num_keys1 + num_keys2);
      }
    }
    else
    {
      if constexpr (IS_FULL_TILE)
      {
        BlockStoreKeysPong(storage.store_keys_pong).Store(keys_out_pong + tile_base, keys_local);
      }
      else
      {
        BlockStoreKeysPong(storage.store_keys_pong).Store(keys_out_pong + tile_base, keys_local, num_keys1 + num_keys2);
      }
    }

    // if items are provided, merge them
    if constexpr (!KEYS_ONLY)
    {
      __syncthreads();

      reg_to_shared<BLOCK_THREADS>(&storage.items_shared[0], items_local);

      __syncthreads();

      // gather items from shared mem
      //
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ITEMS_PER_THREAD; ++item)
      {
        items_local[item] = storage.items_shared[indices[item]];
      }

      __syncthreads();

      // write from reg to gmem
      //
      if (ping)
      {
        if constexpr (IS_FULL_TILE)
        {
          BlockStoreItemsPing(storage.store_items_ping).Store(items_out_ping + tile_base, items_local);
        }
        else
        {
          BlockStoreItemsPing(storage.store_items_ping).Store(items_out_ping + tile_base, items_local, count);
        }
      }
      else
      {
        if constexpr (IS_FULL_TILE)
        {
          BlockStoreItemsPong(storage.store_items_pong).Store(items_out_pong + tile_base, items_local);
        }
        else
        {
          BlockStoreItemsPong(storage.store_items_pong).Store(items_out_pong + tile_base, items_local, count);
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentMerge(
    bool ping_,
    TempStorage& storage_,
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
    OffsetT* merge_partitions_,
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

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process()
  {
    const int tile_idx      = static_cast<int>(blockIdx.x);
    const int num_tiles     = static_cast<int>(gridDim.x);
    const OffsetT tile_base = OffsetT(tile_idx) * ITEMS_PER_TILE;
    const int tid           = static_cast<int>(threadIdx.x);
    const int items_in_tile =
      static_cast<int>((::cuda::std::min) (static_cast<OffsetT>(ITEMS_PER_TILE), keys_count - tile_base));

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

} // namespace merge_sort
} // namespace detail

CUB_NAMESPACE_END
