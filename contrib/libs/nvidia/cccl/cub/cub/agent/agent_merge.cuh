// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/util.h>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
namespace merge
{
template <int ThreadsPerBlock,
          int ItemsPerThread,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadCacheModifier,
          BlockStoreAlgorithm StoreAlgorithm>
struct agent_policy_t
{
  // do not change data member names, policy_wrapper_t depends on it
  static constexpr int BLOCK_THREADS                   = ThreadsPerBlock;
  static constexpr int ITEMS_PER_THREAD                = ItemsPerThread;
  static constexpr int ITEMS_PER_TILE                  = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM   = LoadAlgorithm;
  static constexpr CacheLoadModifier LOAD_MODIFIER     = LoadCacheModifier;
  static constexpr BlockStoreAlgorithm STORE_ALGORITHM = StoreAlgorithm;
};

// TODO(bgruber): can we unify this one with AgentMerge in agent_merge_sort.cuh?
template <typename Policy,
          typename KeysIt1,
          typename ItemsIt1,
          typename KeysIt2,
          typename ItemsIt2,
          typename KeysOutputIt,
          typename ItemsOutputIt,
          typename Offset,
          typename CompareOp>
struct agent_t
{
  using policy = Policy;

  // key and value type are taken from the first input sequence (consistent with old Thrust behavior)
  using key_type  = it_value_t<KeysIt1>;
  using item_type = it_value_t<ItemsIt1>;

  using keys_load_it1  = typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, KeysIt1>::type;
  using keys_load_it2  = typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, KeysIt2>::type;
  using items_load_it1 = typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, ItemsIt1>::type;
  using items_load_it2 = typename THRUST_NS_QUALIFIER::cuda_cub::core::detail::LoadIterator<Policy, ItemsIt2>::type;

  using block_load_keys1  = typename BlockLoadType<Policy, keys_load_it1>::type;
  using block_load_keys2  = typename BlockLoadType<Policy, keys_load_it2>::type;
  using block_load_items1 = typename BlockLoadType<Policy, items_load_it1>::type;
  using block_load_items2 = typename BlockLoadType<Policy, items_load_it2>::type;

  using block_store_keys  = typename BlockStoreType<Policy, KeysOutputIt, key_type>::type;
  using block_store_items = typename BlockStoreType<Policy, ItemsOutputIt, item_type>::type;

  union temp_storages
  {
    typename block_load_keys1::TempStorage load_keys1;
    typename block_load_keys2::TempStorage load_keys2;
    typename block_load_items1::TempStorage load_items1;
    typename block_load_items2::TempStorage load_items2;
    typename block_store_keys::TempStorage store_keys;
    typename block_store_items::TempStorage store_items;

    key_type keys_shared[Policy::ITEMS_PER_TILE + 1];
    item_type items_shared[Policy::ITEMS_PER_TILE + 1];
  };

  struct TempStorage : Uninitialized<temp_storages>
  {};

  static constexpr int items_per_thread  = Policy::ITEMS_PER_THREAD;
  static constexpr int threads_per_block = Policy::BLOCK_THREADS;
  static constexpr Offset items_per_tile = Policy::ITEMS_PER_TILE;

  // Per thread data
  temp_storages& storage;
  keys_load_it1 keys1_in;
  items_load_it1 items1_in;
  Offset keys1_count;
  keys_load_it2 keys2_in;
  items_load_it2 items2_in;
  Offset keys2_count;
  KeysOutputIt keys_out;
  ItemsOutputIt items_out;
  CompareOp compare_op;
  Offset* merge_partitions;

  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_tile(Offset tile_idx, Offset tile_base, int num_remaining)
  {
    const Offset partition_beg = merge_partitions[tile_idx + 0];
    const Offset partition_end = merge_partitions[tile_idx + 1];

    const Offset diag0 = items_per_tile * tile_idx;
    const Offset diag1 = (::cuda::std::min) (keys1_count + keys2_count, diag0 + items_per_tile);

    // compute bounding box for keys1 & keys2
    const Offset keys1_beg = partition_beg;
    const Offset keys1_end = partition_end;
    const Offset keys2_beg = diag0 - keys1_beg;
    const Offset keys2_end = diag1 - keys1_end;

    // number of keys per tile
    const int num_keys1 = static_cast<int>(keys1_end - keys1_beg);
    const int num_keys2 = static_cast<int>(keys2_end - keys2_beg);

    key_type keys_loc[items_per_thread];
    merge_sort::gmem_to_reg<threads_per_block, IsFullTile>(
      keys_loc, keys1_in + keys1_beg, keys2_in + keys2_beg, num_keys1, num_keys2);
    merge_sort::reg_to_shared<threads_per_block>(&storage.keys_shared[0], keys_loc);
    __syncthreads();

    // use binary search in shared memory to find merge path for each of thread.
    // we can use int type here, because the number of items in shared memory is limited
    const int diag0_loc = (::cuda::std::min) (num_keys1 + num_keys2, static_cast<int>(items_per_thread * threadIdx.x));

    const int keys1_beg_loc =
      MergePath(&storage.keys_shared[0], &storage.keys_shared[num_keys1], num_keys1, num_keys2, diag0_loc, compare_op);
    const int keys1_end_loc = num_keys1;
    const int keys2_beg_loc = diag0_loc - keys1_beg_loc;
    const int keys2_end_loc = num_keys2;

    const int num_keys1_loc = keys1_end_loc - keys1_beg_loc;
    const int num_keys2_loc = keys2_end_loc - keys2_beg_loc;

    // perform serial merge
    int indices[items_per_thread];
    cub::SerialMerge(
      &storage.keys_shared[0],
      keys1_beg_loc,
      keys2_beg_loc + num_keys1,
      num_keys1_loc,
      num_keys2_loc,
      keys_loc,
      indices,
      compare_op);
    __syncthreads();

    // write keys
    if (IsFullTile)
    {
      block_store_keys{storage.store_keys}.Store(keys_out + tile_base, keys_loc);
    }
    else
    {
      block_store_keys{storage.store_keys}.Store(keys_out + tile_base, keys_loc, num_remaining);
    }

    // if items are provided, merge them
    static constexpr bool have_items = !::cuda::std::is_same_v<item_type, NullType>;
    if constexpr (have_items)
    {
      item_type items_loc[items_per_thread];
      merge_sort::gmem_to_reg<threads_per_block, IsFullTile>(
        items_loc, items1_in + keys1_beg, items2_in + keys2_beg, num_keys1, num_keys2);
      __syncthreads(); // block_store_keys above uses shared memory, so make sure all threads are done before we write
                       // to it
      merge_sort::reg_to_shared<threads_per_block>(&storage.items_shared[0], items_loc);
      __syncthreads();

      // gather items from shared mem
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        items_loc[i] = storage.items_shared[indices[i]];
      }
      __syncthreads();

      // write from reg to gmem
      if (IsFullTile)
      {
        block_store_items{storage.store_items}.Store(items_out + tile_base, items_loc);
      }
      else
      {
        block_store_items{storage.store_items}.Store(items_out + tile_base, items_loc, num_remaining);
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
  {
    // XXX with 8.5 changing type to Offset (or long long) results in error!
    // TODO(bgruber): is the above still true?
    const int tile_idx     = static_cast<int>(blockIdx.x);
    const Offset tile_base = tile_idx * items_per_tile;
    // TODO(bgruber): random mixing of int and Offset
    const int items_in_tile =
      static_cast<int>((::cuda::std::min) (static_cast<Offset>(items_per_tile), keys1_count + keys2_count - tile_base));
    if (items_in_tile == items_per_tile)
    {
      consume_tile<true>(tile_idx, tile_base, items_per_tile); // full tile
    }
    else
    {
      consume_tile<false>(tile_idx, tile_base, items_in_tile); // partial tile
    }
  }
};
} // namespace merge
} // namespace detail
CUB_NAMESPACE_END
