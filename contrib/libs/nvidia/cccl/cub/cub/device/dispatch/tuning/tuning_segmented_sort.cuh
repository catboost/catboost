/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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

#include <cub/agent/agent_segmented_radix_sort.cuh>
#include <cub/agent/agent_sub_warp_merge_sort.cuh>
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace segmented_sort
{
template <typename KeyT, typename ValueT>
struct policy_hub
{
  using DominantT                = ::cuda::std::_If<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;
  static constexpr int KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 300;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      16,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(7);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(7);
    using SmallAndMediumSegmentedSortPolicyT     = AgentSmallAndMediumSegmentedSortPolicy<
          BLOCK_THREADS,
          // Small policy
          AgentSubWarpMergeSortPolicy<4 /* Threads per segment */, ITEMS_PER_SMALL_THREAD, WARP_LOAD_DIRECT, LOAD_DEFAULT>,
          // Medium policy
          AgentSubWarpMergeSortPolicy<32 /* Threads per segment */, ITEMS_PER_MEDIUM_THREAD, WARP_LOAD_DIRECT, LOAD_DEFAULT>>;
  };

  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      19,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(9);
    using SmallAndMediumSegmentedSortPolicyT     = AgentSmallAndMediumSegmentedSortPolicy<
          BLOCK_THREADS,
          // Small policy
          AgentSubWarpMergeSortPolicy<4 /* Threads per segment */, ITEMS_PER_SMALL_THREAD, WARP_LOAD_DIRECT, LOAD_DEFAULT>,
          // Medium policy
          AgentSubWarpMergeSortPolicy<32 /* Threads per segment */, ITEMS_PER_MEDIUM_THREAD, WARP_LOAD_DIRECT, LOAD_DEFAULT>>;
  };

  struct Policy610 : ChainedPolicy<610, Policy610, Policy600>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(9);
    using SmallAndMediumSegmentedSortPolicyT     = AgentSmallAndMediumSegmentedSortPolicy<
          BLOCK_THREADS,
          // Small policy
          AgentSubWarpMergeSortPolicy<4 /* Threads per segment */, ITEMS_PER_SMALL_THREAD, WARP_LOAD_DIRECT, LOAD_DEFAULT>,
          // Medium policy
          AgentSubWarpMergeSortPolicy<32 /* Threads per segment */, ITEMS_PER_MEDIUM_THREAD, WARP_LOAD_DIRECT, LOAD_DEFAULT>>;
  };

  struct Policy620 : ChainedPolicy<620, Policy620, Policy610>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 5 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      16,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(9);
    using SmallAndMediumSegmentedSortPolicyT     = AgentSmallAndMediumSegmentedSortPolicy<
          BLOCK_THREADS,
          // Small policy
          AgentSubWarpMergeSortPolicy<4 /* Threads per segment */, ITEMS_PER_SMALL_THREAD, WARP_LOAD_DIRECT, LOAD_DEFAULT>,
          // Medium policy
          AgentSubWarpMergeSortPolicy<32 /* Threads per segment */, ITEMS_PER_MEDIUM_THREAD, WARP_LOAD_DIRECT, LOAD_DEFAULT>>;
  };

  struct Policy700 : ChainedPolicy<700, Policy700, Policy620>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(7);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(KEYS_ONLY ? 11 : 7);
    using SmallAndMediumSegmentedSortPolicyT     = AgentSmallAndMediumSegmentedSortPolicy<
          BLOCK_THREADS,
          // Small policy
          AgentSubWarpMergeSortPolicy<KEYS_ONLY ? 4 : 8 /* Threads per segment */,
                                      ITEMS_PER_SMALL_THREAD,
                                      WARP_LOAD_DIRECT,
                                      LOAD_DEFAULT>,
          // Medium policy
          AgentSubWarpMergeSortPolicy<32 /* Threads per segment */, ITEMS_PER_MEDIUM_THREAD, WARP_LOAD_DIRECT, LOAD_DEFAULT>>;
  };

  struct Policy800 : ChainedPolicy<800, Policy800, Policy700>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int PARTITIONING_THRESHOLD = 500;
    using LargeSegmentPolicy                    = AgentRadixSortDownsweepPolicy<
                         BLOCK_THREADS,
                         23,
                         DominantT,
                         BLOCK_LOAD_TRANSPOSE,
                         LOAD_DEFAULT,
                         RADIX_RANK_MEMOIZE,
                         BLOCK_SCAN_WARP_SCANS,
      (sizeof(KeyT) > 1) ? 6 : 4>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(KEYS_ONLY ? 7 : 11);
    using SmallAndMediumSegmentedSortPolicyT     = AgentSmallAndMediumSegmentedSortPolicy<
          BLOCK_THREADS,
          // Small policy
          AgentSubWarpMergeSortPolicy<KEYS_ONLY ? 4 : 2 /* Threads per segment */,
                                      ITEMS_PER_SMALL_THREAD,
                                      WARP_LOAD_TRANSPOSE,
                                      LOAD_DEFAULT>,
          // Medium policy
          AgentSubWarpMergeSortPolicy<32 /* Threads per segment */, ITEMS_PER_MEDIUM_THREAD, WARP_LOAD_TRANSPOSE, LOAD_DEFAULT>>;
  };

  struct Policy860 : ChainedPolicy<860, Policy860, Policy800>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int PARTITIONING_THRESHOLD = 500;
    using LargeSegmentPolicy                    = AgentRadixSortDownsweepPolicy<
                         BLOCK_THREADS,
                         23,
                         DominantT,
                         BLOCK_LOAD_TRANSPOSE,
                         LOAD_DEFAULT,
                         RADIX_RANK_MEMOIZE,
                         BLOCK_SCAN_WARP_SCANS,
      (sizeof(KeyT) > 1) ? 6 : 4>;

    static constexpr bool LARGE_ITEMS            = sizeof(DominantT) > 4;
    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(LARGE_ITEMS ? 7 : 9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(LARGE_ITEMS ? 9 : 7);
    using SmallAndMediumSegmentedSortPolicyT     = AgentSmallAndMediumSegmentedSortPolicy<
          BLOCK_THREADS,
          // Small policy
          AgentSubWarpMergeSortPolicy<LARGE_ITEMS ? 8 : 2 /* Threads per segment */,
                                      ITEMS_PER_SMALL_THREAD,
                                      WARP_LOAD_TRANSPOSE,
                                      LOAD_LDG>,
          // Medium policy
          AgentSubWarpMergeSortPolicy<16 /* Threads per segment */, ITEMS_PER_MEDIUM_THREAD, WARP_LOAD_TRANSPOSE, LOAD_LDG>>;
  };

  using MaxPolicy = Policy860;
};
} // namespace segmented_sort
} // namespace detail

CUB_NAMESPACE_END
