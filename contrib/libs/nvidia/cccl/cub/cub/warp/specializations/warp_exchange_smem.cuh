/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * @file
 * The cub::WarpExchangeSmem class provides [<em>collective</em>](index.html#sec0)
 * methods for rearranging data partitioned across a CUDA warp.
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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/ptx>

CUB_NAMESPACE_BEGIN

namespace detail
{

template <typename InputT, int ITEMS_PER_THREAD, int LOGICAL_WARP_THREADS = warp_threads>
class WarpExchangeSmem
{
  static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE, "LOGICAL_WARP_THREADS must be a power of two");

  static constexpr int ITEMS_PER_TILE = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS + 1;

  static constexpr bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == warp_threads;

  static constexpr int LOG_SMEM_BANKS = log2_smem_banks;

  // Insert padding if the number of items per thread is a power of two
  // and > 4 (otherwise we can typically use 128b loads)
  static constexpr bool INSERT_PADDING = (ITEMS_PER_THREAD > 4) && (PowerOfTwo<ITEMS_PER_THREAD>::VALUE);

  static constexpr int PADDING_ITEMS = INSERT_PADDING ? (ITEMS_PER_TILE >> LOG_SMEM_BANKS) : 0;

  union _TempStorage
  {
    InputT items_shared[ITEMS_PER_TILE + PADDING_ITEMS];
  }; // union TempStorage

  /// Shared storage reference
  _TempStorage& temp_storage;

  const unsigned int lane_id;
  const unsigned int warp_id;
  const unsigned int member_mask;

public:
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  WarpExchangeSmem() = delete;

  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpExchangeSmem(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , lane_id(IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : (::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS))
      , warp_id(IS_ARCH_WARP ? 0 : (::cuda::ptx::get_sreg_laneid() / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {}

  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  BlockedToStriped(const InputT (&input_items)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
  {
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx                  = ITEMS_PER_THREAD * lane_id + item;
      temp_storage.items_shared[idx] = input_items[item];
    }
    __syncwarp(member_mask);

    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx      = LOGICAL_WARP_THREADS * item + lane_id;
      output_items[item] = temp_storage.items_shared[idx];
    }
  }

  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  StripedToBlocked(const InputT (&input_items)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
  {
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx                  = LOGICAL_WARP_THREADS * item + lane_id;
      temp_storage.items_shared[idx] = input_items[item];
    }
    __syncwarp(member_mask);

    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx      = ITEMS_PER_THREAD * lane_id + item;
      output_items[item] = temp_storage.items_shared[idx];
    }
  }

  template <typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScatterToStriped(InputT (&items)[ITEMS_PER_THREAD], OffsetT (&ranks)[ITEMS_PER_THREAD])
  {
    ScatterToStriped(items, items, ranks);
  }

  template <typename OutputT, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(
    const InputT (&input_items)[ITEMS_PER_THREAD],
    OutputT (&output_items)[ITEMS_PER_THREAD],
    OffsetT (&ranks)[ITEMS_PER_THREAD])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
      if (INSERT_PADDING)
      {
        ranks[ITEM] = (ranks[ITEM] >> LOG_SMEM_BANKS) + ranks[ITEM];
      }

      temp_storage.items_shared[ranks[ITEM]] = input_items[ITEM];
    }

    __syncwarp(member_mask);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
      int item_offset = (ITEM * LOGICAL_WARP_THREADS) + lane_id;

      if (INSERT_PADDING)
      {
        item_offset = (item_offset >> LOG_SMEM_BANKS) + item_offset;
      }

      output_items[ITEM] = temp_storage.items_shared[item_offset];
    }
  }
};

} // namespace detail

CUB_NAMESPACE_END
