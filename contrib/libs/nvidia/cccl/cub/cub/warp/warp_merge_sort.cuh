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

#include <cub/block/block_merge_sort.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/ptx>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

//! @rst
//! The WarpMergeSort class provides methods for sorting items partitioned across a CUDA warp
//! using a merge sorting method.
//!
//! Overview
//! ++++++++++++++++
//!
//!   WarpMergeSort arranges items into ascending order using a comparison
//!   functor with less-than semantics. Merge sort can handle arbitrary types
//!   and comparison functors.
//!
//! A Simple Example
//! ++++++++++++++++
//!
//! The code snippet below illustrates a sort of 64 integer keys that are
//! partitioned across 16 threads where each thread owns 4 consecutive items.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>  // or equivalently <cub/warp/warp_merge_sort.cuh>
//!
//!    struct CustomLess
//!    {
//!      template <typename DataType>
//!      __device__ bool operator()(const DataType &lhs, const DataType &rhs)
//!      {
//!        return lhs < rhs;
//!      }
//!    };
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        constexpr int warp_threads = 16;
//!        constexpr int block_threads = 256;
//!        constexpr int items_per_thread = 4;
//!        constexpr int warps_per_block = block_threads / warp_threads;
//!        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
//!
//!        // Specialize WarpMergeSort for a virtual warp of 16 threads
//!        // owning 4 integer items each
//!        using WarpMergeSortT =
//!          cub::WarpMergeSort<int, items_per_thread, warp_threads>;
//!
//!        // Allocate shared memory for WarpMergeSort
//!        __shared__ typename WarpMergeSortT::TempStorage temp_storage[warps_per_block];
//!
//!        // Obtain a segment of consecutive items that are blocked across threads
//!        int thread_keys[items_per_thread];
//!        // ...
//!
//!        WarpMergeSortT(temp_storage[warp_id]).Sort(thread_keys, CustomLess());
//!        // ...
//!    }
//!
//! Suppose the set of input ``thread_keys`` across a warp of threads is
//! ``{ [0,64,1,63], [2,62,3,61], [4,60,5,59], ..., [31,34,32,33] }``.
//! The corresponding output ``thread_keys`` in those threads will be
//! ``{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [31,32,33,34] }``.
//! @endrst
//!
//! @tparam KeyT
//!   Key type
//!
//! @tparam ITEMS_PER_THREAD
//!   The number of items per thread
//!
//! @tparam LOGICAL_WARP_THREADS
//!   <b>[optional]</b> The number of threads per "logical" warp (may be less
//!   than the number of hardware warp threads). Default is the warp size of the
//!   targeted CUDA compute-capability (e.g., 32 threads for SM86). Must be a
//!   power of two.
//!
//! @tparam ValueT
//!   <b>[optional]</b> Value type (default: cub::NullType, which indicates a
//!   keys-only sort)
//!
template <typename KeyT, int ITEMS_PER_THREAD, int LOGICAL_WARP_THREADS = detail::warp_threads, typename ValueT = NullType>
class WarpMergeSort
    : public BlockMergeSortStrategy<KeyT,
                                    ValueT,
                                    LOGICAL_WARP_THREADS,
                                    ITEMS_PER_THREAD,
                                    WarpMergeSort<KeyT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, ValueT>>
{
private:
  static constexpr bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == detail::warp_threads;
  static constexpr bool KEYS_ONLY    = ::cuda::std::is_same_v<ValueT, NullType>;
  static constexpr int TILE_SIZE     = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;

  using BlockMergeSortStrategyT =
    BlockMergeSortStrategy<KeyT, ValueT, LOGICAL_WARP_THREADS, ITEMS_PER_THREAD, WarpMergeSort>;

  const unsigned int warp_id;
  const unsigned int member_mask;

public:
  WarpMergeSort() = delete;

  _CCCL_DEVICE _CCCL_FORCEINLINE WarpMergeSort(typename BlockMergeSortStrategyT::TempStorage& temp_storage)
      : BlockMergeSortStrategyT(
          temp_storage,
          IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : (::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS))
      , warp_id(IS_ARCH_WARP ? 0 : (::cuda::ptx::get_sreg_laneid() / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int get_member_mask() const
  {
    return member_mask;
  }

private:
  _CCCL_DEVICE _CCCL_FORCEINLINE void SyncImplementation() const
  {
    __syncwarp(member_mask);
  }

  friend BlockMergeSortStrategyT;
};

CUB_NAMESPACE_END
