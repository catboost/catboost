/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::BlockReduceRakingCommutativeOnly provides raking-based methods of parallel reduction across
 * a CUDA thread block.  Does not support non-commutative reduction operators.
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

#include <cub/block/specializations/block_reduce_raking.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/std/span>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief BlockReduceRakingCommutativeOnly provides raking-based methods of parallel reduction
 *        across a CUDA thread block. Does not support non-commutative reduction operators. Does not
 *        support block sizes that are not a multiple of the warp size.
 *
 * @tparam T
 *   Data type being reduced
 *
 * @tparam BLOCK_DIM_X
 *   The thread block length in threads along the X dimension
 *
 * @tparam BLOCK_DIM_Y
 *   The thread block length in threads along the Y dimension
 *
 * @tparam BLOCK_DIM_Z
 *   The thread block length in threads along the Z dimension
 */
template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
struct BlockReduceRakingCommutativeOnly
{
  /// Constants
  enum
  {
    /// The thread block size in threads
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
  };

  // The fall-back implementation to use when BLOCK_THREADS is not a multiple of the warp size or not all threads have
  // valid values
  using FallBack = detail::BlockReduceRaking<T, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>;

  /// Constants
  enum
  {
    /// Number of warp threads
    WARP_THREADS = warp_threads,

    /// Whether or not to use fall-back
    USE_FALLBACK = ((BLOCK_THREADS % WARP_THREADS != 0) || (BLOCK_THREADS <= WARP_THREADS)),

    /// Number of raking threads
    RAKING_THREADS = WARP_THREADS,

    /// Number of threads actually sharing items with the raking threads
    SHARING_THREADS = _CUDA_VSTD::max(1, BLOCK_THREADS - RAKING_THREADS),

    /// Number of raking elements per warp synchronous raking thread
    SEGMENT_LENGTH = SHARING_THREADS / WARP_THREADS,
  };

  ///  WarpReduce utility type
  using WarpReduce = WarpReduce<T, RAKING_THREADS>;

  /// Layout type for padded thread block raking grid
  using BlockRakingLayout = BlockRakingLayout<T, SHARING_THREADS>;

  /// Shared memory storage layout type
  union _TempStorage
  {
    struct DefaultStorage
    {
      /// Storage for warp-synchronous reduction
      typename WarpReduce::TempStorage warp_storage;

      /// Padded thread block raking grid
      typename BlockRakingLayout::TempStorage raking_grid;
    } default_storage;

    /// Fall-back storage for non-commutative block reduction
    typename FallBack::TempStorage fallback_storage;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // Thread fields
  _TempStorage& temp_storage;
  unsigned int linear_tid;

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockReduceRakingCommutativeOnly(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @brief Computes a thread block-wide reduction using addition (+) as the reduction operator.
   *        The first num_valid threads each contribute one reduction partial.
   *        The return value is only valid for thread<sub>0</sub>.
   *
   * @param[in] partial
   *   Calling thread's input partial reductions
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <bool FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T partial, int num_valid)
  {
    if (USE_FALLBACK || !FULL_TILE)
    {
      return FallBack(temp_storage.fallback_storage).template Sum<FULL_TILE>(partial, num_valid);
    }
    else
    {
      // Place partial into shared memory grid
      if (linear_tid >= RAKING_THREADS)
      {
        *BlockRakingLayout::PlacementPtr(temp_storage.default_storage.raking_grid, linear_tid - RAKING_THREADS) =
          partial;
      }

      __syncthreads();

      // Reduce parallelism to one warp
      if (linear_tid < RAKING_THREADS)
      {
        // Raking reduction in grid
        T* raking_segment = BlockRakingLayout::RakingPtr(temp_storage.default_storage.raking_grid, linear_tid);
        auto span         = ::cuda::std::span<T, SEGMENT_LENGTH>(raking_segment, SEGMENT_LENGTH);
        partial           = cub::ThreadReduce(span, ::cuda::std::plus<>{}, partial);

        // Warp reduction
        partial = WarpReduce(temp_storage.default_storage.warp_storage).Sum(partial);
      }
    }

    return partial;
  }

  /**
   * @brief Computes a thread block-wide reduction using the specified reduction operator.
   *        The first num_valid threads each contribute one reduction partial.
   *        The return value is only valid for thread<sub>0</sub>.
   *
   * @param[in] partial
   *   Calling thread's input partial reductions
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   */
  template <bool FULL_TILE, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T partial, int num_valid, ReductionOp reduction_op)
  {
    if (USE_FALLBACK || !FULL_TILE)
    {
      return FallBack(temp_storage.fallback_storage).template Reduce<FULL_TILE>(partial, num_valid, reduction_op);
    }
    else
    {
      // Place partial into shared memory grid
      if (linear_tid >= RAKING_THREADS)
      {
        *BlockRakingLayout::PlacementPtr(temp_storage.default_storage.raking_grid, linear_tid - RAKING_THREADS) =
          partial;
      }

      __syncthreads();

      // Reduce parallelism to one warp
      if (linear_tid < RAKING_THREADS)
      {
        // Raking reduction in grid
        T* raking_segment = BlockRakingLayout::RakingPtr(temp_storage.default_storage.raking_grid, linear_tid);
        auto span         = ::cuda::std::span<T, SEGMENT_LENGTH>(raking_segment, SEGMENT_LENGTH);
        partial           = cub::ThreadReduce(span, reduction_op, partial);

        // Warp reduction
        partial = WarpReduce(temp_storage.default_storage.warp_storage).Reduce(partial, reduction_op);
      }
    }

    return partial;
  }
};
} // namespace detail

CUB_NAMESPACE_END
