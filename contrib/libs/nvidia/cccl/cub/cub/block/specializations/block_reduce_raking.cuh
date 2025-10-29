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
 * cub::BlockReduceRaking provides raking-based methods of parallel reduction across a CUDA thread
 * block.  Supports non-commutative reduction operators.
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

#include <cub/block/block_raking_layout.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief BlockReduceRaking provides raking-based methods of parallel reduction across a CUDA thread
 *        block. Supports non-commutative reduction operators.
 *
 * Supports non-commutative binary reduction operators.  Unlike commutative
 * reduction operators (e.g., addition), the application of a non-commutative
 * reduction operator (e.g, string concatenation) across a sequence of inputs must
 * honor the relative ordering of items and partial reductions when applying the
 * reduction operator.
 *
 * Compared to the implementation of BlockReduceRakingCommutativeOnly (which
 * does not support non-commutative operators), this implementation requires a
 * few extra rounds of inter-thread communication.
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
struct BlockReduceRaking
{
  /// The thread block size in threads
  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

  /// Layout type for padded thread block raking grid
  using BlockRakingLayout = BlockRakingLayout<T, BLOCK_THREADS>;

  ///  WarpReduce utility type
  using WarpReduce = typename WarpReduce<T, BlockRakingLayout::RAKING_THREADS>::InternalWarpReduce;

  /// Constants
  /// Number of raking threads
  static constexpr int RAKING_THREADS = BlockRakingLayout::RAKING_THREADS;

  /// Number of raking elements per warp synchronous raking thread
  static constexpr int SEGMENT_LENGTH = BlockRakingLayout::SEGMENT_LENGTH;

  /// Cooperative work can be entirely warp synchronous
  static constexpr bool WARP_SYNCHRONOUS = (RAKING_THREADS == BLOCK_THREADS);

  /// Whether or not warp-synchronous reduction should be unguarded (i.e., the warp-reduction elements is a power of
  /// two
  static constexpr int WARP_SYNCHRONOUS_UNGUARDED = PowerOfTwo<RAKING_THREADS>::VALUE;

  /// Whether or not accesses into smem are unguarded
  static constexpr bool RAKING_UNGUARDED = BlockRakingLayout::UNGUARDED;

  /// Shared memory storage layout type
  union _TempStorage
  {
    /// Storage for warp-synchronous reduction
    typename WarpReduce::TempStorage warp_storage;

    /// Padded thread block raking grid
    typename BlockRakingLayout::TempStorage raking_grid;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // Thread fields
  _TempStorage& temp_storage;
  unsigned int linear_tid;

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockReduceRaking(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] partial
   *   <b>[<em>lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <bool IS_FULL_TILE, typename ReductionOp, int ITERATION>
  _CCCL_DEVICE _CCCL_FORCEINLINE T RakingReduction(
    ReductionOp reduction_op, T* raking_segment, T partial, int num_valid, constant_t<ITERATION> /*iteration*/)
  {
    // Update partial if addend is in range
    if ((IS_FULL_TILE && RAKING_UNGUARDED) || ((linear_tid * SEGMENT_LENGTH) + ITERATION < num_valid))
    {
      T addend = raking_segment[ITERATION];
      partial  = reduction_op(partial, addend);
    }
    return RakingReduction<IS_FULL_TILE>(reduction_op, raking_segment, partial, num_valid, constant_t<ITERATION + 1>());
  }

  /**
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] partial
   *   <b>[<em>lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <bool IS_FULL_TILE, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T RakingReduction(
    ReductionOp /*reduction_op*/,
    T* /*raking_segment*/,
    T partial,
    int /*num_valid*/,
    constant_t<SEGMENT_LENGTH> /*iteration*/)
  {
    return partial;
  }

  /**
   * @brief Computes a thread block-wide reduction using the specified reduction operator. The
   *        first num_valid threads each contribute one reduction partial. The return value is
   *        only valid for thread<sub>0</sub>.
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
  template <bool IS_FULL_TILE, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T partial, int num_valid, ReductionOp reduction_op)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp synchronous reduction (unguarded if active threads is a power-of-two)
      partial = WarpReduce(temp_storage.warp_storage).template Reduce<IS_FULL_TILE>(partial, num_valid, reduction_op);
    }
    else
    {
      // Place partial into shared memory grid.
      *BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid) = partial;

      __syncthreads();

      // Reduce parallelism to one warp
      if (linear_tid < RAKING_THREADS)
      {
        // Raking reduction in grid
        T* raking_segment = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);
        partial           = raking_segment[0];

        partial = RakingReduction<IS_FULL_TILE>(reduction_op, raking_segment, partial, num_valid, constant_v<1>);

        int valid_raking_threads = (IS_FULL_TILE) ? RAKING_THREADS : (num_valid + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH;

        // sync before re-using shmem (warp_storage/raking_grid are aliased)
        static_assert(RAKING_THREADS <= warp_threads, "RAKING_THREADS must be <= warp size.");
        unsigned int mask = static_cast<unsigned int>((1ull << RAKING_THREADS) - 1);
        __syncwarp(mask);

        partial = WarpReduce(temp_storage.warp_storage)
                    .template Reduce<(IS_FULL_TILE && RAKING_UNGUARDED)>(partial, valid_raking_threads, reduction_op);
      }
    }

    return partial;
  }

  /**
   * @brief Computes a thread block-wide reduction using addition (+) as the reduction operator.
   *        The first num_valid threads each contribute one reduction partial. The return value is
   *        only valid for thread<sub>0</sub>.
   *
   * @param[in] partial
   *   Calling thread's input partial reductions
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <bool IS_FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T partial, int num_valid)
  {
    ::cuda::std::plus<> reduction_op;

    return Reduce<IS_FULL_TILE>(partial, num_valid, reduction_op);
  }
};
} // namespace detail

CUB_NAMESPACE_END
