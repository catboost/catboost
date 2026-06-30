/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::BlockReduceWarpReductions provides variants of warp-reduction-based parallel reduction
 * across a CUDA thread block. Supports non-commutative reduction operators.
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

#include <cub/detail/uninitialized_copy.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/ptx>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief BlockReduceWarpReductions provides variants of warp-reduction-based parallel reduction
 *        across a CUDA thread block. Supports non-commutative reduction operators.
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
struct BlockReduceWarpReductions
{
  /// Constants
  enum
  {
    /// The thread block size in threads
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

    /// Number of warp threads
    WARP_THREADS = warp_threads,

    /// Number of active warps
    WARPS = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

    /// The logical warp size for warp reductions
    LOGICAL_WARP_SIZE = (BLOCK_THREADS < WARP_THREADS ? BLOCK_THREADS : WARP_THREADS), // MSVC bug with cuda::std::min

    /// Whether or not the logical warp size evenly divides the thread block size
    EVEN_WARP_MULTIPLE = (BLOCK_THREADS % LOGICAL_WARP_SIZE == 0)
  };

  ///  WarpReduce utility type
  using WarpReduceInternal = typename WarpReduce<T, LOGICAL_WARP_SIZE>::InternalWarpReduce;

  /// Shared memory storage layout type
  struct _TempStorage
  {
    /// Buffer for warp-synchronous reduction
    typename WarpReduceInternal::TempStorage warp_reduce[WARPS];

    /// Shared totals from each warp-synchronous reduction
    T warp_aggregates[WARPS];

    /// Shared prefix for the entire thread block
    T block_prefix;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // Thread fields
  _TempStorage& temp_storage;
  int linear_tid;
  int warp_id;
  int lane_id;

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockReduceWarpReductions(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
      , warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS)
      , lane_id(::cuda::ptx::get_sreg_laneid())
  {}

  /**
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] warp_aggregate
   *   <b>[<em>lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <bool FULL_TILE, typename ReductionOp, int SUCCESSOR_WARP>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregates(
    ReductionOp reduction_op, T warp_aggregate, int num_valid, constant_t<SUCCESSOR_WARP> /*successor_warp*/)
  {
    if (FULL_TILE || (SUCCESSOR_WARP * LOGICAL_WARP_SIZE < num_valid))
    {
      T addend       = temp_storage.warp_aggregates[SUCCESSOR_WARP];
      warp_aggregate = reduction_op(warp_aggregate, addend);
    }
    return ApplyWarpAggregates<FULL_TILE>(reduction_op, warp_aggregate, num_valid, constant_v<SUCCESSOR_WARP + 1>);
  }

  /**
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] warp_aggregate
   *   <b>[<em>lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <bool FULL_TILE, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregates(
    ReductionOp /*reduction_op*/, T warp_aggregate, int /*num_valid*/, constant_t<int{WARPS}> /*successor_warp*/)
  {
    return warp_aggregate;
  }

  /**
   * @brief Returns block-wide aggregate in <em>thread</em><sub>0</sub>.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] warp_aggregate
   *   <b>[<em>lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <bool FULL_TILE, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregates(ReductionOp reduction_op, T warp_aggregate, int num_valid)
  {
    // Share lane aggregates
    if (lane_id == 0)
    {
      detail::uninitialized_copy_single(temp_storage.warp_aggregates + warp_id, warp_aggregate);
    }

    __syncthreads();

    // Update total aggregate in warp 0, lane 0
    if (linear_tid == 0)
    {
      warp_aggregate = ApplyWarpAggregates<FULL_TILE>(reduction_op, warp_aggregate, num_valid, constant_v<1>);
    }

    return warp_aggregate;
  }

  /**
   * @brief Computes a thread block-wide reduction using addition (+) as the reduction operator.
   *        The first num_valid threads each contribute one reduction partial. The return value is
   *        only valid for thread<sub>0</sub>.
   *
   * @param[in] input
   *   Calling thread's input partial reductions
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <bool FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input, int num_valid)
  {
    ::cuda::std::plus<> reduction_op;
    int warp_offset    = (warp_id * LOGICAL_WARP_SIZE);
    int warp_num_valid = ((FULL_TILE && EVEN_WARP_MULTIPLE) || (warp_offset + LOGICAL_WARP_SIZE <= num_valid))
                         ? LOGICAL_WARP_SIZE
                         : num_valid - warp_offset;

    // Warp reduction in every warp
    T warp_aggregate =
      WarpReduceInternal(temp_storage.warp_reduce[warp_id])
        .template Reduce<(FULL_TILE && EVEN_WARP_MULTIPLE)>(input, warp_num_valid, ::cuda::std::plus<>{});

    // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
    return ApplyWarpAggregates<FULL_TILE>(reduction_op, warp_aggregate, num_valid);
  }

  /**
   * @brief Computes a thread block-wide reduction using the specified reduction operator.
   *        The first num_valid threads each contribute one reduction partial.
   *        The return value is only valid for thread<sub>0</sub>.
   *
   * @param[in] input
   *   Calling thread's input partial reductions
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   */
  template <bool FULL_TILE, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, int num_valid, ReductionOp reduction_op)
  {
    int warp_offset    = warp_id * LOGICAL_WARP_SIZE;
    int warp_num_valid = ((FULL_TILE && EVEN_WARP_MULTIPLE) || (warp_offset + LOGICAL_WARP_SIZE <= num_valid))
                         ? LOGICAL_WARP_SIZE
                         : num_valid - warp_offset;

    // Warp reduction in every warp
    T warp_aggregate = WarpReduceInternal(temp_storage.warp_reduce[warp_id])
                         .template Reduce<(FULL_TILE && EVEN_WARP_MULTIPLE)>(input, warp_num_valid, reduction_op);

    // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
    return ApplyWarpAggregates<FULL_TILE>(reduction_op, warp_aggregate, num_valid);
  }
};
} // namespace detail

CUB_NAMESPACE_END
