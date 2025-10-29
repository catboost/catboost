/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
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

#include <cub/agent/agent_reduce.cuh>
#include <cub/device/dispatch/kernels/reduce.cuh> // finalize_and_store_aggregate
#include <cub/iterator/arg_index_input_iterator.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace reduce
{

/// Normalize input iterator to segment offset
template <typename T, typename OffsetT, typename IteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void NormalizeReductionOutput(T& /*val*/, OffsetT /*base_offset*/, IteratorT /*itr*/)
{}

/// Normalize input iterator to segment offset (specialized for arg-index)
template <typename KeyValuePairT, typename OffsetT, typename WrappedIteratorT, typename OutputValueT>
_CCCL_DEVICE _CCCL_FORCEINLINE void NormalizeReductionOutput(
  KeyValuePairT& val, OffsetT base_offset, ArgIndexInputIterator<WrappedIteratorT, OffsetT, OutputValueT> /*itr*/)
{
  val.key -= base_offset;
}

/**
 * Segmented reduction (one block per segment)
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam BeginOffsetIteratorT
 *   Random-access input iterator type for reading segment beginning offsets
 *   @iterator
 *
 * @tparam EndOffsetIteratorT
 *   Random-access input iterator type for reading segment ending offsets
 *   @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `T operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of
 *   length `num_segments`, such that `d_begin_offsets[i]` is the first element
 *   of the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of
 *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
 *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is
 *   considered empty.
 *
 * @param[in] num_segments
 *   The number of segments on which the reduction is performed
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 *
 * @param[in] init
 *   The initial value of the reduction
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS)) void DeviceSegmentedReduceKernel(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  ReductionOpT reduction_op,
  InitT init)
{
  // Thread block type for reducing input tiles
  using AgentReduceT =
    AgentReduce<typename ChainedPolicyT::ActivePolicy::ReducePolicy,
                InputIteratorT,
                OutputIteratorT,
                OffsetT,
                ReductionOpT,
                AccumT>;

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  OffsetT segment_begin = d_begin_offsets[blockIdx.x];
  OffsetT segment_end   = d_end_offsets[blockIdx.x];

  // Check if empty problem
  if (segment_begin == segment_end)
  {
    if (threadIdx.x == 0)
    {
      *(d_out + blockIdx.x) = init;
    }
    return;
  }

  // Consume input tiles
  AccumT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op).ConsumeRange(segment_begin, segment_end);

  // Normalize as needed
  NormalizeReductionOutput(block_aggregate, segment_begin, d_in);

  if (threadIdx.x == 0)
  {
    finalize_and_store_aggregate(d_out + blockIdx.x, reduction_op, init, block_aggregate);
  }
}

/**
 * Fixed Segment Size Segmented reduction
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `T operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] segment_size
 *   The fixed segment size of each the segments
 *
 * @param[in] num_segments
 *   The number of segments on which the reduction is performed
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 *
 * @param[in] init
 *   The initial value of the reduction
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS)) void DeviceFixedSizeSegmentedReduceKernel(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT segment_size,
  int num_segments,
  ReductionOpT reduction_op,
  InitT init)
{
  using ActivePolicyT = typename ChainedPolicyT::ActivePolicy;

  // Thread block type for reducing input tiles
  using AgentReduceT =
    AgentReduce<typename ActivePolicyT::ReducePolicy, InputIteratorT, OutputIteratorT, int, ReductionOpT, AccumT>;

  using AgentMediumReduceT =
    AgentWarpReduce<typename ActivePolicyT::MediumReducePolicy, InputIteratorT, OutputIteratorT, int, ReductionOpT, AccumT>;

  using AgentSmallReduceT =
    AgentWarpReduce<typename ActivePolicyT::SmallReducePolicy, InputIteratorT, OutputIteratorT, int, ReductionOpT, AccumT>;

  constexpr auto segments_per_medium_block = ActivePolicyT::MediumReducePolicy::SEGMENTS_PER_BLOCK;
  constexpr auto medium_threads_per_warp   = ActivePolicyT::MediumReducePolicy::WARP_THREADS;
  constexpr auto medium_items_per_tile     = ActivePolicyT::MediumReducePolicy::ITEMS_PER_TILE;

  constexpr auto segments_per_small_block = ActivePolicyT::SmallReducePolicy::SEGMENTS_PER_BLOCK;
  constexpr auto small_threads_per_warp   = ActivePolicyT::SmallReducePolicy::WARP_THREADS;
  constexpr auto small_items_per_tile     = ActivePolicyT::SmallReducePolicy::ITEMS_PER_TILE;

  // Shared memory storage
  __shared__ union
  {
    typename AgentReduceT::TempStorage large_storage;
    typename AgentMediumReduceT::TempStorage medium_storage[segments_per_medium_block];
    typename AgentSmallReduceT::TempStorage small_storage[segments_per_small_block];
  } temp_storage;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  if (segment_size <= small_items_per_tile)
  {
    const int sid_within_block  = tid / small_threads_per_warp;
    const int lane_id           = tid % small_threads_per_warp;
    const int global_segment_id = bid * segments_per_small_block + sid_within_block;

    const auto segment_begin = static_cast<::cuda::std::int64_t>(global_segment_id) * segment_size;

    if (global_segment_id < num_segments)
    {
      // If empty segment, write out the initial value
      if (segment_size == 0)
      {
        if (lane_id == 0)
        {
          *(d_out + global_segment_id) = detail::reduce::unwrap_empty_problem_init(init);
        }
        return;
      }
      // Consume input tiles
      AccumT warp_aggregate =
        AgentSmallReduceT(temp_storage.small_storage[sid_within_block], d_in + segment_begin, reduction_op)
          .ConsumeRange({}, static_cast<int>(segment_size));

      if (lane_id == 0)
      {
        finalize_and_store_aggregate(d_out + global_segment_id, reduction_op, init, warp_aggregate);
      }
    }
  }
  else if (segment_size <= medium_items_per_tile)
  {
    const int sid_within_block  = tid / medium_threads_per_warp;
    const int lane_id           = tid % medium_threads_per_warp;
    const int global_segment_id = bid * segments_per_medium_block + sid_within_block;

    const auto segment_begin = static_cast<::cuda::std::int64_t>(global_segment_id) * segment_size;

    if (global_segment_id < num_segments)
    {
      // Consume input tiles
      AccumT warp_aggregate =
        AgentMediumReduceT(temp_storage.medium_storage[sid_within_block], d_in + segment_begin, reduction_op)
          .ConsumeRange({}, static_cast<int>(segment_size));

      if (lane_id == 0)
      {
        finalize_and_store_aggregate(d_out + global_segment_id, reduction_op, init, warp_aggregate);
      }
    }
  }
  else
  {
    const auto segment_begin = static_cast<::cuda::std::int64_t>(bid) * segment_size;

    // Consume input tiles
    AccumT block_aggregate = AgentReduceT(temp_storage.large_storage, d_in + segment_begin, reduction_op)
                               .ConsumeRange({}, static_cast<int>(segment_size));

    if (tid == 0)
    {
      finalize_and_store_aggregate(d_out + bid, reduction_op, init, block_aggregate);
    }
  }
}

} // namespace reduce
} // namespace detail

CUB_NAMESPACE_END
