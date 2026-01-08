/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <cub/detail/rfa.cuh>
#include <cub/grid/grid_even_share.cuh>

#include <thrust/type_traits/unwrap_contiguous_iterator.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace reduce
{

/**
 * All cub::DeviceReduce::* algorithms are using the same implementation. Some of them, however,
 * should use initial value only for empty problems. If this struct is used as initial value with
 * one of the `DeviceReduce` algorithms, the `init` value wrapped by this struct will only be used
 * for empty problems; it will not be incorporated into the aggregate of non-empty problems.
 */
template <class T>
struct empty_problem_init_t
{
  T init;

  _CCCL_HOST_DEVICE operator T() const
  {
    return init;
  }
};

template <class InitT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE InitT unwrap_empty_problem_init(InitT init)
{
  return init;
}

template <class InitT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE InitT unwrap_empty_problem_init(empty_problem_init_t<InitT> empty_problem_init)
{
  return empty_problem_init.init;
}

/**
 * @brief Applies initial value to the block aggregate and stores the result to the output iterator.
 *
 * @param d_out Iterator to the output aggregate
 * @param reduction_op Binary reduction functor
 * @param init Initial value
 * @param block_aggregate Aggregate value computed by the block
 */
template <class OutputIteratorT, class ReductionOpT, class InitT, class AccumT>
_CCCL_HOST_DEVICE void
finalize_and_store_aggregate(OutputIteratorT d_out, ReductionOpT reduction_op, InitT init, AccumT block_aggregate)
{
  *d_out = reduction_op(init, block_aggregate);
}

/**
 * @brief Ignores initial value and stores the block aggregate to the output iterator.
 *
 * @param d_out Iterator to the output aggregate
 * @param block_aggregate Aggregate value computed by the block
 */
template <class OutputIteratorT, class ReductionOpT, class InitT, class AccumT>
_CCCL_HOST_DEVICE void
finalize_and_store_aggregate(OutputIteratorT d_out, ReductionOpT, empty_problem_init_t<InitT>, AccumT block_aggregate)
{
  *d_out = block_aggregate;
}

/**
 * @brief Reduce region kernel entry point (multi-block). Computes privatized
 *        reductions, one per thread block.
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 *
 * @tparam AccumT
 *   Accumulator type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] even_share
 *   Even-share descriptor for mapping an equal number of tiles onto each
 *   thread block
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename AccumT,
          typename TransformOpT>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS)) void DeviceReduceKernel(
  InputIteratorT d_in,
  AccumT* d_out,
  OffsetT num_items,
  GridEvenShare<OffsetT> even_share,
  ReductionOpT reduction_op,
  TransformOpT transform_op)
{
  // Thread block type for reducing input tiles
  using AgentReduceT = detail::reduce::AgentReduce<
    typename ChainedPolicyT::ActivePolicy::ReducePolicy,
    InputIteratorT,
    AccumT*,
    OffsetT,
    ReductionOpT,
    AccumT,
    TransformOpT>;

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  // Consume input tiles
  AccumT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op, transform_op).ConsumeTiles(even_share);

  // Output result
  if (threadIdx.x == 0)
  {
    detail::uninitialized_copy_single(d_out + blockIdx.x, block_aggregate);
  }
}

/**
 * @brief Reduce a single tile kernel entry point (single-block). Can be used
 *        to aggregate privatized thread block reductions from a previous
 *        multi-block reduction pass.
 *
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
 * @tparam AccumT
 *   Accumulator type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] num_items
 *   Total number of input data items
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
          typename AccumT,
          typename TransformOpT = ::cuda::std::identity>
CUB_DETAIL_KERNEL_ATTRIBUTES __launch_bounds__(
  int(ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS),
  1) void DeviceReduceSingleTileKernel(InputIteratorT d_in,
                                       OutputIteratorT d_out,
                                       OffsetT num_items,
                                       ReductionOpT reduction_op,
                                       InitT init,
                                       TransformOpT transform_op)
{
  // Thread block type for reducing input tiles
  using AgentReduceT = detail::reduce::AgentReduce<
    typename ChainedPolicyT::ActivePolicy::SingleTilePolicy,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ReductionOpT,
    AccumT,
    TransformOpT>;

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  // Check if empty problem
  if (num_items == 0)
  {
    if (threadIdx.x == 0)
    {
      *d_out = detail::reduce::unwrap_empty_problem_init(init);
    }

    return;
  }

  // Consume input tiles
  AccumT block_aggregate =
    AgentReduceT(temp_storage, d_in, reduction_op, transform_op).ConsumeRange(OffsetT(0), num_items);

  // Output result
  if (threadIdx.x == 0)
  {
    detail::reduce::finalize_and_store_aggregate(d_out, reduction_op, init, block_aggregate);
  }
}

/**
 * @brief Deterministically Reduce region kernel entry point (multi-block). Computes privatized
 *        reductions, one per thread block in deterministic fashion
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 *
 * @tparam AccumT
 *   Accumulator type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] even_share
 *   Even-share descriptor for mapping an equal number of tiles onto each
 *   thread block
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename AccumT,
          typename TransformOpT>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::ReducePolicy::BLOCK_THREADS)) void DeterministicDeviceReduceKernel(
  InputIteratorT d_in,
  AccumT* d_out,
  OffsetT num_items,
  ReductionOpT reduction_op,
  TransformOpT transform_op,
  const int reduce_grid_size)
{
  using BlockReduceT =
    BlockReduce<AccumT,
                ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS,
                ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_ALGORITHM>;
  // Shared memory storage
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  using FloatType                 = typename AccumT::ftype;
  constexpr int BinLength         = AccumT::max_index + AccumT::max_fold;
  constexpr auto ITEMS_PER_THREAD = ChainedPolicyT::ReducePolicy::ITEMS_PER_THREAD;
  constexpr auto BLOCK_THREADS    = ChainedPolicyT::ReducePolicy::BLOCK_THREADS;
  const int GRID_DIM              = reduce_grid_size;
  const int tid                   = BLOCK_THREADS * blockIdx.x + threadIdx.x;

  FloatType* shared_bins = detail::rfa::get_shared_bin_array<FloatType, BinLength>();

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int index = threadIdx.x; index < BinLength; index += ChainedPolicyT::ReducePolicy::BLOCK_THREADS)
  {
    shared_bins[index] = AccumT::initialize_bin(index);
  }

  __syncthreads();

  AccumT thread_aggregate{};
  int count = 0;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = tid; i < num_items; i += ITEMS_PER_THREAD * GRID_DIM * BLOCK_THREADS)
  {
    FloatType items[ITEMS_PER_THREAD] = {};
    for (int j = 0; j < ITEMS_PER_THREAD; j++)
    {
      const int idx = i + j * GRID_DIM * BLOCK_THREADS;
      if (idx < num_items)
      {
        items[j] = transform_op(d_in[idx]);
      }
    }

    FloatType abs_max_val = ::cuda::std::fabs(items[0]);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 1; j < ITEMS_PER_THREAD; j++)
    {
      abs_max_val = ::cuda::std::fmax(::cuda::std::fabs(items[j]), abs_max_val);
    }

    thread_aggregate.set_max_val(abs_max_val);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < ITEMS_PER_THREAD; j++)
    {
      thread_aggregate.unsafe_add(items[j]);
      count++;
      if (count >= thread_aggregate.endurance())
      {
        thread_aggregate.renorm();
        count = 0;
      }
    }
  }

  AccumT block_aggregate = BlockReduceT(temp_storage).Reduce(thread_aggregate, [](AccumT lhs, AccumT rhs) -> AccumT {
    AccumT rtn = lhs;
    rtn += rhs;
    return rtn;
  });

  // Output result
  if (threadIdx.x == 0)
  {
    detail::uninitialized_copy_single(d_out + blockIdx.x, block_aggregate);
  }
}

/**
 * @brief Deterministically Reduce a single tile kernel entry point (single-block). Can be used
 *        to aggregate privatized thread block reductions from a previous
 *        multi-block reduction pass.
 *
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
 * @tparam AccumT
 *   Accumulator type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] num_items
 *   Total number of input data items
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
          typename AccumT,
          typename TransformOpT = ::cuda::std::identity>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::SingleTilePolicy::BLOCK_THREADS), 1) void DeterministicDeviceReduceSingleTileKernel(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  ReductionOpT reduction_op,
  InitT init,
  TransformOpT transform_op)
{
  using BlockReduceT =
    BlockReduce<AccumT,
                ChainedPolicyT::SingleTilePolicy::BLOCK_THREADS,
                ChainedPolicyT::SingleTilePolicy::BLOCK_ALGORITHM>;

  // Shared memory storage
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // Check if empty problem
  if (num_items == 0)
  {
    if (threadIdx.x == 0)
    {
      *d_out = init;
    }
    return;
  }

  using FloatType         = typename AccumT::ftype;
  constexpr int BinLength = AccumT::max_index + AccumT::max_fold;

  FloatType* shared_bins = detail::rfa::get_shared_bin_array<FloatType, BinLength>();

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int index = threadIdx.x; index < BinLength;
       index += ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS)
  {
    shared_bins[index] = AccumT::initialize_bin(index);
  }

  __syncthreads();

  constexpr auto BLOCK_THREADS = ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS;

  AccumT thread_aggregate{};

  // Consume block aggregates of previous kernel
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = threadIdx.x; i < num_items; i += BLOCK_THREADS)
  {
    thread_aggregate += transform_op(d_in[i]);
  }

  AccumT block_aggregate = BlockReduceT(temp_storage).Reduce(thread_aggregate, reduction_op, num_items);

  // Output result
  if (threadIdx.x == 0)
  {
    detail::reduce::finalize_and_store_aggregate(d_out, reduction_op, init, block_aggregate.conv_to_fp());
  }
}

} // namespace reduce
} // namespace detail

CUB_NAMESPACE_END
