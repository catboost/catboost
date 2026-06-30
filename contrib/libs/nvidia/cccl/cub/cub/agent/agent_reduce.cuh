/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * @file cub::AgentReduce implements a stateful abstraction of CUDA thread
 *       blocks for participating in device-wide reduction.
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

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/grid/grid_mapping.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/functional>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentReduce
 * @tparam NOMINAL_BLOCK_THREADS_4B Threads per thread block
 * @tparam NOMINAL_ITEMS_PER_THREAD_4B Items per thread (per tile of input)
 * @tparam ComputeT Dominant compute type
 * @tparam _VECTOR_LOAD_LENGTH Number of items per vectorized load
 * @tparam _BLOCK_ALGORITHM Cooperative block-wide reduction algorithm to use
 * @tparam _LOAD_MODIFIER Cache load modifier for reading input elements
 */
template <
  int NOMINAL_BLOCK_THREADS_4B,
  int NOMINAL_ITEMS_PER_THREAD_4B,
  typename ComputeT,
  int _VECTOR_LOAD_LENGTH,
  BlockReduceAlgorithm _BLOCK_ALGORITHM,
  CacheLoadModifier _LOAD_MODIFIER,
  typename ScalingType = detail::MemBoundScaling<NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT>>
struct AgentReducePolicy : ScalingType
{
  /// Number of items per vectorized load
  static constexpr int VECTOR_LOAD_LENGTH = _VECTOR_LOAD_LENGTH;

  /// Cooperative block-wide reduction algorithm to use
  static constexpr BlockReduceAlgorithm BLOCK_ALGORITHM = _BLOCK_ALGORITHM;

  /// Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;
};

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
namespace detail
{
// Only define this when needed.
// Because of overload woes, this depends on C++20 concepts. util_device.h checks that concepts are available when
// either runtime policies or PTX JSON information are enabled, so if they are, this is always valid. The generic
// version is always defined, and that's the only one needed for regular CUB operations.
//
// TODO: enable this unconditionally once concepts are always available
CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  ReduceAgentPolicy,
  (GenericAgentPolicy),
  (BLOCK_THREADS, BlockThreads, int),
  (ITEMS_PER_THREAD, ItemsPerThread, int),
  (VECTOR_LOAD_LENGTH, VectorLoadLength, int),
  (BLOCK_ALGORITHM, BlockAlgorithm, cub::BlockReduceAlgorithm),
  (LOAD_MODIFIER, LoadModifier, cub::CacheLoadModifier))
} // namespace detail
#endif // defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)

/**
 * Parameterizable tuning policy type for AgentReduce
 * @tparam _BLOCK_THREADS Threads per thread block
 * @tparam _WARP_THREADS Threads per warp
 * @tparam NOMINAL_ITEMS_PER_THREAD_4B Items per thread (per tile of input)
 * @tparam ComputeT Dominant compute type
 * @tparam _VECTOR_LOAD_LENGTH Number of items per vectorized load
 * @tparam _LOAD_MODIFIER Cache load modifier for reading input elements
 */
template <int _BLOCK_THREADS,
          int _WARP_THREADS,
          int NOMINAL_ITEMS_PER_THREAD_4B,
          typename ComputeT,
          int _VECTOR_LOAD_LENGTH,
          CacheLoadModifier _LOAD_MODIFIER>
struct AgentWarpReducePolicy
{
  /// Number of threads per warp
  static constexpr int WARP_THREADS = _WARP_THREADS;

  /// Number of items per vectorized load
  static constexpr int VECTOR_LOAD_LENGTH = _VECTOR_LOAD_LENGTH;

  /// Number of threads per block
  static constexpr int BLOCK_THREADS = _BLOCK_THREADS;

  /// Number of items per thread
  static constexpr int ITEMS_PER_THREAD =
    detail::MemBoundScaling<0, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT>::ITEMS_PER_THREAD;

  /// Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;

  /// Number of items per tile
  constexpr static int ITEMS_PER_TILE = ITEMS_PER_THREAD * WARP_THREADS;

  /// Number of segments per block
  constexpr static int SEGMENTS_PER_BLOCK = BLOCK_THREADS / WARP_THREADS;

  static_assert((BLOCK_THREADS % WARP_THREADS) == 0, "Block should be multiple of warp");
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

namespace detail
{
namespace reduce
{

/**
 * @brief AgentReduceImpl implements a stateful abstraction of CUDA thread blocks
 *        and warps, for participating in device-wide reduction .
 *
 * Each thread reduces only the values it loads. If `FIRST_TILE`, this partial
 * reduction is stored into `thread_aggregate`. Otherwise it is accumulated
 * into `thread_aggregate`.
 *
 * @tparam AgentReducePolicy
 *   Parameterized AgentReducePolicy tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access iterator type for input
 *
 * @tparam OutputIteratorT
 *   Random-access iterator type for output
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOp
 *   Binary reduction operator type having member
 *   `auto operator()(T &&a, U &&b)`
 *
 * @tparam AccumT
 *   The type of intermediate accumulator (according to P2322R6)
 *
 * @tparam TransformOp
 *    Unary operator type having member `auto operator()(T &&a)`
 *
 * @tparam CollectiveReduceT
 *   Block or Warp reduction type
 *
 * @tparam NumThreads
 *   Number of threads participating in the collective reduction
 *
 * @tparam IsWarpReduction
 *   Whether or not this is a warp reduction
 */
template <typename AgentReducePolicy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOp,
          typename AccumT,
          typename TransformOp,
          typename CollectiveReduceT,
          int NumThreads,
          bool IsWarpReduction = false>
struct AgentReduceImpl
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  /// The input value type
  using InputT = it_value_t<InputIteratorT>;

  /// Vector type of InputT for data movement
  using VectorT = typename CubVector<InputT, AgentReducePolicy::VECTOR_LOAD_LENGTH>::Type;

  /// Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using WrappedInputIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<InputIteratorT>,
                     CacheModifiedInputIterator<AgentReducePolicy::LOAD_MODIFIER, InputT, OffsetT>,
                     InputIteratorT>;

  /// Constants
  static constexpr int ITEMS_PER_THREAD   = AgentReducePolicy::ITEMS_PER_THREAD;
  static constexpr int TILE_ITEMS         = NumThreads * ITEMS_PER_THREAD;
  static constexpr int VECTOR_LOAD_LENGTH = _CUDA_VSTD::min(ITEMS_PER_THREAD, AgentReducePolicy::VECTOR_LOAD_LENGTH);

  // Can vectorize according to the policy if the input iterator is a native
  // pointer to a primitive type
  static constexpr bool ATTEMPT_VECTORIZATION =
    (VECTOR_LOAD_LENGTH > 1) && (ITEMS_PER_THREAD % VECTOR_LOAD_LENGTH == 0)
    && (::cuda::std::is_pointer_v<InputIteratorT>) && is_primitive<InputT>::value;

  static constexpr CacheLoadModifier LOAD_MODIFIER = AgentReducePolicy::LOAD_MODIFIER;

  /// Shared memory type required by this thread block
  struct _TempStorage
  {
    typename CollectiveReduceT::TempStorage reduce;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage& temp_storage; ///< Reference to temp_storage
  InputIteratorT d_in; ///< Input data to reduce
  WrappedInputIteratorT d_wrapped_in; ///< Wrapped input data to reduce
  ReductionOp reduction_op; ///< Binary reduction operator
  TransformOp transform_op; ///< Transform operator
  unsigned int lane_id; ///< Local thread index inside a Warp or Block

  //---------------------------------------------------------------------
  // Utility
  //---------------------------------------------------------------------

  // Whether or not the input is aligned with the vector type (specialized for
  // types we can vectorize)
  template <typename Iterator>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool IsAligned(Iterator d_in, ::cuda::std::true_type /*can_vectorize*/)
  {
    return (size_t(d_in) & (sizeof(VectorT) - 1)) == 0;
  }

  // Whether or not the input is aligned with the vector type (specialized for
  // types we cannot vectorize)
  template <typename Iterator>
  static _CCCL_DEVICE _CCCL_FORCEINLINE bool IsAligned(Iterator /*d_in*/, ::cuda::std::false_type /*can_vectorize*/)
  {
    return false;
  }

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  /**
   * @brief Constructor
   * @param temp_storage Reference to temp_storage
   * @param d_in Input data to reduce
   * @param reduction_op Binary reduction operator
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentReduceImpl(
    TempStorage& temp_storage, InputIteratorT d_in, ReductionOp reduction_op, TransformOp transform_op, int lane_id)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_wrapped_in(d_in)
      , reduction_op(reduction_op)
      , transform_op(transform_op)
      , lane_id(lane_id)
  {}

  //---------------------------------------------------------------------
  // Tile consumption
  //---------------------------------------------------------------------

  /**
   * @brief Consume a full tile of input (non-vectorized)
   * @param block_offset The offset the tile to consume
   * @param valid_items The number of valid items in the tile
   * @param is_full_tile Whether or not this is a full tile
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <int IS_FIRST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTile(
    AccumT& thread_aggregate,
    OffsetT block_offset,
    int /*valid_items*/,
    ::cuda::std::true_type /*is_full_tile*/,
    ::cuda::std::false_type /*can_vectorize*/)
  {
    AccumT items[ITEMS_PER_THREAD];

    // Load items in striped fashion
    load_transform_direct_striped<NumThreads>(lane_id, d_wrapped_in + block_offset, items, transform_op);

    // Reduce items within each thread stripe
    thread_aggregate = (IS_FIRST_TILE) ? cub::ThreadReduce(items, reduction_op)
                                       : cub::ThreadReduce(items, reduction_op, thread_aggregate);
  }

  /**
   * Consume a full tile of input (vectorized)
   * @param block_offset The offset the tile to consume
   * @param valid_items The number of valid items in the tile
   * @param is_full_tile Whether or not this is a full tile
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <int IS_FIRST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTile(
    AccumT& thread_aggregate,
    OffsetT block_offset,
    int /*valid_items*/,
    ::cuda::std::true_type /*is_full_tile*/,
    ::cuda::std::true_type /*can_vectorize*/)
  {
    // Alias items as an array of VectorT and load it in striped fashion
    enum
    {
      WORDS = ITEMS_PER_THREAD / VECTOR_LOAD_LENGTH
    };

    // Fabricate a vectorized input iterator
    InputT* d_in_unqualified = const_cast<InputT*>(d_in) + block_offset + (lane_id * VECTOR_LOAD_LENGTH);
    CacheModifiedInputIterator<AgentReducePolicy::LOAD_MODIFIER, VectorT, OffsetT> d_vec_in(
      reinterpret_cast<VectorT*>(d_in_unqualified));

    // Load items as vector items
    InputT input_items[ITEMS_PER_THREAD];
    VectorT* vec_items = reinterpret_cast<VectorT*>(input_items);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < WORDS; ++i)
    {
      vec_items[i] = d_vec_in[NumThreads * i];
    }

    // Convert from input type to output type
    AccumT items[ITEMS_PER_THREAD];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      items[i] = transform_op(input_items[i]);
    }

    // Reduce items within each thread stripe
    thread_aggregate = (IS_FIRST_TILE) ? cub::ThreadReduce(items, reduction_op)
                                       : cub::ThreadReduce(items, reduction_op, thread_aggregate);
  }

  /**
   * Consume a partial tile of input
   * @param block_offset The offset the tile to consume
   * @param valid_items The number of valid items in the tile
   * @param is_full_tile Whether or not this is a full tile
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <int IS_FIRST_TILE, bool CAN_VECTORIZE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTile(
    AccumT& thread_aggregate,
    OffsetT block_offset,
    int valid_items,
    ::cuda::std::false_type /*is_full_tile*/,
    ::cuda::std::bool_constant<CAN_VECTORIZE> /*can_vectorize*/)
  {
    // Partial tile
    int thread_offset = lane_id;

    // Read first item
    if ((IS_FIRST_TILE) && (thread_offset < valid_items))
    {
      thread_aggregate = transform_op(d_wrapped_in[block_offset + thread_offset]);
      thread_offset += NumThreads;
    }

    // Continue reading items (block-striped)
    while (thread_offset < valid_items)
    {
      InputT item(d_wrapped_in[block_offset + thread_offset]);

      thread_aggregate = reduction_op(thread_aggregate, transform_op(item));
      thread_offset += NumThreads;
    }
  }

  //---------------------------------------------------------------
  // Consume a contiguous segment of tiles
  //---------------------------------------------------------------------

  /**
   * @brief Reduce a contiguous segment of input tiles
   * @param even_share GridEvenShare descriptor
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <bool CAN_VECTORIZE>
  _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
  ConsumeRange(GridEvenShare<OffsetT>& even_share, ::cuda::std::bool_constant<CAN_VECTORIZE> can_vectorize)
  {
    AccumT thread_aggregate{};

    if (even_share.block_end - even_share.block_offset < TILE_ITEMS)
    {
      // First tile isn't full (not all threads have valid items)
      int valid_items = even_share.block_end - even_share.block_offset;
      ConsumeTile<true>(
        thread_aggregate, even_share.block_offset, valid_items, ::cuda::std::false_type(), can_vectorize);

      // For Warp Reduction, we need to explicitly handle the valid_items,
      // whereas for Block Reduction it is implicitly handled
      if constexpr (IsWarpReduction)
      {
        valid_items = (NumThreads <= valid_items) ? NumThreads : valid_items;
      }
      return CollectiveReduceT(temp_storage.reduce).Reduce(thread_aggregate, reduction_op, valid_items);
    }

    // Extracting this into a function saves 8% of generated kernel size by allowing to reuse
    // the block reduction below. This also workaround hang in nvcc.
    ConsumeFullTileRange(thread_aggregate, even_share, can_vectorize);

    // Compute block-wide reduction (all threads have valid items)
    return CollectiveReduceT(temp_storage.reduce).Reduce(thread_aggregate, reduction_op);
  }

  /**
   * @brief Reduce a contiguous segment of input tiles
   * @param[in] block_offset Threadblock begin offset (inclusive)
   * @param[in] block_end Threadblock end offset (exclusive)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ConsumeRange(OffsetT block_offset, OffsetT block_end)
  {
    GridEvenShare<OffsetT> even_share;
    even_share.template BlockInit<TILE_ITEMS>(block_offset, block_end);

    return (IsAligned(d_in + block_offset, bool_constant_v<ATTEMPT_VECTORIZATION>))
           ? ConsumeRange(even_share, bool_constant_v<ATTEMPT_VECTORIZATION>)
           : ConsumeRange(even_share, ::cuda::std::false_type{});
  }

  /**
   * Reduce a contiguous segment of input tiles
   * @param[in] even_share GridEvenShare descriptor
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ConsumeTiles(GridEvenShare<OffsetT>& even_share)
  {
    // Initialize GRID_MAPPING_STRIP_MINE even-share descriptor for this thread block
    even_share.template BlockInit<TILE_ITEMS, GRID_MAPPING_STRIP_MINE>();

    return (IsAligned(d_in, bool_constant_v<ATTEMPT_VECTORIZATION>))
           ? ConsumeRange(even_share, bool_constant_v<ATTEMPT_VECTORIZATION>)
           : ConsumeRange(even_share, ::cuda::std::false_type{});
  }

private:
  /**
   * @brief Reduce a contiguous segment of input tiles with more than `TILE_ITEMS` elements
   * @param even_share GridEvenShare descriptor
   * @param can_vectorize Whether or not we can vectorize loads
   */
  template <bool CAN_VECTORIZE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeFullTileRange(
    AccumT& thread_aggregate,
    GridEvenShare<OffsetT>& even_share,
    ::cuda::std::bool_constant<CAN_VECTORIZE> can_vectorize)
  {
    // At least one full block
    ConsumeTile<true>(thread_aggregate, even_share.block_offset, TILE_ITEMS, ::cuda::std::true_type(), can_vectorize);

    if (even_share.block_end - even_share.block_offset < even_share.block_stride)
    {
      // Exit early to handle offset overflow
      return;
    }

    even_share.block_offset += even_share.block_stride;

    // Consume subsequent full tiles of input, at least one full tile was processed, so
    // `even_share.block_end >= TILE_ITEMS`
    while (even_share.block_offset <= even_share.block_end - TILE_ITEMS)
    {
      ConsumeTile<false>(thread_aggregate, even_share.block_offset, TILE_ITEMS, ::cuda::std::true_type(), can_vectorize);

      if (even_share.block_end - even_share.block_offset < even_share.block_stride)
      {
        // Exit early to handle offset overflow
        return;
      }

      even_share.block_offset += even_share.block_stride;
    }

    // Consume a partially-full tile
    if (even_share.block_offset < even_share.block_end)
    {
      int valid_items = even_share.block_end - even_share.block_offset;
      ConsumeTile<false>(
        thread_aggregate, even_share.block_offset, valid_items, ::cuda::std::false_type(), can_vectorize);
    }
  }
};

/**
 * @brief AgentReduce implements a stateful abstraction of CUDA thread blocks
 *        and warps, for participating in device-wide reduction .
 *
 * Each thread reduces only the values it loads. If `FIRST_TILE`, this partial
 * reduction is stored into `thread_aggregate`. Otherwise it is accumulated
 * into `thread_aggregate`.
 *
 * @tparam AgentReducePolicy
 *   Parameterized AgentReducePolicy tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access iterator type for input
 *
 * @tparam OutputIteratorT
 *   Random-access iterator type for output
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOp
 *   Binary reduction operator type having member
 *   `auto operator()(T &&a, U &&b)`
 *
 * @tparam AccumT
 *   The type of intermediate accumulator (according to P2322R6)
 *
 * @tparam TransformOp
 *    Unary operator type having member `auto operator()(T &&a)`
 */
template <typename AgentReducePolicy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOp,
          typename AccumT,
          typename TransformOp = ::cuda::std::identity>
struct AgentReduce
    : AgentReduceImpl<AgentReducePolicy,
                      InputIteratorT,
                      OutputIteratorT,
                      OffsetT,
                      ReductionOp,
                      AccumT,
                      TransformOp,
                      BlockReduce<AccumT, AgentReducePolicy::BLOCK_THREADS, AgentReducePolicy::BLOCK_ALGORITHM>,
                      AgentReducePolicy::BLOCK_THREADS>
{
  using base_t =
    AgentReduceImpl<AgentReducePolicy,
                    InputIteratorT,
                    OutputIteratorT,
                    OffsetT,
                    ReductionOp,
                    AccumT,
                    TransformOp,
                    BlockReduce<AccumT, AgentReducePolicy::BLOCK_THREADS, AgentReducePolicy::BLOCK_ALGORITHM>,
                    AgentReducePolicy::BLOCK_THREADS>;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentReduce(
    typename base_t::TempStorage& temp_storage,
    InputIteratorT d_in,
    ReductionOp reduction_op,
    TransformOp transform_op = {})
      : base_t(temp_storage, d_in, reduction_op, transform_op, threadIdx.x)
  {}
};

/**
 * @brief AgentWarpReduce implements a stateful abstraction of CUDA warps,
 *        for participating in device-wide reduction .
 *
 * Each thread reduces only the values it loads. If `FIRST_TILE`, this partial
 * reduction is stored into `thread_aggregate`. Otherwise it is accumulated
 * into `thread_aggregate`.
 *
 * @tparam AgentReducePolicy
 *   Parameterized AgentReducePolicy tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access iterator type for input
 *
 * @tparam OutputIteratorT
 *   Random-access iterator type for output
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOp
 *   Binary reduction operator type having member
 *   `auto operator()(T &&a, U &&b)`
 *
 * @tparam AccumT
 *   The type of intermediate accumulator (according to P2322R6)
 *
 * @tparam TransformOp
 *    Unary operator type having member `auto operator()(T &&a)`
 */
template <typename AgentReducePolicy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOp,
          typename AccumT,
          typename TransformOp = ::cuda::std::identity>
struct AgentWarpReduce
    : AgentReduceImpl<AgentReducePolicy,
                      InputIteratorT,
                      OutputIteratorT,
                      OffsetT,
                      ReductionOp,
                      AccumT,
                      TransformOp,
                      WarpReduce<AccumT, AgentReducePolicy::WARP_THREADS>,
                      AgentReducePolicy::WARP_THREADS,
                      true>
{
  using base_t =
    AgentReduceImpl<AgentReducePolicy,
                    InputIteratorT,
                    OutputIteratorT,
                    OffsetT,
                    ReductionOp,
                    AccumT,
                    TransformOp,
                    WarpReduce<AccumT, AgentReducePolicy::WARP_THREADS>,
                    AgentReducePolicy::WARP_THREADS,
                    true>;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentWarpReduce(
    typename base_t::TempStorage& temp_storage,
    InputIteratorT d_in,
    ReductionOp reduction_op,
    TransformOp transform_op = {})
      : base_t(temp_storage, d_in, reduction_op, transform_op, threadIdx.x % AgentReducePolicy::WARP_THREADS)
  {}
};

} // namespace reduce
} // namespace detail

CUB_NAMESPACE_END
