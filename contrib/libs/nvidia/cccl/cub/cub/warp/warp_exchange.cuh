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

/**
 * @file
 * The cub::WarpExchange class provides [<em>collective</em>](../index.html#sec0)
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
#include <cub/warp/specializations/warp_exchange_shfl.cuh>
#include <cub/warp/specializations/warp_exchange_smem.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

enum WarpExchangeAlgorithm
{
  WARP_EXCHANGE_SMEM,
  WARP_EXCHANGE_SHUFFLE,
};

namespace detail
{
template <typename InputT, int ITEMS_PER_THREAD, int LOGICAL_WARP_THREADS, WarpExchangeAlgorithm WARP_EXCHANGE_ALGORITHM>
using InternalWarpExchangeImpl =
  ::cuda::std::_If<WARP_EXCHANGE_ALGORITHM == WARP_EXCHANGE_SMEM,
                   WarpExchangeSmem<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>,
                   WarpExchangeShfl<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>>;
} // namespace detail

/**
 * @brief The WarpExchange class provides [<em>collective</em>](../index.html#sec0)
 *        methods for rearranging data partitioned across a CUDA warp.
 *
 * @tparam T
 *   The data type to be exchanged.
 *
 * @tparam ITEMS_PER_THREAD
 *   The number of items partitioned onto each thread.
 *
 * @tparam LOGICAL_WARP_THREADS
 *   <b>[optional]</b> The number of threads per "logical" warp (may be less
 *   than the number of hardware warp threads). Default is the warp size of the
 *   targeted CUDA compute-capability (e.g., 32 threads for SM86). Must be a
 *   power of two.
 *
 * @par Overview
 * - It is commonplace for a warp of threads to rearrange data items between
 *   threads. For example, the global memory accesses prefer patterns where
 *   data items are "striped" across threads (where consecutive threads access
 *   consecutive items), yet most warp-wide operations prefer a "blocked"
 *   partitioning of items across threads (where consecutive items belong to a
 *   single thread).
 * - WarpExchange supports the following types of data exchanges:
 *   - Transposing between [<em>blocked</em>](../index.html#sec5sec3) and
 *     [<em>striped</em>](../index.html#sec5sec3) arrangements
 *   - Scattering ranked items to a
 *     [<em>striped arrangement</em>](../index.html#sec5sec3)
 *
 * @par A Simple Example
 * @par
 * The code snippet below illustrates the conversion from a "blocked" to a
 * "striped" arrangement of 64 integer items partitioned across 16 threads where
 * each thread owns 4 items.
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_exchange.cuh>
 *
 * __global__ void ExampleKernel(int *d_data, ...)
 * {
 *     constexpr int warp_threads = 16;
 *     constexpr int block_threads = 256;
 *     constexpr int items_per_thread = 4;
 *     constexpr int warps_per_block = block_threads / warp_threads;
 *     const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
 *
 *     // Specialize WarpExchange for a virtual warp of 16 threads owning 4 integer items each
 *     using WarpExchangeT =
 *       cub::WarpExchange<int, items_per_thread, warp_threads>;
 *
 *     // Allocate shared memory for WarpExchange
 *     __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];
 *
 *     // Load a tile of data striped across threads
 *     int thread_data[items_per_thread];
 *     // ...
 *
 *     // Collectively exchange data into a blocked arrangement across threads
 *     WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
 * @endcode
 * @par
 * Suppose the set of striped input @p thread_data across the block of threads
 * is <tt>{ [0,16,32,48], [1,17,33,49], ..., [15, 32, 47, 63] }</tt>.
 * The corresponding output @p thread_data in those threads will be
 * <tt>{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [60,61,62,63] }</tt>.
 */
template <typename InputT,
          int ITEMS_PER_THREAD,
          int LOGICAL_WARP_THREADS                      = detail::warp_threads,
          WarpExchangeAlgorithm WARP_EXCHANGE_ALGORITHM = WARP_EXCHANGE_SMEM>
class WarpExchange
    : private detail::InternalWarpExchangeImpl<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, WARP_EXCHANGE_ALGORITHM>
{
  using InternalWarpExchange =
    detail::InternalWarpExchangeImpl<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, WARP_EXCHANGE_ALGORITHM>;

public:
  /// \smemstorage{WarpExchange}
  using TempStorage = typename InternalWarpExchange::TempStorage;

  //! @name Collective constructors
  //! @{

  WarpExchange() = delete;

  /**
   * @brief Collective constructor using the specified memory allocation as
   *        temporary storage.
   */
  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpExchange(TempStorage& temp_storage)
      : InternalWarpExchange(temp_storage)
  {}

  //! @}  end member group
  //! @name Data movement
  //! @{

  /**
   * @brief Transposes data items from <em>blocked</em> arrangement to
   *        <em>striped</em> arrangement.
   *
   * @par
   * @smemwarpreuse
   *
   * @par Snippet
   * The code snippet below illustrates the conversion from a "blocked" to a
   * "striped" arrangement of 64 integer items partitioned across 16 threads
   * where each thread owns 4 items.
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_exchange.cuh>
   *
   * __global__ void ExampleKernel(int *d_data, ...)
   * {
   *     constexpr int warp_threads = 16;
   *     constexpr int block_threads = 256;
   *     constexpr int items_per_thread = 4;
   *     constexpr int warps_per_block = block_threads / warp_threads;
   *     const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
   *
   *     // Specialize WarpExchange for a virtual warp of 16 threads owning 4 integer items each
   *     using WarpExchangeT = cub::WarpExchange<int, items_per_thread, warp_threads>;
   *
   *     // Allocate shared memory for WarpExchange
   *     __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];
   *
   *     // Obtain a segment of consecutive items that are blocked across threads
   *     int thread_data[items_per_thread];
   *     // ...
   *
   *     // Collectively exchange data into a striped arrangement across threads
   *     WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);
   * @endcode
   * @par
   * Suppose the set of striped input @p thread_data across the block of threads
   * is <tt>{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [60,61,62,63] }</tt>.
   * The corresponding output @p thread_data in those threads will be
   * <tt>{ [0,16,32,48], [1,17,33,49], ..., [15, 32, 47, 63] }</tt>.
   *
   * @param[in] input_items
   *   Items to exchange, converting between <em>blocked</em> and
   *   <em>striped</em> arrangements.
   *
   * @param[out] output_items
   *   Items from exchange, converting between <em>striped</em> and
   *   <em>blocked</em> arrangements. May be aliased to @p input_items.
   */
  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  BlockedToStriped(const InputT (&input_items)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
  {
    InternalWarpExchange::BlockedToStriped(input_items, output_items);
  }

  /**
   * @brief Transposes data items from <em>striped</em> arrangement to
   *        <em>blocked</em> arrangement.
   *
   * @par
   * @smemwarpreuse
   *
   * @par Snippet
   * The code snippet below illustrates the conversion from a "striped" to a
   * "blocked" arrangement of 64 integer items partitioned across 16 threads
   * where each thread owns 4 items.
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_exchange.cuh>
   *
   * __global__ void ExampleKernel(int *d_data, ...)
   * {
   *     constexpr int warp_threads = 16;
   *     constexpr int block_threads = 256;
   *     constexpr int items_per_thread = 4;
   *     constexpr int warps_per_block = block_threads / warp_threads;
   *     const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
   *
   *     // Specialize WarpExchange for a virtual warp of 16 threads owning 4 integer items each
   *     using WarpExchangeT = cub::WarpExchange<int, items_per_thread, warp_threads>;
   *
   *     // Allocate shared memory for WarpExchange
   *     __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];
   *
   *     // Load a tile of data striped across threads
   *     int thread_data[items_per_thread];
   *     // ...
   *
   *     // Collectively exchange data into a blocked arrangement across threads
   *     WarpExchangeT(temp_storage[warp_id]).StripedToBlocked(thread_data, thread_data);
   * @endcode
   * @par
   * Suppose the set of striped input @p thread_data across the block of threads
   * is <tt>{ [0,16,32,48], [1,17,33,49], ..., [15, 32, 47, 63] }</tt>.
   * The corresponding output @p thread_data in those threads will be
   * <tt>{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [60,61,62,63] }</tt>.
   *
   * @param[in] input_items
   *   Items to exchange
   *
   * @param[out] output_items
   *   Items from exchange. May be aliased to @p input_items.
   */
  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  StripedToBlocked(const InputT (&input_items)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
  {
    InternalWarpExchange::StripedToBlocked(input_items, output_items);
  }

  /**
   * @brief Exchanges valid data items annotated by rank
   *        into <em>striped</em> arrangement.
   *
   * @par
   * @smemwarpreuse
   *
   * @par Snippet
   * The code snippet below illustrates the conversion from a "scatter" to a
   * "striped" arrangement of 64 integer items partitioned across 16 threads
   * where each thread owns 4 items.
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_exchange.cuh>
   *
   * __global__ void ExampleKernel(int *d_data, ...)
   * {
   *     constexpr int warp_threads = 16;
   *     constexpr int block_threads = 256;
   *     constexpr int items_per_thread = 4;
   *     constexpr int warps_per_block = block_threads / warp_threads;
   *     const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
   *
   *     // Specialize WarpExchange for a virtual warp of 16 threads owning 4 integer items each
   *     using WarpExchangeT = cub::WarpExchange<int, items_per_thread, warp_threads>;
   *
   *     // Allocate shared memory for WarpExchange
   *     __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];
   *
   *     // Obtain a segment of consecutive items that are blocked across threads
   *     int thread_data[items_per_thread];
   *     int thread_ranks[items_per_thread];
   *     // ...
   *
   *     // Collectively exchange data into a striped arrangement across threads
   *     WarpExchangeT(temp_storage[warp_id]).ScatterToStriped(
   *       thread_data, thread_ranks);
   * @endcode
   * @par
   * Suppose the set of input @p thread_data across the block of threads
   * is `{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }`, and the set of
   * @p thread_ranks is `{ [63,62,61,60], ..., [7,6,5,4], [3,2,1,0] }`. The
   * corresponding output @p thread_data in those threads will be
   * `{ [63, 47, 31, 15], [62, 46, 30, 14], ..., [48, 32, 16, 0] }`.
   *
   * @tparam OffsetT <b>[inferred]</b> Signed integer type for local offsets
   *
   * @param[in,out] items Items to exchange
   * @param[in] ranks Corresponding scatter ranks
   */
  template <typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScatterToStriped(InputT (&items)[ITEMS_PER_THREAD], OffsetT (&ranks)[ITEMS_PER_THREAD])
  {
    InternalWarpExchange::ScatterToStriped(items, ranks);
  }

  /**
   * @brief Exchanges valid data items annotated by rank
   *        into <em>striped</em> arrangement.
   *
   * @par
   * @smemwarpreuse
   *
   * @par Snippet
   * The code snippet below illustrates the conversion from a "scatter" to a
   * "striped" arrangement of 64 integer items partitioned across 16 threads
   * where each thread owns 4 items.
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_exchange.cuh>
   *
   * __global__ void ExampleKernel(int *d_data, ...)
   * {
   *     constexpr int warp_threads = 16;
   *     constexpr int block_threads = 256;
   *     constexpr int items_per_thread = 4;
   *     constexpr int warps_per_block = block_threads / warp_threads;
   *     const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
   *
   *     // Specialize WarpExchange for a virtual warp of 16 threads owning 4 integer items each
   *     using WarpExchangeT = cub::WarpExchange<int, items_per_thread, warp_threads>;
   *
   *     // Allocate shared memory for WarpExchange
   *     __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];
   *
   *     // Obtain a segment of consecutive items that are blocked across threads
   *     int thread_input[items_per_thread];
   *     int thread_ranks[items_per_thread];
   *     // ...
   *
   *     // Collectively exchange data into a striped arrangement across threads
   *     int thread_output[items_per_thread];
   *     WarpExchangeT(temp_storage[warp_id]).ScatterToStriped(
   *       thread_input, thread_output, thread_ranks);
   * @endcode
   * @par
   * Suppose the set of input @p thread_input across the block of threads
   * is `{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }`, and the set of
   * @p thread_ranks is `{ [63,62,61,60], ..., [7,6,5,4], [3,2,1,0] }`. The
   * corresponding @p thread_output in those threads will be
   * `{ [63, 47, 31, 15], [62, 46, 30, 14], ..., [48, 32, 16, 0] }`.
   *
   * @tparam OffsetT <b>[inferred]</b> Signed integer type for local offsets
   *
   * @param[in] input_items
   *   Items to exchange
   *
   * @param[out] output_items
   *   Items from exchange. May be aliased to @p input_items.
   *
   * @param[in] ranks
   *   Corresponding scatter ranks
   */
  template <typename OutputT, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(
    const InputT (&input_items)[ITEMS_PER_THREAD],
    OutputT (&output_items)[ITEMS_PER_THREAD],
    OffsetT (&ranks)[ITEMS_PER_THREAD])
  {
    InternalWarpExchange::ScatterToStriped(input_items, output_items, ranks);
  }

  //@}  end member group
};

CUB_NAMESPACE_END
