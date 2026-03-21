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
 * Operations for writing linear segments of data from the CUDA warp
 */

#pragma once
#pragma clang system_header


#include <iterator>
#include <type_traits>

#include <cub/block/block_store.cuh>
#include <cub/config.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_exchange.cuh>


CUB_NAMESPACE_BEGIN


/**
 * @brief cub::WarpStoreAlgorithm enumerates alternative algorithms for
 *        cub::WarpStore to write a blocked arrangement of items across a CUDA
 *        warp to a linear segment of memory.
 */
enum WarpStoreAlgorithm
{
  /**
   * @par Overview
   * A [<em>blocked arrangement</em>](index.html#sec5sec3) of data is written
   * directly to memory.
   *
   * @par Performance Considerations
   * The utilization of memory transactions (coalescing) decreases as the
   * access stride between threads increases (i.e., the number items per thread).
   */
  WARP_STORE_DIRECT,

  /**
   * @par Overview
   * A [<em>striped arrangement</em>](index.html#sec5sec3) of data is written
   * directly to memory.
   *
   * @par Performance Considerations
   * The utilization of memory transactions (coalescing) remains high regardless
   * of items written per thread.
   */
  WARP_STORE_STRIPED,

  /**
   * @par Overview
   *
   * A [<em>blocked arrangement</em>](index.html#sec5sec3) of data is written
   * directly to memory using CUDA's built-in vectorized stores as a coalescing
   * optimization. For example, <tt>st.global.v4.s32</tt> instructions will be
   * generated when @p T = @p int and @p ITEMS_PER_THREAD % 4 == 0.
   *
   * @par Performance Considerations
   * - The utilization of memory transactions (coalescing) remains high until
   *   the the access stride between threads (i.e., the number items per thread)
   *   exceeds the maximum vector store width (typically 4 items or 64B,
   *   whichever is lower).
   * - The following conditions will prevent vectorization and writing will fall
   *   back to cub::WARP_STORE_DIRECT:
   *   - @p ITEMS_PER_THREAD is odd
   *   - The @p OutputIteratorT is not a simple pointer type
   *   - The block output offset is not quadword-aligned
   *   - The data type @p T is not a built-in primitive or CUDA vector type
   *     (e.g., @p short, @p int2, @p double, @p float2, etc.)
   */
  WARP_STORE_VECTORIZE,

  /**
   * @par Overview
   * A [<em>blocked arrangement</em>](index.html#sec5sec3) is locally
   * transposed and then efficiently written to memory as a
   * [<em>striped arrangement</em>](index.html#sec5sec3).
   *
   * @par Performance Considerations
   * - The utilization of memory transactions (coalescing) remains high
   *   regardless of items written per thread.
   * - The local reordering incurs slightly longer latencies and throughput than the
   *   direct cub::WARP_STORE_DIRECT and cub::WARP_STORE_VECTORIZE alternatives.
   */
  WARP_STORE_TRANSPOSE
};


/**
 * @brief The WarpStore class provides [<em>collective</em>](index.html#sec0)
 *        data movement methods for writing a [<em>blocked arrangement</em>](index.html#sec5sec3)
 *        of items partitioned across a CUDA warp to a linear segment of memory.
 * @ingroup WarpModule
 * @ingroup UtilIo
 *
 * @tparam T
 *   The type of data to be written.
 *
 * @tparam ITEMS_PER_THREAD
 *   The number of consecutive items partitioned onto each thread.
 *
 * @tparam ALGORITHM
 *   <b>[optional]</b> cub::WarpStoreAlgorithm tuning policy enumeration.
 *   default: cub::WARP_STORE_DIRECT.
 *
 * @tparam LOGICAL_WARP_THREADS
 *   <b>[optional]</b> The number of threads per "logical" warp (may be less
 *   than the number of hardware warp threads). Default is the warp size of the
 *   targeted CUDA compute-capability (e.g., 32 threads for SM86). Must be a
 *   power of two.
 *
 * @tparam LEGACY_PTX_ARCH
 *   Unused.
 *
 * @par Overview
 * - The WarpStore class provides a single data movement abstraction that can be
 *   specialized to implement different cub::WarpStoreAlgorithm strategies. This
 *   facilitates different performance policies for different architectures,
 *   data types, granularity sizes, etc.
 * - WarpStore can be optionally specialized by different data movement strategies:
 *   -# <b>cub::WARP_STORE_DIRECT</b>. A [<em>blocked arrangement</em>](index.html#sec5sec3)
 *      of data is written directly to memory. [More...](@ref cub::WarpStoreAlgorithm)
 *   -# <b>cub::WARP_STORE_STRIPED</b>. A [<em>striped arrangement</em>](index.html#sec5sec3)
 *      of data is written directly to memory. [More...](@ref cub::WarpStoreAlgorithm)
 *   -# <b>cub::WARP_STORE_VECTORIZE</b>. A [<em>blocked arrangement</em>](index.html#sec5sec3)
 *      of data is written directly to memory using CUDA's built-in vectorized
 *      stores as a coalescing optimization. [More...](@ref cub::WarpStoreAlgorithm)
 *   -# <b>cub::WARP_STORE_TRANSPOSE</b>. A [<em>blocked arrangement</em>](index.html#sec5sec3)
 *      is locally transposed into a [<em>striped arrangement</em>](index.html#sec5sec3)
 *      which is then written to memory. [More...](@ref cub::WarpStoreAlgorithm)
 * - \rowmajor
 *
 * @par A Simple Example
 * @par
 * The code snippet below illustrates the storing of a "blocked" arrangement
 * of 64 integers across 16 threads (where each thread owns 4 consecutive items)
 * into a linear segment of memory. The store is specialized for
 * @p WARP_STORE_TRANSPOSE, meaning items are locally reordered among threads so
 * that memory references will be efficiently coalesced using a warp-striped
 * access pattern.
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_store.cuh>
 *
 * __global__ void ExampleKernel(int *d_data, ...)
 * {
 *     constexpr int warp_threads = 16;
 *     constexpr int block_threads = 256;
 *     constexpr int items_per_thread = 4;
 *
 *     // Specialize WarpStore for a virtual warp of 16 threads owning 4 integer items each
 *     using WarpStoreT = WarpStore<int,
 *                                  items_per_thread,
 *                                  cub::WARP_STORE_TRANSPOSE,
 *                                  warp_threads>;
 *
 *     constexpr int warps_in_block = block_threads / warp_threads;
 *     constexpr int tile_size = items_per_thread * warp_threads;
 *     const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
 *
 *     // Allocate shared memory for WarpStore
 *     __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];
 *
 *     // Obtain a segment of consecutive items that are blocked across threads
 *     int thread_data[4];
 *     ...
 *
 *     // Store items to linear memory
 *     WarpStoreT(temp_storage[warp_id]).Store(d_data + warp_id * tile_size, thread_data);
 * @endcode
 * @par
 * Suppose the set of @p thread_data across the warp threads is
 * <tt>{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }</tt>.
 * The output @p d_data will be <tt>0, 1, 2, 3, 4, 5, ...</tt>.
 */
template <typename           T,
          int                ITEMS_PER_THREAD,
          WarpStoreAlgorithm ALGORITHM            = WARP_STORE_DIRECT,
          int                LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS,
          int                LEGACY_PTX_ARCH      = 0>
class WarpStore
{
  static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE,
                "LOGICAL_WARP_THREADS must be a power of two");

  constexpr static bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == CUB_WARP_THREADS(0);

private:

  /// Store helper
  template <WarpStoreAlgorithm _POLICY, int DUMMY>
  struct StoreInternal;

  template <int DUMMY>
  struct StoreInternal<WARP_STORE_DIRECT, DUMMY>
  {
    typedef NullType TempStorage;

    int linear_tid;

    __device__ __forceinline__ StoreInternal(TempStorage &/*temp_storage*/,
                                             int linear_tid)
      : linear_tid(linear_tid)
    {}

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlocked(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD],
                                          int valid_items)
    {
      StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
    }
  };


  template <int DUMMY>
  struct StoreInternal<WARP_STORE_STRIPED, DUMMY>
  {
    typedef NullType TempStorage;

    int linear_tid;

    __device__ __forceinline__ StoreInternal(TempStorage & /*temp_storage*/,
                                             int linear_tid)
        : linear_tid(linear_tid)
    {}

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD],
                                          int valid_items)
    {
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid,
                                               block_itr,
                                               items,
                                               valid_items);
    }
  };


  template <int DUMMY>
  struct StoreInternal<WARP_STORE_VECTORIZE, DUMMY>
  {
    typedef NullType TempStorage;

    int linear_tid;

    __device__ __forceinline__ StoreInternal(TempStorage & /*temp_storage*/,
                                             int linear_tid)
        : linear_tid(linear_tid)
    {}

    __device__ __forceinline__ void Store(T *block_ptr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlockedVectorized(linear_tid, block_ptr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlocked(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD],
                                          int valid_items)
    {
      StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
    }
  };


  template <int DUMMY>
  struct StoreInternal<WARP_STORE_TRANSPOSE, DUMMY>
  {
    using WarpExchangeT =
      WarpExchange<T, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>;

    struct _TempStorage : WarpExchangeT::TempStorage
    {};

    struct TempStorage : Uninitialized<_TempStorage> {};

    _TempStorage &temp_storage;

    int linear_tid;

    __device__ __forceinline__ StoreInternal(TempStorage &temp_storage,
                                             int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      WarpExchangeT(temp_storage).BlockedToStriped(items, items);
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD],
                                          int valid_items)
    {
      WarpExchangeT(temp_storage).BlockedToStriped(items, items);
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid,
                                               block_itr,
                                               items,
                                               valid_items);
    }
  };


  /// Internal load implementation to use
  using InternalStore = StoreInternal<ALGORITHM, 0>;

  /// Shared memory storage layout type
  using _TempStorage = typename InternalStore::TempStorage;


  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }


  _TempStorage &temp_storage;

  int linear_tid;

public:

  struct TempStorage : Uninitialized<_TempStorage> {};

  /*************************************************************************//**
   * @name Collective constructors
   ****************************************************************************/
  //@{

  /**
   * @brief Collective constructor using a private static allocation of shared
   *        memory as temporary storage.
   */
  __device__ __forceinline__ WarpStore()
      : temp_storage(PrivateStorage())
      , linear_tid(IS_ARCH_WARP ? LaneId() : (LaneId() % LOGICAL_WARP_THREADS))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation as
   *        temporary storage.
   */
  __device__ __forceinline__ WarpStore(TempStorage &temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(IS_ARCH_WARP ? LaneId() : (LaneId() % LOGICAL_WARP_THREADS))
  {}

  //@}  end member group
  /*************************************************************************//**
   * @name Data movement
   ****************************************************************************/
  //@{

  /**
   * @brief Store items into a linear segment of memory.
   *
   * @par
   * \smemreuse
   *
   * @par Snippet
   * @par
   * The code snippet below illustrates the storing of a "blocked" arrangement
   * of 64 integers across 16 threads (where each thread owns 4 consecutive items)
   * into a linear segment of memory. The store is specialized for
   * @p WARP_STORE_TRANSPOSE, meaning items are locally reordered among threads so
   * that memory references will be efficiently coalesced using a warp-striped
   * access pattern.
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_store.cuh>
   *
   * __global__ void ExampleKernel(int *d_data, ...)
   * {
   *     constexpr int warp_threads = 16;
   *     constexpr int block_threads = 256;
   *     constexpr int items_per_thread = 4;
   *
   *     // Specialize WarpStore for a virtual warp of 16 threads owning 4 integer items each
   *     using WarpStoreT = WarpStore<int,
   *                                  items_per_thread,
   *                                  cub::WARP_STORE_TRANSPOSE,
   *                                  warp_threads>;
   *
   *     constexpr int warps_in_block = block_threads / warp_threads;
   *     constexpr int tile_size = items_per_thread * warp_threads;
   *     const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
   *
   *     // Allocate shared memory for WarpStore
   *     __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];
   *
   *     // Obtain a segment of consecutive items that are blocked across threads
   *     int thread_data[4];
   *     ...
   *
   *     // Store items to linear memory
   *     WarpStoreT(temp_storage[warp_id]).Store(d_data + warp_id * tile_size, thread_data);
   * @endcode
   * @par
   * Suppose the set of @p thread_data across the warp threads is
   * <tt>{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }</tt>.
   * The output @p d_data will be <tt>0, 1, 2, 3, 4, 5, ...</tt>.
   *
   * @param[out] block_itr The thread block's base output iterator for storing to
   * @param[in] items Data to store
   */
  template <typename OutputIteratorT>
  __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                        T (&items)[ITEMS_PER_THREAD])
  {
    InternalStore(temp_storage, linear_tid).Store(block_itr, items);
  }

  /**
   * @brief Store items into a linear segment of memory, guarded by range.
   *
   * @par
   * \smemreuse
   *
   * @par Snippet
   * @par
   * The code snippet below illustrates the storing of a "blocked" arrangement
   * of 64 integers across 16 threads (where each thread owns 4 consecutive items)
   * into a linear segment of memory. The store is specialized for
   * @p WARP_STORE_TRANSPOSE, meaning items are locally reordered among threads so
   * that memory references will be efficiently coalesced using a warp-striped
   * access pattern.
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_store.cuh>
   *
   * __global__ void ExampleKernel(int *d_data, int valid_items ...)
   * {
   *     constexpr int warp_threads = 16;
   *     constexpr int block_threads = 256;
   *     constexpr int items_per_thread = 4;
   *
   *     // Specialize WarpStore for a virtual warp of 16 threads owning 4 integer items each
   *     using WarpStoreT = WarpStore<int,
   *                                  items_per_thread,
   *                                  cub::WARP_STORE_TRANSPOSE,
   *                                  warp_threads>;
   *
   *     constexpr int warps_in_block = block_threads / warp_threads;
   *     constexpr int tile_size = items_per_thread * warp_threads;
   *     const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
   *
   *     // Allocate shared memory for WarpStore
   *     __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];
   *
   *     // Obtain a segment of consecutive items that are blocked across threads
   *     int thread_data[4];
   *     ...
   *
   *     // Store items to linear memory
   *     WarpStoreT(temp_storage[warp_id]).Store(
   *       d_data + warp_id * tile_size, thread_data, valid_items);
   * @endcode
   * @par
   * Suppose the set of @p thread_data across the warp threads is
   * <tt>{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }</tt> and @p valid_items
   * is @p 5.. The output @p d_data will be <tt>0, 1, 2, 3, 4, ?, ?, ...</tt>,
   * with only the first two threads being unmasked to store portions of valid
   * data.
   *
   * @param[out] block_itr The thread block's base output iterator for storing to
   * @param[in] items Data to store
   * @param[in] valid_items Number of valid items to write
   */
  template <typename OutputIteratorT>
  __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                        T (&items)[ITEMS_PER_THREAD],
                                        int valid_items)
  {
    InternalStore(temp_storage, linear_tid).Store(block_itr, items, valid_items);
  }

  //@}  end member group
};


CUB_NAMESPACE_END
