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

//! @file
//! Operations for writing linear segments of data from the CUDA warp

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_store.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_exchange.cuh>

#include <cuda/ptx>

CUB_NAMESPACE_BEGIN

//! @rst
//! ``cub::WarpStoreAlgorithm`` enumerates alternative algorithms for :cpp:struct:`cub::WarpStore`
//! to write a blocked arrangement of items across a CUDA warp to a linear segment of memory.
//! @endrst
enum WarpStoreAlgorithm
{
  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly
  //! to memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) decreases as the
  //! access stride between threads increases (i.e., the number items per thread).
  //! @endrst
  WARP_STORE_DIRECT,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is written
  //! directly to memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) remains high regardless
  //! of items written per thread.
  //! @endrst
  WARP_STORE_STRIPED,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written
  //! directly to memory using CUDA's built-in vectorized stores as a coalescing
  //! optimization. For example, ``st.global.v4.s32`` instructions will be
  //! generated when ``T = int`` and ``ITEMS_PER_THREAD % 4 == 0``.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! * The utilization of memory transactions (coalescing) remains high until
  //!   the the access stride between threads (i.e., the number items per thread)
  //!   exceeds the maximum vector store width (typically 4 items or 64B,
  //!   whichever is lower).
  //! * The following conditions will prevent vectorization and writing will fall
  //!   back to ``cub::WARP_STORE_DIRECT``:
  //!
  //!   * ``ITEMS_PER_THREAD`` is odd
  //!   * The ``OutputIteratorT`` is not a simple pointer type
  //!   * The block output offset is not quadword-aligned
  //!   * The data type ``T`` is not a built-in primitive or CUDA vector type
  //!     (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
  //!
  //! @endrst
  WARP_STORE_VECTORIZE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` is locally
  //! transposed and then efficiently written to memory as a
  //! :ref:`striped arrangement <flexible-data-arrangement>`.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! * The utilization of memory transactions (coalescing) remains high
  //!   regardless of items written per thread.
  //! * The local reordering incurs slightly longer latencies and throughput than the
  //!   direct ``cub::WARP_STORE_DIRECT`` and ``cub::WARP_STORE_VECTORIZE`` alternatives.
  //!
  //! @endrst
  WARP_STORE_TRANSPOSE
};

//! @rst
//! The WarpStore class provides :ref:`collective <collective-primitives>`
//! data movement methods for writing a :ref:`blocked arrangement <flexible-data-arrangement>`
//! of items partitioned across a CUDA warp to a linear segment of memory.
//!
//! Overview
//! ++++++++++++++++
//!
//! * The WarpStore class provides a single data movement abstraction that can be
//!   specialized to implement different cub::WarpStoreAlgorithm strategies. This
//!   facilitates different performance policies for different architectures,
//!   data types, granularity sizes, etc.
//! * WarpStore can be optionally specialized by different data movement strategies:
//!
//!   #. :cpp:enumerator:`cub::WARP_STORE_DIRECT`:
//!      a :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly to
//!      memory.
//!   #. :cpp:enumerator:`cub::WARP_STORE_STRIPED`:
//!      a :ref:`striped arrangement <flexible-data-arrangement>` of data is written directly to
//!      memory.
//!   #. :cpp:enumerator:`cub::WARP_STORE_VECTORIZE`:
//!      a :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly to
//!      memory using CUDA's built-in vectorized stores as a coalescing optimization.
//!   #. :cpp:enumerator:`cub::WARP_STORE_TRANSPOSE`:
//!      a :ref:`blocked arrangement <flexible-data-arrangement>` is locally transposed into a
//!      :ref:`striped arrangement <flexible-data-arrangement>` which is then written to memory.
//!
//! * @rowmajor
//!
//! A Simple Example
//! ++++++++++++++++
//!
//! The code snippet below illustrates the storing of a "blocked" arrangement
//! of 64 integers across 16 threads (where each thread owns 4 consecutive items)
//! into a linear segment of memory. The store is specialized for
//! ``WARP_STORE_TRANSPOSE``, meaning items are locally reordered among threads so
//! that memory references will be efficiently coalesced using a warp-striped
//! access pattern.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_store.cuh>
//!
//!    __global__ void ExampleKernel(int *d_data, ...)
//!    {
//!        constexpr int warp_threads = 16;
//!        constexpr int block_threads = 256;
//!        constexpr int items_per_thread = 4;
//!
//!        // Specialize WarpStore for a virtual warp of 16 threads owning 4 integer items each
//!        using WarpStoreT = WarpStore<int,
//!                                     items_per_thread,
//!                                     cub::WARP_STORE_TRANSPOSE,
//!                                     warp_threads>;
//!
//!        constexpr int warps_in_block = block_threads / warp_threads;
//!        constexpr int tile_size = items_per_thread * warp_threads;
//!        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
//!
//!        // Allocate shared memory for WarpStore
//!        __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];
//!
//!        // Obtain a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        ...
//!
//!        // Store items to linear memory
//!        WarpStoreT(temp_storage[warp_id]).Store(d_data + warp_id * tile_size, thread_data);
//!
//! Suppose the set of ``thread_data`` across the warp threads is
//! ``{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }``.
//! The output ``d_data`` will be ``0, 1, 2, 3, 4, 5, ...``.
//! @endrst
//!
//! @tparam T
//!   The type of data to be written.
//!
//! @tparam ITEMS_PER_THREAD
//!   The number of consecutive items partitioned onto each thread.
//!
//! @tparam ALGORITHM
//!   <b>[optional]</b> cub::WarpStoreAlgorithm tuning policy enumeration.
//!   default: cub::WARP_STORE_DIRECT.
//!
//! @tparam LOGICAL_WARP_THREADS
//!   <b>[optional]</b> The number of threads per "logical" warp (may be less
//!   than the number of hardware warp threads). Default is the warp size of the
//!   targeted CUDA compute-capability (e.g., 32 threads for SM86). Must be a
//!   power of two.
//!
template <typename T,
          int ITEMS_PER_THREAD,
          WarpStoreAlgorithm ALGORITHM = WARP_STORE_DIRECT,
          int LOGICAL_WARP_THREADS     = detail::warp_threads>
class WarpStore
{
  static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE, "LOGICAL_WARP_THREADS must be a power of two");

  static constexpr bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == detail::warp_threads;

private:
  /// Store helper
  template <WarpStoreAlgorithm _POLICY, int DUMMY>
  struct StoreInternal;

  template <int DUMMY>
  struct StoreInternal<WARP_STORE_DIRECT, DUMMY>
  {
    using TempStorage = NullType;

    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlocked(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
    }
  };

  template <int DUMMY>
  struct StoreInternal<WARP_STORE_STRIPED, DUMMY>
  {
    using TempStorage = NullType;

    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items, valid_items);
    }
  };

  template <int DUMMY>
  struct StoreInternal<WARP_STORE_VECTORIZE, DUMMY>
  {
    using TempStorage = NullType;

    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(T* block_ptr, T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlockedVectorized(linear_tid, block_ptr, items);
    }

    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlocked(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
    }
  };

  template <int DUMMY>
  struct StoreInternal<WARP_STORE_TRANSPOSE, DUMMY>
  {
    using WarpExchangeT = WarpExchange<T, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>;

    struct _TempStorage : WarpExchangeT::TempStorage
    {};

    struct TempStorage : Uninitialized<_TempStorage>
    {};

    _TempStorage& temp_storage;

    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      WarpExchangeT(temp_storage).BlockedToStriped(items, items);
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      WarpExchangeT(temp_storage).BlockedToStriped(items, items);
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items, valid_items);
    }
  };

  /// Internal load implementation to use
  using InternalStore = StoreInternal<ALGORITHM, 0>;

  /// Shared memory storage layout type
  using _TempStorage = typename InternalStore::TempStorage;

  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  _TempStorage& temp_storage;

  int linear_tid;

public:
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared
  //!        memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpStore()
      : temp_storage(PrivateStorage())
      , linear_tid(
          IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : (::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS))
  {}

  //! @brief Collective constructor using the specified memory allocation as
  //!        temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpStore(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(
          IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : (::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS))
  {}

  //! @}  end member group
  //! @name Data movement
  //! @{

  //! @rst
  //! Store items into a linear segment of memory.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the storing of a "blocked" arrangement
  //! of 64 integers across 16 threads (where each thread owns 4 consecutive items)
  //! into a linear segment of memory. The store is specialized for
  //! ``WARP_STORE_TRANSPOSE``, meaning items are locally reordered among threads so
  //! that memory references will be efficiently coalesced using a warp-striped
  //! access pattern.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_store.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, ...)
  //!    {
  //!        constexpr int warp_threads = 16;
  //!        constexpr int block_threads = 256;
  //!        constexpr int items_per_thread = 4;
  //!
  //!        // Specialize WarpStore for a virtual warp of 16 threads owning 4 integer items each
  //!        using WarpStoreT = WarpStore<int,
  //!                                     items_per_thread,
  //!                                     cub::WARP_STORE_TRANSPOSE,
  //!                                     warp_threads>;
  //!
  //!        constexpr int warps_in_block = block_threads / warp_threads;
  //!        constexpr int tile_size = items_per_thread * warp_threads;
  //!        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
  //!
  //!        // Allocate shared memory for WarpStore
  //!        __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Store items to linear memory
  //!        WarpStoreT(temp_storage[warp_id]).Store(d_data + warp_id * tile_size, thread_data);
  //!
  //! Suppose the set of ``thread_data`` across the warp threads is
  //! ``{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }``.
  //! The output ``d_data`` will be ``0, 1, 2, 3, 4, 5, ...``.
  //! @endrst
  //!
  //! @param[out] block_itr The thread block's base output iterator for storing to
  //! @param[in] items Data to store
  template <typename OutputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
  {
    InternalStore(temp_storage, linear_tid).Store(block_itr, items);
  }

  //! @rst
  //! Store items into a linear segment of memory, guarded by range.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the storing of a "blocked" arrangement
  //! of 64 integers across 16 threads (where each thread owns 4 consecutive items)
  //! into a linear segment of memory. The store is specialized for
  //! ``WARP_STORE_TRANSPOSE``, meaning items are locally reordered among threads so
  //! that memory references will be efficiently coalesced using a warp-striped
  //! access pattern.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_store.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int valid_items ...)
  //!    {
  //!        constexpr int warp_threads = 16;
  //!        constexpr int block_threads = 256;
  //!        constexpr int items_per_thread = 4;
  //!
  //!        // Specialize WarpStore for a virtual warp of 16 threads owning 4 integer items each
  //!        using WarpStoreT = WarpStore<int,
  //!                                     items_per_thread,
  //!                                     cub::WARP_STORE_TRANSPOSE,
  //!                                     warp_threads>;
  //!
  //!        constexpr int warps_in_block = block_threads / warp_threads;
  //!        constexpr int tile_size = items_per_thread * warp_threads;
  //!        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
  //!
  //!        // Allocate shared memory for WarpStore
  //!        __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Store items to linear memory
  //!        WarpStoreT(temp_storage[warp_id]).Store(
  //!          d_data + warp_id * tile_size, thread_data, valid_items);
  //!
  //! Suppose the set of ``thread_data`` across the warp threads is
  //! ``{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }`` and ``valid_items``
  //! is ``5``. The output ``d_data`` will be ``0, 1, 2, 3, 4, ?, ?, ...``,
  //! with only the first two threads being unmasked to store portions of valid
  //! data.
  //! @endrst
  //!
  //! @param[out] block_itr The thread block's base output iterator for storing to
  //! @param[in] items Data to store
  //! @param[in] valid_items Number of valid items to write
  //!
  template <typename OutputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
  {
    InternalStore(temp_storage, linear_tid).Store(block_itr, items, valid_items);
  }

  //! @}  end member group
};

CUB_NAMESPACE_END
