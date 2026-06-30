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
//! Operations for reading linear tiles of data into the CUDA warp.

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
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_exchange.cuh>

#include <cuda/ptx>

CUB_NAMESPACE_BEGIN

//! @rst
//! ``cub::WarpLoadAlgorithm`` enumerates alternative algorithms for :cpp:struct:`cub::WarpLoad` to
//! read a linear segment of data from memory into a CUDA warp.
//! @endrst
enum WarpLoadAlgorithm
{
  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) decreases as the
  //! access stride between threads increases (i.e., the number items per thread).
  //! @endrst
  WARP_LOAD_DIRECT,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) doesn't depend on
  //! the number of items per thread.
  //! @endrst
  WARP_LOAD_STRIPED,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read from memory using
  //! CUDA's built-in vectorized loads as a coalescing optimization.
  //! For example, ``ld.global.v4.s32`` instructions will be generated when ``T = int`` and
  //! ``ITEMS_PER_THREAD % 4 == 0``.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high until the the
  //!   access stride between threads (i.e., the number items per thread) exceeds the
  //!   maximum vector load width (typically 4 items or 64B, whichever is lower).
  //! - The following conditions will prevent vectorization and loading will fall
  //!   back to cub::WARP_LOAD_DIRECT:
  //!
  //!   - ``ITEMS_PER_THREAD`` is odd
  //!   - The ``InputIteratorT`` is not a simple pointer type
  //!   - The block input offset is not quadword-aligned
  //!   - The data type ``T`` is not a built-in primitive or CUDA vector type
  //!     (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
  //! @endrst
  WARP_LOAD_VECTORIZE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is read efficiently from
  //! memory and then locally transposed into a
  //! :ref:`blocked arrangement <flexible-data-arrangement>`.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high
  //!   regardless of items loaded per thread.
  //! - The local reordering incurs slightly longer latencies and throughput than the direct
  //!   ``cub::WARP_LOAD_DIRECT`` and ``cub::WARP_LOAD_VECTORIZE`` alternatives.
  //! @endrst
  WARP_LOAD_TRANSPOSE
};

//! @rst
//! The WarpLoad class provides :ref:`collective <collective-primitives>` data movement methods for
//! loading a linear segment of items from memory into a
//! :ref:`blocked arrangement <flexible-data-arrangement>` across a CUDA thread warp.
//!
//! Overview
//! ++++++++++++++++
//!
//! - The WarpLoad class provides a single data movement abstraction that can be
//!   specialized to implement different cub::WarpLoadAlgorithm strategies. This
//!   facilitates different performance policies for different architectures, data
//!   types, granularity sizes, etc.
//! - WarpLoad can be optionally specialized by different data movement strategies:
//!
//!   #. :cpp:enumerator:`cub::WARP_LOAD_DIRECT`:
//!      a :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from
//!      memory.
//!   #. :cpp:enumerator:`cub::WARP_LOAD_STRIPED`:
//!      a :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from
//!      memory.
//!   #. :cpp:enumerator:`cub::WARP_LOAD_VECTORIZE`:
//!      a :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from
//!      memory using CUDA's built-in vectorized loads as a coalescing optimization.
//!   #. :cpp:enumerator:`cub::WARP_LOAD_TRANSPOSE`:
//!      a :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from
//!      memory and is then locally transposed into a
//!      :ref:`blocked arrangement <flexible-data-arrangement>`.
//!
//! A Simple Example
//! ++++++++++++++++
//!
//! The code snippet below illustrates the loading of a linear segment of 64
//! integers into a "blocked" arrangement across 16 threads where each thread
//! owns 4 consecutive items. The load is specialized for ``WARP_LOAD_TRANSPOSE``,
//! meaning memory references are efficiently coalesced using a warp-striped access
//! pattern (after which items are locally reordered among threads).
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_load.cuh>
//!
//!    __global__ void ExampleKernel(int *d_data, ...)
//!    {
//!        constexpr int warp_threads = 16;
//!        constexpr int block_threads = 256;
//!        constexpr int items_per_thread = 4;
//!
//!        // Specialize WarpLoad for a warp of 16 threads owning 4 integer items each
//!        using WarpLoadT = WarpLoad<int,
//!                                   items_per_thread,
//!                                   cub::WARP_LOAD_TRANSPOSE,
//!                                   warp_threads>;
//!
//!        constexpr int warps_in_block = block_threads / warp_threads;
//!        constexpr int tile_size = items_per_thread * warp_threads;
//!        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
//!
//!        // Allocate shared memory for WarpLoad
//!        __shared__ typename WarpLoadT::TempStorage temp_storage[warps_in_block];
//!
//!        // Load a segment of consecutive items that are blocked across threads
//!        int thread_data[items_per_thread];
//!        WarpLoadT(temp_storage[warp_id]).Load(d_data + warp_id * tile_size,
//!                                           thread_data);
//!
//! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...``.
//! The set of ``thread_data`` across the first logical warp of threads in those
//! threads will be: ``{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }``.
//! @endrst
//!
//! @tparam InputT
//!   The data type to read into (which must be convertible from the input
//!   iterator's value type).
//!
//! @tparam ITEMS_PER_THREAD
//!   The number of consecutive items partitioned onto each thread.
//!
//! @tparam ALGORITHM
//!   <b>[optional]</b> cub::WarpLoadAlgorithm tuning policy.
//!   default: cub::WARP_LOAD_DIRECT.
//!
//! @tparam LOGICAL_WARP_THREADS
//!   <b>[optional]</b> The number of threads per "logical" warp (may be less
//!   than the number of hardware warp threads). Default is the warp size of the
//!   targeted CUDA compute-capability (e.g., 32 threads for SM86). Must be a
//!   power of two.
//!
template <typename InputT,
          int ITEMS_PER_THREAD,
          WarpLoadAlgorithm ALGORITHM = WARP_LOAD_DIRECT,
          int LOGICAL_WARP_THREADS    = detail::warp_threads>
class WarpLoad
{
  static constexpr bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == detail::warp_threads;

  static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE, "LOGICAL_WARP_THREADS must be a power of two");

private:
  /*****************************************************************************
   * Algorithmic variants
   ****************************************************************************/

  /// Load helper
  template <WarpLoadAlgorithm _POLICY, int DUMMY>
  struct LoadInternal;

  template <int DUMMY>
  struct LoadInternal<WARP_LOAD_DIRECT, DUMMY>
  {
    using TempStorage = NullType;

    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectBlocked(linear_tid, block_itr, items);
    }

    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
    }

    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default);
    }
  };

  template <int DUMMY>
  struct LoadInternal<WARP_LOAD_STRIPED, DUMMY>
  {
    using TempStorage = NullType;

    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
    }

    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items, valid_items);
    }

    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items, valid_items, oob_default);
    }
  };

  template <int DUMMY>
  struct LoadInternal<WARP_LOAD_VECTORIZE, DUMMY>
  {
    using TempStorage = NullType;

    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputT* block_ptr, InputT (&items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, block_ptr, items);
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(const InputT* block_ptr, InputT (&items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, block_ptr, items);
    }

    template <CacheLoadModifier MODIFIER, typename ValueType, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(CacheModifiedInputIterator<MODIFIER, ValueType, OffsetT> block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<MODIFIER>(linear_tid, block_itr.ptr, items);
    }

    template <typename _InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(_InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectBlocked(linear_tid, block_itr, items);
    }

    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
    }

    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default);
    }
  };

  template <int DUMMY>
  struct LoadInternal<WARP_LOAD_TRANSPOSE, DUMMY>
  {
    using WarpExchangeT = WarpExchange<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>;

    struct _TempStorage : WarpExchangeT::TempStorage
    {};

    struct TempStorage : Uninitialized<_TempStorage>
    {};

    _TempStorage& temp_storage;

    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
      WarpExchangeT(temp_storage).StripedToBlocked(items, items);
    }

    template <typename InputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items, valid_items);
      WarpExchangeT(temp_storage).StripedToBlocked(items, items);
    }

    template <typename InputIteratorT, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items, valid_items, oob_default);
      WarpExchangeT(temp_storage).StripedToBlocked(items, items);
    }
  };

  /*****************************************************************************
   * Type definitions
   ****************************************************************************/

  /// Internal load implementation to use
  using InternalLoad = LoadInternal<ALGORITHM, 0>;

  /// Shared memory storage layout type
  using _TempStorage = typename InternalLoad::TempStorage;

  /*****************************************************************************
   * Utility methods
   ****************************************************************************/

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  /*****************************************************************************
   * Thread fields
   ****************************************************************************/

  /// Thread reference to shared storage
  _TempStorage& temp_storage;

  /// Linear thread-id
  int linear_tid;

public:
  /// @smemstorage{WarpLoad}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of
  //!        shared memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpLoad()
      : temp_storage(PrivateStorage())
      , linear_tid(
          IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : (::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS))
  {}

  //! @brief Collective constructor using the specified memory allocation as
  //!        temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpLoad(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(
          IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : (::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS))
  {}

  //! @} end member group
  //! @name Data movement
  //! @{

  //! @rst
  //! Load a linear segment of items from memory.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, ...)
  //!    {
  //!        constexpr int warp_threads = 16;
  //!        constexpr int block_threads = 256;
  //!        constexpr int items_per_thread = 4;
  //!
  //!        // Specialize WarpLoad for a warp of 16 threads owning 4 integer items each
  //!        using WarpLoadT = WarpLoad<int,
  //!                                   items_per_thread,
  //!                                   cub::WARP_LOAD_TRANSPOSE,
  //!                                   warp_threads>;
  //!
  //!        constexpr int warps_in_block = block_threads / warp_threads;
  //!        constexpr int tile_size = items_per_thread * warp_threads;
  //!        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
  //!
  //!        // Allocate shared memory for WarpLoad
  //!        __shared__ typename WarpLoadT::TempStorage temp_storage[warps_in_block];
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[items_per_thread];
  //!        WarpLoadT(temp_storage[warp_id]).Load(d_data + warp_id * tile_size,
  //!                                              thread_data);
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...``,
  //! The set of ``thread_data`` across the first logical warp of threads in those
  //! threads will be: ``{ [0,1,2,3], [4,5,6,7], ..., [60,61,62,63] }``.
  //! @endrst
  //!
  //! @param[in] block_itr The thread block's base input iterator for loading from
  //! @param[out] items Data to load
  template <typename InputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD])
  {
    InternalLoad(temp_storage, linear_tid).Load(block_itr, items);
  }

  //! @rst
  //! Load a linear segment of items from memory, guarded by range.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int valid_items, ...)
  //!    {
  //!        constexpr int warp_threads = 16;
  //!        constexpr int block_threads = 256;
  //!        constexpr int items_per_thread = 4;
  //!
  //!        // Specialize WarpLoad for a warp of 16 threads owning 4 integer items each
  //!        using WarpLoadT = WarpLoad<int,
  //!                                   items_per_thread,
  //!                                   cub::WARP_LOAD_TRANSPOSE,
  //!                                   warp_threads>;
  //!
  //!        constexpr int warps_in_block = block_threads / warp_threads;
  //!        constexpr int tile_size = items_per_thread * warp_threads;
  //!        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
  //!
  //!        // Allocate shared memory for WarpLoad
  //!        __shared__ typename WarpLoadT::TempStorage temp_storage[warps_in_block];
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[items_per_thread];
  //!        WarpLoadT(temp_storage[warp_id]).Load(d_data + warp_id * tile_size,
  //!                                              thread_data,
  //!                                              valid_items);
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...`` and ``valid_items`` is ``5``.
  //! The set of ``thread_data`` across the first logical warp of threads in those threads will be:
  //! ``{ [0,1,2,3], [4,?,?,?], ..., [?,?,?,?] }`` with only the first two threads being unmasked to
  //! load portions of valid data (and other items remaining unassigned).
  //! @endrst
  //!
  //! @param[in] block_itr The thread block's base input iterator for loading from
  //! @param[out] items Data to load
  //! @param[in] valid_items Number of valid items to load
  template <typename InputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items)
  {
    InternalLoad(temp_storage, linear_tid).Load(block_itr, items, valid_items);
  }

  //! @rst
  //! Load a linear segment of items from memory, guarded by range.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/warp/warp_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int valid_items, ...)
  //!    {
  //!        constexpr int warp_threads = 16;
  //!        constexpr int block_threads = 256;
  //!        constexpr int items_per_thread = 4;
  //!
  //!        // Specialize WarpLoad for a warp of 16 threads owning 4 integer items each
  //!        using WarpLoadT = WarpLoad<int,
  //!                                   items_per_thread,
  //!                                   cub::WARP_LOAD_TRANSPOSE,
  //!                                   warp_threads>;
  //!
  //!        constexpr int warps_in_block = block_threads / warp_threads;
  //!        constexpr int tile_size = items_per_thread * warp_threads;
  //!        const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
  //!
  //!        // Allocate shared memory for WarpLoad
  //!        __shared__ typename WarpLoadT::TempStorage temp_storage[warps_in_block];
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[items_per_thread];
  //!        WarpLoadT(temp_storage[warp_id]).Load(d_data + warp_id * tile_size,
  //!                                              thread_data,
  //!                                              valid_items,
  //!                                              -1);
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...``, ``valid_items`` is ``5``, and the
  //! out-of-bounds default is ``-1``. The set of ``thread_data`` across the first logical warp of
  //! threads in those threads will be: ``{ [0,1,2,3], [4,-1,-1,-1], ..., [-1,-1,-1,-1] }`` with
  //! only the first two threads being unmasked to load portions of valid data (and other items
  //! are assigned ``-1``).
  //! @endrst
  //!
  //! @param[in] block_itr The thread block's base input iterator for loading from
  //! @param[out] items Data to load
  //! @param[in] valid_items Number of valid items to load
  //! @param[in] oob_default Default value to assign out-of-bound items
  template <typename InputIteratorT, typename DefaultT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default)
  {
    InternalLoad(temp_storage, linear_tid).Load(block_itr, items, valid_items, oob_default);
  }

  //! @} end member group
};

CUB_NAMESPACE_END
