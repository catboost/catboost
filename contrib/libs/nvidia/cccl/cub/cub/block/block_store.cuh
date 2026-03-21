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

//! @file
//! Operations for writing linear segments of data from the CUDA thread block

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_exchange.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

//! @name Blocked arrangement I/O (direct)
//! @{

//! @rst
//! Store a blocked arrangement of items across a thread block into a linear segment of items
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., ``(threadIdx.y * blockDim.x) + linear_tid`` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
template <typename T, int ITEMS_PER_THREAD, typename OutputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StoreDirectBlocked(int linear_tid, OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
{
  OutputIteratorT thread_itr = block_itr + (linear_tid * ITEMS_PER_THREAD);

  // Store directly in thread-blocked order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    thread_itr[ITEM] = items[ITEM];
  }
}

//! @rst
//! Store a blocked arrangement of items across a
//! thread block into a linear segment of items, guarded by range
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
//!
//! @param[in] valid_items
//!   Number of valid items to write
template <typename T, int ITEMS_PER_THREAD, typename OutputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StoreDirectBlocked(int linear_tid, OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
{
  OutputIteratorT thread_itr = block_itr + (linear_tid * ITEMS_PER_THREAD);

  // Store directly in thread-blocked order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    if (ITEM + (linear_tid * ITEMS_PER_THREAD) < valid_items)
    {
      thread_itr[ITEM] = items[ITEM];
    }
  }
}

//! @rst
//! Store a blocked arrangement of items across a
//! thread block into a linear segment of items.
//!
//! @blocked
//!
//! The output offset (``block_ptr + block_offset``) must be quad-item aligned,
//! which is the default starting offset returned by ``cudaMalloc()``
//!
//! The following conditions will prevent vectorization and storing will
//! fall back to cub::BLOCK_STORE_DIRECT:
//!
//!   - ``ITEMS_PER_THREAD`` is odd
//!   - The data type ``T`` is not a built-in primitive or CUDA vector type
//!     (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., ``(threadIdx.y * blockDim.x) + linear_tid`` for 2D thread blocks)
//!
//! @param[in] block_ptr
//!   Input pointer for storing from
//!
//! @param[in] items
//!   Data to store
template <typename T, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StoreDirectBlockedVectorized(int linear_tid, T* block_ptr, T (&items)[ITEMS_PER_THREAD])
{
  enum
  {
    // Maximum CUDA vector size is 4 elements
    MAX_VEC_SIZE = _CUDA_VSTD::min(4, ITEMS_PER_THREAD),

    // Vector size must be a power of two and an even divisor of the items per thread
    VEC_SIZE =
      ((((MAX_VEC_SIZE - 1) & MAX_VEC_SIZE) == 0) && ((ITEMS_PER_THREAD % MAX_VEC_SIZE) == 0)) ? MAX_VEC_SIZE : 1,

    VECTORS_PER_THREAD = ITEMS_PER_THREAD / VEC_SIZE,
  };

  // Vector type
  using Vector = typename CubVector<T, VEC_SIZE>::Type;

  // Add the alignment check to ensure the vectorized storing can proceed.
  if (reinterpret_cast<uintptr_t>(block_ptr) % (alignof(Vector)) == 0)
  {
    // Alias global pointer
    Vector* block_ptr_vectors = reinterpret_cast<Vector*>(const_cast<T*>(block_ptr));

    // Alias pointers (use "raw" array here which should get optimized away to prevent conservative PTXAS lmem spilling)
    Vector raw_vector[VECTORS_PER_THREAD];
    T* raw_items = reinterpret_cast<T*>(raw_vector);

    // Copy
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
      raw_items[ITEM] = items[ITEM];
    }

    // Direct-store using vector types
    StoreDirectBlocked(linear_tid, block_ptr_vectors, raw_vector);
  }
  else
  {
    // Direct-store using original type when the address is misaligned
    StoreDirectBlocked(linear_tid, block_ptr, items);
  }
}

//! @}  end member group
//! @name Striped arrangement I/O (direct)
//! @{

//! @rst
//! Store a striped arrangement of data across the thread block into a
//! linear segment of items.
//!
//! @striped
//!
//! @endrst
//!
//! @tparam BLOCK_THREADS
//!   The thread block size in threads
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
template <int BLOCK_THREADS, typename T, int ITEMS_PER_THREAD, typename OutputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StoreDirectStriped(int linear_tid, OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
{
  OutputIteratorT thread_itr = block_itr + linear_tid;

  // Store directly in striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    thread_itr[(ITEM * BLOCK_THREADS)] = items[ITEM];
  }
}

//! @rst
//! Store a striped arrangement of data across the thread block into
//! a linear segment of items, guarded by range
//!
//! @striped
//!
//! @endrst
//!
//! @tparam BLOCK_THREADS
//!   The thread block size in threads
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
//!
//! @param[in] valid_items
//!   Number of valid items to write
template <int BLOCK_THREADS, typename T, int ITEMS_PER_THREAD, typename OutputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StoreDirectStriped(int linear_tid, OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
{
  OutputIteratorT thread_itr = block_itr + linear_tid;

  // Store directly in striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    if ((ITEM * BLOCK_THREADS) + linear_tid < valid_items)
    {
      thread_itr[(ITEM * BLOCK_THREADS)] = items[ITEM];
    }
  }
}

//! @}  end member group
//! @name Warp-striped arrangement I/O (direct)
//! @{

//! @rst
//! Store a warp-striped arrangement of data across the
//! thread block into a linear segment of items.
//!
//! @warpstriped
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[out] items
//!   Data to load
template <typename T, int ITEMS_PER_THREAD, typename OutputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StoreDirectWarpStriped(int linear_tid, OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
{
  int tid         = linear_tid & (detail::warp_threads - 1);
  int wid         = linear_tid >> detail::log2_warp_threads;
  int warp_offset = wid * detail::warp_threads * ITEMS_PER_THREAD;

  OutputIteratorT thread_itr = block_itr + warp_offset + tid;

  // Store directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    thread_itr[(ITEM * detail::warp_threads)] = items[ITEM];
  }
}

//! @rst
//! Store a warp-striped arrangement of data across the thread block into a
//! linear segment of items, guarded by range
//!
//! @warpstriped
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to store.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam OutputIteratorT
//!   **[inferred]** The random-access iterator type for output @iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread
//!   (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D thread blocks)
//!
//! @param[in] block_itr
//!   The thread block's base output iterator for storing to
//!
//! @param[in] items
//!   Data to store
//!
//! @param[in] valid_items
//!   Number of valid items to write
template <typename T, int ITEMS_PER_THREAD, typename OutputIteratorT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
StoreDirectWarpStriped(int linear_tid, OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
{
  int tid         = linear_tid & (detail::warp_threads - 1);
  int wid         = linear_tid >> detail::log2_warp_threads;
  int warp_offset = wid * detail::warp_threads * ITEMS_PER_THREAD;

  OutputIteratorT thread_itr = block_itr + warp_offset + tid;

  // Store directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
  {
    if (warp_offset + tid + (ITEM * detail::warp_threads) < valid_items)
    {
      thread_itr[(ITEM * detail::warp_threads)] = items[ITEM];
    }
  }
}

//! @}  end member group

//-----------------------------------------------------------------------------
// Generic BlockStore abstraction
//-----------------------------------------------------------------------------

//! cub::BlockStoreAlgorithm enumerates alternative algorithms for cub::BlockStore to write a
//! blocked arrangement of items across a CUDA thread block to a linear segment of memory.
enum BlockStoreAlgorithm
{
  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly to memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) decreases as the
  //!   access stride between threads increases (i.e., the number items per thread).
  //!
  //! @endrst
  BLOCK_STORE_DIRECT,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is written directly to memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) remains high regardless
  //! of items written per thread.
  //!
  //! @endrst
  BLOCK_STORE_STRIPED,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly
  //! to memory using CUDA's built-in vectorized stores as a coalescing optimization.
  //! For example, ``st.global.v4.s32`` instructions will be generated
  //! when ``T = int`` and ``ITEMS_PER_THREAD % 4 == 0``.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high until the the
  //!   access stride between threads (i.e., the number items per thread) exceeds the
  //!   maximum vector store width (typically 4 items or 64B, whichever is lower).
  //! - The following conditions will prevent vectorization and writing will fall back to cub::BLOCK_STORE_DIRECT:
  //!
  //!   - ``ITEMS_PER_THREAD`` is odd
  //!   - The ``OutputIteratorT`` is not a simple pointer type
  //!   - The block output offset is not quadword-aligned
  //!   - The data type ``T`` is not a built-in primitive or CUDA vector type
  //!     (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
  //!
  //! @endrst
  BLOCK_STORE_VECTORIZE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` is locally
  //! transposed and then efficiently written to memory as a :ref:`striped arrangement <flexible-data-arrangement>`.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless
  //!   of items written per thread.
  //! - The local reordering incurs slightly longer latencies and throughput than the
  //!   direct cub::BLOCK_STORE_DIRECT and cub::BLOCK_STORE_VECTORIZE alternatives.
  //!
  //! @endrst
  BLOCK_STORE_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` is locally
  //! transposed and then efficiently written to memory as a
  //! :ref:`warp-striped arrangement <flexible-data-arrangement>`.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - BLOCK_THREADS must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless
  //!   of items written per thread.
  //! - The local reordering incurs slightly longer latencies and throughput than the
  //!   direct cub::BLOCK_STORE_DIRECT and cub::BLOCK_STORE_VECTORIZE alternatives.
  //!
  //! @endrst
  BLOCK_STORE_WARP_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` is locally
  //! transposed and then efficiently written to memory as a
  //! :ref:`warp-striped arrangement <flexible-data-arrangement>`.
  //! To reduce the shared memory requirement, only one warp's worth of shared
  //! memory is provisioned and is subsequently time-sliced among warps.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - BLOCK_THREADS must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless
  //!   of items written per thread.
  //! - Provisions less shared memory temporary storage, but incurs larger
  //!   latencies than the BLOCK_STORE_WARP_TRANSPOSE alternative.
  //!
  //! @endrst
  BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
};

//! @rst
//! The BlockStore class provides :ref:`collective <collective-primitives>` data movement
//! methods for writing a :ref:`blocked arrangement <flexible-data-arrangement>` of items
//! partitioned across a CUDA thread block to a linear segment of memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - The BlockStore class provides a single data movement abstraction that can be specialized
//!   to implement different cub::BlockStoreAlgorithm strategies. This facilitates different
//!   performance policies for different architectures, data types, granularity sizes, etc.
//! - BlockStore can be optionally specialized by different data movement strategies:
//!
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_DIRECT`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly to memory.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_STRIPED`:
//!      A :ref:`striped arrangement <flexible-data-arrangement>` of data is written directly to memory.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_VECTORIZE`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is written directly to memory
//!      using CUDA's built-in vectorized stores as a coalescing optimization.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_TRANSPOSE`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` is locally transposed into
//!      a :ref:`striped arrangement <flexible-data-arrangement>` which is then written to memory.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_WARP_TRANSPOSE`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` is locally transposed into
//!      a :ref:`warp-striped arrangement <flexible-data-arrangement>` which is then written to memory.
//!   #. :cpp:enumerator:`cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` is locally transposed into
//!      a :ref:`warp-striped arrangement <flexible-data-arrangement>` which is then written to memory.
//!      To reduce the shared memory requireent, only one warp's worth of shared memory is provisioned and is
//!      subsequently time-sliced among warps.
//!
//! - @rowmajor
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockStore}
//!
//! The code snippet below illustrates the storing of a "blocked" arrangement
//! of 512 integers across 128 threads (where each thread owns 4 consecutive items)
//! into a linear segment of memory. The store is specialized for ``BLOCK_STORE_WARP_TRANSPOSE``,
//! meaning items are locally reordered among threads so that memory references will be
//! efficiently coalesced using a warp-striped access pattern.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_store.cuh>
//!
//!    __global__ void ExampleKernel(int *d_data, ...)
//!    {
//!        // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
//!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_WARP_TRANSPOSE>;
//!
//!        // Allocate shared memory for BlockStore
//!        __shared__ typename BlockStore::TempStorage temp_storage;
//!
//!        // Obtain a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        ...
//!
//!        // Store items to linear memory
//!        BlockStore(temp_storage).Store(d_data, thread_data);
//!
//! Suppose the set of ``thread_data`` across the block of threads is
//! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
//! The output ``d_data`` will be ``0, 1, 2, 3, 4, 5, ...``.
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of
//! dynamically shared memory with BlockReduce and how to re-purpose the same memory region.
//! This example can be easily adapted to the storage required by BlockStore.
//!
//! @endrst
//!
//! @tparam T
//!   The type of data to be written.
//!
//! @tparam BLOCK_DIM_X
//!   The thread block length in threads along the X dimension
//!
//! @tparam ITEMS_PER_THREAD
//!   The number of consecutive items partitioned onto each thread.
//!
//! @tparam ALGORITHM
//!   **[optional]** cub::BlockStoreAlgorithm tuning policy enumeration (default: cub::BLOCK_STORE_DIRECT)
//!
//! @tparam BLOCK_DIM_Y
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1)
//!
//! @tparam BLOCK_DIM_Z
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
template <typename T,
          int BLOCK_DIM_X,
          int ITEMS_PER_THREAD,
          BlockStoreAlgorithm ALGORITHM = BLOCK_STORE_DIRECT,
          int BLOCK_DIM_Y               = 1,
          int BLOCK_DIM_Z               = 1>
class BlockStore
{
private:
  enum
  {
    /// The thread block size in threads
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
  };

  /// Store helper
  template <BlockStoreAlgorithm _POLICY, int DUMMY>
  struct StoreInternal;

  template <int DUMMY>
  struct StoreInternal<BLOCK_STORE_DIRECT, DUMMY>
  {
    /// Shared memory storage layout type
    using TempStorage = NullType;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    /**
     * @brief Store items into a linear segment of memory
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlocked(linear_tid, block_itr, items);
    }

    /**
     * @brief Store items into a linear segment of memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     *
     * @param[in] valid_items
     *   Number of valid items to write
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
    }
  };

  /**
   * BLOCK_STORE_STRIPED specialization of store helper
   */
  template <int DUMMY>
  struct StoreInternal<BLOCK_STORE_STRIPED, DUMMY>
  {
    /// Shared memory storage layout type
    using TempStorage = NullType;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    /**
     * @brief Store items into a linear segment of memory
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items);
    }

    /**
     * @brief Store items into a linear segment of memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     *
     * @param[in] valid_items
     *   Number of valid items to write
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, valid_items);
    }
  };

  /**
   * BLOCK_STORE_VECTORIZE specialization of store helper
   */
  template <int DUMMY>
  struct StoreInternal<BLOCK_STORE_VECTORIZE, DUMMY>
  {
    /// Shared memory storage layout type
    using TempStorage = NullType;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    /**
     * @brief Store items into a linear segment of memory,
     *        specialized for native pointer types (attempts vectorization)
     *
     * @param[in] block_ptr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     */
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(T* block_ptr, T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlockedVectorized(linear_tid, block_ptr, items);
    }

    /**
     * @brief Store items into a linear segment of memory,
     *        specialized for opaque input iterators (skips vectorization)
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlocked(linear_tid, block_itr, items);
    }

    /**
     * @brief Store items into a linear segment of memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     *
     * @param[in] valid_items
     *   Number of valid items to write
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
    }
  };

  /**
   * BLOCK_STORE_TRANSPOSE specialization of store helper
   */
  template <int DUMMY>
  struct StoreInternal<BLOCK_STORE_TRANSPOSE, DUMMY>
  {
    // BlockExchange utility type for keys
    using BlockExchange = BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z>;

    /// Shared memory storage layout type
    struct _TempStorage : BlockExchange::TempStorage
    {
      /// Temporary storage for partially-full block guard
      volatile int valid_items;
    };

    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage>
    {};

    /// Thread reference to shared storage
    _TempStorage& temp_storage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    /**
     * @brief Store items into a linear segment of memory
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      BlockExchange(temp_storage).BlockedToStriped(items);
      StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items);
    }

    /**
     * @brief Store items into a linear segment of memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     *
     * @param[in] valid_items
     *   Number of valid items to write
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      BlockExchange(temp_storage).BlockedToStriped(items);
      if (linear_tid == 0)
      {
        // Move through volatile smem as a workaround to prevent RF spilling on
        // subsequent loads
        temp_storage.valid_items = valid_items;
      }
      __syncthreads();
      StoreDirectStriped<BLOCK_THREADS>(linear_tid, block_itr, items, temp_storage.valid_items);
    }
  };

  /**
   * BLOCK_STORE_WARP_TRANSPOSE specialization of store helper
   */
  template <int DUMMY>
  struct StoreInternal<BLOCK_STORE_WARP_TRANSPOSE, DUMMY>
  {
    enum
    {
      WARP_THREADS = detail::warp_threads
    };

    // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
    static_assert(int(BLOCK_THREADS) % int(WARP_THREADS) == 0, "BLOCK_THREADS must be a multiple of WARP_THREADS");

    // BlockExchange utility type for keys
    using BlockExchange = BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z>;

    /// Shared memory storage layout type
    struct _TempStorage : BlockExchange::TempStorage
    {
      /// Temporary storage for partially-full block guard
      volatile int valid_items;
    };

    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage>
    {};

    /// Thread reference to shared storage
    _TempStorage& temp_storage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    /**
     * @brief Store items into a linear segment of memory
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      BlockExchange(temp_storage).BlockedToWarpStriped(items);
      StoreDirectWarpStriped(linear_tid, block_itr, items);
    }

    /**
     * @brief Store items into a linear segment of memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     *
     * @param[in] valid_items
     *   Number of valid items to write
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      BlockExchange(temp_storage).BlockedToWarpStriped(items);
      if (linear_tid == 0)
      {
        // Move through volatile smem as a workaround to prevent RF spilling on
        // subsequent loads
        temp_storage.valid_items = valid_items;
      }
      __syncthreads();
      StoreDirectWarpStriped(linear_tid, block_itr, items, temp_storage.valid_items);
    }
  };

  /**
   * BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED specialization of store helper
   */
  template <int DUMMY>
  struct StoreInternal<BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, DUMMY>
  {
    enum
    {
      WARP_THREADS = detail::warp_threads
    };

    // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
    static_assert(int(BLOCK_THREADS) % int(WARP_THREADS) == 0, "BLOCK_THREADS must be a multiple of WARP_THREADS");

    // BlockExchange utility type for keys
    using BlockExchange = BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, true, BLOCK_DIM_Y, BLOCK_DIM_Z>;

    /// Shared memory storage layout type
    struct _TempStorage : BlockExchange::TempStorage
    {
      /// Temporary storage for partially-full block guard
      volatile int valid_items;
    };

    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage>
    {};

    /// Thread reference to shared storage
    _TempStorage& temp_storage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE StoreInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    /**
     * @brief Store items into a linear segment of memory
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
      BlockExchange(temp_storage).BlockedToWarpStriped(items);
      StoreDirectWarpStriped(linear_tid, block_itr, items);
    }

    /**
     * @brief Store items into a linear segment of memory, guarded by range
     *
     * @param[in] block_itr
     *   The thread block's base output iterator for storing to
     *
     * @param[in] items
     *   Data to store
     *
     * @param[in] valid_items
     *   Number of valid items to write
     */
    template <typename OutputIteratorT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
    {
      BlockExchange(temp_storage).BlockedToWarpStriped(items);
      if (linear_tid == 0)
      {
        // Move through volatile smem as a workaround to prevent RF spilling on
        // subsequent loads
        temp_storage.valid_items = valid_items;
      }
      __syncthreads();
      StoreDirectWarpStriped(linear_tid, block_itr, items, temp_storage.valid_items);
    }
  };

  /// Internal load implementation to use
  using InternalStore = StoreInternal<ALGORITHM, 0>;

  /// Shared memory storage layout type
  using _TempStorage = typename InternalStore::TempStorage;

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  /// Thread reference to shared storage
  _TempStorage& temp_storage;

  /// Linear thread-id
  int linear_tid;

public:
  //! @smemstorage{BlockStore}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  /**
   * @brief Collective constructor using a private static allocation of shared memory as temporary storage.
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockStore()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation as temporary storage.
   *
   * @param temp_storage[in]
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockStore(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @}  end member group
  //! @name Data movement
  //! @{

  //! @rst
  //! Store items into a linear segment of memory
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the storing of a "blocked" arrangement
  //! of 512 integers across 128 threads (where each thread owns 4 consecutive items)
  //! into a linear segment of memory. The store is specialized for ``BLOCK_STORE_WARP_TRANSPOSE``,
  //! meaning items are locally reordered among threads so that memory references will be
  //! efficiently coalesced using a warp-striped access pattern.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_store.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, ...)
  //!    {
  //!        // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockStore
  //!        __shared__ typename BlockStore::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Store items to linear memory
  //!        int thread_data[4];
  //!        BlockStore(temp_storage).Store(d_data, thread_data);
  //!
  //! Suppose the set of ``thread_data`` across the block of threads is
  //! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
  //! The output ``d_data`` will be ``0, 1, 2, 3, 4, 5, ...``.
  //!
  //! @endrst
  //!
  //! @param block_itr[out]
  //!   The thread block's base output iterator for storing to
  //!
  //! @param items[in]
  //!   Data to store
  template <typename OutputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
  {
    InternalStore(temp_storage, linear_tid).Store(block_itr, items);
  }

  //! @rst
  //! Store items into a linear segment of memory, guarded by range.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the guarded storing of a "blocked" arrangement
  //! of 512 integers across 128 threads (where each thread owns 4 consecutive items)
  //! into a linear segment of memory. The store is specialized for ``BLOCK_STORE_WARP_TRANSPOSE``,
  //! meaning items are locally reordered among threads so that memory references will be
  //! efficiently coalesced using a warp-striped access pattern.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_store.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int valid_items, ...)
  //!    {
  //!        // Specialize BlockStore for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockStore = cub::BlockStore<int, 128, 4, BLOCK_STORE_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockStore
  //!        __shared__ typename BlockStore::TempStorage temp_storage;
  //!
  //!        // Obtain a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        ...
  //!
  //!        // Store items to linear memory
  //!        int thread_data[4];
  //!        BlockStore(temp_storage).Store(d_data, thread_data, valid_items);
  //!
  //! Suppose the set of ``thread_data`` across the block of threads is
  //! ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }`` and ``valid_items`` is ``5``.
  //! The output ``d_data`` will be ``0, 1, 2, 3, 4, ?, ?, ?, ...``, with
  //! only the first two threads being unmasked to store portions of valid data.
  //!
  //! @endrst
  //!
  //! @param block_itr[out]
  //!   The thread block's base output iterator for storing to
  //!
  //! @param items[in]
  //!   Data to store
  //!
  //! @param valid_items[in]
  //!   Number of valid items to write
  template <typename OutputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD], int valid_items)
  {
    InternalStore(temp_storage, linear_tid).Store(block_itr, items, valid_items);
  }

  //! @}  end member group
};

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
template <class Policy, class It, class T = cub::detail::it_value_t<It>>
struct BlockStoreType
{
  using type = cub::BlockStore<T, Policy::BLOCK_THREADS, Policy::ITEMS_PER_THREAD, Policy::STORE_ALGORITHM>;
};
#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
