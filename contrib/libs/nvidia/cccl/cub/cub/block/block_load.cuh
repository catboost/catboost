/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
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
//! block_load.cuh Operations for reading linear tiles of data into the CUDA thread block.

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
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

//! @name Blocked arrangement I/O (direct)
//! @{

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block.
//!
//! @blocked
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base input iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
template <typename T, int ITEMS_PER_THREAD, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectBlocked(int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
{
  // Load directly in thread-blocked order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    dst_items[i] = block_src_it[linear_tid * ITEMS_PER_THREAD + i];
  }
}

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block, guarded by range.
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   First out-of-bounds index when loading from block_src_it
template <typename T, int ITEMS_PER_THREAD, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectBlocked(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    const auto src_pos = linear_tid * ITEMS_PER_THREAD + i;
    if (src_pos < block_items_end)
    {
      dst_items[i] = block_src_it[src_pos];
    }
  }
}

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block, guarded
//! by range, with a fall-back assignment of out-of-bound elements.
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **[inferred]** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base input iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   First out-of-bounds index when loading from block_src_it
//!
//! @param[in] oob_default
//!   Default value to assign out-of-bound items
template <typename T, typename DefaultT, int ITEMS_PER_THREAD, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectBlocked(
  int linear_tid,
  RandomAccessIterator block_src_it,
  T (&dst_items)[ITEMS_PER_THREAD],
  int block_items_end,
  DefaultT oob_default)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    dst_items[i] = oob_default;
  }

  LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end);
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

//! @brief Internal implementation for load vectorization
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_ptr
//!   Input pointer for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
template <CacheLoadModifier MODIFIER, typename T, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
InternalLoadDirectBlockedVectorized(int linear_tid, const T* block_src_ptr, T (&dst_items)[ITEMS_PER_THREAD])
{
  // Find biggest memory access word that T is a whole multiple of
  using device_word_t = typename UnitWord<T>::DeviceWord;
  _CCCL_DIAG_PUSH
#  if _CCCL_COMPILER(CLANG, >=, 10)
  _CCCL_DIAG_SUPPRESS_CLANG("-Wsizeof-array-div")
#  endif // _CCCL_COMPILER(CLANG, >=, 10)
  constexpr int total_words = static_cast<int>(sizeof(dst_items) / sizeof(device_word_t));
  _CCCL_DIAG_POP
  constexpr int vector_size        = (total_words % 4 == 0) ? 4 : (total_words % 2 == 0) ? 2 : 1;
  constexpr int vectors_per_thread = total_words / vector_size;

  // Load into an array of vectors in thread-blocked order
  using vector_t = typename CubVector<device_word_t, vector_size>::Type;

  // Add the alignment check to ensure the vectorized loading can proceed.
  if (reinterpret_cast<uintptr_t>(block_src_ptr) % (alignof(vector_t)) == 0)
  {
    vector_t vec_items[vectors_per_thread];
    // Load into an array of vectors in thread-blocked order
    const vector_t* vec_ptr = reinterpret_cast<const vector_t*>(block_src_ptr) + linear_tid * vectors_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < vectors_per_thread; i++)
    {
      vec_items[i] = ThreadLoad<MODIFIER>(vec_ptr + i);
    }

    // Copy to destination
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
      dst_items[i] = *(reinterpret_cast<T*>(vec_items) + i);
    }
  }
  else
  {
    LoadDirectBlocked(linear_tid, block_src_ptr, dst_items);
  }
}

#endif // _CCCL_DOXYGEN_INVOKED

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block.
//!
//! @blocked
//!
//! The input offset (``block_ptr + block_offset``) must be quad-item aligned
//!
//! The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
//!
//! - ``ITEMS_PER_THREAD`` is odd
//! - The data type ``T`` is not a built-in primitive or CUDA vector type
//!   (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//!   linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_ptr
//!   The thread block's base pointer for loading from
//!
//! @param[out] dst_items
//!  destination to load data into
template <typename T, int ITEMS_PER_THREAD>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectBlockedVectorized(int linear_tid, T* block_src_ptr, T (&dst_items)[ITEMS_PER_THREAD])
{
  InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, block_src_ptr, dst_items);
}

//! @} end member group
//! @name Striped arrangement I/O (direct)
//! @{

//! @rst
//! Load a linear segment of items into a striped arrangement across the thread block.
//!
//! @striped
//!
//! @endrst
//!
//! @tparam BLOCK_THREADS
//!   The thread block size in threads
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
template <int BLOCK_THREADS, typename T, int ITEMS_PER_THREAD, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectStriped(int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    dst_items[i] = block_src_it[linear_tid + i * BLOCK_THREADS];
  }
}

namespace detail
{
template <int BLOCK_THREADS, typename T, int ITEMS_PER_THREAD, typename RandomAccessIterator, typename TransformOpT>
_CCCL_DEVICE _CCCL_FORCEINLINE void load_transform_direct_striped(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], TransformOpT transform_op)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    dst_items[i] = transform_op(block_src_it[linear_tid + i * BLOCK_THREADS]);
  }
}
} // namespace detail

//! @rst
//! Load a linear segment of items into a striped arrangement across the thread block, guarded by range
//!
//! @striped
//!
//! @endrst
//!
//! @tparam BLOCK_THREADS
//!   The thread block size in threads
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) +
//! linear_tid</tt> for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   Number of valid items to load
template <int BLOCK_THREADS, typename T, int ITEMS_PER_THREAD, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectStriped(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    const auto src_pos = linear_tid + i * BLOCK_THREADS;
    if (src_pos < block_items_end)
    {
      dst_items[i] = block_src_it[src_pos];
    }
  }
}

//! @rst
//! Load a linear segment of items into a striped arrangement across the thread block, guarded
//! by range, with a fall-back assignment of out-of-bound elements.
//!
//! @striped
//!
//! @endrst
//!
//! @tparam BLOCK_THREADS
//!   The thread block size in threads
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//! linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   Number of valid items to load
//!
//! @param[in] oob_default
//!   Default value to assign out-of-bound items
template <int BLOCK_THREADS, typename T, typename DefaultT, int ITEMS_PER_THREAD, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectStriped(
  int linear_tid,
  RandomAccessIterator block_src_it,
  T (&dst_items)[ITEMS_PER_THREAD],
  int block_items_end,
  DefaultT oob_default)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    dst_items[i] = oob_default;
  }

  LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_src_it, dst_items, block_items_end);
}

//! @} end member group
//! @name Warp-striped arrangement I/O (direct)
//! @{

//! @rst
//! Load a linear segment of items into a warp-striped arrangement across the thread block.
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
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//! linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
template <typename T, int ITEMS_PER_THREAD, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectWarpStriped(int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
{
  const int tid         = linear_tid & (detail::warp_threads - 1);
  const int wid         = linear_tid >> detail::log2_warp_threads;
  const int warp_offset = wid * detail::warp_threads * ITEMS_PER_THREAD;

  // Load directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    new (&dst_items[i]) T(block_src_it[warp_offset + tid + (i * detail::warp_threads)]);
  }
}

//! @rst
//! Load a linear segment of items into a warp-striped arrangement across the thread block, guarded by range
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
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//! linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   Number of valid items to load
template <typename T, int ITEMS_PER_THREAD, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectWarpStriped(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
{
  const int tid         = linear_tid & (detail::warp_threads - 1);
  const int wid         = linear_tid >> detail::log2_warp_threads;
  const int warp_offset = wid * detail::warp_threads * ITEMS_PER_THREAD;

  // Load directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    const auto src_pos = warp_offset + tid + (i * detail::warp_threads);
    if (src_pos < block_items_end)
    {
      new (&dst_items[i]) T(block_src_it[src_pos]);
    }
  }
}

//! @rst
//! Load a linear segment of items into a warp-striped arrangement across the thread block,
//! guarded by range, with a fall-back assignment of out-of-bound elements.
//!
//! @warpstriped
//!
//! @endrst
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ITEMS_PER_THREAD
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//! linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   Number of valid items to load
//!
//! @param[in] oob_default
//!   Default value to assign out-of-bound items
template <typename T, typename DefaultT, int ITEMS_PER_THREAD, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectWarpStriped(
  int linear_tid,
  RandomAccessIterator block_src_it,
  T (&dst_items)[ITEMS_PER_THREAD],
  int block_items_end,
  DefaultT oob_default)
{
  // Load directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ITEMS_PER_THREAD; i++)
  {
    dst_items[i] = oob_default;
  }

  LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end);
}

//! @} end member group

//! @brief cub::BlockLoadAlgorithm enumerates alternative algorithms for cub::BlockLoad to read a linear segment of data
//!        from memory into a blocked arrangement across a CUDA thread block.
enum BlockLoadAlgorithm
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
  //! The utilization of memory transactions (coalescing) decreases as the access stride between threads increases
  //! (i.e., the number items per thread).
  //! @endrst
  BLOCK_LOAD_DIRECT,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) doesn't depend on the number of items per thread.
  //!
  //! @endrst
  BLOCK_LOAD_STRIPED,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read from memory using CUDA's built-in
  //! vectorized loads as a coalescing optimization. For example, ``ld.global.v4.s32`` instructions will be generated
  //! when ``T = int`` and ``ITEMS_PER_THREAD % 4 == 0``.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high until the the access stride between threads
  //!   (i.e., the number items per thread) exceeds the maximum vector load width (typically 4 items or 64B, whichever
  //!   is lower).
  //! - The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
  //!
  //!   - ``ITEMS_PER_THREAD`` is odd
  //!   - The ``RandomAccessIterator`` is not a simple pointer type
  //!   - The block input offset is not quadword-aligned
  //!   - The data type ``T`` is not a built-in primitive or CUDA vector type
  //!     (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
  //!
  //! @endrst
  BLOCK_LOAD_VECTORIZE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is read efficiently from memory and then locally
  //! transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless of items loaded per thread.
  //! - The local reordering incurs slightly longer latencies and throughput than the direct cub::BLOCK_LOAD_DIRECT and
  //!   cub::BLOCK_LOAD_VECTORIZE alternatives.
  //!
  //! @endrst
  BLOCK_LOAD_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read efficiently from memory and then
  //! locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - BLOCK_THREADS must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless of items loaded per thread.
  //! - The local reordering incurs slightly larger latencies than the direct cub::BLOCK_LOAD_DIRECT and
  //!   cub::BLOCK_LOAD_VECTORIZE alternatives.
  //! - Provisions more shared storage, but incurs smaller latencies than the
  //!   BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED alternative.
  //!
  //! @endrst
  BLOCK_LOAD_WARP_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! Like ``BLOCK_LOAD_WARP_TRANSPOSE``, a :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read
  //! directly from memory and then is locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
  //! To reduce the shared memory requirement, only one warp's worth of shared memory is provisioned and is subsequently
  //! time-sliced among warps.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - BLOCK_THREADS must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless of items loaded per thread.
  //! - Provisions less shared memory temporary storage, but incurs larger latencies than the BLOCK_LOAD_WARP_TRANSPOSE
  //!   alternative.
  //!
  //! @endrst
  BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
};

//! @rst
//! The BlockLoad class provides :ref:`collective <collective-primitives>` data movement methods for loading a linear
//! segment of items from memory into a :ref:`blocked arrangement <flexible-data-arrangement>` across a CUDA thread
//! block.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - The BlockLoad class provides a single data movement abstraction that can be specialized to implement different
//!   cub::BlockLoadAlgorithm strategies. This facilitates different performance policies for different architectures,
//!   data types, granularity sizes, etc.
//! - BlockLoad can be optionally specialized by different data movement strategies:
//!
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_DIRECT`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from memory.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_STRIPED`:
//!      A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_VECTORIZE`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from memory
//!      using CUDA's built-in vectorized loads as a coalescing optimization.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_TRANSPOSE`:
//!      A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory and is then
//!      locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_WARP_TRANSPOSE`:
//!      A :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read directly from memory and is then
//!      locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED`:
//!      A :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read directly from memory and is then
//!      locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>` one warp at a time.
//!
//! - @rowmajor
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockLoad}
//!
//! The code snippet below illustrates the loading of a linear segment of 512 integers into a "blocked" arrangement
//! across 128 threads where each thread owns 4 consecutive items. The load is specialized for
//! ``BLOCK_LOAD_WARP_TRANSPOSE``, meaning memory references are efficiently coalesced using a warp-striped access
//! pattern (after which items are locally reordered among threads).
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
//!
//!    __global__ void ExampleKernel(int *d_data, ...)
//!    {
//!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
//!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
//!
//!        // Allocate shared memory for BlockLoad
//!        __shared__ typename BlockLoad::TempStorage temp_storage;
//!
//!        // Load a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        BlockLoad(temp_storage).Load(d_data, thread_data);
//!
//! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...``. The set of ``thread_data`` across the block of threads in
//! those threads will be ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of dynamically shared memory with
//! BlockReduce and how to re-purpose the same memory region. This example can be easily adapted to the storage required
//! by BlockLoad.
//!
//! @endrst
//!
//! @tparam T
// The data type to read into (which must be convertible from the input iterator's value type).
//!
//! @tparam BLOCK_DIM_X
//!   The thread block length in threads along the X dimension
//!
//! @tparam ITEMS_PER_THREAD
//!   The number of consecutive items partitioned onto each thread.
//!
//! @tparam ALGORITHM
//!   **[optional]** cub::BlockLoadAlgorithm tuning policy. default: ``cub::BLOCK_LOAD_DIRECT``.
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
          BlockLoadAlgorithm ALGORITHM = BLOCK_LOAD_DIRECT,
          int BLOCK_DIM_Y              = 1,
          int BLOCK_DIM_Z              = 1>
class BlockLoad
{
  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z; // total threads in the block

  template <BlockLoadAlgorithm _POLICY, int DUMMY>
  struct LoadInternal; // helper to dispatch the load algorithm

  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_DIRECT, DUMMY>
  {
    using TempStorage = NullType;
    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
    {
      LoadDirectBlocked(linear_tid, block_src_it, dst_items);
    }

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
    {
      LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end);
    }

    template <typename RandomAccessIterator, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end, DefaultT oob_default)
    {
      LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
    }
  };

  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_STRIPED, DUMMY>
  {
    using TempStorage = NullType;
    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_src_it, dst_items);
    }

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_src_it, dst_items, block_items_end);
    }

    template <typename RandomAccessIterator, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end, DefaultT oob_default)
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
    }
  };

  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_VECTORIZE, DUMMY>
  {
    using TempStorage = NullType;
    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& /*temp_storage*/, int linear_tid)
        : linear_tid(linear_tid)
    {}

    // attempts vectorization (pointer)
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(const T* block_ptr, T (&dst_items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, block_ptr, dst_items);
    }
    // NOTE: This function is necessary for pointers to non-const types.
    // The core reason is that the compiler will not deduce 'T*' to 'const T*' automatically.
    // Otherwise, when the pointer type is 'T*', the compiler will prefer the overloaded version
    // Load(RandomAccessIterator...) over Load(const T*...), which means it will never perform vectorized loading for
    // pointers to non-const types.
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(T* block_ptr, T (&dst_items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, block_ptr, dst_items);
    }

    // any other iterator, no vectorization
    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
    {
      LoadDirectBlocked(linear_tid, block_src_it, dst_items);
    }

    // attempts vectorization (cache modified iterator)
    template <CacheLoadModifier MODIFIER, typename ValueType, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(CacheModifiedInputIterator<MODIFIER, ValueType, OffsetT> block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<MODIFIER>(linear_tid, block_src_it.ptr, dst_items);
    }

    // skips vectorization
    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
    {
      LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end);
    }

    // skips vectorization
    template <typename RandomAccessIterator, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end, DefaultT oob_default)
    {
      LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
    }
  };

  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_TRANSPOSE, DUMMY>
  {
    using BlockExchange = BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z>;
    using _TempStorage  = typename BlockExchange::TempStorage;
    using TempStorage   = Uninitialized<_TempStorage>;

    _TempStorage& temp_storage;
    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_src_it, dst_items);
      BlockExchange(temp_storage).StripedToBlocked(dst_items, dst_items);
    }

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_src_it, dst_items, block_items_end);
      BlockExchange(temp_storage).StripedToBlocked(dst_items, dst_items);
    }

    template <typename RandomAccessIterator, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end, DefaultT oob_default)
    {
      LoadDirectStriped<BLOCK_THREADS>(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
      BlockExchange(temp_storage).StripedToBlocked(dst_items, dst_items);
    }
  };

  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_WARP_TRANSPOSE, DUMMY>
  {
    static constexpr int WARP_THREADS = detail::warp_threads;
    static_assert(BLOCK_THREADS % WARP_THREADS == 0, "BLOCK_THREADS must be a multiple of WARP_THREADS");

    using BlockExchange = BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z>;
    using _TempStorage  = typename BlockExchange::TempStorage;
    using TempStorage   = Uninitialized<_TempStorage>;

    _TempStorage& temp_storage;
    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
    {
      LoadDirectWarpStriped(linear_tid, block_src_it, dst_items);
      BlockExchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
    {
      LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end);
      BlockExchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }

    template <typename RandomAccessIterator, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end, DefaultT oob_default)
    {
      LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
      BlockExchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }
  };

  template <int DUMMY>
  struct LoadInternal<BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED, DUMMY>
  {
    static constexpr int WARP_THREADS = detail::warp_threads;
    static_assert(BLOCK_THREADS % WARP_THREADS == 0, "BLOCK_THREADS must be a multiple of WARP_THREADS");

    using BlockExchange = BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, true, BLOCK_DIM_Y, BLOCK_DIM_Z>;
    using _TempStorage  = typename BlockExchange::TempStorage;
    using TempStorage   = Uninitialized<_TempStorage>;

    _TempStorage& temp_storage;
    int linear_tid;

    _CCCL_DEVICE _CCCL_FORCEINLINE LoadInternal(TempStorage& temp_storage, int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
    {
      LoadDirectWarpStriped(linear_tid, block_src_it, dst_items);
      BlockExchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }

    template <typename RandomAccessIterator>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
    {
      LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end);
      BlockExchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }

    template <typename RandomAccessIterator, typename DefaultT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end, DefaultT oob_default)
    {
      LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
      BlockExchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }
  };

  using InternalLoad = LoadInternal<ALGORITHM, 0>; // load implementation to use
  using _TempStorage = typename InternalLoad::TempStorage;

  // Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  _TempStorage& temp_storage;
  int linear_tid;

public:
  /// @smemstorage{BlockLoad}
  using TempStorage = Uninitialized<_TempStorage>;

  //! @name Collective constructors
  //! @{

  /// @brief Collective constructor using a private static allocation of shared memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoad()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /// @brief Collective constructor using the specified memory allocation as temporary storage.
  /// @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoad(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @} end member group
  //! @name Data movement
  //! @{

  //! @rst
  //! Load a linear segment of items from memory.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the loading of a linear segment of 512 integers into a "blocked" arrangement
  //! across 128 threads where each thread owns 4 consecutive items. The load is specialized for
  //! ``BLOCK_LOAD_WARP_TRANSPOSE``, meaning memory references are efficiently coalesced using a warp-striped access
  //! pattern (after which items are locally reordered among threads).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, ...)
  //!    {
  //!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockLoad
  //!        __shared__ typename BlockLoad::TempStorage temp_storage;
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        BlockLoad(temp_storage).Load(d_data, thread_data);
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...``. The set of ``thread_data`` across the block of threads
  //! in those threads will be ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
  //!
  //! @endrst
  //!
  //! @param[in] block_src_it
  //!   The thread block's base iterator for loading from
  //!
  //! @param[out] dst_items
  //!   Destination to load data into
  template <typename RandomAccessIterator>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD])
  {
    InternalLoad(temp_storage, linear_tid).Load(block_src_it, dst_items);
  }

  //! @rst
  //!
  //! Load a linear segment of items from memory, guarded by range.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the guarded loading of a linear segment of 512 integers into a "blocked"
  //! arrangement across 128 threads where each thread owns 4 consecutive items. The load is specialized for
  //! ``BLOCK_LOAD_WARP_TRANSPOSE``, meaning memory references are efficiently coalesced using a warp-striped access
  //! pattern (after which items are locally reordered among threads).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int block_items_end, ...)
  //!    {
  //!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockLoad
  //!        __shared__ typename BlockLoad::TempStorage temp_storage;
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        BlockLoad(temp_storage).Load(d_data, thread_data, block_items_end);
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, 6...`` and ``block_items_end`` is ``5``. The set of
  //! ``thread_data`` across the block of threads in those threads will be ``{ [0,1,2,3], [4,?,?,?], ..., [?,?,?,?] }``,
  //! with only the first two threads being unmasked to load portions of valid data (and other items remaining
  //! unassigned).
  //!
  //! @endrst
  //!
  //! @param[in] block_src_it
  //!   The thread block's base iterator for loading from
  //!
  //! @param[out] dst_items
  //!   Destination to load data into
  //!
  //! @param[in] block_items_end
  //!   Number of valid items to load
  template <typename RandomAccessIterator>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end)
  {
    InternalLoad(temp_storage, linear_tid).Load(block_src_it, dst_items, block_items_end);
  }

  //! @rst
  //! Load a linear segment of items from memory, guarded by range, with a fall-back assignment of out-of-bound elements
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the guarded loading of a linear segment of 512 integers into a "blocked"
  //! arrangement across 128 threads where each thread owns 4 consecutive items. The load is specialized for
  //! ``BLOCK_LOAD_WARP_TRANSPOSE``, meaning memory references are efficiently coalesced using a warp-striped access
  //! pattern (after which items are locally reordered among threads).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int block_items_end, ...)
  //!    {
  //!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockLoad
  //!        __shared__ typename BlockLoad::TempStorage temp_storage;
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        BlockLoad(temp_storage).Load(d_data, thread_data, block_items_end, -1);
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, 6...``, ``block_items_end`` is ``5``, and the out-of-bounds
  //! default is ``-1``. The set of ``thread_data`` across the block of threads in those threads will be
  //! ``{ [0,1,2,3], [4,-1,-1,-1], ..., [-1,-1,-1,-1] }``, with only the first two threads being unmasked to load
  //! portions of valid data (and other items are assigned ``-1``)
  //!
  //! @endrst
  //!
  //! @param[in] block_src_it
  //!   The thread block's base iterator for loading from
  //!
  //! @param[out] dst_items
  //!   Destination to load data into
  //!
  //! @param[in] block_items_end
  //!   Number of valid items to load
  //!
  //! @param[in] oob_default
  //!   Default value to assign out-of-bound items
  template <typename RandomAccessIterator, typename DefaultT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Load(RandomAccessIterator block_src_it, T (&dst_items)[ITEMS_PER_THREAD], int block_items_end, DefaultT oob_default)
  {
    InternalLoad(temp_storage, linear_tid).Load(block_src_it, dst_items, block_items_end, oob_default);
  }

  //! @}  end member group
};

template <class Policy, class It, class T = cub::detail::it_value_t<It>>
struct BlockLoadType
{
  using type = cub::BlockLoad<T, Policy::BLOCK_THREADS, Policy::ITEMS_PER_THREAD, Policy::LOAD_ALGORITHM>;
};

CUB_NAMESPACE_END
