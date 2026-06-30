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

/**
 * @file
 * The cub::BlockHistogram class provides [<em>collective</em>](../index.html#sec0) methods for
 * constructing block-wide histograms from data samples partitioned across a CUDA thread block.
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

#include <cub/block/specializations/block_histogram_atomic.cuh>
#include <cub/block/specializations/block_histogram_sort.cuh>
#include <cub/util_ptx.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

//! @brief BlockHistogramAlgorithm enumerates alternative algorithms for the parallel construction of
//!        block-wide histograms.
enum BlockHistogramAlgorithm
{

  //! @rst
  //!
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! Sorting followed by differentiation. Execution is comprised of two phases:
  //!
  //! #. Sort the data using efficient radix sort
  //! #. Look for "runs" of same-valued keys by detecting discontinuities; the run-lengths are histogram bin counts.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! Delivers consistent throughput regardless of sample bin distribution.
  //!
  //! @endrst
  BLOCK_HISTO_SORT,

  //! @rst
  //!
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! Use atomic addition to update byte counts directly
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! Performance is strongly tied to the hardware implementation of atomic
  //! addition, and may be significantly degraded for non uniformly-random
  //! input distributions where many concurrent updates are likely to be
  //! made to the same bin counter.
  //!
  //! @endrst
  BLOCK_HISTO_ATOMIC,
};

//! @rst
//! The BlockHistogram class provides :ref:`collective <collective-primitives>` methods for
//! constructing block-wide histograms from data samples partitioned across a CUDA thread block.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - A `histogram <http://en.wikipedia.org/wiki/Histogram>`_ counts the number of observations that fall into
//!   each of the disjoint categories (known as *bins*).
//! - The ``T`` type must be implicitly castable to an integer type.
//! - BlockHistogram expects each integral ``input[i]`` value to satisfy
//!   ``0 <= input[i] < BINS``. Values outside of this range result in undefined behavior.
//! - BlockHistogram can be optionally specialized to use different algorithms:
//!
//!   #. :cpp:enumerator:`cub::BLOCK_HISTO_SORT`: Sorting followed by differentiation.
//!   #. :cpp:enumerator:`cub::BLOCK_HISTO_ATOMIC`: Use atomic addition to update byte counts directly.
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockHistogram}
//!
//! The code snippet below illustrates a 256-bin histogram of 512 integer samples that
//! are partitioned across 128 threads where each thread owns 4 samples.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_histogram.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize a 256-bin BlockHistogram type for a 1D block of 128 threads having 4 character samples each
//!        using BlockHistogram = cub::BlockHistogram<unsigned char, 128, 4, 256>;
//!
//!        // Allocate shared memory for BlockHistogram
//!        __shared__ typename BlockHistogram::TempStorage temp_storage;
//!
//!        // Allocate shared memory for block-wide histogram bin counts
//!        __shared__ unsigned int smem_histogram[256];
//!
//!        // Obtain input samples per thread
//!        unsigned char data[4];
//!        ...
//!
//!        // Compute the block-wide histogram
//!        BlockHistogram(temp_storage).Histogram(data, smem_histogram);
//!
//! Performance and Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - @granularity
//! - All input values must fall between ``[0, BINS)``, or behavior is undefined.
//! - The histogram output can be constructed in shared or device-accessible memory
//! - See ``cub::BlockHistogramAlgorithm`` for performance details regarding algorithmic alternatives
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of dynamically shared memory with
//! BlockReduce and how to re-purpose the same memory region. This example can be easily adapted to the storage
//! required by BlockHistogram.
//! @endrst
//!
//! @tparam T
//!   The sample type being histogrammed (must be castable to an integer bin identifier)
//!
//! @tparam BLOCK_DIM_X
//!   The thread block length in threads along the X dimension
//!
//! @tparam ITEMS_PER_THREAD
//!   The number of items per thread
//!
//! @tparam BINS
//!   The number bins within the histogram
//!
//! @tparam ALGORITHM
//!   **[optional]** cub::BlockHistogramAlgorithm enumerator specifying the underlying algorithm to use
//!   (default: cub::BLOCK_HISTO_SORT)
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
          int BINS,
          BlockHistogramAlgorithm ALGORITHM = BLOCK_HISTO_SORT,
          int BLOCK_DIM_Y                   = 1,
          int BLOCK_DIM_Z                   = 1>
class BlockHistogram
{
private:
  /// Constants
  enum
  {
    /// The thread block size in threads
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
  };

  /// Internal specialization.
  using InternalBlockHistogram =
    ::cuda::std::_If<ALGORITHM == BLOCK_HISTO_SORT,
                     detail::BlockHistogramSort<T, BLOCK_DIM_X, ITEMS_PER_THREAD, BINS, BLOCK_DIM_Y, BLOCK_DIM_Z>,
                     detail::BlockHistogramAtomic<BINS>>;

  /// Shared memory storage layout type for BlockHistogram
  using _TempStorage = typename InternalBlockHistogram::TempStorage;

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  unsigned int linear_tid;

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

public:
  /// @smemstorage{BlockHistogram}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockHistogram()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation as temporary storage.
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockHistogram(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //! @}  end member group
  //! @name Histogram operations
  //! @{

  //! @rst
  //! Initialize the shared histogram counters to zero.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a the initialization and update of a
  //! histogram of 512 integer samples that are partitioned across 128 threads
  //! where each thread owns 4 samples.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_histogram.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!      // Specialize a 256-bin BlockHistogram type for a 1D block of 128 threads having 4 character samples each
  //!      using BlockHistogram = cub::BlockHistogram<unsigned char, 128, 4, 256>;
  //!
  //!      // Allocate shared memory for BlockHistogram
  //!      __shared__ typename BlockHistogram::TempStorage temp_storage;
  //!
  //!      // Allocate shared memory for block-wide histogram bin counts
  //!      __shared__ unsigned int smem_histogram[256];
  //!
  //!      // Obtain input samples per thread
  //!      unsigned char thread_samples[4];
  //!      ...
  //!
  //!      // Initialize the block-wide histogram
  //!      BlockHistogram(temp_storage).InitHistogram(smem_histogram);
  //!
  //!      // Update the block-wide histogram
  //!      BlockHistogram(temp_storage).Composite(thread_samples, smem_histogram);
  //!
  //! @endrst
  //!
  //! @tparam CounterT
  //!   **[inferred]** Histogram counter type
  template <typename CounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitHistogram(CounterT histogram[BINS])
  {
    // Initialize histogram bin counts to zeros
    int histo_offset = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + BLOCK_THREADS <= BINS; histo_offset += BLOCK_THREADS)
    {
      histogram[histo_offset + linear_tid] = 0;
    }
    // Finish up with guarded initialization if necessary
    if ((BINS % BLOCK_THREADS != 0) && (histo_offset + linear_tid < BINS))
    {
      histogram[histo_offset + linear_tid] = 0;
    }
  }

  //! @rst
  //! Constructs a block-wide histogram in shared/device-accessible memory.
  //! Each thread contributes an array of input elements.
  //!
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a 256-bin histogram of 512 integer samples that
  //! are partitioned across 128 threads where each thread owns 4 samples.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_histogram.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize a 256-bin BlockHistogram type for a 1D block of 128 threads having 4 character samples each
  //!        using BlockHistogram = cub::BlockHistogram<unsigned char, 128, 4, 256>;
  //!
  //!        // Allocate shared memory for BlockHistogram
  //!        __shared__ typename BlockHistogram::TempStorage temp_storage;
  //!
  //!        // Allocate shared memory for block-wide histogram bin counts
  //!        __shared__ unsigned int smem_histogram[256];
  //!
  //!        // Obtain input samples per thread
  //!        unsigned char thread_samples[4];
  //!        ...
  //!
  //!        // Compute the block-wide histogram
  //!        BlockHistogram(temp_storage).Histogram(thread_samples, smem_histogram);
  //!
  //! @endrst
  //!
  //! @tparam CounterT
  //!   **[inferred]** Histogram counter type
  //!
  //! @param[in] items
  //!   Calling thread's input values to histogram
  //!
  //! @param[out] histogram
  //!   Reference to shared/device-accessible memory histogram
  template <typename CounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Histogram(T (&items)[ITEMS_PER_THREAD], CounterT histogram[BINS])
  {
    // Initialize histogram bin counts to zeros
    InitHistogram(histogram);

    __syncthreads();

    // Composite the histogram
    InternalBlockHistogram(temp_storage).Composite(items, histogram);
  }

  //! @rst
  //! Updates an existing block-wide histogram in shared/device-accessible memory.
  //! Each thread composites an array of input elements.
  //!
  //! - @granularity
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates a the initialization and update of a
  //! histogram of 512 integer samples that are partitioned across 128 threads
  //! where each thread owns 4 samples.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_histogram.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        // Specialize a 256-bin BlockHistogram type for a 1D block of 128 threads having 4 character samples each
  //!        using BlockHistogram = cub::BlockHistogram<unsigned char, 128, 4, 256>;
  //!
  //!        // Allocate shared memory for BlockHistogram
  //!        __shared__ typename BlockHistogram::TempStorage temp_storage;
  //!
  //!        // Allocate shared memory for block-wide histogram bin counts
  //!        __shared__ unsigned int smem_histogram[256];
  //!
  //!        // Obtain input samples per thread
  //!        unsigned char thread_samples[4];
  //!        ...
  //!
  //!        // Initialize the block-wide histogram
  //!        BlockHistogram(temp_storage).InitHistogram(smem_histogram);
  //!
  //!        // Update the block-wide histogram
  //!        BlockHistogram(temp_storage).Composite(thread_samples, smem_histogram);
  //!
  //! @endrst
  //!
  //! @tparam CounterT
  //!   **[inferred]** Histogram counter type
  //!
  //! @param[in] items
  //!   Calling thread's input values to histogram
  //!
  //! @param[out] histogram
  //!   Reference to shared/device-accessible memory histogram
  template <typename CounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Composite(T (&items)[ITEMS_PER_THREAD], CounterT histogram[BINS])
  {
    InternalBlockHistogram(temp_storage).Composite(items, histogram);
  }
};

CUB_NAMESPACE_END
