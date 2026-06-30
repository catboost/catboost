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
 * The cub::BlockHistogramSort class provides sorting-based methods for constructing block-wide
 * histograms from data samples partitioned across a CUDA thread block.
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

#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/util_ptx.cuh>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief The BlockHistogramSort class provides sorting-based methods for constructing block-wide
 *        histograms from data samples partitioned across a CUDA thread block.
 *
 * @tparam T
 *   Sample type
 *
 * @tparam BLOCK_DIM_X
 *   The thread block length in threads along the X dimension
 *
 * @tparam ITEMS_PER_THREAD
 *   The number of samples per thread
 *
 * @tparam BINS
 *   The number of bins into which histogram samples may fall
 *
 * @tparam BLOCK_DIM_Y
 *   The thread block length in threads along the Y dimension
 *
 * @tparam BLOCK_DIM_Z
 *   The thread block length in threads along the Z dimension
 */
template <typename T, int BLOCK_DIM_X, int ITEMS_PER_THREAD, int BINS, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
struct BlockHistogramSort
{
  /// Constants
  enum
  {
    /// The thread block size in threads
    BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
  };

  // Parameterize BlockRadixSort type for our thread block
  using BlockRadixSortT =
    BlockRadixSort<T,
                   BLOCK_DIM_X,
                   ITEMS_PER_THREAD,
                   NullType,
                   4,
                   true,
                   BLOCK_SCAN_WARP_SCANS,
                   cudaSharedMemBankSizeFourByte,
                   BLOCK_DIM_Y,
                   BLOCK_DIM_Z>;

  // Parameterize BlockDiscontinuity type for our thread block
  using BlockDiscontinuityT = BlockDiscontinuity<T, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>;

  /// Shared memory
  union _TempStorage
  {
    // Storage for sorting bin values
    typename BlockRadixSortT::TempStorage sort;

    struct Discontinuities
    {
      // Storage for detecting discontinuities in the tile of sorted bin values
      typename BlockDiscontinuityT::TempStorage flag;

      // Storage for noting begin/end offsets of bin runs in the tile of sorted bin values
      unsigned int run_begin[BINS];
      unsigned int run_end[BINS];
    } discontinuities;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // Thread fields
  _TempStorage& temp_storage;
  unsigned int linear_tid;

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockHistogramSort(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  // Discontinuity functor
  struct DiscontinuityOp
  {
    // Reference to temp_storage
    _TempStorage& temp_storage;

    // Constructor
    _CCCL_DEVICE _CCCL_FORCEINLINE DiscontinuityOp(_TempStorage& temp_storage)
        : temp_storage(temp_storage)
    {}

    // Discontinuity predicate
    _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(const T& a, const T& b, int b_index)
    {
      if (a != b)
      {
        // Note the begin/end offsets in shared storage
        temp_storage.discontinuities.run_begin[b] = b_index;
        temp_storage.discontinuities.run_end[a]   = b_index;

        return true;
      }
      else
      {
        return false;
      }
    }
  };

  /**
   * @brief Composite data onto an existing histogram
   *
   * @param[in] items
   *   Calling thread's input values to histogram
   *
   * @param[out] histogram
   *   Reference to shared/device-accessible memory histogram
   */
  template <typename CounterT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Composite(T (&items)[ITEMS_PER_THREAD], CounterT histogram[BINS])
  {
    enum
    {
      TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD
    };

    // Sort bytes in blocked arrangement
    BlockRadixSortT(temp_storage.sort).Sort(items);

    __syncthreads();

    // Initialize the shared memory's run_begin and run_end for each bin
    int histo_offset = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + BLOCK_THREADS <= BINS; histo_offset += BLOCK_THREADS)
    {
      temp_storage.discontinuities.run_begin[histo_offset + linear_tid] = TILE_SIZE;
      temp_storage.discontinuities.run_end[histo_offset + linear_tid]   = TILE_SIZE;
    }
    // Finish up with guarded initialization if necessary
    if ((BINS % BLOCK_THREADS != 0) && (histo_offset + linear_tid < BINS))
    {
      temp_storage.discontinuities.run_begin[histo_offset + linear_tid] = TILE_SIZE;
      temp_storage.discontinuities.run_end[histo_offset + linear_tid]   = TILE_SIZE;
    }

    __syncthreads();

    int flags[ITEMS_PER_THREAD]; // unused

    // Compute head flags to demarcate contiguous runs of the same bin in the sorted tile
    DiscontinuityOp flag_op(temp_storage);
    BlockDiscontinuityT(temp_storage.discontinuities.flag).FlagHeads(flags, items, flag_op);

    // Update begin for first item
    if (linear_tid == 0)
    {
      temp_storage.discontinuities.run_begin[items[0]] = 0;
    }

    __syncthreads();

    // Composite into histogram
    histo_offset = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (; histo_offset + BLOCK_THREADS <= BINS; histo_offset += BLOCK_THREADS)
    {
      int thread_offset = histo_offset + linear_tid;
      CounterT count =
        temp_storage.discontinuities.run_end[thread_offset] - temp_storage.discontinuities.run_begin[thread_offset];
      histogram[thread_offset] += count;
    }

    // Finish up with guarded composition if necessary
    if ((BINS % BLOCK_THREADS != 0) && (histo_offset + linear_tid < BINS))
    {
      int thread_offset = histo_offset + linear_tid;
      CounterT count =
        temp_storage.discontinuities.run_end[thread_offset] - temp_storage.discontinuities.run_begin[thread_offset];
      histogram[thread_offset] += count;
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
