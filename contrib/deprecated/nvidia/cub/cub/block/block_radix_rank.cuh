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
 * \file
 * cub::BlockRadixRank provides operations for ranking unsigned integer types within a CUDA thread block
 */

#pragma once
#pragma clang system_header


#include <stdint.h>

#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../block/block_scan.cuh"
#include "../block/radix_rank_sort_operations.cuh"
#include "../config.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"


CUB_NAMESPACE_BEGIN


/**
 * \brief Radix ranking algorithm, the algorithm used to implement stable ranking of the
 * keys from a single tile. Note that different ranking algorithms require different
 * initial arrangements of keys to function properly.
 */
enum RadixRankAlgorithm
{
    /** Ranking using the BlockRadixRank algorithm with MEMOIZE_OUTER_SCAN == false. It
     * uses thread-private histograms, and thus uses more shared memory. Requires blocked
     * arrangement of keys. Does not support count callbacks. */
    RADIX_RANK_BASIC,
    /** Ranking using the BlockRadixRank algorithm with MEMOIZE_OUTER_SCAN ==
     * true. Similar to RADIX_RANK BASIC, it requires blocked arrangement of
     * keys and does not support count callbacks.*/
    RADIX_RANK_MEMOIZE,
    /** Ranking using the BlockRadixRankMatch algorithm. It uses warp-private
     * histograms and matching for ranking the keys in a single warp. Therefore,
     * it uses less shared memory compared to RADIX_RANK_BASIC. It requires
     * warp-striped key arrangement and supports count callbacks. */
    RADIX_RANK_MATCH,
    /** Ranking using the BlockRadixRankMatchEarlyCounts algorithm with
     * MATCH_ALGORITHM == WARP_MATCH_ANY. An alternative implementation of
     * match-based ranking that computes bin counts early. Because of this, it
     * works better with onesweep sorting, which requires bin counts for
     * decoupled look-back. Assumes warp-striped key arrangement and supports
     * count callbacks.*/
    RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
    /** Ranking using the BlockRadixRankEarlyCounts algorithm with
     * MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR. It uses extra space in shared
     * memory to generate warp match masks using atomicOr(). This is faster when
     * there are few matches, but can lead to slowdowns if the number of
     * matching keys among warp lanes is high. Assumes warp-striped key
     * arrangement and supports count callbacks. */
    RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR
};


/** Empty callback implementation */
template <int BINS_PER_THREAD>
struct BlockRadixRankEmptyCallback
{
    __device__ __forceinline__ void operator()(int (&bins)[BINS_PER_THREAD]) {}
};


namespace detail
{

template <int Bits, int PartialWarpThreads, int PartialWarpId>
struct warp_in_block_matcher_t
{
  static __device__ unsigned int match_any(unsigned int label, unsigned int warp_id)
  {
    if (warp_id == static_cast<unsigned int>(PartialWarpId)) 
    {
      return MatchAny<Bits, PartialWarpThreads>(label);
    }

    return MatchAny<Bits>(label);
  }
};

template <int Bits, int PartialWarpId>
struct warp_in_block_matcher_t<Bits, 0, PartialWarpId>
{
  static __device__ unsigned int match_any(unsigned int label, unsigned int warp_id)
  {
    return MatchAny<Bits>(label);
  }
};

} // namespace detail


/**
 * \brief BlockRadixRank provides operations for ranking unsigned integer types within a CUDA thread block.
 * \ingroup BlockModule
 *
 * \tparam BLOCK_DIM_X          The thread block length in threads along the X dimension
 * \tparam RADIX_BITS           The number of radix bits per digit place
 * \tparam IS_DESCENDING           Whether or not the sorted-order is high-to-low
 * \tparam MEMOIZE_OUTER_SCAN   <b>[optional]</b> Whether or not to buffer outer raking scan partials to incur fewer shared memory reads at the expense of higher register pressure (default: true for architectures SM35 and newer, false otherwise).  See BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE for more details.
 * \tparam INNER_SCAN_ALGORITHM <b>[optional]</b> The cub::BlockScanAlgorithm algorithm to use (default: cub::BLOCK_SCAN_WARP_SCANS)
 * \tparam SMEM_CONFIG          <b>[optional]</b> Shared memory bank mode (default: \p cudaSharedMemBankSizeFourByte)
 * \tparam BLOCK_DIM_Y          <b>[optional]</b> The thread block length in threads along the Y dimension (default: 1)
 * \tparam BLOCK_DIM_Z          <b>[optional]</b> The thread block length in threads along the Z dimension (default: 1)
 * \tparam LEGACY_PTX_ARCH      <b>[optional]</b> Unused.
 *
 * \par Overview
 * Blah...
 * - Keys must be in a form suitable for radix ranking (i.e., unsigned bits).
 * - \blocked
 *
 * \par Performance Considerations
 * - \granularity
 *
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *   constexpr int block_threads = 2;
 *   constexpr int radix_bits = 5;
 *
 *   // Specialize BlockRadixRank for a 1D block of 2 threads 
 *   using block_radix_rank = cub::BlockRadixRank<block_threads, radix_bits>;
 *   using storage_t = typename block_radix_rank::TempStorage;
 *
 *   // Allocate shared memory for BlockRadixSort
 *   __shared__ storage_t temp_storage;
 *
 *   // Obtain a segment of consecutive items that are blocked across threads
 *   int keys[2];
 *   int ranks[2];
 *   ...
 *
 *   cub::BFEDigitExtractor<int> extractor(0, radix_bits);
 *   block_radix_rank(temp_storage).RankKeys(keys, ranks, extractor);
 *
 *   ...
 * \endcode
 * Suppose the set of input `keys` across the block of threads is `{ [16,10], [9,11] }`.  
 * The corresponding output `ranks` in those threads will be `{ [3,1], [0,2] }`.
 *
 * \par Re-using dynamically allocating shared memory
 * The following example under the examples/block folder illustrates usage of
 * dynamically shared memory with BlockReduce and how to re-purpose
 * the same memory region:
 * <a href="../../examples/block/example_block_reduce_dyn_smem.cu">example_block_reduce_dyn_smem.cu</a>
 *
 * This example can be easily adapted to the storage required by BlockRadixRank.
 */
template <
    int                     BLOCK_DIM_X,
    int                     RADIX_BITS,
    bool                    IS_DESCENDING,
    bool                    MEMOIZE_OUTER_SCAN      = true,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM    = BLOCK_SCAN_WARP_SCANS,
    cudaSharedMemConfig     SMEM_CONFIG             = cudaSharedMemBankSizeFourByte,
    int                     BLOCK_DIM_Y             = 1,
    int                     BLOCK_DIM_Z             = 1,
    int                     LEGACY_PTX_ARCH         = 0>
class BlockRadixRank
{
private:

    /******************************************************************************
     * Type definitions and constants
     ******************************************************************************/

    // Integer type for digit counters (to be packed into words of type PackedCounters)
    typedef unsigned short DigitCounter;

    // Integer type for packing DigitCounters into columns of shared memory banks
    using PackedCounter =
      cub::detail::conditional_t<SMEM_CONFIG == cudaSharedMemBankSizeEightByte,
                                 unsigned long long,
                                 unsigned int>;

    enum
    {
        // The thread block size in threads
        BLOCK_THREADS               = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

        RADIX_DIGITS                = 1 << RADIX_BITS,

        LOG_WARP_THREADS            = CUB_LOG_WARP_THREADS(0),
        WARP_THREADS                = 1 << LOG_WARP_THREADS,
        WARPS                       = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        BYTES_PER_COUNTER           = sizeof(DigitCounter),
        LOG_BYTES_PER_COUNTER       = Log2<BYTES_PER_COUNTER>::VALUE,

        PACKING_RATIO               = static_cast<int>(sizeof(PackedCounter) / sizeof(DigitCounter)),
        LOG_PACKING_RATIO           = Log2<PACKING_RATIO>::VALUE,

        LOG_COUNTER_LANES           = CUB_MAX((int(RADIX_BITS) - int(LOG_PACKING_RATIO)), 0),                // Always at least one lane
        COUNTER_LANES               = 1 << LOG_COUNTER_LANES,

        // The number of packed counters per thread (plus one for padding)
        PADDED_COUNTER_LANES        = COUNTER_LANES + 1,
        RAKING_SEGMENT              = PADDED_COUNTER_LANES,
    };

public:

    enum
    {
        /// Number of bin-starting offsets tracked per thread
        BINS_TRACKED_PER_THREAD = CUB_MAX(1, (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS),
    };

private:


    /// BlockScan type
    typedef BlockScan<
            PackedCounter,
            BLOCK_DIM_X,
            INNER_SCAN_ALGORITHM,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z>
        BlockScan;


    /// Shared memory storage layout type for BlockRadixRank
    struct __align__(16) _TempStorage
    {
        union Aliasable
        {
            DigitCounter            digit_counters[PADDED_COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
            PackedCounter           raking_grid[BLOCK_THREADS][RAKING_SEGMENT];

        } aliasable;

        // Storage for scanning local ranks
        typename BlockScan::TempStorage block_scan;
    };


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;

    /// Copy of raking segment, promoted to registers
    PackedCounter cached_segment[RAKING_SEGMENT];


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /**
     * Internal storage allocator
     */
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    /**
     * Performs upsweep raking reduction, returning the aggregate
     */
    __device__ __forceinline__ PackedCounter Upsweep()
    {
        PackedCounter *smem_raking_ptr = temp_storage.aliasable.raking_grid[linear_tid];
        PackedCounter *raking_ptr;

        if (MEMOIZE_OUTER_SCAN)
        {
            // Copy data into registers
            #pragma unroll
            for (int i = 0; i < RAKING_SEGMENT; i++)
            {
                cached_segment[i] = smem_raking_ptr[i];
            }
            raking_ptr = cached_segment;
        }
        else
        {
            raking_ptr = smem_raking_ptr;
        }

        return internal::ThreadReduce<RAKING_SEGMENT>(raking_ptr, Sum());
    }


    /// Performs exclusive downsweep raking scan
    __device__ __forceinline__ void ExclusiveDownsweep(
        PackedCounter raking_partial)
    {
        PackedCounter *smem_raking_ptr = temp_storage.aliasable.raking_grid[linear_tid];

        PackedCounter *raking_ptr = (MEMOIZE_OUTER_SCAN) ?
            cached_segment :
            smem_raking_ptr;

        // Exclusive raking downsweep scan
        internal::ThreadScanExclusive<RAKING_SEGMENT>(raking_ptr, raking_ptr, Sum(), raking_partial);

        if (MEMOIZE_OUTER_SCAN)
        {
            // Copy data back to smem
            #pragma unroll
            for (int i = 0; i < RAKING_SEGMENT; i++)
            {
                smem_raking_ptr[i] = cached_segment[i];
            }
        }
    }


    /**
     * Reset shared memory digit counters
     */
    __device__ __forceinline__ void ResetCounters()
    {
        // Reset shared memory digit counters
        #pragma unroll
        for (int LANE = 0; LANE < PADDED_COUNTER_LANES; LANE++)
        {
            *((PackedCounter*) temp_storage.aliasable.digit_counters[LANE][linear_tid]) = 0;
        }
    }


    /**
     * Block-scan prefix callback
     */
    struct PrefixCallBack
    {
        __device__ __forceinline__ PackedCounter operator()(PackedCounter block_aggregate)
        {
            PackedCounter block_prefix = 0;

            // Propagate totals in packed fields
            #pragma unroll
            for (int PACKED = 1; PACKED < PACKING_RATIO; PACKED++)
            {
                block_prefix += block_aggregate << (sizeof(DigitCounter) * 8 * PACKED);
            }

            return block_prefix;
        }
    };


    /**
     * Scan shared memory digit counters.
     */
    __device__ __forceinline__ void ScanCounters()
    {
        // Upsweep scan
        PackedCounter raking_partial = Upsweep();

        // Compute exclusive sum
        PackedCounter exclusive_partial;
        PrefixCallBack prefix_call_back;
        BlockScan(temp_storage.block_scan).ExclusiveSum(raking_partial, exclusive_partial, prefix_call_back);

        // Downsweep scan with exclusive partial
        ExclusiveDownsweep(exclusive_partial);
    }

public:

    /// \smemstorage{BlockScan}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    __device__ __forceinline__ BlockRadixRank()
    :
        temp_storage(PrivateStorage()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    __device__ __forceinline__ BlockRadixRank(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    //@}  end member group
    /******************************************************************//**
     * \name Raking
     *********************************************************************/
    //@{

    /**
     * \brief Rank keys.
     */
    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
        int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile
        DigitExtractorT digit_extractor)                    ///< [in] The digit extractor
    {
        DigitCounter    thread_prefixes[KEYS_PER_THREAD];   // For each key, the count of previous keys in this tile having the same digit
        DigitCounter*   digit_counters[KEYS_PER_THREAD];    // For each key, the byte-offset of its corresponding digit counter in smem

        // Reset shared memory digit counters
        ResetCounters();

        #pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
        {
            // Get digit
            unsigned int digit = digit_extractor.Digit(keys[ITEM]);

            // Get sub-counter
            unsigned int sub_counter = digit >> LOG_COUNTER_LANES;

            // Get counter lane
            unsigned int counter_lane = digit & (COUNTER_LANES - 1);

            if (IS_DESCENDING)
            {
                sub_counter = PACKING_RATIO - 1 - sub_counter;
                counter_lane = COUNTER_LANES - 1 - counter_lane;
            }

            // Pointer to smem digit counter
            digit_counters[ITEM] = &temp_storage.aliasable.digit_counters[counter_lane][linear_tid][sub_counter];

            // Load thread-exclusive prefix
            thread_prefixes[ITEM] = *digit_counters[ITEM];

            // Store inclusive prefix
            *digit_counters[ITEM] = thread_prefixes[ITEM] + 1;
        }

        CTA_SYNC();

        // Scan shared memory counters
        ScanCounters();

        CTA_SYNC();

        // Extract the local ranks of each key
        #pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
        {
            // Add in thread block exclusive prefix
            ranks[ITEM] = thread_prefixes[ITEM] + *digit_counters[ITEM];
        }
    }


    /**
     * \brief Rank keys.  For the lower \p RADIX_DIGITS threads, digit counts for each digit are provided for the corresponding thread.
     */
    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
        int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile (out parameter)
        DigitExtractorT digit_extractor,                    ///< [in] The digit extractor
        int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD])            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
    {
        // Rank keys
        RankKeys(keys, ranks, digit_extractor);

        // Get the inclusive and exclusive digit totals corresponding to the calling thread.
        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (linear_tid * BINS_TRACKED_PER_THREAD) + track;

            if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
            {
                if (IS_DESCENDING)
                    bin_idx = RADIX_DIGITS - bin_idx - 1;

                // Obtain ex/inclusive digit counts.  (Unfortunately these all reside in the
                // first counter column, resulting in unavoidable bank conflicts.)
                unsigned int counter_lane   = (bin_idx & (COUNTER_LANES - 1));
                unsigned int sub_counter    = bin_idx >> (LOG_COUNTER_LANES);

                exclusive_digit_prefix[track] = temp_storage.aliasable.digit_counters[counter_lane][0][sub_counter];
            }
        }
    }
};





/**
 * Radix-rank using match.any
 */
template <
    int                     BLOCK_DIM_X,
    int                     RADIX_BITS,
    bool                    IS_DESCENDING,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM    = BLOCK_SCAN_WARP_SCANS,
    int                     BLOCK_DIM_Y             = 1,
    int                     BLOCK_DIM_Z             = 1,
    int                     LEGACY_PTX_ARCH         = 0>
class BlockRadixRankMatch
{
private:

    /******************************************************************************
     * Type definitions and constants
     ******************************************************************************/

    typedef int32_t    RankT;
    typedef int32_t    DigitCounterT;

    enum
    {
        // The thread block size in threads
        BLOCK_THREADS               = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

        RADIX_DIGITS                = 1 << RADIX_BITS,

        LOG_WARP_THREADS            = CUB_LOG_WARP_THREADS(0),
        WARP_THREADS                = 1 << LOG_WARP_THREADS,
        PARTIAL_WARP_THREADS        = BLOCK_THREADS % WARP_THREADS,
        WARPS                       = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        PADDED_WARPS            = ((WARPS & 0x1) == 0) ?
                                    WARPS + 1 :
                                    WARPS,

        COUNTERS                = PADDED_WARPS * RADIX_DIGITS,
        RAKING_SEGMENT          = (COUNTERS + BLOCK_THREADS - 1) / BLOCK_THREADS,
        PADDED_RAKING_SEGMENT   = ((RAKING_SEGMENT & 0x1) == 0) ?
                                    RAKING_SEGMENT + 1 :
                                    RAKING_SEGMENT,
    };

public:

    enum
    {
        /// Number of bin-starting offsets tracked per thread
        BINS_TRACKED_PER_THREAD = CUB_MAX(1, (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS),
    };

private:

    /// BlockScan type
    typedef BlockScan<
            DigitCounterT,
            BLOCK_THREADS,
            INNER_SCAN_ALGORITHM,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z>
        BlockScanT;


    /// Shared memory storage layout type for BlockRadixRank
    struct __align__(16) _TempStorage
    {
        typename BlockScanT::TempStorage            block_scan;

        union __align__(16) Aliasable
        {
            volatile DigitCounterT                  warp_digit_counters[RADIX_DIGITS][PADDED_WARPS];
            DigitCounterT                           raking_grid[BLOCK_THREADS][PADDED_RAKING_SEGMENT];

        } aliasable;
    };


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;



public:

    /// \smemstorage{BlockScan}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    __device__ __forceinline__ BlockRadixRankMatch(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    //@}  end member group
    /******************************************************************//**
     * \name Raking
     *********************************************************************/
    //@{

    /** \brief Computes the count of keys for each digit value, and calls the
     * callback with the array of key counts.

     * @tparam CountsCallback The callback type. It should implement an instance
     * overload of operator()(int (&bins)[BINS_TRACKED_PER_THREAD]), where bins
     * is an array of key counts for each digit value distributed in block
     * distribution among the threads of the thread block. Key counts can be
     * used, to update other data structures in global or shared
     * memory. Depending on the implementation of the ranking algoirhtm
     * (see BlockRadixRankMatchEarlyCounts), key counts may become available
     * early, therefore, they are returned through a callback rather than a
     * separate output parameter of RankKeys().
     */
    template <int KEYS_PER_THREAD, typename CountsCallback>
    __device__ __forceinline__ void CallBack(CountsCallback callback)
    {
        int bins[BINS_TRACKED_PER_THREAD];
        // Get count for each digit
        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (linear_tid * BINS_TRACKED_PER_THREAD) + track;
            const int TILE_ITEMS = KEYS_PER_THREAD * BLOCK_THREADS;

            if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
            {
                if (IS_DESCENDING)
                {
                    bin_idx = RADIX_DIGITS - bin_idx - 1;
                    bins[track] = (bin_idx > 0 ?
                        temp_storage.aliasable.warp_digit_counters[bin_idx - 1][0] : TILE_ITEMS) -
                        temp_storage.aliasable.warp_digit_counters[bin_idx][0];
                }
                else
                {
                    bins[track] = (bin_idx < RADIX_DIGITS - 1 ?
                        temp_storage.aliasable.warp_digit_counters[bin_idx + 1][0] : TILE_ITEMS) -
                        temp_storage.aliasable.warp_digit_counters[bin_idx][0];
                }
            }
        }
        callback(bins);
    }

    /**
     * \brief Rank keys.
     */
    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT,
        typename        CountsCallback>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
        int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile
        DigitExtractorT digit_extractor,                    ///< [in] The digit extractor
        CountsCallback    callback)
    {
        // Initialize shared digit counters

        #pragma unroll
        for (int ITEM = 0; ITEM < PADDED_RAKING_SEGMENT; ++ITEM)
            temp_storage.aliasable.raking_grid[linear_tid][ITEM] = 0;

        CTA_SYNC();

        // Each warp will strip-mine its section of input, one strip at a time

        volatile DigitCounterT  *digit_counters[KEYS_PER_THREAD];
        uint32_t                warp_id         = linear_tid >> LOG_WARP_THREADS;
        uint32_t                lane_mask_lt    = LaneMaskLt();

        #pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
        {
            // My digit
            uint32_t digit = digit_extractor.Digit(keys[ITEM]);

            if (IS_DESCENDING)
                digit = RADIX_DIGITS - digit - 1;

            // Mask of peers who have same digit as me
            uint32_t peer_mask =
              detail::warp_in_block_matcher_t<
                RADIX_BITS, 
                PARTIAL_WARP_THREADS, 
                WARPS - 1>::match_any(digit, warp_id);

            // Pointer to smem digit counter for this key
            digit_counters[ITEM] = &temp_storage.aliasable.warp_digit_counters[digit][warp_id];

            // Number of occurrences in previous strips
            DigitCounterT warp_digit_prefix = *digit_counters[ITEM];

            // Warp-sync
            WARP_SYNC(0xFFFFFFFF);

            // Number of peers having same digit as me
            int32_t digit_count = __popc(peer_mask);

            // Number of lower-ranked peers having same digit seen so far
            int32_t peer_digit_prefix = __popc(peer_mask & lane_mask_lt);

            if (peer_digit_prefix == 0)
            {
                // First thread for each digit updates the shared warp counter
                *digit_counters[ITEM] = DigitCounterT(warp_digit_prefix + digit_count);
            }

            // Warp-sync
            WARP_SYNC(0xFFFFFFFF);

            // Number of prior keys having same digit
            ranks[ITEM] = warp_digit_prefix + DigitCounterT(peer_digit_prefix);
        }

        CTA_SYNC();

        // Scan warp counters

        DigitCounterT scan_counters[PADDED_RAKING_SEGMENT];

        #pragma unroll
        for (int ITEM = 0; ITEM < PADDED_RAKING_SEGMENT; ++ITEM)
            scan_counters[ITEM] = temp_storage.aliasable.raking_grid[linear_tid][ITEM];

        BlockScanT(temp_storage.block_scan).ExclusiveSum(scan_counters, scan_counters);

        #pragma unroll
        for (int ITEM = 0; ITEM < PADDED_RAKING_SEGMENT; ++ITEM)
            temp_storage.aliasable.raking_grid[linear_tid][ITEM] = scan_counters[ITEM];

        CTA_SYNC();
        if (!std::is_same<
              CountsCallback,
              BlockRadixRankEmptyCallback<BINS_TRACKED_PER_THREAD>>::value)
        {
            CallBack<KEYS_PER_THREAD>(callback);
        }

        // Seed ranks with counter values from previous warps
        #pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
            ranks[ITEM] += *digit_counters[ITEM];
    }

    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD], int (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor)
    {
        RankKeys(keys, ranks, digit_extractor,
                 BlockRadixRankEmptyCallback<BINS_TRACKED_PER_THREAD>());
    }

    /**
     * \brief Rank keys.  For the lower \p RADIX_DIGITS threads, digit counts for each digit are provided for the corresponding thread.
     */
    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT,
        typename        CountsCallback>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
        int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile (out parameter)
        DigitExtractorT digit_extractor,                    ///< [in] The digit extractor
        int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD],            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
        CountsCallback callback)
    {
        RankKeys(keys, ranks, digit_extractor, callback);

        // Get exclusive count for each digit
        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (linear_tid * BINS_TRACKED_PER_THREAD) + track;

            if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
            {
                if (IS_DESCENDING)
                    bin_idx = RADIX_DIGITS - bin_idx - 1;

                exclusive_digit_prefix[track] = temp_storage.aliasable.warp_digit_counters[bin_idx][0];
            }
        }
    }

    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],           ///< [in] Keys for this tile
        int             (&ranks)[KEYS_PER_THREAD],          ///< [out] For each key, the local rank within the tile (out parameter)
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD])            ///< [out] The exclusive prefix sum for the digits [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD - 1]
    {
        RankKeys(keys, ranks, digit_extractor, exclusive_digit_prefix,
                 BlockRadixRankEmptyCallback<BINS_TRACKED_PER_THREAD>());
    }
};

enum WarpMatchAlgorithm
{
    WARP_MATCH_ANY,
    WARP_MATCH_ATOMIC_OR
};

/**
 * Radix-rank using matching which computes the counts of keys for each digit
 * value early, at the expense of doing more work. This may be useful e.g. for
 * decoupled look-back, where it reduces the time other thread blocks need to
 * wait for digit counts to become available.
 */
template <int BLOCK_DIM_X, int RADIX_BITS, bool IS_DESCENDING,
          BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS,
          WarpMatchAlgorithm MATCH_ALGORITHM = WARP_MATCH_ANY, int NUM_PARTS = 1>
struct BlockRadixRankMatchEarlyCounts
{
    // constants
    enum
    {
        BLOCK_THREADS = BLOCK_DIM_X,
        RADIX_DIGITS = 1 << RADIX_BITS,
        BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS,
        BINS_TRACKED_PER_THREAD = BINS_PER_THREAD,
        FULL_BINS = BINS_PER_THREAD * BLOCK_THREADS == RADIX_DIGITS,
        WARP_THREADS = CUB_PTX_WARP_THREADS,
        PARTIAL_WARP_THREADS = BLOCK_THREADS % WARP_THREADS,
        BLOCK_WARPS = BLOCK_THREADS / WARP_THREADS,
        PARTIAL_WARP_ID = BLOCK_WARPS - 1,
        WARP_MASK = ~0,
        NUM_MATCH_MASKS = MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR ? BLOCK_WARPS : 0,
        // Guard against declaring zero-sized array:
        MATCH_MASKS_ALLOC_SIZE = NUM_MATCH_MASKS < 1 ? 1 : NUM_MATCH_MASKS,
    };

    // types
    typedef cub::BlockScan<int, BLOCK_THREADS, INNER_SCAN_ALGORITHM> BlockScan;

    

    // temporary storage
    struct TempStorage
    {
        union
        {
            int warp_offsets[BLOCK_WARPS][RADIX_DIGITS];
            int warp_histograms[BLOCK_WARPS][RADIX_DIGITS][NUM_PARTS];
        };

        int match_masks[MATCH_MASKS_ALLOC_SIZE][RADIX_DIGITS];

        typename BlockScan::TempStorage prefix_tmp;
    };

    TempStorage& temp_storage;

    // internal ranking implementation
    template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT,
              typename CountsCallback>
    struct BlockRadixRankMatchInternal
    {
        TempStorage& s;
        DigitExtractorT digit_extractor;
        CountsCallback callback;
        int warp;
        int lane;

        __device__ __forceinline__ int Digit(UnsignedBits key)
        {
            int digit =  digit_extractor.Digit(key);
            return IS_DESCENDING ? RADIX_DIGITS - 1 - digit : digit;
        }

        __device__ __forceinline__ int ThreadBin(int u)
        {
            int bin = threadIdx.x * BINS_PER_THREAD + u;
            return IS_DESCENDING ? RADIX_DIGITS - 1 - bin : bin;
        }

        __device__ __forceinline__
        void ComputeHistogramsWarp(UnsignedBits (&keys)[KEYS_PER_THREAD])
        {
            //int* warp_offsets = &s.warp_offsets[warp][0];
            int (&warp_histograms)[RADIX_DIGITS][NUM_PARTS] = s.warp_histograms[warp];
            // compute warp-private histograms
            #pragma unroll
            for (int bin = lane; bin < RADIX_DIGITS; bin += WARP_THREADS)
            {
                #pragma unroll
                for (int part = 0; part < NUM_PARTS; ++part)
                {
                    warp_histograms[bin][part] = 0;
                }
            }
            if (MATCH_ALGORITHM == WARP_MATCH_ATOMIC_OR)
            {
                int* match_masks = &s.match_masks[warp][0];
                #pragma unroll
                for (int bin = lane; bin < RADIX_DIGITS; bin += WARP_THREADS)
                {
                    match_masks[bin] = 0;
                }                    
            }
            WARP_SYNC(WARP_MASK);

            // compute private per-part histograms
            int part = lane % NUM_PARTS;
            #pragma unroll
            for (int u = 0; u < KEYS_PER_THREAD; ++u)
            {
                atomicAdd(&warp_histograms[Digit(keys[u])][part], 1);
            }
            
            // sum different parts;
            // no extra work is necessary if NUM_PARTS == 1
            if (NUM_PARTS > 1)
            {
                WARP_SYNC(WARP_MASK);
                // TODO: handle RADIX_DIGITS % WARP_THREADS != 0 if it becomes necessary
                const int WARP_BINS_PER_THREAD = RADIX_DIGITS / WARP_THREADS;
                int bins[WARP_BINS_PER_THREAD];
                #pragma unroll
                for (int u = 0; u < WARP_BINS_PER_THREAD; ++u)
                {
                    int bin = lane + u * WARP_THREADS;
                    bins[u] = internal::ThreadReduce(warp_histograms[bin], Sum());
                }
                CTA_SYNC();

                // store the resulting histogram in shared memory
                int* warp_offsets = &s.warp_offsets[warp][0];
                #pragma unroll
                for (int u = 0; u < WARP_BINS_PER_THREAD; ++u)
                {
                    int bin = lane + u * WARP_THREADS;
                    warp_offsets[bin] = bins[u];
                }
            }
        }

        __device__ __forceinline__
        void ComputeOffsetsWarpUpsweep(int (&bins)[BINS_PER_THREAD])
        {
            // sum up warp-private histograms
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u) 
            {
                bins[u] = 0;
                int bin = ThreadBin(u);
                if (FULL_BINS || (bin >= 0 && bin < RADIX_DIGITS))
                {
                    #pragma unroll
                    for (int j_warp = 0; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        int warp_offset = s.warp_offsets[j_warp][bin];
                        s.warp_offsets[j_warp][bin] = bins[u];
                        bins[u] += warp_offset;
                    }
                }
            }
        }

        __device__ __forceinline__
        void ComputeOffsetsWarpDownsweep(int (&offsets)[BINS_PER_THREAD])
        {
            #pragma unroll
            for (int u = 0; u < BINS_PER_THREAD; ++u)
            {
                int bin = ThreadBin(u);
                if (FULL_BINS || (bin >= 0 && bin < RADIX_DIGITS))
                {
                    int digit_offset = offsets[u];
                    #pragma unroll
                    for (int j_warp = 0; j_warp < BLOCK_WARPS; ++j_warp)
                    {
                        s.warp_offsets[j_warp][bin] += digit_offset;
                    }
                }
            }
        }

        __device__ __forceinline__
        void ComputeRanksItem(
            UnsignedBits (&keys)[KEYS_PER_THREAD], int (&ranks)[KEYS_PER_THREAD],
            Int2Type<WARP_MATCH_ATOMIC_OR>)
        {
            // compute key ranks
            int lane_mask = 1 << lane;
            int* warp_offsets = &s.warp_offsets[warp][0];
            int* match_masks = &s.match_masks[warp][0];
            #pragma unroll
            for (int u = 0; u < KEYS_PER_THREAD; ++u)
            {
                int bin = Digit(keys[u]);
                int* p_match_mask = &match_masks[bin];
                atomicOr(p_match_mask, lane_mask);
                WARP_SYNC(WARP_MASK);
                int bin_mask = *p_match_mask;
                int leader = (WARP_THREADS - 1) - __clz(bin_mask);
                int warp_offset = 0;
                int popc = __popc(bin_mask & LaneMaskLe());
                if (lane == leader)
                {
                    // atomic is a bit faster
                    warp_offset = atomicAdd(&warp_offsets[bin], popc);
                }
                warp_offset = SHFL_IDX_SYNC(warp_offset, leader, WARP_MASK);
                if (lane == leader) *p_match_mask = 0;
                WARP_SYNC(WARP_MASK);
                ranks[u] = warp_offset + popc - 1;
            }
        }

        __device__ __forceinline__
        void ComputeRanksItem(
            UnsignedBits (&keys)[KEYS_PER_THREAD], int (&ranks)[KEYS_PER_THREAD],
            Int2Type<WARP_MATCH_ANY>)
        {
            // compute key ranks
            int* warp_offsets = &s.warp_offsets[warp][0];
            #pragma unroll
            for (int u = 0; u < KEYS_PER_THREAD; ++u)
            {
                int bin = Digit(keys[u]);
                int bin_mask = detail::warp_in_block_matcher_t<RADIX_BITS,
                                                               PARTIAL_WARP_THREADS,
                                                               BLOCK_WARPS - 1>::match_any(bin,
                                                                                           warp);
                int leader = (WARP_THREADS - 1) - __clz(bin_mask);
                int warp_offset = 0;
                int popc = __popc(bin_mask & LaneMaskLe());
                if (lane == leader)
                {
                    // atomic is a bit faster
                    warp_offset = atomicAdd(&warp_offsets[bin], popc);
                }
                warp_offset = SHFL_IDX_SYNC(warp_offset, leader, WARP_MASK);
                ranks[u] = warp_offset + popc - 1;
            }
        }

        __device__ __forceinline__ void RankKeys(
            UnsignedBits (&keys)[KEYS_PER_THREAD],
            int (&ranks)[KEYS_PER_THREAD],
            int (&exclusive_digit_prefix)[BINS_PER_THREAD])
        {
            ComputeHistogramsWarp(keys);
            
            CTA_SYNC();
            int bins[BINS_PER_THREAD];
            ComputeOffsetsWarpUpsweep(bins);
            callback(bins);
            
            BlockScan(s.prefix_tmp).ExclusiveSum(bins, exclusive_digit_prefix);

            ComputeOffsetsWarpDownsweep(exclusive_digit_prefix);
            CTA_SYNC();
            ComputeRanksItem(keys, ranks, Int2Type<MATCH_ALGORITHM>());
        }

        __device__ __forceinline__ BlockRadixRankMatchInternal
        (TempStorage& temp_storage, DigitExtractorT digit_extractor, CountsCallback callback)
            : s(temp_storage), digit_extractor(digit_extractor),
              callback(callback), warp(threadIdx.x / WARP_THREADS), lane(LaneId())
            {}
    };

    __device__ __forceinline__ BlockRadixRankMatchEarlyCounts
    (TempStorage& temp_storage) : temp_storage(temp_storage) {}

    /**
     * \brief Rank keys.  For the lower \p RADIX_DIGITS threads, digit counts for each digit are provided for the corresponding thread.
     */
    template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT,
        typename CountsCallback>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_PER_THREAD],
        CountsCallback  callback)
    {
        BlockRadixRankMatchInternal<UnsignedBits, KEYS_PER_THREAD, DigitExtractorT, CountsCallback>
            internal(temp_storage, digit_extractor, callback);
        internal.RankKeys(keys, ranks, exclusive_digit_prefix);        
    }

    template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_PER_THREAD])
    {
        typedef BlockRadixRankEmptyCallback<BINS_PER_THREAD> CountsCallback;
        BlockRadixRankMatchInternal<UnsignedBits, KEYS_PER_THREAD, DigitExtractorT, CountsCallback>
            internal(temp_storage, digit_extractor, CountsCallback());
        internal.RankKeys(keys, ranks, exclusive_digit_prefix);
    }

    template <typename UnsignedBits, int KEYS_PER_THREAD, typename DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor)
    {
        int exclusive_digit_prefix[BINS_PER_THREAD];
        RankKeys(keys, ranks, digit_extractor, exclusive_digit_prefix);
    }
};


namespace detail 
{

// `BlockRadixRank` doesn't conform to the typical pattern, not exposing the algorithm 
// template parameter. Other algorithms don't provide the same template parameters, not allowing 
// multi-dimensional thread block specializations. 
// 
// TODO(senior-zero) for 3.0:
// - Put existing implementations into the detail namespace
// - Support multi-dimensional thread blocks in the rest of implementations
// - Repurpose BlockRadixRank as an entry name with the algorithm template parameter
template <RadixRankAlgorithm RankAlgorithm,
          int BlockDimX,
          int RadixBits,
          bool IsDescending,
          BlockScanAlgorithm ScanAlgorithm>
using block_radix_rank_t = cub::detail::conditional_t<
  RankAlgorithm == RADIX_RANK_BASIC,
  BlockRadixRank<BlockDimX, RadixBits, IsDescending, false, ScanAlgorithm>,
  cub::detail::conditional_t<
    RankAlgorithm == RADIX_RANK_MEMOIZE,
    BlockRadixRank<BlockDimX, RadixBits, IsDescending, true, ScanAlgorithm>,
    cub::detail::conditional_t<
      RankAlgorithm == RADIX_RANK_MATCH,
      BlockRadixRankMatch<BlockDimX, RadixBits, IsDescending, ScanAlgorithm>,
      cub::detail::conditional_t<
        RankAlgorithm == RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BlockRadixRankMatchEarlyCounts<BlockDimX,
                                       RadixBits,
                                       IsDescending,
                                       ScanAlgorithm,
                                       WARP_MATCH_ANY>,
        BlockRadixRankMatchEarlyCounts<BlockDimX,
                                       RadixBits,
                                       IsDescending,
                                       ScanAlgorithm,
                                       WARP_MATCH_ATOMIC_OR>>>>>;

} // namespace detail


CUB_NAMESPACE_END
