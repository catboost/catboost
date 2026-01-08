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
 * cub::DeviceRadixSort provides device-wide, parallel operations for computing a radix sort across a sequence of data items residing within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <stdio.h>
#include <iterator>

#include <cub/agent/agent_radix_sort_downsweep.cuh>
#include <cub/agent/agent_radix_sort_histogram.cuh>
#include <cub/agent/agent_radix_sort_onesweep.cuh>
#include <cub/agent/agent_radix_sort_upsweep.cuh>
#include <cub/agent/agent_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/config.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

// suppress warnings triggered by #pragma unroll:
// "warning: loop not unrolled: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering [-Wpass-failed=transform-warning]"
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wpass-failed"
#endif

CUB_NAMESPACE_BEGIN


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Upsweep digit-counting kernel entry point (multi-block).  Computes privatized digit histograms, one per block.
 */
template <
    typename                ChainedPolicyT,                 ///< Chained tuning policy
    bool                    ALT_DIGIT_BITS,                 ///< Whether or not to use the alternate (lower-bits) policy
    bool                    IS_DESCENDING,                  ///< Whether or not the sorted-order is high-to-low
    typename                KeyT,                           ///< Key type
    typename                OffsetT>                        ///< Signed integer type for global offsets
__launch_bounds__ (int((ALT_DIGIT_BITS) ?
    int(ChainedPolicyT::ActivePolicy::AltUpsweepPolicy::BLOCK_THREADS) :
    int(ChainedPolicyT::ActivePolicy::UpsweepPolicy::BLOCK_THREADS)))
__global__ void DeviceRadixSortUpsweepKernel(
    const KeyT              *d_keys,                        ///< [in] Input keys buffer
    OffsetT                 *d_spine,                       ///< [out] Privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    OffsetT                 /*num_items*/,                  ///< [in] Total number of input data items
    int                     current_bit,                    ///< [in] Bit position of current radix digit
    int                     num_bits,                       ///< [in] Number of bits of current radix digit
    GridEvenShare<OffsetT>  even_share)                     ///< [in] Even-share descriptor for mapan equal number of tiles onto each thread block
{
    using ActiveUpsweepPolicyT =
      cub::detail::conditional_t<
        ALT_DIGIT_BITS,
        typename ChainedPolicyT::ActivePolicy::AltUpsweepPolicy,
        typename ChainedPolicyT::ActivePolicy::UpsweepPolicy>;

    using ActiveDownsweepPolicyT =
      cub::detail::conditional_t<
        ALT_DIGIT_BITS,
        typename ChainedPolicyT::ActivePolicy::AltDownsweepPolicy,
        typename ChainedPolicyT::ActivePolicy::DownsweepPolicy>;

    enum {
        TILE_ITEMS = CUB_MAX(
            ActiveUpsweepPolicyT::BLOCK_THREADS * ActiveUpsweepPolicyT::ITEMS_PER_THREAD,
            ActiveDownsweepPolicyT::BLOCK_THREADS * ActiveDownsweepPolicyT::ITEMS_PER_THREAD)
    };

    // Parameterize AgentRadixSortUpsweep type for the current configuration
    typedef AgentRadixSortUpsweep<
            ActiveUpsweepPolicyT,
            KeyT,
            OffsetT>
        AgentRadixSortUpsweepT;

    // Shared memory storage
    __shared__ typename AgentRadixSortUpsweepT::TempStorage temp_storage;

    // Initialize GRID_MAPPING_RAKE even-share descriptor for this thread block
    even_share.template BlockInit<TILE_ITEMS, GRID_MAPPING_RAKE>();

    AgentRadixSortUpsweepT upsweep(temp_storage, d_keys, current_bit, num_bits);

    upsweep.ProcessRegion(even_share.block_offset, even_share.block_end);

    CTA_SYNC();

    // Write out digit counts (striped)
    upsweep.template ExtractCounts<IS_DESCENDING>(d_spine, gridDim.x, blockIdx.x);
}


/**
 * Spine scan kernel entry point (single-block).  Computes an exclusive prefix sum over the privatized digit histograms
 */
template <
    typename                ChainedPolicyT,                 ///< Chained tuning policy
    typename                OffsetT>                        ///< Signed integer type for global offsets
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::ScanPolicy::BLOCK_THREADS), 1)
__global__ void RadixSortScanBinsKernel(
    OffsetT                 *d_spine,                       ///< [in,out] Privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    int                     num_counts)                     ///< [in] Total number of bin-counts
{
    // Parameterize the AgentScan type for the current configuration
    typedef AgentScan<
            typename ChainedPolicyT::ActivePolicy::ScanPolicy,
            OffsetT*,
            OffsetT*,
            cub::Sum,
            OffsetT,
            OffsetT,
            OffsetT>
        AgentScanT;

    // Shared memory storage
    __shared__ typename AgentScanT::TempStorage temp_storage;

    // Block scan instance
    AgentScanT block_scan(temp_storage, d_spine, d_spine, cub::Sum(), OffsetT(0)) ;

    // Process full input tiles
    int block_offset = 0;
    BlockScanRunningPrefixOp<OffsetT, Sum> prefix_op(0, Sum());
    while (block_offset + AgentScanT::TILE_ITEMS <= num_counts)
    {
        block_scan.template ConsumeTile<false, false>(block_offset, prefix_op);
        block_offset += AgentScanT::TILE_ITEMS;
    }

    // Process the remaining partial tile (if any).
    if (block_offset < num_counts)
    {
        block_scan.template ConsumeTile<false, true>(block_offset, prefix_op,
                                                     num_counts - block_offset);
    }
}


/**
 * Downsweep pass kernel entry point (multi-block).  Scatters keys (and values) into corresponding bins for the current digit place.
 */
template <
    typename                ChainedPolicyT,                 ///< Chained tuning policy
    bool                    ALT_DIGIT_BITS,                 ///< Whether or not to use the alternate (lower-bits) policy
    bool                    IS_DESCENDING,                  ///< Whether or not the sorted-order is high-to-low
    typename                KeyT,                           ///< Key type
    typename                ValueT,                         ///< Value type
    typename                OffsetT>                        ///< Signed integer type for global offsets
__launch_bounds__ (int((ALT_DIGIT_BITS) ?
    int(ChainedPolicyT::ActivePolicy::AltDownsweepPolicy::BLOCK_THREADS) :
    int(ChainedPolicyT::ActivePolicy::DownsweepPolicy::BLOCK_THREADS)))
__global__ void DeviceRadixSortDownsweepKernel(
    const KeyT              *d_keys_in,                     ///< [in] Input keys buffer
    KeyT                    *d_keys_out,                    ///< [in] Output keys buffer
    const ValueT            *d_values_in,                   ///< [in] Input values buffer
    ValueT                  *d_values_out,                  ///< [in] Output values buffer
    OffsetT                 *d_spine,                       ///< [in] Scan of privatized (per block) digit histograms (striped, i.e., 0s counts from each block, then 1s counts from each block, etc.)
    OffsetT                 num_items,                      ///< [in] Total number of input data items
    int                     current_bit,                    ///< [in] Bit position of current radix digit
    int                     num_bits,                       ///< [in] Number of bits of current radix digit
    GridEvenShare<OffsetT>  even_share)                     ///< [in] Even-share descriptor for mapan equal number of tiles onto each thread block
{
    using ActiveUpsweepPolicyT =
      cub::detail::conditional_t<
        ALT_DIGIT_BITS,
        typename ChainedPolicyT::ActivePolicy::AltUpsweepPolicy,
        typename ChainedPolicyT::ActivePolicy::UpsweepPolicy>;

    using ActiveDownsweepPolicyT =
      cub::detail::conditional_t<
        ALT_DIGIT_BITS,
        typename ChainedPolicyT::ActivePolicy::AltDownsweepPolicy,
        typename ChainedPolicyT::ActivePolicy::DownsweepPolicy>;

    enum {
        TILE_ITEMS = CUB_MAX(
            ActiveUpsweepPolicyT::BLOCK_THREADS * ActiveUpsweepPolicyT::ITEMS_PER_THREAD,
            ActiveDownsweepPolicyT::BLOCK_THREADS * ActiveDownsweepPolicyT::ITEMS_PER_THREAD)
    };

    // Parameterize AgentRadixSortDownsweep type for the current configuration
    typedef AgentRadixSortDownsweep<
            ActiveDownsweepPolicyT,
            IS_DESCENDING,
            KeyT,
            ValueT,
            OffsetT>
        AgentRadixSortDownsweepT;

    // Shared memory storage
    __shared__  typename AgentRadixSortDownsweepT::TempStorage temp_storage;

    // Initialize even-share descriptor for this thread block
    even_share.template BlockInit<TILE_ITEMS, GRID_MAPPING_RAKE>();

    // Process input tiles
    AgentRadixSortDownsweepT(temp_storage, num_items, d_spine, d_keys_in, d_keys_out, d_values_in, d_values_out, current_bit, num_bits).ProcessRegion(
        even_share.block_offset,
        even_share.block_end);
}


/**
 * Single pass kernel entry point (single-block).  Fully sorts a tile of input.
 */
template <
    typename                ChainedPolicyT,                 ///< Chained tuning policy
    bool                    IS_DESCENDING,                  ///< Whether or not the sorted-order is high-to-low
    typename                KeyT,                           ///< Key type
    typename                ValueT,                         ///< Value type
    typename                OffsetT>                        ///< Signed integer type for global offsets
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS), 1)
__global__ void DeviceRadixSortSingleTileKernel(
    const KeyT              *d_keys_in,                     ///< [in] Input keys buffer
    KeyT                    *d_keys_out,                    ///< [in] Output keys buffer
    const ValueT            *d_values_in,                   ///< [in] Input values buffer
    ValueT                  *d_values_out,                  ///< [in] Output values buffer
    OffsetT                 num_items,                      ///< [in] Total number of input data items
    int                     current_bit,                    ///< [in] Bit position of current radix digit
    int                     end_bit)                        ///< [in] The past-the-end (most-significant) bit index needed for key comparison
{
    // Constants
    enum
    {
        BLOCK_THREADS           = ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD        = ChainedPolicyT::ActivePolicy::SingleTilePolicy::ITEMS_PER_THREAD,
        KEYS_ONLY               = std::is_same<ValueT, NullType>::value,
    };

    // BlockRadixSort type
    typedef BlockRadixSort<
            KeyT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            ValueT,
            ChainedPolicyT::ActivePolicy::SingleTilePolicy::RADIX_BITS,
            (ChainedPolicyT::ActivePolicy::SingleTilePolicy::RANK_ALGORITHM == RADIX_RANK_MEMOIZE),
            ChainedPolicyT::ActivePolicy::SingleTilePolicy::SCAN_ALGORITHM>
        BlockRadixSortT;

    // BlockLoad type (keys)
    typedef BlockLoad<
        KeyT,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        ChainedPolicyT::ActivePolicy::SingleTilePolicy::LOAD_ALGORITHM> BlockLoadKeys;

    // BlockLoad type (values)
    typedef BlockLoad<
        ValueT,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        ChainedPolicyT::ActivePolicy::SingleTilePolicy::LOAD_ALGORITHM> BlockLoadValues;

    // Unsigned word for key bits
    typedef typename Traits<KeyT>::UnsignedBits UnsignedBitsT;

    // Shared memory storage
    __shared__ union TempStorage
    {
        typename BlockRadixSortT::TempStorage       sort;
        typename BlockLoadKeys::TempStorage         load_keys;
        typename BlockLoadValues::TempStorage       load_values;

    } temp_storage;

    // Keys and values for the block
    KeyT            keys[ITEMS_PER_THREAD];
    ValueT          values[ITEMS_PER_THREAD];

    // Get default (min/max) value for out-of-bounds keys
    UnsignedBitsT   default_key_bits = (IS_DESCENDING) ? Traits<KeyT>::LOWEST_KEY : Traits<KeyT>::MAX_KEY;
    KeyT            default_key = reinterpret_cast<KeyT&>(default_key_bits);

    // Load keys
    BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in, keys, num_items, default_key);

    CTA_SYNC();

    // Load values
    if (!KEYS_ONLY)
    {
        // Register pressure work-around: moving num_items through shfl prevents compiler
        // from reusing guards/addressing from prior guarded loads
        num_items = ShuffleIndex<CUB_PTX_WARP_THREADS>(num_items, 0, 0xffffffff);

        BlockLoadValues(temp_storage.load_values).Load(d_values_in, values, num_items);

        CTA_SYNC();
    }

    // Sort tile
    BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(
        keys,
        values,
        current_bit,
        end_bit,
        Int2Type<IS_DESCENDING>(),
        Int2Type<KEYS_ONLY>());

    // Store keys and values
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
        int item_offset = ITEM * BLOCK_THREADS + threadIdx.x;
        if (item_offset < num_items)
        {
            d_keys_out[item_offset] = keys[ITEM];
            if (!KEYS_ONLY)
                d_values_out[item_offset] = values[ITEM];
        }
    }
}


/**
 * Segmented radix sorting pass (one block per segment)
 */
template <
    typename                ChainedPolicyT,                 ///< Chained tuning policy
    bool                    ALT_DIGIT_BITS,                 ///< Whether or not to use the alternate (lower-bits) policy
    bool                    IS_DESCENDING,                  ///< Whether or not the sorted-order is high-to-low
    typename                KeyT,                           ///< Key type
    typename                ValueT,                         ///< Value type
    typename                BeginOffsetIteratorT,           ///< Random-access input iterator type for reading segment beginning offsets \iterator
    typename                EndOffsetIteratorT,             ///< Random-access input iterator type for reading segment ending offsets \iterator
    typename                OffsetT>                        ///< Signed integer type for global offsets
__launch_bounds__ (int((ALT_DIGIT_BITS) ?
    ChainedPolicyT::ActivePolicy::AltSegmentedPolicy::BLOCK_THREADS :
    ChainedPolicyT::ActivePolicy::SegmentedPolicy::BLOCK_THREADS))
__global__ void DeviceSegmentedRadixSortKernel(
    const KeyT              *d_keys_in,                     ///< [in] Input keys buffer
    KeyT                    *d_keys_out,                    ///< [in] Output keys buffer
    const ValueT            *d_values_in,                   ///< [in] Input values buffer
    ValueT                  *d_values_out,                  ///< [in] Output values buffer
    BeginOffsetIteratorT    d_begin_offsets,                ///< [in] Random-access input iterator to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
    EndOffsetIteratorT      d_end_offsets,                  ///< [in] Random-access input iterator to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
    int                     /*num_segments*/,               ///< [in] The number of segments that comprise the sorting data
    int                     current_bit,                    ///< [in] Bit position of current radix digit
    int                     pass_bits)                      ///< [in] Number of bits of current radix digit
{
    //
    // Constants
    //

    using SegmentedPolicyT = cub::detail::conditional_t<
      ALT_DIGIT_BITS,
      typename ChainedPolicyT::ActivePolicy::AltSegmentedPolicy,
      typename ChainedPolicyT::ActivePolicy::SegmentedPolicy>;

    enum
    {
        BLOCK_THREADS       = SegmentedPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD    = SegmentedPolicyT::ITEMS_PER_THREAD,
        RADIX_BITS          = SegmentedPolicyT::RADIX_BITS,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
        RADIX_DIGITS        = 1 << RADIX_BITS,
        KEYS_ONLY           = std::is_same<ValueT, NullType>::value,
    };

    // Upsweep type
    using BlockUpsweepT =
      AgentRadixSortUpsweep<SegmentedPolicyT, KeyT, OffsetT>;

    // Digit-scan type
    using DigitScanT = BlockScan<OffsetT, BLOCK_THREADS>;

    // Downsweep type
    using BlockDownsweepT = AgentRadixSortDownsweep<SegmentedPolicyT,
                                                    IS_DESCENDING,
                                                    KeyT,
                                                    ValueT,
                                                    OffsetT>;

    enum
    {
        /// Number of bin-starting offsets tracked per thread
        BINS_TRACKED_PER_THREAD = BlockDownsweepT::BINS_TRACKED_PER_THREAD
    };

    //
    // Process input tiles
    //

    // Shared memory storage
    __shared__ union
    {
        typename BlockUpsweepT::TempStorage     upsweep;
        typename BlockDownsweepT::TempStorage   downsweep;
        struct
        {
            volatile OffsetT                        reverse_counts_in[RADIX_DIGITS];
            volatile OffsetT                        reverse_counts_out[RADIX_DIGITS];
            typename DigitScanT::TempStorage        scan;
        };

    } temp_storage;

    OffsetT segment_begin   = d_begin_offsets[blockIdx.x];
    OffsetT segment_end     = d_end_offsets[blockIdx.x];
    OffsetT num_items       = segment_end - segment_begin;

    // Check if empty segment
    if (num_items <= 0)
        return;

    // Upsweep
    BlockUpsweepT upsweep(temp_storage.upsweep, d_keys_in, current_bit, pass_bits);
    upsweep.ProcessRegion(segment_begin, segment_end);

    CTA_SYNC();

    // The count of each digit value in this pass (valid in the first RADIX_DIGITS threads)
    OffsetT bin_count[BINS_TRACKED_PER_THREAD];
    upsweep.ExtractCounts(bin_count);

    CTA_SYNC();

    if (IS_DESCENDING)
    {
        // Reverse bin counts
        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

            if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
                temp_storage.reverse_counts_in[bin_idx] = bin_count[track];
        }

        CTA_SYNC();

        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

            if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
                bin_count[track] = temp_storage.reverse_counts_in[RADIX_DIGITS - bin_idx - 1];
        }
    }

    // Scan
    OffsetT bin_offset[BINS_TRACKED_PER_THREAD];     // The global scatter base offset for each digit value in this pass (valid in the first RADIX_DIGITS threads)
    DigitScanT(temp_storage.scan).ExclusiveSum(bin_count, bin_offset);

    #pragma unroll
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
        bin_offset[track] += segment_begin;
    }

    if (IS_DESCENDING)
    {
        // Reverse bin offsets
        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

            if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
                temp_storage.reverse_counts_out[threadIdx.x] = bin_offset[track];
        }

        CTA_SYNC();

        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

            if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
                bin_offset[track] = temp_storage.reverse_counts_out[RADIX_DIGITS - bin_idx - 1];
        }
    }

    CTA_SYNC();

    // Downsweep
    BlockDownsweepT downsweep(temp_storage.downsweep, bin_offset, num_items, d_keys_in, d_keys_out, d_values_in, d_values_out, current_bit, pass_bits);
    downsweep.ProcessRegion(segment_begin, segment_end);
}


/******************************************************************************
 * Onesweep kernels
 ******************************************************************************/

/** 
 * Kernel for computing multiple histograms
 */

/**
 * Histogram kernel
 */
template <
    typename ChainedPolicyT,
    bool IS_DESCENDING,
    typename KeyT,
    typename OffsetT>
__global__ void __launch_bounds__(ChainedPolicyT::ActivePolicy::HistogramPolicy::BLOCK_THREADS)
DeviceRadixSortHistogramKernel
    (OffsetT* d_bins_out, const KeyT* d_keys_in, OffsetT num_items, int start_bit, int end_bit)
{
    typedef typename ChainedPolicyT::ActivePolicy::HistogramPolicy HistogramPolicyT;
    typedef AgentRadixSortHistogram<HistogramPolicyT, IS_DESCENDING, KeyT, OffsetT> AgentT;
    __shared__ typename AgentT::TempStorage temp_storage;
    AgentT agent(temp_storage, d_bins_out, d_keys_in, num_items, start_bit, end_bit);
    agent.Process();
}

template <
    typename ChainedPolicyT,
    bool IS_DESCENDING,
    typename KeyT,
    typename ValueT,
    typename OffsetT,
    typename PortionOffsetT,
    typename AtomicOffsetT = PortionOffsetT>
__global__ void __launch_bounds__(ChainedPolicyT::ActivePolicy::OnesweepPolicy::BLOCK_THREADS)
DeviceRadixSortOnesweepKernel
    (AtomicOffsetT* d_lookback, AtomicOffsetT* d_ctrs, OffsetT* d_bins_out,
     const OffsetT* d_bins_in, KeyT* d_keys_out, const KeyT* d_keys_in, ValueT* d_values_out,
     const ValueT* d_values_in, PortionOffsetT num_items, int current_bit, int num_bits)
{
    typedef typename ChainedPolicyT::ActivePolicy::OnesweepPolicy OnesweepPolicyT;
    typedef AgentRadixSortOnesweep<OnesweepPolicyT, IS_DESCENDING, KeyT, ValueT, OffsetT,
                                   PortionOffsetT> AgentT;
    __shared__ typename AgentT::TempStorage s;

    AgentT agent(s, d_lookback, d_ctrs, d_bins_out, d_bins_in, d_keys_out, d_keys_in,
                 d_values_out, d_values_in, num_items, current_bit, num_bits);
    agent.Process();
}


/** 
 * Exclusive sum kernel
 */
template <
    typename ChainedPolicyT,
    typename OffsetT>
__global__ void DeviceRadixSortExclusiveSumKernel(OffsetT* d_bins)
{
    typedef typename ChainedPolicyT::ActivePolicy::ExclusiveSumPolicy ExclusiveSumPolicyT;
    const int RADIX_BITS = ExclusiveSumPolicyT::RADIX_BITS;
    const int RADIX_DIGITS = 1 << RADIX_BITS;
    const int BLOCK_THREADS = ExclusiveSumPolicyT::BLOCK_THREADS;
    const int BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS;
    typedef cub::BlockScan<OffsetT, BLOCK_THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // load the bins
    OffsetT bins[BINS_PER_THREAD];
    int bin_start = blockIdx.x * RADIX_DIGITS;
    #pragma unroll
    for (int u = 0; u < BINS_PER_THREAD; ++u)
    {
        int bin = threadIdx.x * BINS_PER_THREAD + u;
        if (bin >= RADIX_DIGITS) break;
        bins[u] = d_bins[bin_start + bin];
    }

    // compute offsets
    BlockScan(temp_storage).ExclusiveSum(bins, bins);

    // store the offsets
    #pragma unroll
    for (int u = 0; u < BINS_PER_THREAD; ++u)
    {
        int bin = threadIdx.x * BINS_PER_THREAD + u;
        if (bin >= RADIX_DIGITS) break;
        d_bins[bin_start + bin] = bins[u];
    }
}


/******************************************************************************
 * Policy
 ******************************************************************************/

/**
 * Tuning policy for kernel specialization
 */
template <
    typename KeyT,          ///< Key type
    typename ValueT,        ///< Value type
    typename OffsetT>       ///< Signed integer type for global offsets
struct DeviceRadixSortPolicy
{
    //------------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------------

    // Whether this is a keys-only (or key-value) sort
    constexpr static bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

    // Dominant-sized key/value type
    using DominantT =
      cub::detail::conditional_t<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

    //------------------------------------------------------------------------------
    // Architecture-specific tuning policies
    //------------------------------------------------------------------------------

    /// SM35
    struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
    {
        enum {
            PRIMARY_RADIX_BITS      = (sizeof(KeyT) > 1) ? 6 : 5,    // 1.72B 32b keys/s, 1.17B 32b pairs/s, 1.55B 32b segmented keys/s (K40m)
            ONESWEEP = false,
            ONESWEEP_RADIX_BITS = 8,
        };

        // Histogram policy
        typedef AgentRadixSortHistogramPolicy <256, 8, 1, KeyT, ONESWEEP_RADIX_BITS> HistogramPolicy;
        
        // Exclusive sum policy
        typedef AgentRadixSortExclusiveSumPolicy <256, ONESWEEP_RADIX_BITS> ExclusiveSumPolicy;
        
        // Onesweep policy
        typedef AgentRadixSortOnesweepPolicy <256, 21, DominantT, 1,
            RADIX_RANK_MATCH_EARLY_COUNTS_ANY, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_STORE_DIRECT,
            ONESWEEP_RADIX_BITS> OnesweepPolicy;

        // Scan policy
        typedef AgentScanPolicy <1024, 4, OffsetT, BLOCK_LOAD_VECTORIZE, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, BLOCK_SCAN_WARP_SCANS> ScanPolicy;

        // Keys-only downsweep policies
        typedef AgentRadixSortDownsweepPolicy <128, 9, DominantT, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_LDG, RADIX_RANK_MATCH, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS> DownsweepPolicyKeys;
        typedef AgentRadixSortDownsweepPolicy <64, 18, DominantT, BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS - 1> AltDownsweepPolicyKeys;

        // Key-value pairs downsweep policies
        typedef DownsweepPolicyKeys DownsweepPolicyPairs;
        typedef AgentRadixSortDownsweepPolicy <128, 15, DominantT, BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS - 1> AltDownsweepPolicyPairs;

        // Downsweep policies
        using DownsweepPolicy =
          cub::detail::conditional_t<
            KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>;

        using AltDownsweepPolicy =
          cub::detail::conditional_t<KEYS_ONLY,
                                   AltDownsweepPolicyKeys,
                                   AltDownsweepPolicyPairs>;

        // Upsweep policies
        using UpsweepPolicy    = DownsweepPolicy;
        using AltUpsweepPolicy = AltDownsweepPolicy;

        // Single-tile policy
        using SingleTilePolicy = DownsweepPolicy;

        // Segmented policies
        using SegmentedPolicy    = DownsweepPolicy;
        using AltSegmentedPolicy = AltDownsweepPolicy;
    };


    /// SM50
    struct Policy500 : ChainedPolicy<500, Policy500, Policy350>
    {
        enum {
            PRIMARY_RADIX_BITS      = (sizeof(KeyT) > 1) ? 7 : 5,    // 3.5B 32b keys/s, 1.92B 32b pairs/s (TitanX)
            SINGLE_TILE_RADIX_BITS  = (sizeof(KeyT) > 1) ? 6 : 5,
            SEGMENTED_RADIX_BITS    = (sizeof(KeyT) > 1) ? 6 : 5,    // 3.1B 32b segmented keys/s (TitanX)
            ONESWEEP = false,
            ONESWEEP_RADIX_BITS = 8,
        };

        // Histogram policy
        typedef AgentRadixSortHistogramPolicy <256, 8, 1, KeyT, ONESWEEP_RADIX_BITS> HistogramPolicy;
        
        // Exclusive sum policy
        typedef AgentRadixSortExclusiveSumPolicy <256, ONESWEEP_RADIX_BITS> ExclusiveSumPolicy;
        
        // Onesweep policy
        typedef AgentRadixSortOnesweepPolicy <256, 21, DominantT, 1,
            RADIX_RANK_MATCH_EARLY_COUNTS_ANY, BLOCK_SCAN_WARP_SCANS, RADIX_SORT_STORE_DIRECT,
            ONESWEEP_RADIX_BITS> OnesweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <512, 23, OffsetT, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // Downsweep policies
        typedef AgentRadixSortDownsweepPolicy <160, 39, DominantT,  BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_BASIC, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS>  DownsweepPolicy;
        typedef AgentRadixSortDownsweepPolicy <256, 16, DominantT,  BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_RAKING_MEMOIZE, PRIMARY_RADIX_BITS - 1>   AltDownsweepPolicy;

        // Upsweep policies
        typedef DownsweepPolicy UpsweepPolicy;
        typedef AltDownsweepPolicy AltUpsweepPolicy;

        // Single-tile policy
        typedef AgentRadixSortDownsweepPolicy <256, 19, DominantT,  BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SINGLE_TILE_RADIX_BITS> SingleTilePolicy;

        // Segmented policies
        typedef AgentRadixSortDownsweepPolicy <192, 31, DominantT,  BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS>   SegmentedPolicy;
        typedef AgentRadixSortDownsweepPolicy <256, 11, DominantT,  BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS - 1>       AltSegmentedPolicy;
    };


    /// SM60 (GP100)
    struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
    {
        enum {
            PRIMARY_RADIX_BITS      = (sizeof(KeyT) > 1) ? 7 : 5,    // 6.9B 32b keys/s (Quadro P100)
            SINGLE_TILE_RADIX_BITS  = (sizeof(KeyT) > 1) ? 6 : 5,
            SEGMENTED_RADIX_BITS    = (sizeof(KeyT) > 1) ? 6 : 5,    // 5.9B 32b segmented keys/s (Quadro P100)
            ONESWEEP = sizeof(KeyT) >= sizeof(uint32_t),  // 10.0B 32b keys/s (GP100, 64M random keys)
            ONESWEEP_RADIX_BITS = 8,
            OFFSET_64BIT            = sizeof(OffsetT) == 8,
        };

        // Histogram policy
        typedef AgentRadixSortHistogramPolicy <256, 8, 8, KeyT, ONESWEEP_RADIX_BITS> HistogramPolicy;
        
        // Exclusive sum policy
        typedef AgentRadixSortExclusiveSumPolicy <256, ONESWEEP_RADIX_BITS> ExclusiveSumPolicy;
        
        // Onesweep policy
        typedef AgentRadixSortOnesweepPolicy <256, OFFSET_64BIT ? 29 : 30, DominantT, 2,
            RADIX_RANK_MATCH_EARLY_COUNTS_ANY, BLOCK_SCAN_WARP_SCANS,
            RADIX_SORT_STORE_DIRECT, ONESWEEP_RADIX_BITS> OnesweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <512, 23, OffsetT, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // Downsweep policies
        typedef AgentRadixSortDownsweepPolicy <256, 25, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MATCH, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS>   DownsweepPolicy;
        typedef AgentRadixSortDownsweepPolicy <192, OFFSET_64BIT ? 32 : 39, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS - 1>   AltDownsweepPolicy;

        // Upsweep policies
        typedef DownsweepPolicy UpsweepPolicy;
        typedef AltDownsweepPolicy AltUpsweepPolicy;

        // Single-tile policy
        typedef AgentRadixSortDownsweepPolicy <256, 19, DominantT,  BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SINGLE_TILE_RADIX_BITS>          SingleTilePolicy;

        // Segmented policies
        typedef AgentRadixSortDownsweepPolicy <192, 39, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS>     SegmentedPolicy;
        typedef AgentRadixSortDownsweepPolicy <384, 11, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS - 1> AltSegmentedPolicy;

    };


    /// SM61 (GP104)
    struct Policy610 : ChainedPolicy<610, Policy610, Policy600>
    {
        enum {
            PRIMARY_RADIX_BITS      = (sizeof(KeyT) > 1) ? 7 : 5,    // 3.4B 32b keys/s, 1.83B 32b pairs/s (1080)
            SINGLE_TILE_RADIX_BITS  = (sizeof(KeyT) > 1) ? 6 : 5,
            SEGMENTED_RADIX_BITS    = (sizeof(KeyT) > 1) ? 6 : 5,    // 3.3B 32b segmented keys/s (1080)
            ONESWEEP = sizeof(KeyT) >= sizeof(uint32_t),
            ONESWEEP_RADIX_BITS = 8,
        };

        // Histogram policy
        typedef AgentRadixSortHistogramPolicy <256, 8, 8, KeyT, ONESWEEP_RADIX_BITS> HistogramPolicy;
        
        // Exclusive sum policy
        typedef AgentRadixSortExclusiveSumPolicy <256, ONESWEEP_RADIX_BITS> ExclusiveSumPolicy;
        
        // Onesweep policy
        typedef AgentRadixSortOnesweepPolicy <256, 30, DominantT, 2,
            RADIX_RANK_MATCH_EARLY_COUNTS_ANY, BLOCK_SCAN_WARP_SCANS,
            RADIX_SORT_STORE_DIRECT, ONESWEEP_RADIX_BITS> OnesweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <512, 23, OffsetT, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // Downsweep policies
        typedef AgentRadixSortDownsweepPolicy <384, 31, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MATCH, BLOCK_SCAN_RAKING_MEMOIZE, PRIMARY_RADIX_BITS>   DownsweepPolicy;
        typedef AgentRadixSortDownsweepPolicy <256, 35, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_RAKING_MEMOIZE, PRIMARY_RADIX_BITS - 1>   AltDownsweepPolicy;

        // Upsweep policies
        typedef AgentRadixSortUpsweepPolicy <128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS>        UpsweepPolicy;
        typedef AgentRadixSortUpsweepPolicy <128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS - 1>    AltUpsweepPolicy;

        // Single-tile policy
        typedef AgentRadixSortDownsweepPolicy <256, 19, DominantT,  BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SINGLE_TILE_RADIX_BITS>          SingleTilePolicy;

        // Segmented policies
        typedef AgentRadixSortDownsweepPolicy <192, 39, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS>     SegmentedPolicy;
        typedef AgentRadixSortDownsweepPolicy <384, 11, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS - 1> AltSegmentedPolicy;
    };


    /// SM62 (Tegra, less RF)
    struct Policy620 : ChainedPolicy<620, Policy620, Policy610>
    {
        enum {
            PRIMARY_RADIX_BITS      = 5,
            ALT_RADIX_BITS          = PRIMARY_RADIX_BITS - 1,
            ONESWEEP = sizeof(KeyT) >= sizeof(uint32_t),
            ONESWEEP_RADIX_BITS = 8,
        };

        // Histogram policy
        typedef AgentRadixSortHistogramPolicy <256, 8, 8, KeyT, ONESWEEP_RADIX_BITS> HistogramPolicy;
        
        // Exclusive sum policy
        typedef AgentRadixSortExclusiveSumPolicy <256, ONESWEEP_RADIX_BITS> ExclusiveSumPolicy;
        
        // Onesweep policy
        typedef AgentRadixSortOnesweepPolicy <256, 30, DominantT, 2,
            RADIX_RANK_MATCH_EARLY_COUNTS_ANY, BLOCK_SCAN_WARP_SCANS,
            RADIX_SORT_STORE_DIRECT, ONESWEEP_RADIX_BITS> OnesweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <512, 23, OffsetT, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // Downsweep policies
        typedef AgentRadixSortDownsweepPolicy <256, 16, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_RAKING_MEMOIZE, PRIMARY_RADIX_BITS>   DownsweepPolicy;
        typedef AgentRadixSortDownsweepPolicy <256, 16, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_RAKING_MEMOIZE, ALT_RADIX_BITS>       AltDownsweepPolicy;

        // Upsweep policies
        typedef DownsweepPolicy UpsweepPolicy;
        typedef AltDownsweepPolicy AltUpsweepPolicy;

        // Single-tile policy
        typedef AgentRadixSortDownsweepPolicy <256, 19, DominantT,  BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS> SingleTilePolicy;

        // Segmented policies
        typedef DownsweepPolicy     SegmentedPolicy;
        typedef AltDownsweepPolicy  AltSegmentedPolicy;
    };


    /// SM70 (GV100)
    struct Policy700 : ChainedPolicy<700, Policy700, Policy620>
    {
        enum {
            PRIMARY_RADIX_BITS      = (sizeof(KeyT) > 1) ? 7 : 5,    // 7.62B 32b keys/s (GV100)
            SINGLE_TILE_RADIX_BITS  = (sizeof(KeyT) > 1) ? 6 : 5,
            SEGMENTED_RADIX_BITS    = (sizeof(KeyT) > 1) ? 6 : 5,    // 8.7B 32b segmented keys/s (GV100)
            ONESWEEP = sizeof(KeyT) >= sizeof(uint32_t),  // 15.8B 32b keys/s (V100-SXM2, 64M random keys)
            ONESWEEP_RADIX_BITS = 8,
            OFFSET_64BIT            = sizeof(OffsetT) == 8,
        };

        // Histogram policy
        typedef AgentRadixSortHistogramPolicy <256, 8, 8, KeyT, ONESWEEP_RADIX_BITS> HistogramPolicy;
        
        // Exclusive sum policy
        typedef AgentRadixSortExclusiveSumPolicy <256, ONESWEEP_RADIX_BITS> ExclusiveSumPolicy;
        
        // Onesweep policy
        typedef AgentRadixSortOnesweepPolicy <256,
            sizeof(KeyT) == 4 && sizeof(ValueT) == 4 ? 46 : 23, DominantT, 4,
            RADIX_RANK_MATCH_EARLY_COUNTS_ANY, BLOCK_SCAN_WARP_SCANS,
            RADIX_SORT_STORE_DIRECT, ONESWEEP_RADIX_BITS> OnesweepPolicy;


        // ScanPolicy
        typedef AgentScanPolicy <512, 23, OffsetT, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // Downsweep policies
        typedef AgentRadixSortDownsweepPolicy <512, 23, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MATCH, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS>   DownsweepPolicy;
        typedef AgentRadixSortDownsweepPolicy <(sizeof(KeyT) > 1) ? 256 : 128, OFFSET_64BIT ? 46 : 47, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS - 1>   AltDownsweepPolicy;

        // Upsweep policies
        typedef AgentRadixSortUpsweepPolicy <256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>     UpsweepPolicy;
        typedef AgentRadixSortUpsweepPolicy <256, OFFSET_64BIT ? 46 : 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1> AltUpsweepPolicy;

        // Single-tile policy
        typedef AgentRadixSortDownsweepPolicy <256, 19, DominantT,  BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SINGLE_TILE_RADIX_BITS>          SingleTilePolicy;

        // Segmented policies
        typedef AgentRadixSortDownsweepPolicy <192, 39, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS>     SegmentedPolicy;
        typedef AgentRadixSortDownsweepPolicy <384, 11, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS - 1> AltSegmentedPolicy;
    };


    /// SM80
    struct Policy800 : ChainedPolicy<800, Policy800, Policy700>
    {
        enum {
            PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5,
            SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
            SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5,
            ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t),
            ONESWEEP_RADIX_BITS    = 8,
            OFFSET_64BIT           = sizeof(OffsetT) == 8,
        };

        // Histogram policy
        typedef AgentRadixSortHistogramPolicy <128, 16, 1, KeyT, ONESWEEP_RADIX_BITS> HistogramPolicy;
        
        // Exclusive sum policy
        typedef AgentRadixSortExclusiveSumPolicy <256, ONESWEEP_RADIX_BITS> ExclusiveSumPolicy;
        
        // Onesweep policy
        typedef AgentRadixSortOnesweepPolicy <384,
            OFFSET_64BIT && sizeof(KeyT) == 4 && !KEYS_ONLY ? 17 : 21, DominantT, 1,
            RADIX_RANK_MATCH_EARLY_COUNTS_ANY, BLOCK_SCAN_RAKING_MEMOIZE,
            RADIX_SORT_STORE_DIRECT, ONESWEEP_RADIX_BITS> OnesweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <512, 23, OffsetT, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // Downsweep policies
        typedef AgentRadixSortDownsweepPolicy <512, 23, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MATCH, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS>   DownsweepPolicy;
        typedef AgentRadixSortDownsweepPolicy <(sizeof(KeyT) > 1) ? 256 : 128, 47, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS - 1>   AltDownsweepPolicy;

        // Upsweep policies
        typedef AgentRadixSortUpsweepPolicy <256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>     UpsweepPolicy;
        typedef AgentRadixSortUpsweepPolicy <256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1> AltUpsweepPolicy;

        // Single-tile policy
        typedef AgentRadixSortDownsweepPolicy <256, 19, DominantT,  BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SINGLE_TILE_RADIX_BITS>          SingleTilePolicy;

        // Segmented policies
        typedef AgentRadixSortDownsweepPolicy <192, 39, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS>     SegmentedPolicy;
        typedef AgentRadixSortDownsweepPolicy <384, 11, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS - 1> AltSegmentedPolicy;
    };

    /// SM90
    struct Policy900 : ChainedPolicy<900, Policy900, Policy800>
    {
        enum {
            PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5,
            SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
            SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5,
            ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t),
            ONESWEEP_RADIX_BITS    = 8,
            OFFSET_64BIT           = sizeof(OffsetT) == 8 ? 1 : 0,
            FLOAT_KEYS             = std::is_same<KeyT, float>::value ? 1 : 0,
        };

        // Histogram policy
        typedef AgentRadixSortHistogramPolicy <128, 16, 1, KeyT, ONESWEEP_RADIX_BITS> HistogramPolicy;

        // Exclusive sum policy
        typedef AgentRadixSortExclusiveSumPolicy <256, ONESWEEP_RADIX_BITS> ExclusiveSumPolicy;

        typedef AgentRadixSortOnesweepPolicy <384,
            KEYS_ONLY ? 20 - OFFSET_64BIT - FLOAT_KEYS :
            (sizeof(ValueT) < 8 ? (OFFSET_64BIT ? 17 : 23) : (OFFSET_64BIT ? 29 : 30)),
            DominantT, 1, RADIX_RANK_MATCH_EARLY_COUNTS_ANY, BLOCK_SCAN_RAKING_MEMOIZE,
            RADIX_SORT_STORE_DIRECT, ONESWEEP_RADIX_BITS> OnesweepPolicyKey32;

        typedef AgentRadixSortOnesweepPolicy <384, sizeof(ValueT) < 8 ? 30 : 24, DominantT, 1,
            RADIX_RANK_MATCH_EARLY_COUNTS_ANY, BLOCK_SCAN_RAKING_MEMOIZE,
            RADIX_SORT_STORE_DIRECT, ONESWEEP_RADIX_BITS> OnesweepPolicyKey64;

        typedef typename std::conditional<sizeof(KeyT) == 4,
            OnesweepPolicyKey32, OnesweepPolicyKey64>::type OnesweepPolicy;

        // ScanPolicy
        typedef AgentScanPolicy <512, 23, OffsetT, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE> ScanPolicy;

        // Downsweep policies
        typedef AgentRadixSortDownsweepPolicy <512, 23, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MATCH, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS>   DownsweepPolicy;
        typedef AgentRadixSortDownsweepPolicy <(sizeof(KeyT) > 1) ? 256 : 128, 47, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS - 1>   AltDownsweepPolicy;

        // Upsweep policies
        typedef AgentRadixSortUpsweepPolicy <256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>     UpsweepPolicy;
        typedef AgentRadixSortUpsweepPolicy <256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1> AltUpsweepPolicy;

        // Single-tile policy
        typedef AgentRadixSortDownsweepPolicy <256, 19, DominantT,  BLOCK_LOAD_DIRECT, LOAD_LDG, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SINGLE_TILE_RADIX_BITS>          SingleTilePolicy;

        // Segmented policies
        typedef AgentRadixSortDownsweepPolicy <192, 39, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS>     SegmentedPolicy;
        typedef AgentRadixSortDownsweepPolicy <384, 11, DominantT,  BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS - 1> AltSegmentedPolicy;
    };


    /// MaxPolicy
    typedef Policy900 MaxPolicy;
};



/******************************************************************************
 * Single-problem dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for device-wide radix sort
 */
template <
    bool     IS_DESCENDING, ///< Whether or not the sorted-order is high-to-low
    typename KeyT,          ///< Key type
    typename ValueT,        ///< Value type
    typename OffsetT,       ///< Signed integer type for global offsets
    typename SelectedPolicy = DeviceRadixSortPolicy<KeyT, ValueT, OffsetT> >
struct DispatchRadixSort :
    SelectedPolicy
{
    //------------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------------

    // Whether this is a keys-only (or key-value) sort
    constexpr static bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

    //------------------------------------------------------------------------------
    // Problem state
    //------------------------------------------------------------------------------

    void                    *d_temp_storage;        ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t                  &temp_storage_bytes;    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    DoubleBuffer<KeyT>      &d_keys;                ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
    DoubleBuffer<ValueT>    &d_values;              ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
    OffsetT                 num_items;              ///< [in] Number of items to sort
    int                     begin_bit;              ///< [in] The beginning (least-significant) bit index needed for key comparison
    int                     end_bit;                ///< [in] The past-the-end (most-significant) bit index needed for key comparison
    cudaStream_t            stream;                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    int                     ptx_version;            ///< [in] PTX version
    bool                    is_overwrite_okay;      ///< [in] Whether is okay to overwrite source buffers


    //------------------------------------------------------------------------------
    // Constructor
    //------------------------------------------------------------------------------

    /// Constructor
    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchRadixSort(
        void*                   d_temp_storage,
        size_t                  &temp_storage_bytes,
        DoubleBuffer<KeyT>      &d_keys,
        DoubleBuffer<ValueT>    &d_values,
        OffsetT                 num_items,
        int                     begin_bit,
        int                     end_bit,
        bool                    is_overwrite_okay,
        cudaStream_t            stream,
        int                     ptx_version)
    :
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_keys(d_keys),
        d_values(d_values),
        num_items(num_items),
        begin_bit(begin_bit),
        end_bit(end_bit),
        stream(stream),
        ptx_version(ptx_version),
        is_overwrite_okay(is_overwrite_okay)
    {}

    CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchRadixSort(
        void*                   d_temp_storage,
        size_t                  &temp_storage_bytes,
        DoubleBuffer<KeyT>      &d_keys,
        DoubleBuffer<ValueT>    &d_values,
        OffsetT                 num_items,
        int                     begin_bit,
        int                     end_bit,
        bool                    is_overwrite_okay,
        cudaStream_t            stream,
        bool                    debug_synchronous,
        int                     ptx_version)
    :
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_keys(d_keys),
        d_values(d_values),
        num_items(num_items),
        begin_bit(begin_bit),
        end_bit(end_bit),
        stream(stream),
        ptx_version(ptx_version),
        is_overwrite_okay(is_overwrite_okay)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
    }


    //------------------------------------------------------------------------------
    // Small-problem (single tile) invocation
    //------------------------------------------------------------------------------

    /// Invoke a single block to sort in-core
    template <
        typename                ActivePolicyT,          ///< Umbrella policy active for the target device
        typename                SingleTileKernelT>      ///< Function type of cub::DeviceRadixSortSingleTileKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokeSingleTile(
        SingleTileKernelT       single_tile_kernel)     ///< [in] Kernel function pointer to parameterization of cub::DeviceRadixSortSingleTileKernel
    {
        cudaError error = cudaSuccess;
        do
        {
            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
            {
                temp_storage_bytes = 1;
                break;
            }

            // Log single_tile_kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking single_tile_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy, current bit %d, bit_grain %d\n",
                1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, (long long) stream,
                ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD, 1, begin_bit, ActivePolicyT::SingleTilePolicy::RADIX_BITS);
            #endif

            // Invoke upsweep_kernel with same grid size as downsweep_kernel
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream
            ).doit(single_tile_kernel,
                d_keys.Current(),
                d_keys.Alternate(),
                d_values.Current(),
                d_values.Alternate(),
                num_items,
                begin_bit,
                end_bit);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError()))
            {
                break;
            }

            // Sync the stream if specified to flush runtime errors
            error = detail::DebugSyncStream(stream);
            if (CubDebug(error))
            {
              break;
            }

            // Update selector
            d_keys.selector ^= 1;
            d_values.selector ^= 1;
        }
        while (0);

        return error;
    }


    //------------------------------------------------------------------------------
    // Normal problem size invocation
    //------------------------------------------------------------------------------

    /**
     * Invoke a three-kernel sorting pass at the current bit.
     */
    template <typename PassConfigT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokePass(
        const KeyT      *d_keys_in,
        KeyT            *d_keys_out,
        const ValueT    *d_values_in,
        ValueT          *d_values_out,
        OffsetT         *d_spine,
        int             /*spine_length*/,
        int             &current_bit,
        PassConfigT     &pass_config)
    {
        cudaError error = cudaSuccess;
        do
        {
            int pass_bits = CUB_MIN(pass_config.radix_bits, (end_bit - current_bit));

            // Log upsweep_kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking upsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy, current bit %d, bit_grain %d\n",
              pass_config.even_share.grid_size, pass_config.upsweep_config.block_threads, (long long) stream,
              pass_config.upsweep_config.items_per_thread, pass_config.upsweep_config.sm_occupancy, current_bit, pass_bits);
            #endif

            // Spine length written by the upsweep kernel in the current pass.
            int pass_spine_length = pass_config.even_share.grid_size * pass_config.radix_digits;

            // Invoke upsweep_kernel with same grid size as downsweep_kernel
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                pass_config.even_share.grid_size,
                pass_config.upsweep_config.block_threads, 0, stream
            ).doit(pass_config.upsweep_kernel,
                d_keys_in,
                d_spine,
                num_items,
                current_bit,
                pass_bits,
                pass_config.even_share);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError()))
            {
                break;
            }

            // Sync the stream if specified to flush runtime errors
            error = detail::DebugSyncStream(stream);
            if (CubDebug(error))
            {
              break;
            }

            // Log scan_kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking scan_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
                1, pass_config.scan_config.block_threads, (long long) stream, pass_config.scan_config.items_per_thread);
            #endif

            // Invoke scan_kernel
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                1, pass_config.scan_config.block_threads, 0, stream
            ).doit(pass_config.scan_kernel,
                d_spine,
                pass_spine_length);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError()))
            {
                break;
            }

            // Sync the stream if specified to flush runtime errors
            error = detail::DebugSyncStream(stream);
            if (CubDebug(error))
            {
              break;
            }

            // Log downsweep_kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking downsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                pass_config.even_share.grid_size, pass_config.downsweep_config.block_threads, (long long) stream,
                pass_config.downsweep_config.items_per_thread, pass_config.downsweep_config.sm_occupancy);
            #endif

            // Invoke downsweep_kernel
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                pass_config.even_share.grid_size,
                pass_config.downsweep_config.block_threads, 0, stream
            ).doit(pass_config.downsweep_kernel,
                d_keys_in,
                d_keys_out,
                d_values_in,
                d_values_out,
                d_spine,
                num_items,
                current_bit,
                pass_bits,
                pass_config.even_share);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError()))
            {
                break;
            }

            // Sync the stream if specified to flush runtime errors
            error = detail::DebugSyncStream(stream);
            if (CubDebug(error))
            {
              break;
            }

            // Update current bit
            current_bit += pass_bits;
        }
        while (0);

        return error;
    }



    /// Pass configuration structure
    template <
        typename UpsweepKernelT,
        typename ScanKernelT,
        typename DownsweepKernelT>
    struct PassConfig
    {
        UpsweepKernelT          upsweep_kernel;
        KernelConfig            upsweep_config;
        ScanKernelT             scan_kernel;
        KernelConfig            scan_config;
        DownsweepKernelT        downsweep_kernel;
        KernelConfig            downsweep_config;
        int                     radix_bits;
        int                     radix_digits;
        int                     max_downsweep_grid_size;
        GridEvenShare<OffsetT>  even_share;

        /// Initialize pass configuration
        template <
            typename UpsweepPolicyT,
            typename ScanPolicyT,
            typename DownsweepPolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        cudaError_t InitPassConfig(
            UpsweepKernelT      upsweep_kernel,
            ScanKernelT         scan_kernel,
            DownsweepKernelT    downsweep_kernel,
            int                 /*ptx_version*/,
            int                 sm_count,
            OffsetT             num_items)
        {
            cudaError error = cudaSuccess;
            do
            {
                this->upsweep_kernel    = upsweep_kernel;
                this->scan_kernel       = scan_kernel;
                this->downsweep_kernel  = downsweep_kernel;
                radix_bits              = DownsweepPolicyT::RADIX_BITS;
                radix_digits            = 1 << radix_bits;

                if (CubDebug(error = upsweep_config.Init<UpsweepPolicyT>(upsweep_kernel))) break;
                if (CubDebug(error = scan_config.Init<ScanPolicyT>(scan_kernel))) break;
                if (CubDebug(error = downsweep_config.Init<DownsweepPolicyT>(downsweep_kernel))) break;

                max_downsweep_grid_size = (downsweep_config.sm_occupancy * sm_count) * CUB_SUBSCRIPTION_FACTOR(0);

                even_share.DispatchInit(
                    num_items,
                    max_downsweep_grid_size,
                    CUB_MAX(downsweep_config.tile_size, upsweep_config.tile_size));

            }
            while (0);
            return error;
        }

    };

    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokeOnesweep()
    {
        typedef typename DispatchRadixSort::MaxPolicy MaxPolicyT;
        // PortionOffsetT is used for offsets within a portion, and must be signed.
        typedef int PortionOffsetT;
        typedef PortionOffsetT AtomicOffsetT;

        // compute temporary storage size
        const int RADIX_BITS = ActivePolicyT::ONESWEEP_RADIX_BITS;
        const int RADIX_DIGITS = 1 << RADIX_BITS;
        const int ONESWEEP_ITEMS_PER_THREAD = ActivePolicyT::OnesweepPolicy::ITEMS_PER_THREAD;
        const int ONESWEEP_BLOCK_THREADS = ActivePolicyT::OnesweepPolicy::BLOCK_THREADS;
        const int ONESWEEP_TILE_ITEMS = ONESWEEP_ITEMS_PER_THREAD * ONESWEEP_BLOCK_THREADS;
        // portions handle inputs with >=2**30 elements, due to the way lookback works
        // for testing purposes, one portion is <= 2**28 elements
        const PortionOffsetT PORTION_SIZE = ((1 << 28) - 1) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;
        int num_passes = cub::DivideAndRoundUp(end_bit - begin_bit, RADIX_BITS);
        OffsetT num_portions = static_cast<OffsetT>(cub::DivideAndRoundUp(num_items, PORTION_SIZE));
        PortionOffsetT max_num_blocks = cub::DivideAndRoundUp(
          static_cast<int>(
            CUB_MIN(num_items, static_cast<OffsetT>(PORTION_SIZE))),
          ONESWEEP_TILE_ITEMS);

        size_t value_size = KEYS_ONLY ? 0 : sizeof(ValueT);
        size_t allocation_sizes[] =
        {
            // bins
            num_portions * num_passes * RADIX_DIGITS * sizeof(OffsetT),
            // lookback
            max_num_blocks * RADIX_DIGITS * sizeof(AtomicOffsetT),
            // extra key buffer
            is_overwrite_okay || num_passes <= 1 ? 0 : num_items * sizeof(KeyT),
            // extra value buffer
            is_overwrite_okay || num_passes <= 1 ? 0 : num_items * value_size,
            // counters
            num_portions * num_passes * sizeof(AtomicOffsetT),
        };
        const int NUM_ALLOCATIONS = sizeof(allocation_sizes) / sizeof(allocation_sizes[0]);
        void* allocations[NUM_ALLOCATIONS] = {};
        AliasTemporaries<NUM_ALLOCATIONS>(d_temp_storage, temp_storage_bytes,
                                          allocations, allocation_sizes);

        // just return if no temporary storage is provided
        cudaError_t error = cudaSuccess;
        if (d_temp_storage == NULL) return error;

        OffsetT* d_bins = (OffsetT*)allocations[0];
        AtomicOffsetT* d_lookback = (AtomicOffsetT*)allocations[1];
        KeyT* d_keys_tmp2 = (KeyT*)allocations[2];
        ValueT* d_values_tmp2 = (ValueT*)allocations[3];
        AtomicOffsetT* d_ctrs = (AtomicOffsetT*)allocations[4];

        do {
            // initialization
            if (CubDebug(error = cudaMemsetAsync(
                   d_ctrs, 0, num_portions * num_passes * sizeof(AtomicOffsetT), stream))) break;

            // compute num_passes histograms with RADIX_DIGITS bins each
            if (CubDebug(error = cudaMemsetAsync
                   (d_bins, 0, num_passes * RADIX_DIGITS * sizeof(OffsetT), stream))) break;
            int device = -1;
            int num_sms = 0;
            if (CubDebug(error = cudaGetDevice(&device))) break;
            if (CubDebug(error = cudaDeviceGetAttribute(
                   &num_sms, cudaDevAttrMultiProcessorCount, device))) break;

            const int HISTO_BLOCK_THREADS = ActivePolicyT::HistogramPolicy::BLOCK_THREADS;
            int histo_blocks_per_sm = 1;
            auto histogram_kernel = DeviceRadixSortHistogramKernel<
                MaxPolicyT, IS_DESCENDING, KeyT, OffsetT>;
            if (CubDebug(error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &histo_blocks_per_sm, histogram_kernel, HISTO_BLOCK_THREADS, 0))) break;

            // log histogram_kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking histogram_kernel<<<%d, %d, 0, %lld>>>(), %d items per iteration, "
                    "%d SM occupancy, bit_grain %d\n",
                    histo_blocks_per_sm * num_sms, HISTO_BLOCK_THREADS,
                    reinterpret_cast<long long>(stream),
                    ActivePolicyT::HistogramPolicy::ITEMS_PER_THREAD, histo_blocks_per_sm,
                    ActivePolicyT::HistogramPolicy::RADIX_BITS);
            #endif

            error = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
              histo_blocks_per_sm * num_sms, HISTO_BLOCK_THREADS, 0, stream
            ).doit(histogram_kernel,
                   d_bins, d_keys.Current(), num_items, begin_bit, end_bit);
            if (CubDebug(error))
            {
                break;
            }

            error = detail::DebugSyncStream(stream);
            if (CubDebug(error))
            {
              break;
            }

            // exclusive sums to determine starts
            const int SCAN_BLOCK_THREADS = ActivePolicyT::ExclusiveSumPolicy::BLOCK_THREADS;

            // log exclusive_sum_kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking exclusive_sum_kernel<<<%d, %d, 0, %lld>>>(), bit_grain %d\n",
                    num_passes, SCAN_BLOCK_THREADS, reinterpret_cast<long long>(stream),
                    ActivePolicyT::ExclusiveSumPolicy::RADIX_BITS);
            #endif

            error = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                      num_passes, SCAN_BLOCK_THREADS, 0, stream
                      ).doit(DeviceRadixSortExclusiveSumKernel<MaxPolicyT, OffsetT>,
                            d_bins);
            if (CubDebug(error))
            {
                break;
            }

            error = detail::DebugSyncStream(stream);
            if (CubDebug(error))
            {
              break;
            }

            // use the other buffer if no overwrite is allowed
            KeyT* d_keys_tmp = d_keys.Alternate();
            ValueT* d_values_tmp = d_values.Alternate();
            if (!is_overwrite_okay && num_passes % 2 == 0)
            {
                d_keys.d_buffers[1] = d_keys_tmp2;
                d_values.d_buffers[1] = d_values_tmp2;
            }

            for (int current_bit = begin_bit, pass = 0; current_bit < end_bit;
                 current_bit += RADIX_BITS, ++pass)
            {
                int num_bits = CUB_MIN(end_bit - current_bit, RADIX_BITS);
                for (OffsetT portion = 0; portion < num_portions; ++portion)
                {
                    PortionOffsetT portion_num_items =
                      static_cast<PortionOffsetT>(
                        CUB_MIN(num_items - portion * PORTION_SIZE,
                                static_cast<OffsetT>(PORTION_SIZE)));
                    PortionOffsetT num_blocks =
                        cub::DivideAndRoundUp(portion_num_items, ONESWEEP_TILE_ITEMS);
                    if (CubDebug(error = cudaMemsetAsync(
                           d_lookback, 0, num_blocks * RADIX_DIGITS * sizeof(AtomicOffsetT),
                           stream))) break;

                    // log onesweep_kernel configuration
                    #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
                    _CubLog("Invoking onesweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, "
                            "current bit %d, bit_grain %d, portion %d/%d\n",
                            num_blocks, ONESWEEP_BLOCK_THREADS, reinterpret_cast<long long>(stream),
                            ActivePolicyT::OnesweepPolicy::ITEMS_PER_THREAD, current_bit,
                            num_bits, static_cast<int>(portion), static_cast<int>(num_portions));
                    #endif

                    auto onesweep_kernel = DeviceRadixSortOnesweepKernel<
                        MaxPolicyT, IS_DESCENDING, KeyT, ValueT, OffsetT, PortionOffsetT>;

                    error = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                      num_blocks, ONESWEEP_BLOCK_THREADS, 0, stream
                    ).doit(onesweep_kernel,
                           d_lookback, d_ctrs + portion * num_passes + pass,
                           portion < num_portions - 1 ?
                             d_bins + ((portion + 1) * num_passes + pass) * RADIX_DIGITS : NULL,
                           d_bins + (portion * num_passes + pass) * RADIX_DIGITS,
                           d_keys.Alternate(),
                           d_keys.Current() + portion * PORTION_SIZE,
                           d_values.Alternate(),
                           d_values.Current() + portion * PORTION_SIZE,
                           portion_num_items, current_bit, num_bits);
                    if (CubDebug(error))
                    {
                      break;
                    }

                    error = detail::DebugSyncStream(stream);
                    if (CubDebug(error))
                    {
                      break;
                    }
                }

                if (error != cudaSuccess)
                {
                    break;
                }
                
                // use the temporary buffers if no overwrite is allowed
                if (!is_overwrite_okay && pass == 0)
                {
                    d_keys = num_passes % 2 == 0 ?
                        DoubleBuffer<KeyT>(d_keys_tmp, d_keys_tmp2) :
                        DoubleBuffer<KeyT>(d_keys_tmp2, d_keys_tmp);
                    d_values = num_passes % 2 == 0 ?
                        DoubleBuffer<ValueT>(d_values_tmp, d_values_tmp2) :
                        DoubleBuffer<ValueT>(d_values_tmp2, d_values_tmp);
                }
                d_keys.selector ^= 1;
                d_values.selector ^= 1;
            }
        } while (0);
        
        return error;
    }

    /// Invocation (run multiple digit passes)
    template <
        typename            ActivePolicyT,          ///< Umbrella policy active for the target device
        typename            UpsweepKernelT,         ///< Function type of cub::DeviceRadixSortUpsweepKernel
        typename            ScanKernelT,            ///< Function type of cub::SpineScanKernel
        typename            DownsweepKernelT>       ///< Function type of cub::DeviceRadixSortDownsweepKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokePasses(
        UpsweepKernelT      upsweep_kernel,         ///< [in] Kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
        UpsweepKernelT      alt_upsweep_kernel,     ///< [in] Alternate kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
        ScanKernelT         scan_kernel,            ///< [in] Kernel function pointer to parameterization of cub::SpineScanKernel
        DownsweepKernelT    downsweep_kernel,       ///< [in] Kernel function pointer to parameterization of cub::DeviceRadixSortDownsweepKernel
        DownsweepKernelT    alt_downsweep_kernel)   ///< [in] Alternate kernel function pointer to parameterization of cub::DeviceRadixSortDownsweepKernel
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Init regular and alternate-digit kernel configurations
            PassConfig<UpsweepKernelT, ScanKernelT, DownsweepKernelT> pass_config, alt_pass_config;
            error = pass_config.template InitPassConfig<
              typename ActivePolicyT::UpsweepPolicy,
              typename ActivePolicyT::ScanPolicy,
              typename ActivePolicyT::DownsweepPolicy>(upsweep_kernel,
                                                       scan_kernel,
                                                       downsweep_kernel,
                                                       ptx_version,
                                                       sm_count,
                                                       num_items);
            if (error)
            {
              break;
            }

            error = alt_pass_config.template InitPassConfig<
              typename ActivePolicyT::AltUpsweepPolicy,
              typename ActivePolicyT::ScanPolicy,
              typename ActivePolicyT::AltDownsweepPolicy>(alt_upsweep_kernel,
                                                          scan_kernel,
                                                          alt_downsweep_kernel,
                                                          ptx_version,
                                                          sm_count,
                                                          num_items);
            if (error)
            {
              break;
            }

            // Get maximum spine length
            int max_grid_size       = CUB_MAX(pass_config.max_downsweep_grid_size, alt_pass_config.max_downsweep_grid_size);
            int spine_length        = (max_grid_size * pass_config.radix_digits) + pass_config.scan_config.tile_size;

            // Temporary storage allocation requirements
            void* allocations[3] = {};
            size_t allocation_sizes[3] =
            {
                spine_length * sizeof(OffsetT),                                         // bytes needed for privatized block digit histograms
                (is_overwrite_okay) ? 0 : num_items * sizeof(KeyT),                     // bytes needed for 3rd keys buffer
                (is_overwrite_okay || (KEYS_ONLY)) ? 0 : num_items * sizeof(ValueT),    // bytes needed for 3rd values buffer
            };

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
                return cudaSuccess;

            // Pass planning.  Run passes of the alternate digit-size configuration until we have an even multiple of our preferred digit size
            int num_bits            = end_bit - begin_bit;
            int num_passes          = cub::DivideAndRoundUp(num_bits, pass_config.radix_bits);
            bool is_num_passes_odd  = num_passes & 1;
            int max_alt_passes      = (num_passes * pass_config.radix_bits) - num_bits;
            int alt_end_bit         = CUB_MIN(end_bit, begin_bit + (max_alt_passes * alt_pass_config.radix_bits));

            // Alias the temporary storage allocations
            OffsetT *d_spine = static_cast<OffsetT*>(allocations[0]);

            DoubleBuffer<KeyT> d_keys_remaining_passes(
                (is_overwrite_okay || is_num_passes_odd) ? d_keys.Alternate() : static_cast<KeyT*>(allocations[1]),
                (is_overwrite_okay) ? d_keys.Current() : (is_num_passes_odd) ? static_cast<KeyT*>(allocations[1]) : d_keys.Alternate());

            DoubleBuffer<ValueT> d_values_remaining_passes(
                (is_overwrite_okay || is_num_passes_odd) ? d_values.Alternate() : static_cast<ValueT*>(allocations[2]),
                (is_overwrite_okay) ? d_values.Current() : (is_num_passes_odd) ? static_cast<ValueT*>(allocations[2]) : d_values.Alternate());

            // Run first pass, consuming from the input's current buffers
            int current_bit = begin_bit;
            if (CubDebug(error = InvokePass(
                d_keys.Current(), d_keys_remaining_passes.Current(),
                d_values.Current(), d_values_remaining_passes.Current(),
                d_spine, spine_length, current_bit,
                (current_bit < alt_end_bit) ? alt_pass_config : pass_config))) break;

            // Run remaining passes
            while (current_bit < end_bit)
            {
                if (CubDebug(error = InvokePass(
                    d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector],    d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
                    d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector],  d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
                    d_spine, spine_length, current_bit,
                    (current_bit < alt_end_bit) ? alt_pass_config : pass_config))) break;;

                // Invert selectors
                d_keys_remaining_passes.selector ^= 1;
                d_values_remaining_passes.selector ^= 1;
            }

            // Update selector
            if (!is_overwrite_okay) {
                num_passes = 1; // Sorted data always ends up in the other vector
            }

            d_keys.selector = (d_keys.selector + num_passes) & 1;
            d_values.selector = (d_values.selector + num_passes) & 1;
        }
        while (0);

        return error;
    }


    //------------------------------------------------------------------------------
    // Chained policy invocation
    //------------------------------------------------------------------------------

    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokeManyTiles(Int2Type<false>)
    {
        // Invoke upsweep-downsweep
        typedef typename DispatchRadixSort::MaxPolicy       MaxPolicyT;
        return InvokePasses<ActivePolicyT>(
            DeviceRadixSortUpsweepKernel<   MaxPolicyT, false,   IS_DESCENDING, KeyT, OffsetT>,
            DeviceRadixSortUpsweepKernel<   MaxPolicyT, true,    IS_DESCENDING, KeyT, OffsetT>,
            RadixSortScanBinsKernel<        MaxPolicyT, OffsetT>,
            DeviceRadixSortDownsweepKernel< MaxPolicyT, false,   IS_DESCENDING, KeyT, ValueT, OffsetT>,
            DeviceRadixSortDownsweepKernel< MaxPolicyT, true,    IS_DESCENDING, KeyT, ValueT, OffsetT>);        
    }

    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokeManyTiles(Int2Type<true>)
    {
        // Invoke onesweep
        return InvokeOnesweep<ActivePolicyT>();
    }

    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokeCopy()
    {
        // is_overwrite_okay == false here
        // Return the number of temporary bytes if requested
        if (d_temp_storage == nullptr)
        {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        
        // Copy keys
        #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking async copy of %lld keys on stream %lld\n", (long long)num_items,
                (long long)stream);
        #endif
        cudaError_t error = cudaSuccess;
        error = cudaMemcpyAsync(d_keys.Alternate(), d_keys.Current(), num_items * sizeof(KeyT),
                                cudaMemcpyDefault, stream);
        if (CubDebug(error))
        {
            return error;
        }
        if (CubDebug(error = detail::DebugSyncStream(stream)))
        {
            return error;
        }
        d_keys.selector ^= 1;

        // Copy values if necessary
        if (!KEYS_ONLY)
        {
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking async copy of %lld values on stream %lld\n",
                    (long long)num_items, (long long)stream);
            #endif
            error = cudaMemcpyAsync(d_values.Alternate(), d_values.Current(),
                                    num_items * sizeof(ValueT), cudaMemcpyDefault, stream);
            if (CubDebug(error))
            {
                return error;
            }
            if (CubDebug(error = detail::DebugSyncStream(stream)))
            {
                return error;
            }
        }
        d_values.selector ^= 1;

        return error;
    }

    /// Invocation
    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t Invoke()
    {
        typedef typename DispatchRadixSort::MaxPolicy       MaxPolicyT;
        typedef typename ActivePolicyT::SingleTilePolicy    SingleTilePolicyT;

        // Return if empty problem, or if no bits to sort and double-buffering is used
        if (num_items == 0 || (begin_bit == end_bit && is_overwrite_okay))
        {
            if (d_temp_storage == nullptr)
            {
                temp_storage_bytes = 1;
            }
            return cudaSuccess;
        }

        // Check if simple copy suffices (is_overwrite_okay == false at this point)
        if (begin_bit == end_bit)
        {
            bool has_uva = false;
            cudaError_t error = detail::HasUVA(has_uva);
            if (error != cudaSuccess) return error;
            if (has_uva)
            {
                return InvokeCopy();
            }
        }

        // Force kernel code-generation in all compiler passes
        if (num_items <= (SingleTilePolicyT::BLOCK_THREADS * SingleTilePolicyT::ITEMS_PER_THREAD))
        {
            // Small, single tile size
            return InvokeSingleTile<ActivePolicyT>(
                DeviceRadixSortSingleTileKernel<MaxPolicyT, IS_DESCENDING, KeyT, ValueT, OffsetT>);
        }
        else
        {
            // Regular size
            return InvokeManyTiles<ActivePolicyT>(Int2Type<ActivePolicyT::ONESWEEP>());
        }
    }


    //------------------------------------------------------------------------------
    // Dispatch entrypoints
    //------------------------------------------------------------------------------

    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<KeyT>      &d_keys,                ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<ValueT>    &d_values,              ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        OffsetT                 num_items,              ///< [in] Number of items to sort
        int                     begin_bit,              ///< [in] The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                ///< [in] The past-the-end (most-significant) bit index needed for key comparison
        bool                    is_overwrite_okay,      ///< [in] Whether is okay to overwrite source buffers
        cudaStream_t            stream)                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    {
        typedef typename DispatchRadixSort::MaxPolicy MaxPolicyT;

        cudaError_t error;
        do {
            // Get PTX version
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Create dispatch functor
            DispatchRadixSort dispatch(
                d_temp_storage, temp_storage_bytes,
                d_keys, d_values,
                num_items, begin_bit, end_bit, is_overwrite_okay,
                stream, ptx_version);

            // Dispatch to chained policy
            if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) break;

        } while (0);

        return error;
    }

    CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
    CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
    Dispatch(void *d_temp_storage,
             size_t &temp_storage_bytes,
             DoubleBuffer<KeyT> &d_keys,
             DoubleBuffer<ValueT> &d_values,
             OffsetT num_items,
             int begin_bit,
             int end_bit,
             bool is_overwrite_okay,
             cudaStream_t stream,
             bool debug_synchronous)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

      return Dispatch(d_temp_storage,
                      temp_storage_bytes,
                      d_keys,
                      d_values,
                      num_items,
                      begin_bit,
                      end_bit,
                      is_overwrite_okay,
                      stream);
    }
};




/******************************************************************************
 * Segmented dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for segmented device-wide radix sort
 */
template <
    bool     IS_DESCENDING,     ///< Whether or not the sorted-order is high-to-low
    typename KeyT,              ///< Key type
    typename ValueT,            ///< Value type
    typename BeginOffsetIteratorT,   ///< Random-access input iterator type for reading segment beginning offsets \iterator
    typename EndOffsetIteratorT,   ///< Random-access input iterator type for reading segment ending offsets \iterator
    typename OffsetT,           ///< Signed integer type for global offsets
    typename SelectedPolicy = DeviceRadixSortPolicy<KeyT, ValueT, OffsetT> >
struct DispatchSegmentedRadixSort :
    SelectedPolicy
{
    //------------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------------

    // Whether this is a keys-only (or key-value) sort
    constexpr static bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

    //------------------------------------------------------------------------------
    // Parameter members
    //------------------------------------------------------------------------------

    void                    *d_temp_storage;        ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t                  &temp_storage_bytes;    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    DoubleBuffer<KeyT>      &d_keys;                ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
    DoubleBuffer<ValueT>    &d_values;              ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
    OffsetT                 num_items;              ///< [in] Number of items to sort
    OffsetT                 num_segments;           ///< [in] The number of segments that comprise the sorting data
    BeginOffsetIteratorT    d_begin_offsets;        ///< [in] Random-access input iterator to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
    EndOffsetIteratorT      d_end_offsets;          ///< [in] Random-access input iterator to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
    int                     begin_bit;              ///< [in] The beginning (least-significant) bit index needed for key comparison
    int                     end_bit;                ///< [in] The past-the-end (most-significant) bit index needed for key comparison
    cudaStream_t            stream;                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    int                     ptx_version;            ///< [in] PTX version
    bool                    is_overwrite_okay;      ///< [in] Whether is okay to overwrite source buffers


    //------------------------------------------------------------------------------
    // Constructors
    //------------------------------------------------------------------------------

    /// Constructor
    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchSegmentedRadixSort(
        void*                   d_temp_storage,
        size_t                  &temp_storage_bytes,
        DoubleBuffer<KeyT>      &d_keys,
        DoubleBuffer<ValueT>    &d_values,
        OffsetT                 num_items,
        OffsetT                 num_segments,
        BeginOffsetIteratorT    d_begin_offsets,
        EndOffsetIteratorT      d_end_offsets,
        int                     begin_bit,
        int                     end_bit,
        bool                    is_overwrite_okay,
        cudaStream_t            stream,
        int                     ptx_version)
    :
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_keys(d_keys),
        d_values(d_values),
        num_items(num_items),
        num_segments(num_segments),
        d_begin_offsets(d_begin_offsets),
        d_end_offsets(d_end_offsets),
        begin_bit(begin_bit),
        end_bit(end_bit),
        stream(stream),
        ptx_version(ptx_version),
        is_overwrite_okay(is_overwrite_okay)
    {}

    CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchSegmentedRadixSort(
        void*                   d_temp_storage,
        size_t                  &temp_storage_bytes,
        DoubleBuffer<KeyT>      &d_keys,
        DoubleBuffer<ValueT>    &d_values,
        OffsetT                 num_items,
        OffsetT                 num_segments,
        BeginOffsetIteratorT    d_begin_offsets,
        EndOffsetIteratorT      d_end_offsets,
        int                     begin_bit,
        int                     end_bit,
        bool                    is_overwrite_okay,
        cudaStream_t            stream,
        bool                    debug_synchronous,
        int                     ptx_version)
    :
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_keys(d_keys),
        d_values(d_values),
        num_items(num_items),
        num_segments(num_segments),
        d_begin_offsets(d_begin_offsets),
        d_end_offsets(d_end_offsets),
        begin_bit(begin_bit),
        end_bit(end_bit),
        stream(stream),
        ptx_version(ptx_version),
        is_overwrite_okay(is_overwrite_okay)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
    }


    //------------------------------------------------------------------------------
    // Multi-segment invocation
    //------------------------------------------------------------------------------

    /// Invoke a three-kernel sorting pass at the current bit.
    template <typename PassConfigT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokePass(
        const KeyT      *d_keys_in,
        KeyT            *d_keys_out,
        const ValueT    *d_values_in,
        ValueT          *d_values_out,
        int             &current_bit,
        PassConfigT     &pass_config)
    {
        cudaError error = cudaSuccess;
        do
        {
            int pass_bits = CUB_MIN(pass_config.radix_bits, (end_bit - current_bit));

            // Log kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking segmented_kernels<<<%lld, %lld, 0, %lld>>>(), "
                    "%lld items per thread, %lld SM occupancy, "
                    "current bit %d, bit_grain %d\n",
                    (long long)num_segments,
                    (long long)pass_config.segmented_config.block_threads,
                    (long long)stream,
                    (long long)pass_config.segmented_config.items_per_thread,
                    (long long)pass_config.segmented_config.sm_occupancy,
                    current_bit,
                    pass_bits);
            #endif

            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                num_segments, pass_config.segmented_config.block_threads, 0,
                stream
            ).doit(pass_config.segmented_kernel,
                d_keys_in, d_keys_out,
                d_values_in,  d_values_out,
                d_begin_offsets, d_end_offsets, num_segments,
                current_bit, pass_bits);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError()))
            {
                break;
            }

            // Sync the stream if specified to flush runtime errors
            error = detail::DebugSyncStream(stream);
            if (CubDebug(error))
            {
              break;
            }

            // Update current bit
            current_bit += pass_bits;
        }
        while (0);

        return error;
    }


    /// PassConfig data structure
    template <typename SegmentedKernelT>
    struct PassConfig
    {
        SegmentedKernelT    segmented_kernel;
        KernelConfig        segmented_config;
        int                 radix_bits;
        int                 radix_digits;

        /// Initialize pass configuration
        template <typename SegmentedPolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        cudaError_t InitPassConfig(SegmentedKernelT segmented_kernel)
        {
            this->segmented_kernel  = segmented_kernel;
            this->radix_bits        = SegmentedPolicyT::RADIX_BITS;
            this->radix_digits      = 1 << radix_bits;

            return CubDebug(segmented_config.Init<SegmentedPolicyT>(segmented_kernel));
        }
    };


    /// Invocation (run multiple digit passes)
    template <
        typename                ActivePolicyT,          ///< Umbrella policy active for the target device
        typename                SegmentedKernelT>       ///< Function type of cub::DeviceSegmentedRadixSortKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokePasses(
        SegmentedKernelT     segmented_kernel,          ///< [in] Kernel function pointer to parameterization of cub::DeviceSegmentedRadixSortKernel
        SegmentedKernelT     alt_segmented_kernel)      ///< [in] Alternate kernel function pointer to parameterization of cub::DeviceSegmentedRadixSortKernel
    {
        cudaError error = cudaSuccess;
        do
        {
            // Init regular and alternate kernel configurations
            PassConfig<SegmentedKernelT> pass_config, alt_pass_config;
            if ((error = pass_config.template       InitPassConfig<typename ActivePolicyT::SegmentedPolicy>(segmented_kernel))) break;
            if ((error = alt_pass_config.template   InitPassConfig<typename ActivePolicyT::AltSegmentedPolicy>(alt_segmented_kernel))) break;

            // Temporary storage allocation requirements
            void* allocations[2] = {};
            size_t allocation_sizes[2] =
            {
                (is_overwrite_okay) ? 0 : num_items * sizeof(KeyT),                      // bytes needed for 3rd keys buffer
                (is_overwrite_okay || (KEYS_ONLY)) ? 0 : num_items * sizeof(ValueT),     // bytes needed for 3rd values buffer
            };

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
            {
                if (temp_storage_bytes == 0)
                    temp_storage_bytes = 1;
                return cudaSuccess;
            }

            // Pass planning.  Run passes of the alternate digit-size configuration until we have an even multiple of our preferred digit size
            int radix_bits          = ActivePolicyT::SegmentedPolicy::RADIX_BITS;
            int alt_radix_bits      = ActivePolicyT::AltSegmentedPolicy::RADIX_BITS;
            int num_bits            = end_bit - begin_bit;
            int num_passes          = CUB_MAX(DivideAndRoundUp(num_bits, radix_bits), 1);
            bool is_num_passes_odd  = num_passes & 1;
            int max_alt_passes      = (num_passes * radix_bits) - num_bits;
            int alt_end_bit         = CUB_MIN(end_bit, begin_bit + (max_alt_passes * alt_radix_bits));

            DoubleBuffer<KeyT> d_keys_remaining_passes(
                (is_overwrite_okay || is_num_passes_odd) ? d_keys.Alternate() : static_cast<KeyT*>(allocations[0]),
                (is_overwrite_okay) ? d_keys.Current() : (is_num_passes_odd) ? static_cast<KeyT*>(allocations[0]) : d_keys.Alternate());

            DoubleBuffer<ValueT> d_values_remaining_passes(
                (is_overwrite_okay || is_num_passes_odd) ? d_values.Alternate() : static_cast<ValueT*>(allocations[1]),
                (is_overwrite_okay) ? d_values.Current() : (is_num_passes_odd) ? static_cast<ValueT*>(allocations[1]) : d_values.Alternate());

            // Run first pass, consuming from the input's current buffers
            int current_bit = begin_bit;

            if (CubDebug(error = InvokePass(
                d_keys.Current(), d_keys_remaining_passes.Current(),
                d_values.Current(), d_values_remaining_passes.Current(),
                current_bit,
                (current_bit < alt_end_bit) ? alt_pass_config : pass_config))) break;

            // Run remaining passes
            while (current_bit < end_bit)
            {
                if (CubDebug(error = InvokePass(
                    d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector],    d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
                    d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector],  d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
                    current_bit,
                    (current_bit < alt_end_bit) ? alt_pass_config : pass_config))) break;

                // Invert selectors and update current bit
                d_keys_remaining_passes.selector ^= 1;
                d_values_remaining_passes.selector ^= 1;
            }

            // Update selector
            if (!is_overwrite_okay) {
                num_passes = 1; // Sorted data always ends up in the other vector
            }

            d_keys.selector = (d_keys.selector + num_passes) & 1;
            d_values.selector = (d_values.selector + num_passes) & 1;
        }
        while (0);

        return error;
    }


    //------------------------------------------------------------------------------
    // Chained policy invocation
    //------------------------------------------------------------------------------

    /// Invocation
    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t Invoke()
    {
        typedef typename DispatchSegmentedRadixSort::MaxPolicy MaxPolicyT;

        // Return if empty problem, or if no bits to sort and double-buffering is used
        if (num_items == 0 || (begin_bit == end_bit && is_overwrite_okay))
        {
            if (d_temp_storage == nullptr)
            {
                temp_storage_bytes = 1;
            }
            return cudaSuccess;
        }

        // Force kernel code-generation in all compiler passes
        return InvokePasses<ActivePolicyT>(
            DeviceSegmentedRadixSortKernel<MaxPolicyT, false,   IS_DESCENDING, KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT, OffsetT>,
            DeviceSegmentedRadixSortKernel<MaxPolicyT, true,    IS_DESCENDING, KeyT, ValueT, BeginOffsetIteratorT, EndOffsetIteratorT, OffsetT>);
    }


    //------------------------------------------------------------------------------
    // Dispatch entrypoints
    //------------------------------------------------------------------------------


    /// Internal dispatch routine
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                  &temp_storage_bytes,    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        DoubleBuffer<KeyT>      &d_keys,                ///< [in,out] Double-buffer whose current buffer contains the unsorted input keys and, upon return, is updated to point to the sorted output keys
        DoubleBuffer<ValueT>    &d_values,              ///< [in,out] Double-buffer whose current buffer contains the unsorted input values and, upon return, is updated to point to the sorted output values
        int                     num_items,              ///< [in] Number of items to sort
        int                     num_segments,           ///< [in] The number of segments that comprise the sorting data
        BeginOffsetIteratorT    d_begin_offsets,        ///< [in] Random-access input iterator to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
        EndOffsetIteratorT      d_end_offsets,          ///< [in] Random-access input iterator to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
        int                     begin_bit,              ///< [in] The beginning (least-significant) bit index needed for key comparison
        int                     end_bit,                ///< [in] The past-the-end (most-significant) bit index needed for key comparison
        bool                    is_overwrite_okay,      ///< [in] Whether is okay to overwrite source buffers
        cudaStream_t            stream)                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    {
        typedef typename DispatchSegmentedRadixSort::MaxPolicy MaxPolicyT;

        cudaError_t error;
        do {
            // Get PTX version
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Create dispatch functor
            DispatchSegmentedRadixSort dispatch(
                d_temp_storage, temp_storage_bytes,
                d_keys, d_values,
                num_items, num_segments, d_begin_offsets, d_end_offsets,
                begin_bit, end_bit, is_overwrite_okay,
                stream, ptx_version);

            // Dispatch to chained policy
            if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) break;

        } while (0);

        return error;
    }

    CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
    CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
    Dispatch(void *d_temp_storage,
             size_t &temp_storage_bytes,
             DoubleBuffer<KeyT> &d_keys,
             DoubleBuffer<ValueT> &d_values,
             int num_items,
             int num_segments,
             BeginOffsetIteratorT d_begin_offsets,
             EndOffsetIteratorT d_end_offsets,
             int begin_bit,
             int end_bit,
             bool is_overwrite_okay,
             cudaStream_t stream,
             bool debug_synchronous)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

      return Dispatch(d_temp_storage,
                      temp_storage_bytes,
                      d_keys,
                      d_values,
                      num_items,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
                      begin_bit,
                      end_bit,
                      is_overwrite_okay,
                      stream);
    }
};


CUB_NAMESPACE_END


#if defined(__clang__)
#  pragma clang diagnostic pop
#endif
