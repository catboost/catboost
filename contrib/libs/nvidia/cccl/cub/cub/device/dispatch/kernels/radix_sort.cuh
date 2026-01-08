// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_radix_sort_downsweep.cuh>
#include <cub/agent/agent_radix_sort_histogram.cuh>
#include <cub/agent/agent_radix_sort_onesweep.cuh>
#include <cub/agent/agent_radix_sort_upsweep.cuh>
#include <cub/agent/agent_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/grid/grid_even_share.cuh>

#include <cuda/std/__algorithm_>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::radix_sort
{

/**
 * @brief Upsweep digit-counting kernel entry point (multi-block).
 *        Computes privatized digit histograms, one per block.
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam ALT_DIGIT_BITS
 *   Whether or not to use the alternate (lower-bits) policy
 *
 * @tparam SortOrder
 *   Whether to sort in ascending or descending order
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in] d_keys
 *   Input keys buffer
 *
 * @param[out] d_spine
 *   Privatized (per block) digit histograms (striped, i.e., 0s counts from each block,
 *   then 1s counts from each block, etc.)
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] current_bit
 *   Bit position of current radix digit
 *
 * @param[in] num_bits
 *   Number of bits of current radix digit
 *
 * @param[in] even_share
 *   Even-share descriptor for mapan equal number of tiles onto each thread block
 */
template <typename ChainedPolicyT,
          bool ALT_DIGIT_BITS,
          SortOrder Order,
          typename KeyT,
          typename OffsetT,
          typename DecomposerT = detail::identity_decomposer_t>
__launch_bounds__(int((ALT_DIGIT_BITS) ? int(ChainedPolicyT::ActivePolicy::AltUpsweepPolicy::BLOCK_THREADS)
                                       : int(ChainedPolicyT::ActivePolicy::UpsweepPolicy::BLOCK_THREADS)))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRadixSortUpsweepKernel(
    const KeyT* d_keys,
    OffsetT* d_spine,
    OffsetT /*num_items*/,
    int current_bit,
    int num_bits,
    GridEvenShare<OffsetT> even_share,
    DecomposerT decomposer = {})
{
  using ActiveUpsweepPolicyT =
    ::cuda::std::_If<ALT_DIGIT_BITS,
                     typename ChainedPolicyT::ActivePolicy::AltUpsweepPolicy,
                     typename ChainedPolicyT::ActivePolicy::UpsweepPolicy>;

  using ActiveDownsweepPolicyT =
    ::cuda::std::_If<ALT_DIGIT_BITS,
                     typename ChainedPolicyT::ActivePolicy::AltDownsweepPolicy,
                     typename ChainedPolicyT::ActivePolicy::DownsweepPolicy>;

  enum
  {
    TILE_ITEMS = _CUDA_VSTD::max(ActiveUpsweepPolicyT::BLOCK_THREADS * ActiveUpsweepPolicyT::ITEMS_PER_THREAD,
                                 ActiveDownsweepPolicyT::BLOCK_THREADS * ActiveDownsweepPolicyT::ITEMS_PER_THREAD)
  };

  // Parameterize AgentRadixSortUpsweep type for the current configuration
  using AgentRadixSortUpsweepT =
    detail::radix_sort::AgentRadixSortUpsweep<ActiveUpsweepPolicyT, KeyT, OffsetT, DecomposerT>;

  // Shared memory storage
  __shared__ typename AgentRadixSortUpsweepT::TempStorage temp_storage;

  // Initialize GRID_MAPPING_RAKE even-share descriptor for this thread block
  even_share.template BlockInit<TILE_ITEMS, GRID_MAPPING_RAKE>();

  AgentRadixSortUpsweepT upsweep(temp_storage, d_keys, current_bit, num_bits, decomposer);

  upsweep.ProcessRegion(even_share.block_offset, even_share.block_end);

  __syncthreads();

  // Write out digit counts (striped)
  upsweep.template ExtractCounts<Order == SortOrder::Descending>(d_spine, gridDim.x, blockIdx.x);
}

/**
 * @brief Spine scan kernel entry point (single-block).
 *        Computes an exclusive prefix sum over the privatized digit histograms
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in,out] d_spine
 *   Privatized (per block) digit histograms (striped, i.e., 0s counts from each block,
 *   then 1s counts from each block, etc.)
 *
 * @param[in] num_counts
 *   Total number of bin-counts
 */
template <typename ChainedPolicyT, typename OffsetT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ScanPolicy::BLOCK_THREADS), 1)
  CUB_DETAIL_KERNEL_ATTRIBUTES void RadixSortScanBinsKernel(OffsetT* d_spine, int num_counts)
{
  // Parameterize the AgentScan type for the current configuration
  using AgentScanT =
    scan::AgentScan<typename ChainedPolicyT::ActivePolicy::ScanPolicy,
                    OffsetT*,
                    OffsetT*,
                    ::cuda::std::plus<>,
                    OffsetT,
                    OffsetT,
                    OffsetT>;

  // Shared memory storage
  __shared__ typename AgentScanT::TempStorage temp_storage;

  // Block scan instance
  AgentScanT block_scan(temp_storage, d_spine, d_spine, ::cuda::std::plus<>{}, OffsetT(0));

  // Process full input tiles
  int block_offset = 0;
  BlockScanRunningPrefixOp<OffsetT, ::cuda::std::plus<>> prefix_op(0, ::cuda::std::plus<>{});
  while (block_offset + AgentScanT::TILE_ITEMS <= num_counts)
  {
    block_scan.template ConsumeTile<false, false>(block_offset, prefix_op);
    block_offset += AgentScanT::TILE_ITEMS;
  }

  // Process the remaining partial tile (if any).
  if (block_offset < num_counts)
  {
    block_scan.template ConsumeTile<false, true>(block_offset, prefix_op, num_counts - block_offset);
  }
}

/**
 * @brief Downsweep pass kernel entry point (multi-block).
 *        Scatters keys (and values) into corresponding bins for the current digit place.
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam ALT_DIGIT_BITS
 *   Whether or not to use the alternate (lower-bits) policy
 *
 * @tparam SortOrder
 *   Whether to sort in ascending or descending order
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in] d_keys_in
 *   Input keys buffer
 *
 * @param[in] d_keys_out
 *   Output keys buffer
 *
 * @param[in] d_values_in
 *   Input values buffer
 *
 * @param[in] d_values_out
 *   Output values buffer
 *
 * @param[in] d_spine
 *   Scan of privatized (per block) digit histograms (striped, i.e., 0s counts from each block,
 *   then 1s counts from each block, etc.)
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] current_bit
 *   Bit position of current radix digit
 *
 * @param[in] num_bits
 *   Number of bits of current radix digit
 *
 * @param[in] even_share
 *   Even-share descriptor for mapan equal number of tiles onto each thread block
 */
template <typename ChainedPolicyT,
          bool ALT_DIGIT_BITS,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename DecomposerT = detail::identity_decomposer_t>
__launch_bounds__(int((ALT_DIGIT_BITS) ? int(ChainedPolicyT::ActivePolicy::AltDownsweepPolicy::BLOCK_THREADS)
                                       : int(ChainedPolicyT::ActivePolicy::DownsweepPolicy::BLOCK_THREADS)))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRadixSortDownsweepKernel(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    OffsetT* d_spine,
    OffsetT num_items,
    int current_bit,
    int num_bits,
    GridEvenShare<OffsetT> even_share,
    DecomposerT decomposer = {})
{
  using ActiveUpsweepPolicyT =
    ::cuda::std::_If<ALT_DIGIT_BITS,
                     typename ChainedPolicyT::ActivePolicy::AltUpsweepPolicy,
                     typename ChainedPolicyT::ActivePolicy::UpsweepPolicy>;

  using ActiveDownsweepPolicyT =
    ::cuda::std::_If<ALT_DIGIT_BITS,
                     typename ChainedPolicyT::ActivePolicy::AltDownsweepPolicy,
                     typename ChainedPolicyT::ActivePolicy::DownsweepPolicy>;

  enum
  {
    TILE_ITEMS = _CUDA_VSTD::max(ActiveUpsweepPolicyT::BLOCK_THREADS * ActiveUpsweepPolicyT::ITEMS_PER_THREAD,
                                 ActiveDownsweepPolicyT::BLOCK_THREADS * ActiveDownsweepPolicyT::ITEMS_PER_THREAD)
  };

  // Parameterize AgentRadixSortDownsweep type for the current configuration
  using AgentRadixSortDownsweepT = radix_sort::
    AgentRadixSortDownsweep<ActiveDownsweepPolicyT, Order == SortOrder::Descending, KeyT, ValueT, OffsetT, DecomposerT>;

  // Shared memory storage
  __shared__ typename AgentRadixSortDownsweepT::TempStorage temp_storage;

  // Initialize even-share descriptor for this thread block
  even_share.template BlockInit<TILE_ITEMS, GRID_MAPPING_RAKE>();

  // Process input tiles
  AgentRadixSortDownsweepT(
    temp_storage, num_items, d_spine, d_keys_in, d_keys_out, d_values_in, d_values_out, current_bit, num_bits, decomposer)
    .ProcessRegion(even_share.block_offset, even_share.block_end);
}

/**
 * @brief Single pass kernel entry point (single-block).
 *        Fully sorts a tile of input.
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam SortOrder
 *   Whether or not to use the alternate (lower-bits) policy
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in] d_keys_in
 *   Input keys buffer
 *
 * @param[in] d_keys_out
 *   Output keys buffer
 *
 * @param[in] d_values_in
 *   Input values buffer
 *
 * @param[in] d_values_out
 *   Output values buffer
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] current_bit
 *   Bit position of current radix digit
 *
 * @param[in] end_bit
 *   The past-the-end (most-significant) bit index needed for key comparison
 */
template <typename ChainedPolicyT,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename DecomposerT = identity_decomposer_t>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS), 1)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRadixSortSingleTileKernel(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    OffsetT num_items,
    int current_bit,
    int end_bit,
    DecomposerT decomposer = {})
{
  // Constants
  enum
  {
    BLOCK_THREADS    = ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS,
    ITEMS_PER_THREAD = ChainedPolicyT::ActivePolicy::SingleTilePolicy::ITEMS_PER_THREAD,
    KEYS_ONLY        = ::cuda::std::is_same_v<ValueT, NullType>,
  };

  // BlockRadixSort type
  using BlockRadixSortT =
    BlockRadixSort<KeyT,
                   BLOCK_THREADS,
                   ITEMS_PER_THREAD,
                   ValueT,
                   ChainedPolicyT::ActivePolicy::SingleTilePolicy::RADIX_BITS,
                   (ChainedPolicyT::ActivePolicy::SingleTilePolicy::RANK_ALGORITHM == RADIX_RANK_MEMOIZE),
                   ChainedPolicyT::ActivePolicy::SingleTilePolicy::SCAN_ALGORITHM>;

  // BlockLoad type (keys)
  using BlockLoadKeys =
    BlockLoad<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, ChainedPolicyT::ActivePolicy::SingleTilePolicy::LOAD_ALGORITHM>;

  // BlockLoad type (values)
  using BlockLoadValues =
    BlockLoad<ValueT, BLOCK_THREADS, ITEMS_PER_THREAD, ChainedPolicyT::ActivePolicy::SingleTilePolicy::LOAD_ALGORITHM>;

  // Unsigned word for key bits
  using traits           = detail::radix::traits_t<KeyT>;
  using bit_ordered_type = typename traits::bit_ordered_type;

  // Shared memory storage
  __shared__ union TempStorage
  {
    typename BlockRadixSortT::TempStorage sort;
    typename BlockLoadKeys::TempStorage load_keys;
    typename BlockLoadValues::TempStorage load_values;

  } temp_storage;

  // Keys and values for the block
  KeyT keys[ITEMS_PER_THREAD];
  ValueT values[ITEMS_PER_THREAD];

  // Get default (min/max) value for out-of-bounds keys
  bit_ordered_type default_key_bits =
    Order == SortOrder::Descending ? traits::min_raw_binary_key(decomposer) : traits::max_raw_binary_key(decomposer);

  KeyT default_key = reinterpret_cast<KeyT&>(default_key_bits);

  // Load keys
  BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in, keys, num_items, default_key);

  __syncthreads();

  // Load values
  if (!KEYS_ONLY)
  {
    // Register pressure work-around: moving num_items through shfl prevents compiler
    // from reusing guards/addressing from prior guarded loads
    num_items = ShuffleIndex<warp_threads>(num_items, 0, 0xffffffff);

    BlockLoadValues(temp_storage.load_values).Load(d_values_in, values, num_items);

    __syncthreads();
  }

  // Sort tile
  BlockRadixSortT(temp_storage.sort)
    .SortBlockedToStriped(
      keys,
      values,
      current_bit,
      end_bit,
      bool_constant_v < Order == SortOrder::Descending >,
      bool_constant_v<KEYS_ONLY>,
      decomposer);

  // Store keys and values
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    int item_offset = ITEM * BLOCK_THREADS + threadIdx.x;
    if (item_offset < num_items)
    {
      d_keys_out[item_offset] = keys[ITEM];
      if (!KEYS_ONLY)
      {
        d_values_out[item_offset] = values[ITEM];
      }
    }
  }
}

/**
 * @brief Segmented radix sorting pass (one block per segment)
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam ALT_DIGIT_BITS
 *   Whether or not to use the alternate (lower-bits) policy
 *
 * @tparam SortOrder
 *   Whether to sort in ascending or descending order
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam BeginOffsetIteratorT
 *   Random-access input iterator type for reading segment beginning offsets @iterator
 *
 * @tparam EndOffsetIteratorT
 *   Random-access input iterator type for reading segment ending offsets @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in] d_keys_in
 *   Input keys buffer
 *
 * @param[in] d_keys_out
 *   Output keys buffer
 *
 * @param[in] d_values_in
 *   Input values buffer
 *
 * @param[in] d_values_out
 *   Output values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length `num_segments`,
 *   such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup>
 *   data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length `num_segments`,
 *   such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup>
 *   data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.
 *   If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>,
 *   the <em>i</em><sup>th</sup> is considered empty.
 *
 * @param[in] num_segments
 *   The number of segments that comprise the sorting data
 *
 * @param[in] current_bit
 *   Bit position of current radix digit
 *
 * @param[in] pass_bits
 *   Number of bits of current radix digit
 */
template <typename ChainedPolicyT,
          bool ALT_DIGIT_BITS,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename SegmentSizeT,
          typename DecomposerT = detail::identity_decomposer_t>
__launch_bounds__(int((ALT_DIGIT_BITS) ? ChainedPolicyT::ActivePolicy::AltSegmentedPolicy::BLOCK_THREADS
                                       : ChainedPolicyT::ActivePolicy::SegmentedPolicy::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedRadixSortKernel(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int current_bit,
    int pass_bits,
    DecomposerT decomposer = {})
{
  //
  // Constants
  //

  using SegmentedPolicyT =
    ::cuda::std::_If<ALT_DIGIT_BITS,
                     typename ChainedPolicyT::ActivePolicy::AltSegmentedPolicy,
                     typename ChainedPolicyT::ActivePolicy::SegmentedPolicy>;

  enum
  {
    BLOCK_THREADS    = SegmentedPolicyT::BLOCK_THREADS,
    ITEMS_PER_THREAD = SegmentedPolicyT::ITEMS_PER_THREAD,
    RADIX_BITS       = SegmentedPolicyT::RADIX_BITS,
    TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD,
    RADIX_DIGITS     = 1 << RADIX_BITS,
    KEYS_ONLY        = ::cuda::std::is_same_v<ValueT, NullType>,
  };

  // Upsweep type
  using BlockUpsweepT = detail::radix_sort::AgentRadixSortUpsweep<SegmentedPolicyT, KeyT, SegmentSizeT, DecomposerT>;

  // Digit-scan type
  using DigitScanT = BlockScan<SegmentSizeT, BLOCK_THREADS>;

  // Downsweep type
  using BlockDownsweepT = detail::radix_sort::
    AgentRadixSortDownsweep<SegmentedPolicyT, Order == SortOrder::Descending, KeyT, ValueT, SegmentSizeT, DecomposerT>;

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
    typename BlockUpsweepT::TempStorage upsweep;
    typename BlockDownsweepT::TempStorage downsweep;
    struct
    {
      volatile SegmentSizeT reverse_counts_in[RADIX_DIGITS];
      volatile SegmentSizeT reverse_counts_out[RADIX_DIGITS];
      typename DigitScanT::TempStorage scan;
    };

  } temp_storage;

  const auto segment_id = blockIdx.x;

  // Ensure the size of the current segment does not overflow SegmentSizeT
  _CCCL_ASSERT(static_cast<decltype(d_end_offsets[segment_id] - d_begin_offsets[segment_id])>(
                 ::cuda::std::numeric_limits<SegmentSizeT>::max())
                 > (d_end_offsets[segment_id] - d_begin_offsets[segment_id]),
               "A single segment size is limited to the maximum value representable by SegmentSizeT");
  const auto num_items = static_cast<SegmentSizeT>(d_end_offsets[segment_id] - d_begin_offsets[segment_id]);

  // Check if empty segment
  if (num_items <= 0)
  {
    return;
  }

  // Upsweep
  BlockUpsweepT upsweep(
    temp_storage.upsweep, d_keys_in + d_begin_offsets[segment_id], current_bit, pass_bits, decomposer);
  upsweep.ProcessRegion(SegmentSizeT{0}, num_items);

  __syncthreads();

  // The count of each digit value in this pass (valid in the first RADIX_DIGITS threads)
  SegmentSizeT bin_count[BINS_TRACKED_PER_THREAD];
  upsweep.ExtractCounts(bin_count);

  __syncthreads();

  if (Order == SortOrder::Descending)
  {
    // Reverse bin counts
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        temp_storage.reverse_counts_in[bin_idx] = bin_count[track];
      }
    }

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        bin_count[track] = temp_storage.reverse_counts_in[RADIX_DIGITS - bin_idx - 1];
      }
    }
  }

  // Scan
  SegmentSizeT bin_offset[BINS_TRACKED_PER_THREAD]; // The scatter base offset within the segment for each digit value
                                                    // in this pass (valid in the first RADIX_DIGITS threads)
  DigitScanT(temp_storage.scan).ExclusiveSum(bin_count, bin_offset);

  if (Order == SortOrder::Descending)
  {
    // Reverse bin offsets
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        temp_storage.reverse_counts_out[threadIdx.x] = bin_offset[track];
      }
    }

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        bin_offset[track] = temp_storage.reverse_counts_out[RADIX_DIGITS - bin_idx - 1];
      }
    }
  }

  __syncthreads();

  // Downsweep
  BlockDownsweepT downsweep(
    temp_storage.downsweep,
    bin_offset,
    num_items,
    d_keys_in + d_begin_offsets[segment_id],
    d_keys_out + d_begin_offsets[segment_id],
    d_values_in + d_begin_offsets[segment_id],
    d_values_out + d_begin_offsets[segment_id],
    current_bit,
    pass_bits,
    decomposer);
  downsweep.ProcessRegion(SegmentSizeT{0}, num_items);
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
template <typename ChainedPolicyT,
          SortOrder Order,
          typename KeyT,
          typename OffsetT,
          typename DecomposerT = identity_decomposer_t>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(ChainedPolicyT::ActivePolicy::HistogramPolicy::BLOCK_THREADS) void DeviceRadixSortHistogramKernel(
  OffsetT* d_bins_out, const KeyT* d_keys_in, OffsetT num_items, int start_bit, int end_bit, DecomposerT decomposer = {})
{
  using HistogramPolicyT = typename ChainedPolicyT::ActivePolicy::HistogramPolicy;
  using AgentT = AgentRadixSortHistogram<HistogramPolicyT, Order == SortOrder::Descending, KeyT, OffsetT, DecomposerT>;
  __shared__ typename AgentT::TempStorage temp_storage;
  AgentT agent(temp_storage, d_bins_out, d_keys_in, num_items, start_bit, end_bit, decomposer);
  agent.Process();
}

template <typename ChainedPolicyT,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename PortionOffsetT,
          typename AtomicOffsetT = PortionOffsetT,
          typename DecomposerT   = identity_decomposer_t>
CUB_DETAIL_KERNEL_ATTRIBUTES void __launch_bounds__(ChainedPolicyT::ActivePolicy::OnesweepPolicy::BLOCK_THREADS)
  DeviceRadixSortOnesweepKernel(
    AtomicOffsetT* d_lookback,
    AtomicOffsetT* d_ctrs,
    OffsetT* d_bins_out,
    const OffsetT* d_bins_in,
    KeyT* d_keys_out,
    const KeyT* d_keys_in,
    ValueT* d_values_out,
    const ValueT* d_values_in,
    PortionOffsetT num_items,
    int current_bit,
    int num_bits,
    DecomposerT decomposer = {})
{
  using OnesweepPolicyT = typename ChainedPolicyT::ActivePolicy::OnesweepPolicy;
  using AgentT =
    AgentRadixSortOnesweep<OnesweepPolicyT,
                           Order == SortOrder::Descending,
                           KeyT,
                           ValueT,
                           OffsetT,
                           PortionOffsetT,
                           DecomposerT>;
  __shared__ typename AgentT::TempStorage s;

  AgentT agent(
    s,
    d_lookback,
    d_ctrs,
    d_bins_out,
    d_bins_in,
    d_keys_out,
    d_keys_in,
    d_values_out,
    d_values_in,
    num_items,
    current_bit,
    num_bits,
    decomposer);
  agent.Process();
}

/**
 * Exclusive sum kernel
 */
template <typename ChainedPolicyT, typename OffsetT>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRadixSortExclusiveSumKernel(OffsetT* d_bins)
{
  using ExclusiveSumPolicyT     = typename ChainedPolicyT::ActivePolicy::ExclusiveSumPolicy;
  constexpr int RADIX_BITS      = ExclusiveSumPolicyT::RADIX_BITS;
  constexpr int RADIX_DIGITS    = 1 << RADIX_BITS;
  constexpr int BLOCK_THREADS   = ExclusiveSumPolicyT::BLOCK_THREADS;
  constexpr int BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS;
  using BlockScan               = cub::BlockScan<OffsetT, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // load the bins
  OffsetT bins[BINS_PER_THREAD];
  int bin_start = blockIdx.x * RADIX_DIGITS;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int u = 0; u < BINS_PER_THREAD; ++u)
  {
    int bin = threadIdx.x * BINS_PER_THREAD + u;
    if (bin >= RADIX_DIGITS)
    {
      break;
    }
    bins[u] = d_bins[bin_start + bin];
  }

  // compute offsets
  BlockScan(temp_storage).ExclusiveSum(bins, bins);

  // store the offsets
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int u = 0; u < BINS_PER_THREAD; ++u)
  {
    int bin = threadIdx.x * BINS_PER_THREAD + u;
    if (bin >= RADIX_DIGITS)
    {
      break;
    }
    d_bins[bin_start + bin] = bins[u];
  }
}

} // namespace detail::radix_sort

CUB_NAMESPACE_END
