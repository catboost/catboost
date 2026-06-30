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
 * cub::AgentHistogram implements a stateful abstraction of CUDA thread blocks for participating in device-wide
 * histogram .
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

#include <cub/block/block_load.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 *
 */
enum BlockHistogramMemoryPreference
{
  GMEM,
  SMEM,
  BLEND
};

/**
 * Parameterizable tuning policy type for AgentHistogram
 *
 * @tparam _BLOCK_THREADS
 *   Threads per thread block
 *
 * @tparam _PIXELS_PER_THREAD
 *   Pixels per thread (per tile of input)
 *
 * @tparam _LOAD_ALGORITHM
 *   The BlockLoad algorithm to use
 *
 * @tparam _LOAD_MODIFIER
 *   Cache load modifier for reading input elements
 *
 * @tparam _RLE_COMPRESS
 *   Whether to perform localized RLE to compress samples before histogramming
 *
 * @tparam _MEM_PREFERENCE
 *   Whether to prefer privatized shared-memory bins (versus privatized global-memory bins)
 *
 * @tparam _WORK_STEALING
 *   Whether to dequeue tiles from a global work queue
 *
 * @tparam _VEC_SIZE
 *   Vector size for samples loading (1, 2, 4)
 */
template <int _BLOCK_THREADS,
          int _PIXELS_PER_THREAD,
          BlockLoadAlgorithm _LOAD_ALGORITHM,
          CacheLoadModifier _LOAD_MODIFIER,
          bool _RLE_COMPRESS,
          BlockHistogramMemoryPreference _MEM_PREFERENCE,
          bool _WORK_STEALING,
          int _VEC_SIZE = 4>
struct AgentHistogramPolicy
{
  /// Threads per thread block
  static constexpr int BLOCK_THREADS = _BLOCK_THREADS;
  /// Pixels per thread (per tile of input)
  static constexpr int PIXELS_PER_THREAD = _PIXELS_PER_THREAD;

  /// Whether to perform localized RLE to compress samples before histogramming
  static constexpr bool IS_RLE_COMPRESS = _RLE_COMPRESS;

  /// Whether to prefer privatized shared-memory bins (versus privatized global-memory bins)
  static constexpr BlockHistogramMemoryPreference MEM_PREFERENCE = _MEM_PREFERENCE;

  /// Whether to dequeue tiles from a global work queue
  static constexpr bool IS_WORK_STEALING = _WORK_STEALING;

  /// Vector size for samples loading (1, 2, 4)
  static constexpr int VEC_SIZE = _VEC_SIZE;

  ///< The BlockLoad algorithm to use
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;

  ///< Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = _LOAD_MODIFIER;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

namespace detail
{
namespace histogram
{

/**
 * @brief AgentHistogram implements a stateful abstraction of CUDA thread blocks for participating
 * in device-wide histogram .
 *
 * @tparam AgentHistogramPolicyT
 *   Parameterized AgentHistogramPolicy tuning policy type
 *
 * @tparam PRIVATIZED_SMEM_BINS
 *   Number of privatized shared-memory histogram bins of any channel.  Zero indicates privatized
 * counters to be maintained in device-accessible memory.
 *
 * @tparam NUM_CHANNELS
 *   Number of channels interleaved in the input data.  Supports up to four channels.
 *
 * @tparam NUM_ACTIVE_CHANNELS
 *   Number of channels actively being histogrammed
 *
 * @tparam SampleIteratorT
 *   Random-access input iterator type for reading samples
 *
 * @tparam CounterT
 *   Integer type for counting sample occurrences per histogram bin
 *
 * @tparam PrivatizedDecodeOpT
 *   The transform operator type for determining privatized counter indices from samples, one for
 * each channel
 *
 * @tparam OutputDecodeOpT
 *   The transform operator type for determining output bin-ids from privatized counter indices, one
 * for each channel
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <typename AgentHistogramPolicyT,
          int PRIVATIZED_SMEM_BINS,
          int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT>
struct AgentHistogram
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  /// The sample type of the input iterator
  using SampleT = cub::detail::it_value_t<SampleIteratorT>;

  /// The pixel type of SampleT
  using PixelT = typename CubVector<SampleT, NUM_CHANNELS>::Type;

  /// The vec type of SampleT
  static constexpr int VecSize = AgentHistogramPolicyT::VEC_SIZE;
  using VecT                   = typename CubVector<SampleT, VecSize>::Type;

  /// Constants
  static constexpr int BLOCK_THREADS = AgentHistogramPolicyT::BLOCK_THREADS;

  static constexpr int PIXELS_PER_THREAD  = AgentHistogramPolicyT::PIXELS_PER_THREAD;
  static constexpr int SAMPLES_PER_THREAD = PIXELS_PER_THREAD * NUM_CHANNELS;
  static constexpr int VECS_PER_THREAD    = SAMPLES_PER_THREAD / VecSize;

  static constexpr int TILE_PIXELS  = PIXELS_PER_THREAD * BLOCK_THREADS;
  static constexpr int TILE_SAMPLES = SAMPLES_PER_THREAD * BLOCK_THREADS;

  static constexpr bool IS_RLE_COMPRESS = AgentHistogramPolicyT::IS_RLE_COMPRESS;

  static constexpr BlockHistogramMemoryPreference MEM_PREFERENCE =
    (PRIVATIZED_SMEM_BINS > 0) ? AgentHistogramPolicyT::MEM_PREFERENCE : GMEM;

  static constexpr bool IS_WORK_STEALING = AgentHistogramPolicyT::IS_WORK_STEALING;

  /// Cache load modifier for reading input elements
  static constexpr CacheLoadModifier LOAD_MODIFIER = AgentHistogramPolicyT::LOAD_MODIFIER;

  /// Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using WrappedSampleIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<SampleIteratorT>,
                     CacheModifiedInputIterator<LOAD_MODIFIER, SampleT, OffsetT>,
                     SampleIteratorT>;

  /// Pixel input iterator type (for applying cache modifier)
  using WrappedPixelIteratorT = CacheModifiedInputIterator<LOAD_MODIFIER, PixelT, OffsetT>;

  /// Qaud input iterator type (for applying cache modifier)
  using WrappedVecsIteratorT = CacheModifiedInputIterator<LOAD_MODIFIER, VecT, OffsetT>;

  /// Parameterized BlockLoad type for samples
  using BlockLoadSampleT = BlockLoad<SampleT, BLOCK_THREADS, SAMPLES_PER_THREAD, AgentHistogramPolicyT::LOAD_ALGORITHM>;

  /// Parameterized BlockLoad type for pixels
  using BlockLoadPixelT = BlockLoad<PixelT, BLOCK_THREADS, PIXELS_PER_THREAD, AgentHistogramPolicyT::LOAD_ALGORITHM>;

  /// Parameterized BlockLoad type for vecs
  using BlockLoadVecT = BlockLoad<VecT, BLOCK_THREADS, VECS_PER_THREAD, AgentHistogramPolicyT::LOAD_ALGORITHM>;

  /// Shared memory type required by this thread block
  struct _TempStorage
  {
    // Smem needed for block-privatized smem histogram (with 1 word of padding)
    CounterT histograms[NUM_ACTIVE_CHANNELS][PRIVATIZED_SMEM_BINS + 1];

    int tile_idx;

    // Aliasable storage layout
    union Aliasable
    {
      // Smem needed for loading a tile of samples
      typename BlockLoadSampleT::TempStorage sample_load;

      // Smem needed for loading a tile of pixels
      typename BlockLoadPixelT::TempStorage pixel_load;

      // Smem needed for loading a tile of vecs
      typename BlockLoadVecT::TempStorage vec_load;

    } aliasable;
  };

  /// Temporary storage type (unionable)
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  /// Reference to temp_storage
  _TempStorage& temp_storage;

  /// Sample input iterator (with cache modifier applied, if possible)
  WrappedSampleIteratorT d_wrapped_samples;

  /// Native pointer for input samples (possibly nullptr if unavailable)
  SampleT* d_native_samples;

  /// The number of output bins for each channel
  int* num_output_bins;

  /// The number of privatized bins for each channel
  int* num_privatized_bins;

  /// Copy of gmem privatized histograms for each channel
  CounterT* d_privatized_histograms[NUM_ACTIVE_CHANNELS];

  /// Reference to final output histograms (gmem)
  CounterT** d_output_histograms;

  /// The transform operator for determining output bin-ids from privatized counter indices, one for each channel
  OutputDecodeOpT* output_decode_op;

  /// The transform operator for determining privatized counter indices from samples, one for each channel
  PrivatizedDecodeOpT* privatized_decode_op;

  /// Whether to prefer privatized smem counters vs privatized global counters
  bool prefer_smem;

  //---------------------------------------------------------------------
  // Initialize privatized bin counters
  //---------------------------------------------------------------------

  // Initialize privatized bin counters
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitBinCounters(CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS])
  {
    // Initialize histogram bin counts to zeros
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
      for (int privatized_bin = threadIdx.x; privatized_bin < num_privatized_bins[CHANNEL];
           privatized_bin += BLOCK_THREADS)
      {
        privatized_histograms[CHANNEL][privatized_bin] = 0;
      }
    }

    // Barrier to make sure all threads are done updating counters
    __syncthreads();
  }

  // Initialize privatized bin counters.  Specialized for privatized shared-memory counters
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitSmemBinCounters()
  {
    CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS];

    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
      privatized_histograms[CHANNEL] = temp_storage.histograms[CHANNEL];
    }

    InitBinCounters(privatized_histograms);
  }

  // Initialize privatized bin counters.  Specialized for privatized global-memory counters
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitGmemBinCounters()
  {
    InitBinCounters(d_privatized_histograms);
  }

  //---------------------------------------------------------------------
  // Update final output histograms
  //---------------------------------------------------------------------

  // Update final output histograms from privatized histograms
  _CCCL_DEVICE _CCCL_FORCEINLINE void StoreOutput(CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS])
  {
    // Barrier to make sure all threads are done updating counters
    __syncthreads();

    // Apply privatized bin counts to output bin counts
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
      int channel_bins = num_privatized_bins[CHANNEL];
      for (int privatized_bin = threadIdx.x; privatized_bin < channel_bins; privatized_bin += BLOCK_THREADS)
      {
        int output_bin = -1;
        CounterT count = privatized_histograms[CHANNEL][privatized_bin];
        bool is_valid  = count > 0;

        output_decode_op[CHANNEL].template BinSelect<LOAD_MODIFIER>((SampleT) privatized_bin, output_bin, is_valid);

        if (output_bin >= 0)
        {
          atomicAdd(&d_output_histograms[CHANNEL][output_bin], count);
        }
      }
    }
  }

  // Update final output histograms from privatized histograms.  Specialized for privatized shared-memory counters
  _CCCL_DEVICE _CCCL_FORCEINLINE void StoreSmemOutput()
  {
    CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS];
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
      privatized_histograms[CHANNEL] = temp_storage.histograms[CHANNEL];
    }

    StoreOutput(privatized_histograms);
  }

  // Update final output histograms from privatized histograms.  Specialized for privatized global-memory counters
  _CCCL_DEVICE _CCCL_FORCEINLINE void StoreGmemOutput()
  {
    StoreOutput(d_privatized_histograms);
  }

  //---------------------------------------------------------------------
  // Tile accumulation
  //---------------------------------------------------------------------

  // Accumulate pixels.  Specialized for RLE compression.
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulatePixels(
    SampleT samples[PIXELS_PER_THREAD][NUM_CHANNELS],
    bool is_valid[PIXELS_PER_THREAD],
    CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS],
    ::cuda::std::true_type is_rle_compress)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
      // Bin pixels
      int bins[PIXELS_PER_THREAD];

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
      {
        bins[PIXEL] = -1;
        privatized_decode_op[CHANNEL].template BinSelect<LOAD_MODIFIER>(
          samples[PIXEL][CHANNEL], bins[PIXEL], is_valid[PIXEL]);
      }

      CounterT accumulator = 1;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD - 1; ++PIXEL)
      {
        if (bins[PIXEL] != bins[PIXEL + 1])
        {
          if (bins[PIXEL] >= 0)
          {
            NV_IF_TARGET(NV_PROVIDES_SM_60,
                         (atomicAdd_block(privatized_histograms[CHANNEL] + bins[PIXEL], accumulator);),
                         (atomicAdd(privatized_histograms[CHANNEL] + bins[PIXEL], accumulator);));
          }

          accumulator = 0;
        }
        accumulator++;
      }

      // Last pixel
      if (bins[PIXELS_PER_THREAD - 1] >= 0)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_60,
                     (atomicAdd_block(privatized_histograms[CHANNEL] + bins[PIXELS_PER_THREAD - 1], accumulator);),
                     (atomicAdd(privatized_histograms[CHANNEL] + bins[PIXELS_PER_THREAD - 1], accumulator);));
      }
    }
  }

  // Accumulate pixels.  Specialized for individual accumulation of each pixel.
  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulatePixels(
    SampleT samples[PIXELS_PER_THREAD][NUM_CHANNELS],
    bool is_valid[PIXELS_PER_THREAD],
    CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS],
    ::cuda::std::false_type is_rle_compress)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
      {
        int bin = -1;
        privatized_decode_op[CHANNEL].template BinSelect<LOAD_MODIFIER>(samples[PIXEL][CHANNEL], bin, is_valid[PIXEL]);
        if (bin >= 0)
        {
          NV_IF_TARGET(NV_PROVIDES_SM_60,
                       (atomicAdd_block(privatized_histograms[CHANNEL] + bin, 1);),
                       (atomicAdd(privatized_histograms[CHANNEL] + bin, 1);));
        }
      }
    }
  }

  /**
   * Accumulate pixel, specialized for smem privatized histogram
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  AccumulateSmemPixels(SampleT samples[PIXELS_PER_THREAD][NUM_CHANNELS], bool is_valid[PIXELS_PER_THREAD])
  {
    CounterT* privatized_histograms[NUM_ACTIVE_CHANNELS];

    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
      privatized_histograms[CHANNEL] = temp_storage.histograms[CHANNEL];
    }

    AccumulatePixels(samples, is_valid, privatized_histograms, ::cuda::std::bool_constant<IS_RLE_COMPRESS>{});
  }

  /**
   * Accumulate pixel, specialized for gmem privatized histogram
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  AccumulateGmemPixels(SampleT samples[PIXELS_PER_THREAD][NUM_CHANNELS], bool is_valid[PIXELS_PER_THREAD])
  {
    AccumulatePixels(samples, is_valid, d_privatized_histograms, ::cuda::std::bool_constant<IS_RLE_COMPRESS>{});
  }

  //---------------------------------------------------------------------
  // Tile loading
  //---------------------------------------------------------------------

  // Load full, aligned tile using pixel iterator (multi-channel)
  template <int _NUM_ACTIVE_CHANNELS>
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadFullAlignedTile(
    OffsetT block_offset,
    int valid_samples,
    SampleT (&samples)[PIXELS_PER_THREAD][NUM_CHANNELS],
    constant_t<_NUM_ACTIVE_CHANNELS> num_active_channels)
  {
    using AliasedPixels = PixelT[PIXELS_PER_THREAD];

    WrappedPixelIteratorT d_wrapped_pixels((PixelT*) (d_native_samples + block_offset));

    // Load using a wrapped pixel iterator
    BlockLoadPixelT(temp_storage.aliasable.pixel_load).Load(d_wrapped_pixels, reinterpret_cast<AliasedPixels&>(samples));
  }

  // Load full, aligned tile using vec iterator (single-channel)
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadFullAlignedTile(
    OffsetT block_offset,
    int valid_samples,
    SampleT (&samples)[PIXELS_PER_THREAD][NUM_CHANNELS],
    constant_t<1> num_active_channels)
  {
    using AliasedVecs = VecT[VECS_PER_THREAD];

    WrappedVecsIteratorT d_wrapped_vecs((VecT*) (d_native_samples + block_offset));

    // Load using a wrapped vec iterator
    BlockLoadVecT(temp_storage.aliasable.vec_load).Load(d_wrapped_vecs, reinterpret_cast<AliasedVecs&>(samples));
  }

  // Load full, aligned tile
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadTile(
    OffsetT block_offset,
    int valid_samples,
    SampleT (&samples)[PIXELS_PER_THREAD][NUM_CHANNELS],
    ::cuda::std::true_type is_full_tile,
    ::cuda::std::true_type is_aligned)
  {
    LoadFullAlignedTile(block_offset, valid_samples, samples, constant_v<NUM_ACTIVE_CHANNELS>);
  }

  // Load full, mis-aligned tile using sample iterator
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadTile(
    OffsetT block_offset,
    int valid_samples,
    SampleT (&samples)[PIXELS_PER_THREAD][NUM_CHANNELS],
    ::cuda::std::true_type is_full_tile,
    ::cuda::std::false_type is_aligned)
  {
    using AliasedSamples = SampleT[SAMPLES_PER_THREAD];

    // Load using sample iterator
    BlockLoadSampleT(temp_storage.aliasable.sample_load)
      .Load(d_wrapped_samples + block_offset, reinterpret_cast<AliasedSamples&>(samples));
  }

  // Load partially-full, aligned tile using the pixel iterator
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadTile(
    OffsetT block_offset,
    int valid_samples,
    SampleT (&samples)[PIXELS_PER_THREAD][NUM_CHANNELS],
    ::cuda::std::false_type is_full_tile,
    ::cuda::std::true_type is_aligned)
  {
    using AliasedPixels = PixelT[PIXELS_PER_THREAD];

    WrappedPixelIteratorT d_wrapped_pixels((PixelT*) (d_native_samples + block_offset));

    int valid_pixels = valid_samples / NUM_CHANNELS;

    // Load using a wrapped pixel iterator
    BlockLoadPixelT(temp_storage.aliasable.pixel_load)
      .Load(d_wrapped_pixels, reinterpret_cast<AliasedPixels&>(samples), valid_pixels);
  }

  // Load partially-full, mis-aligned tile using sample iterator
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadTile(
    OffsetT block_offset,
    int valid_samples,
    SampleT (&samples)[PIXELS_PER_THREAD][NUM_CHANNELS],
    ::cuda::std::false_type is_full_tile,
    ::cuda::std::false_type is_aligned)
  {
    using AliasedSamples = SampleT[SAMPLES_PER_THREAD];

    BlockLoadSampleT(temp_storage.aliasable.sample_load)
      .Load(d_wrapped_samples + block_offset, reinterpret_cast<AliasedSamples&>(samples), valid_samples);
  }

  template <bool IS_FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  MarkValid(bool (&is_valid)[PIXELS_PER_THREAD], int valid_samples, ::cuda::std::false_type /* is_striped = false */)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
    {
      is_valid[PIXEL] = IS_FULL_TILE || (((threadIdx.x * PIXELS_PER_THREAD + PIXEL) * NUM_CHANNELS) < valid_samples);
    }
  }

  template <bool IS_FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  MarkValid(bool (&is_valid)[PIXELS_PER_THREAD], int valid_samples, ::cuda::std::true_type /* is_striped = true */)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int PIXEL = 0; PIXEL < PIXELS_PER_THREAD; ++PIXEL)
    {
      is_valid[PIXEL] = IS_FULL_TILE || (((threadIdx.x + BLOCK_THREADS * PIXEL) * NUM_CHANNELS) < valid_samples);
    }
  }

  //---------------------------------------------------------------------
  // Tile processing
  //---------------------------------------------------------------------

  /**
   * @brief Consume a tile of data samples
   *
   * @tparam IS_ALIGNED
   *   Whether the tile offset is aligned (vec-aligned for single-channel, pixel-aligned for multi-channel)
   *
   * @tparam IS_FULL_TILE
      Whether the tile is full
   */
  template <bool IS_ALIGNED, bool IS_FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTile(OffsetT block_offset, int valid_samples)
  {
    SampleT samples[PIXELS_PER_THREAD][NUM_CHANNELS];
    bool is_valid[PIXELS_PER_THREAD];

    // Load tile
    LoadTile(block_offset, valid_samples, samples, bool_constant_v<IS_FULL_TILE>, bool_constant_v<IS_ALIGNED>);

    // Set valid flags
    MarkValid<IS_FULL_TILE>(
      is_valid, valid_samples, bool_constant_v < AgentHistogramPolicyT::LOAD_ALGORITHM == BLOCK_LOAD_STRIPED >);

    // Accumulate samples
    if (prefer_smem)
    {
      AccumulateSmemPixels(samples, is_valid);
    }
    else
    {
      AccumulateGmemPixels(samples, is_valid);
    }
  }

  /**
   * @brief Consume row tiles. Specialized for work-stealing from queue
   *
   * @param num_row_pixels
   *   The number of multi-channel pixels per row in the region of interest
   *
   * @param num_rows
   *   The number of rows in the region of interest
   *
   * @param row_stride_samples
   *   The number of samples between starts of consecutive rows in the region of interest
   *
   * @param tiles_per_row
   *   Number of image tiles per row
   */
  template <bool IS_ALIGNED>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTiles(
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    int tiles_per_row,
    GridQueue<int> tile_queue,
    ::cuda::std::true_type is_work_stealing)
  {
    int num_tiles                = num_rows * tiles_per_row;
    int tile_idx                 = (blockIdx.y * gridDim.x) + blockIdx.x;
    OffsetT num_even_share_tiles = gridDim.x * gridDim.y;

    while (tile_idx < num_tiles)
    {
      int row             = tile_idx / tiles_per_row;
      int col             = tile_idx - (row * tiles_per_row);
      OffsetT row_offset  = row * row_stride_samples;
      OffsetT col_offset  = (col * TILE_SAMPLES);
      OffsetT tile_offset = row_offset + col_offset;

      if (col == tiles_per_row - 1)
      {
        // Consume a partially-full tile at the end of the row
        OffsetT num_remaining = (num_row_pixels * NUM_CHANNELS) - col_offset;
        ConsumeTile<IS_ALIGNED, false>(tile_offset, num_remaining);
      }
      else
      {
        // Consume full tile
        ConsumeTile<IS_ALIGNED, true>(tile_offset, TILE_SAMPLES);
      }

      __syncthreads();

      // Get next tile
      if (threadIdx.x == 0)
      {
        temp_storage.tile_idx = tile_queue.Drain(1) + num_even_share_tiles;
      }

      __syncthreads();

      tile_idx = temp_storage.tile_idx;
    }
  }

  /**
   * @brief Consume row tiles.  Specialized for even-share (striped across thread blocks)
   *
   * @param num_row_pixels
   *   The number of multi-channel pixels per row in the region of interest
   *
   * @param num_rows
   *   The number of rows in the region of interest
   *
   * @param row_stride_samples
   *   The number of samples between starts of consecutive rows in the region of interest
   *
   * @param tiles_per_row
   *   Number of image tiles per row
   */
  template <bool IS_ALIGNED>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTiles(
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    int tiles_per_row,
    GridQueue<int> tile_queue,
    ::cuda::std::false_type is_work_stealing)
  {
    for (int row = blockIdx.y; row < num_rows; row += gridDim.y)
    {
      OffsetT row_begin   = row * row_stride_samples;
      OffsetT row_end     = row_begin + (num_row_pixels * NUM_CHANNELS);
      OffsetT tile_offset = row_begin + (blockIdx.x * TILE_SAMPLES);

      while (tile_offset < row_end)
      {
        OffsetT num_remaining = row_end - tile_offset;

        if (num_remaining < TILE_SAMPLES)
        {
          // Consume partial tile
          ConsumeTile<IS_ALIGNED, false>(tile_offset, num_remaining);
          break;
        }

        // Consume full tile
        ConsumeTile<IS_ALIGNED, true>(tile_offset, TILE_SAMPLES);
        tile_offset += gridDim.x * TILE_SAMPLES;
      }
    }
  }

  //---------------------------------------------------------------------
  // Parameter extraction
  //---------------------------------------------------------------------

  // Return a native pixel pointer (specialized for CacheModifiedInputIterator types)
  template <CacheLoadModifier _MODIFIER, typename _ValueT, typename _OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE SampleT* NativePointer(CacheModifiedInputIterator<_MODIFIER, _ValueT, _OffsetT> itr)
  {
    return itr.ptr;
  }

  // Return a native pixel pointer (specialized for other types)
  template <typename IteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE SampleT* NativePointer(IteratorT itr)
  {
    return nullptr;
  }

  //---------------------------------------------------------------------
  // Interface
  //---------------------------------------------------------------------

  /**
   * @brief Constructor
   *
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param d_samples
   *   Input data to reduce
   *
   * @param num_output_bins
   *   The number bins per final output histogram
   *
   * @param num_privatized_bins
   *   The number bins per privatized histogram
   *
   * @param d_output_histograms
   *   Reference to final output histograms
   *
   * @param d_privatized_histograms
   *   Reference to privatized histograms
   *
   * @param output_decode_op
   *   The transform operator for determining output bin-ids from privatized counter indices, one for each channel
   *
   * @param privatized_decode_op
   *   The transform operator for determining privatized counter indices from samples, one for each channel
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentHistogram(
    TempStorage& temp_storage,
    SampleIteratorT d_samples,
    int* num_output_bins,
    int* num_privatized_bins,
    CounterT** d_output_histograms,
    CounterT** d_privatized_histograms,
    OutputDecodeOpT* output_decode_op,
    PrivatizedDecodeOpT* privatized_decode_op)
      : temp_storage(temp_storage.Alias())
      , d_wrapped_samples(d_samples)
      , d_native_samples(NativePointer(d_wrapped_samples))
      , num_output_bins(num_output_bins)
      , num_privatized_bins(num_privatized_bins)
      , d_output_histograms(d_output_histograms)
      , output_decode_op(output_decode_op)
      , privatized_decode_op(privatized_decode_op)
      , prefer_smem((MEM_PREFERENCE == SMEM) ? true : // prefer smem privatized histograms
                      (MEM_PREFERENCE == GMEM) ? false
                                               : // prefer gmem privatized histograms
                      blockIdx.x & 1) // prefer blended privatized histograms
  {
    int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;

    // Initialize the locations of this block's privatized histograms
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
      this->d_privatized_histograms[CHANNEL] =
        d_privatized_histograms[CHANNEL] + (blockId * num_privatized_bins[CHANNEL]);
    }
  }

  /**
   * @brief Consume image
   *
   * @param num_row_pixels
   *   The number of multi-channel pixels per row in the region of interest
   *
   * @param num_rows
   *   The number of rows in the region of interest
   *
   * @param row_stride_samples
   *   The number of samples between starts of consecutive rows in the region of interest
   *
   * @param tiles_per_row
   *   Number of image tiles per row
   *
   * @param tile_queue
   *   Queue descriptor for assigning tiles of work to thread blocks
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeTiles(
    OffsetT num_row_pixels, OffsetT num_rows, OffsetT row_stride_samples, int tiles_per_row, GridQueue<int> tile_queue)
  {
    // Check whether all row starting offsets are vec-aligned (in single-channel) or pixel-aligned (in multi-channel)
    int vec_mask     = AlignBytes<VecT>::ALIGN_BYTES - 1;
    int pixel_mask   = AlignBytes<PixelT>::ALIGN_BYTES - 1;
    size_t row_bytes = sizeof(SampleT) * row_stride_samples;

    bool vec_aligned_rows =
      (NUM_CHANNELS == 1) && (SAMPLES_PER_THREAD % VecSize == 0) && // Single channel
      ((size_t(d_native_samples) & vec_mask) == 0) && // ptr is quad-aligned
      ((num_rows == 1) || ((row_bytes & vec_mask) == 0)); // number of row-samples is a multiple of the alignment of the
                                                          // quad

    bool pixel_aligned_rows =
      (NUM_CHANNELS > 1) && // Multi channel
      ((size_t(d_native_samples) & pixel_mask) == 0) && // ptr is pixel-aligned
      ((row_bytes & pixel_mask) == 0); // number of row-samples is a multiple of the alignment of the pixel

    // Whether rows are aligned and can be vectorized
    if ((d_native_samples != nullptr) && (vec_aligned_rows || pixel_aligned_rows))
    {
      ConsumeTiles<true>(
        num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue, bool_constant_v<IS_WORK_STEALING>);
    }
    else
    {
      ConsumeTiles<false>(
        num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue, bool_constant_v<IS_WORK_STEALING>);
    }
  }

  /**
   * Initialize privatized bin counters.  Specialized for privatized shared-memory counters
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void InitBinCounters()
  {
    if (prefer_smem)
    {
      InitSmemBinCounters();
    }
    else
    {
      InitGmemBinCounters();
    }
  }

  /**
   * Store privatized histogram to device-accessible memory.  Specialized for privatized shared-memory counters
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void StoreOutput()
  {
    if (prefer_smem)
    {
      StoreSmemOutput();
    }
    else
    {
      StoreGmemOutput();
    }
  }
};

} // namespace histogram
} // namespace detail

CUB_NAMESPACE_END
