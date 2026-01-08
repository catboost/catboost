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
 *   cub::DeviceHistogram provides device-wide parallel operations for constructing histogram(s)
 *   from a sequence of samples data residing within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#include <cuda/std/__type_traits/is_void.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_histogram.cuh>
#include <cub/device/dispatch/kernels/histogram.cuh>
#include <cub/device/dispatch/tuning/tuning_histogram.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/functional>
#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::histogram
{
template <int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS, typename SampleIteratorT, typename CounterT, typename OffsetT>
struct DeviceHistogramKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(HistogramInitKernel, DeviceHistogramInitKernel<NUM_ACTIVE_CHANNELS, CounterT, OffsetT>);

  // We define this differently than the other kernel getters because there are
  // different dispatch paths which affect which policy is used and the decode operators.
  template <typename PolicyT, int PRIVATIZED_SMEM_BINS, typename PrivatizedDecodeOpT, typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepKernel()
  {
    return &DeviceHistogramSweepKernel<
      PolicyT,
      PRIVATIZED_SMEM_BINS,
      NUM_CHANNELS,
      NUM_ACTIVE_CHANNELS,
      SampleIteratorT,
      CounterT,
      PrivatizedDecodeOpT,
      OutputDecodeOpT,
      OffsetT>;
  }

  CUB_RUNTIME_FUNCTION static constexpr size_t CounterSize()
  {
    return sizeof(CounterT);
  }
};

template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          int PRIVATIZED_SMEM_BINS,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT,
          typename MaxPolicyT,
          typename KernelSource,
          typename KernelLauncherFactory>
struct dispatch_histogram
{
  void* d_temp_storage;
  size_t& temp_storage_bytes;
  SampleIteratorT d_samples;
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms;
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels;
  ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op;
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels;
  ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op;
  int max_num_output_bins;
  OffsetT num_row_pixels;
  OffsetT num_rows;
  OffsetT row_stride_samples;
  cudaStream_t stream;
  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;

  template <typename ActivePolicyT, typename DeviceHistogramInitKernelT, typename DeviceHistogramSweepKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  Invoke(DeviceHistogramInitKernelT histogram_init_kernel,
         DeviceHistogramSweepKernelT histogram_sweep_kernel,
         ActivePolicyT policy = {})
  {
    cudaError error = cudaSuccess;

    auto wrapped_policy = detail::histogram::MakeHistogramPolicyWrapper(policy);

    const int block_threads     = wrapped_policy.BlockThreads();
    const int pixels_per_thread = wrapped_policy.PixelsPerThread();

    do
    {
      // Get SM count
      int sm_count;
      error = CubDebug(launcher_factory.MultiProcessorCount(sm_count));

      if (cudaSuccess != error)
      {
        break;
      }

      // Get SM occupancy for histogram_sweep_kernel
      int histogram_sweep_sm_occupancy;
      error =
        CubDebug(launcher_factory.MaxSmOccupancy(histogram_sweep_sm_occupancy, histogram_sweep_kernel, block_threads));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get device occupancy for histogram_sweep_kernel
      int histogram_sweep_occupancy = histogram_sweep_sm_occupancy * sm_count;

      if (num_row_pixels * NUM_CHANNELS == row_stride_samples)
      {
        // Treat as a single linear array of samples
        num_row_pixels *= num_rows;
        num_rows           = 1;
        row_stride_samples = num_row_pixels * NUM_CHANNELS;
      }

      // Get grid dimensions, trying to keep total blocks ~histogram_sweep_occupancy
      int pixels_per_tile = block_threads * pixels_per_thread;
      int tiles_per_row   = static_cast<int>(::cuda::ceil_div(num_row_pixels, pixels_per_tile));
      int blocks_per_row  = _CUDA_VSTD::min(histogram_sweep_occupancy, tiles_per_row);
      int blocks_per_col =
        (blocks_per_row > 0)
          ? int(_CUDA_VSTD::min(static_cast<OffsetT>(histogram_sweep_occupancy / blocks_per_row), num_rows))
          : 0;
      int num_thread_blocks = blocks_per_row * blocks_per_col;

      dim3 sweep_grid_dims;
      sweep_grid_dims.x = (unsigned int) blocks_per_row;
      sweep_grid_dims.y = (unsigned int) blocks_per_col;
      sweep_grid_dims.z = 1;

      // Temporary storage allocation requirements
      constexpr int NUM_ALLOCATIONS      = NUM_ACTIVE_CHANNELS + 1;
      void* allocations[NUM_ALLOCATIONS] = {};
      size_t allocation_sizes[NUM_ALLOCATIONS];

      for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
      {
        allocation_sizes[CHANNEL] =
          size_t(num_thread_blocks) * (num_privatized_levels[CHANNEL] - 1) * kernel_source.CounterSize();
      }

      allocation_sizes[NUM_ALLOCATIONS - 1] = GridQueue<int>::AllocationSize();

      // Alias the temporary allocations from the single storage blob (or compute the
      // necessary size of the blob)
      error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      // Construct the grid queue descriptor
      GridQueue<int> tile_queue(allocations[NUM_ALLOCATIONS - 1]);

      // Wrap arrays so we can pass them by-value to the kernel
      ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_privatized_histograms_wrapper;
      ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_bins_wrapper;
      ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_wrapper;

      auto* typedAllocations = reinterpret_cast<CounterT**>(allocations);
      ::cuda::std::copy(
        typedAllocations, typedAllocations + NUM_ACTIVE_CHANNELS, d_privatized_histograms_wrapper.begin());

      auto minus_one = ::cuda::proclaim_return_type<int>([](int levels) {
        return levels - 1;
      });
      ::cuda::std::transform(
        num_privatized_levels.begin(), num_privatized_levels.end(), num_privatized_bins_wrapper.begin(), minus_one);
      ::cuda::std::transform(
        num_output_levels.begin(), num_output_levels.end(), num_output_bins_wrapper.begin(), minus_one);
      int histogram_init_block_threads = 256;

      int histogram_init_grid_dims =
        (max_num_output_bins + histogram_init_block_threads - 1) / histogram_init_block_threads;

// Log DeviceHistogramInitKernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking DeviceHistogramInitKernel<<<%d, %d, 0, %lld>>>()\n",
              histogram_init_grid_dims,
              histogram_init_block_threads,
              (long long) stream);
#endif // CUB_DEBUG_LOG

      // Invoke histogram_init_kernel
      launcher_factory(histogram_init_grid_dims, histogram_init_block_threads, 0, stream)
        .doit(histogram_init_kernel, num_output_bins_wrapper, d_output_histograms, tile_queue);

      // Return if empty problem
      if ((blocks_per_row == 0) || (blocks_per_col == 0))
      {
        break;
      }

// Log histogram_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking histogram_sweep_kernel<<<{%d, %d, %d}, %d, 0, %lld>>>(), %d pixels "
              "per thread, %d SM occupancy\n",
              sweep_grid_dims.x,
              sweep_grid_dims.y,
              sweep_grid_dims.z,
              block_threads,
              (long long) stream,
              pixels_per_thread,
              histogram_sweep_sm_occupancy);
#endif // CUB_DEBUG_LOG

      // Invoke histogram_sweep_kernel
      launcher_factory(sweep_grid_dims, block_threads, 0, stream)
        .doit(histogram_sweep_kernel,
              d_samples,
              num_output_bins_wrapper,
              num_privatized_bins_wrapper,
              d_output_histograms,
              d_privatized_histograms_wrapper,
              output_decode_op,
              privatized_decode_op,
              num_row_pixels,
              num_rows,
              row_stride_samples,
              tiles_per_row,
              tile_queue);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT active_policy = {})
  {
    return Invoke<ActivePolicyT>(
      kernel_source.HistogramInitKernel(),
      kernel_source
        .template HistogramSweepKernel<MaxPolicyT, PRIVATIZED_SMEM_BINS, PrivatizedDecodeOpT, OutputDecodeOpT>(),
      active_policy);
  }
};

} // namespace detail::histogram

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceHistogram
 *
 * @tparam NUM_CHANNELS
 *   Number of channels interleaved in the input data (may be greater than the number of channels
 *   being actively histogrammed)
 *
 * @tparam NUM_ACTIVE_CHANNELS
 *   Number of channels actively being histogrammed
 *
 * @tparam SampleIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam CounterT
 *   Integer type for counting sample occurrences per histogram bin
 *
 * @tparam LevelT
 *   Type for specifying bin level boundaries
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam PolicyHub
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          typename PolicyHub    = void, // if user passes a custom Policy this should not be void
          typename KernelSource = detail::histogram::
            DeviceHistogramKernelSource<NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleIteratorT, CounterT, OffsetT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
          typename SampleT = cub::detail::it_value_t<SampleIteratorT>, /// The sample value type of the input iterator
          typename TransformsT = detail::histogram::Transforms<LevelT, OffsetT, SampleT>>
struct DispatchHistogram
{
  static_assert(NUM_CHANNELS <= 4, "Histograms only support up to 4 channels");
  static_assert(NUM_ACTIVE_CHANNELS <= NUM_CHANNELS,
                "Active channels must be at most the number of total channels of the input samples");

public:
  //---------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------

  /**
   * Dispatch routine for HistogramRange, specialized for sample types larger than 8bit
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to `temp_storage_bytes` and
   *   no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_samples
   *   The pointer to the multi-channel input sequence of data samples.
   *   The samples from different channels are assumed to be interleaved
   *   (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
   *
   * @param d_output_histograms
   *   The pointers to the histogram counter output arrays, one for each active channel.
   *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
   *   `num_output_levels[i] - 1`.
   *
   * @param num_output_levels
   *   The number of boundaries (levels) for delineating histogram samples in each active channel.
   *   Implies that the number of bins for channel<sub><em>i</em></sub> is
   *   `num_output_levels[i] - 1`.
   *
   * @param d_levels
   *   The pointers to the arrays of boundaries (levels), one for each active channel.
   *   Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are
   *   inclusive and upper sample value boundaries are exclusive.
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
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   *
   * @param is_byte_sample
   *   type indicating whether or not SampleT is a 8b type
   */
  // Should we call DispatchHistogram<....., PolicyHub=void> in DeviceHistogram?
  template <typename MaxPolicyT = typename ::cuda::std::_If<
              ::cuda::std::is_void_v<PolicyHub>,
              /* fallback_policy_hub */
              detail::histogram::policy_hub<SampleT, CounterT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, /* isEven */ 0>,
              PolicyHub>::MaxPolicy>
  CUB_RUNTIME_FUNCTION static cudaError_t DispatchRange(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    SampleIteratorT d_samples,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
    ::cuda::std::array<const LevelT*, NUM_ACTIVE_CHANNELS> d_levels,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    cudaStream_t stream,
    ::cuda::std::false_type /*is_byte_sample*/,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(launcher_factory.PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Use the search transform op for converting samples to privatized bins
      using PrivatizedDecodeOpT = typename TransformsT::template SearchTransform<const LevelT*>;

      // Use the pass-thru transform op for converting privatized bins to output bins
      using OutputDecodeOpT = typename TransformsT::PassThruTransform;

      ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
      ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
      int max_levels = num_output_levels[0];

      for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
      {
        privatized_decode_op[channel].Init(d_levels[channel], num_output_levels[channel]);
        if (num_output_levels[channel] > max_levels)
        {
          max_levels = num_output_levels[channel];
        }
      }
      int max_num_output_bins = max_levels - 1;

      // Dispatch
      if (max_num_output_bins > TransformsT::MAX_PRIVATIZED_SMEM_BINS)
      {
        // Too many bins to keep in shared memory.
        constexpr int PRIVATIZED_SMEM_BINS = 0;

        detail::histogram::dispatch_histogram<
          NUM_CHANNELS,
          NUM_ACTIVE_CHANNELS,
          PRIVATIZED_SMEM_BINS,
          SampleIteratorT,
          CounterT,
          PrivatizedDecodeOpT,
          OutputDecodeOpT,
          OffsetT,
          MaxPolicyT,
          KernelSource,
          KernelLauncherFactory>
          dispatch{
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_output_levels,
            privatized_decode_op,
            num_output_levels,
            output_decode_op,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            kernel_source,
            launcher_factory};

        error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
        if (cudaSuccess != error)
        {
          break;
        }
      }
      else
      {
        // Dispatch shared-privatized approach
        constexpr int PRIVATIZED_SMEM_BINS = TransformsT::MAX_PRIVATIZED_SMEM_BINS;

        detail::histogram::dispatch_histogram<
          NUM_CHANNELS,
          NUM_ACTIVE_CHANNELS,
          PRIVATIZED_SMEM_BINS,
          SampleIteratorT,
          CounterT,
          PrivatizedDecodeOpT,
          OutputDecodeOpT,
          OffsetT,
          MaxPolicyT,
          KernelSource,
          KernelLauncherFactory>
          dispatch{
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_output_levels,
            privatized_decode_op,
            num_output_levels,
            output_decode_op,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            kernel_source,
            launcher_factory};

        error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
        if (cudaSuccess != error)
        {
          break;
        }
      }
    } while (0);

    return error;
  }

  /**
   * Dispatch routine for HistogramRange, specialized for 8-bit sample types
   * (computes 256-bin privatized histograms and then reduces to user-specified levels)
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to `temp_storage_bytes` and
   *   no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_samples
   *   The pointer to the multi-channel input sequence of data samples.
   *   The samples from different channels are assumed to be interleaved
   *   (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
   *
   * @param d_output_histograms
   *   The pointers to the histogram counter output arrays, one for each active channel.
   *   For channel<sub><em>i</em></sub>, the allocation length of
   *   `d_histograms[i]` should be `num_output_levels[i] - 1`.
   *
   * @param num_output_levels
   *   The number of boundaries (levels) for delineating histogram samples in each active channel.
   *   Implies that the number of bins for channel<sub><em>i</em></sub> is
   *   `num_output_levels[i] - 1`.
   *
   * @param d_levels
   *   The pointers to the arrays of boundaries (levels), one for each active channel.
   *   Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are
   *   inclusive and upper sample value boundaries are exclusive.
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
   * @param stream
   *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
   *
   * @param is_byte_sample
   *   Marker type indicating whether or not SampleT is a 8b type
   */
  template <typename MaxPolicyT = typename ::cuda::std::_If<
              ::cuda::std::is_void_v<PolicyHub>,
              /* fallback_policy_hub */
              detail::histogram::policy_hub<SampleT, CounterT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, /* isEven */ 0>,
              PolicyHub>::MaxPolicy>
  CUB_RUNTIME_FUNCTION static cudaError_t DispatchRange(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    SampleIteratorT d_samples,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
    ::cuda::std::array<const LevelT*, NUM_ACTIVE_CHANNELS> d_levels,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    cudaStream_t stream,
    ::cuda::std::true_type /*is_byte_sample*/,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(launcher_factory.PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Use the pass-thru transform op for converting samples to privatized bins
      using PrivatizedDecodeOpT = typename TransformsT::PassThruTransform;

      // Use the search transform op for converting privatized bins to output bins
      using OutputDecodeOpT = typename TransformsT::template SearchTransform<const LevelT*>;

      ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels;
      ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
      ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
      int max_levels = num_output_levels[0]; // Maximum number of levels in any channel

      for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
      {
        num_privatized_levels[channel] = 257;
        output_decode_op[channel].Init(d_levels[channel], num_output_levels[channel]);

        if (num_output_levels[channel] > max_levels)
        {
          max_levels = num_output_levels[channel];
        }
      }
      int max_num_output_bins = max_levels - 1;

      constexpr int PRIVATIZED_SMEM_BINS = 256;

      detail::histogram::dispatch_histogram<
        NUM_CHANNELS,
        NUM_ACTIVE_CHANNELS,
        PRIVATIZED_SMEM_BINS,
        SampleIteratorT,
        CounterT,
        PrivatizedDecodeOpT,
        OutputDecodeOpT,
        OffsetT,
        MaxPolicyT,
        KernelSource,
        KernelLauncherFactory>
        dispatch{
          d_temp_storage,
          temp_storage_bytes,
          d_samples,
          d_output_histograms,
          num_privatized_levels,
          privatized_decode_op,
          num_output_levels,
          output_decode_op,
          max_num_output_bins,
          num_row_pixels,
          num_rows,
          row_stride_samples,
          stream,
          kernel_source,
          launcher_factory};

      error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  /**
   * Dispatch routine for HistogramEven, specialized for sample types larger than 8-bit
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to
   *   `temp_storage_bytes` and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_samples
   *   The pointer to the input sequence of sample items.
   *   The samples from different channels are assumed to be interleaved
   *   (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
   *
   * @param d_output_histograms
   *   The pointers to the histogram counter output arrays, one for each active channel.
   *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
   *   `num_output_levels[i] - 1`.
   *
   * @param num_output_levels
   *   The number of bin level boundaries for delineating histogram samples in each active channel.
   *   Implies that the number of bins for channel<sub><em>i</em></sub> is
   *   `num_output_levels[i] - 1`.
   *
   * @param lower_level
   *   The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
   *
   * @param upper_level
   *   The upper sample value bound (exclusive) for the highest histogram bin in each active
   * channel.
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
   * @param stream
   *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
   *
   * @param is_byte_sample
   *   Marker type indicating whether or not SampleT is a 8b type
   */
  template <typename MaxPolicyT = typename ::cuda::std::_If<
              ::cuda::std::is_void_v<PolicyHub>,
              /* fallback_policy_hub */
              detail::histogram::policy_hub<SampleT, CounterT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, /* isEven */ 1>,
              PolicyHub>::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t DispatchEven(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    SampleIteratorT d_samples,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> lower_level,
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> upper_level,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    cudaStream_t stream,
    ::cuda::std::false_type /*is_byte_sample*/,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(launcher_factory.PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Use the scale transform op for converting samples to privatized bins
      using PrivatizedDecodeOpT = typename TransformsT::ScaleTransform;

      // Use the pass-thru transform op for converting privatized bins to output bins
      using OutputDecodeOpT = typename TransformsT::PassThruTransform;

      ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
      ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
      int max_levels = num_output_levels[0];

      for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
      {
        error = CubDebug(
          privatized_decode_op[channel].Init(num_output_levels[channel], upper_level[channel], lower_level[channel]));
        if (error != cudaSuccess)
        {
          // Make sure to also return a reasonable value for `temp_storage_bytes` in case of
          // an overflow of the bin computation, in which case a subsequent algorithm
          // invocation will also fail
          if (!d_temp_storage)
          {
            temp_storage_bytes = 1U;
          }
          return error;
        }

        if (num_output_levels[channel] > max_levels)
        {
          max_levels = num_output_levels[channel];
        }
      }
      int max_num_output_bins = max_levels - 1;

      if (max_num_output_bins > TransformsT::MAX_PRIVATIZED_SMEM_BINS)
      {
        // Dispatch shared-privatized approach
        constexpr int PRIVATIZED_SMEM_BINS = 0;

        detail::histogram::dispatch_histogram<
          NUM_CHANNELS,
          NUM_ACTIVE_CHANNELS,
          PRIVATIZED_SMEM_BINS,
          SampleIteratorT,
          CounterT,
          PrivatizedDecodeOpT,
          OutputDecodeOpT,
          OffsetT,
          MaxPolicyT,
          KernelSource,
          KernelLauncherFactory>
          dispatch{
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_output_levels,
            privatized_decode_op,
            num_output_levels,
            output_decode_op,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            kernel_source,
            launcher_factory};

        error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
        if (cudaSuccess != error)
        {
          break;
        }
      }
      else
      {
        // Dispatch shared-privatized approach
        constexpr int PRIVATIZED_SMEM_BINS = TransformsT::MAX_PRIVATIZED_SMEM_BINS;

        detail::histogram::dispatch_histogram<
          NUM_CHANNELS,
          NUM_ACTIVE_CHANNELS,
          PRIVATIZED_SMEM_BINS,
          SampleIteratorT,
          CounterT,
          PrivatizedDecodeOpT,
          OutputDecodeOpT,
          OffsetT,
          MaxPolicyT,
          KernelSource,
          KernelLauncherFactory>
          dispatch{
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            d_output_histograms,
            num_output_levels,
            privatized_decode_op,
            num_output_levels,
            output_decode_op,
            max_num_output_bins,
            num_row_pixels,
            num_rows,
            row_stride_samples,
            stream,
            kernel_source,
            launcher_factory};

        error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
        if (cudaSuccess != error)
        {
          break;
        }
      }
    } while (0);

    return error;
  }

  /**
   * Dispatch routine for HistogramEven, specialized for 8-bit sample types
   * (computes 256-bin privatized histograms and then reduces to user-specified levels)
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When nullptr, the required allocation size is written to `temp_storage_bytes` and
   *   no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_samples
   *   The pointer to the input sequence of sample items. The samples from different channels are
   *   assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of
   *   four RGBA 8-bit samples).
   *
   * @param d_output_histograms
   *   The pointers to the histogram counter output arrays, one for each active channel.
   *   For channel<sub><em>i</em></sub>, the allocation length of `d_histograms[i]` should be
   *   `num_output_levels[i] - 1`.
   *
   * @param num_output_levels
   *   The number of bin level boundaries for delineating histogram samples in each active channel.
   *   Implies that the number of bins for channel<sub><em>i</em></sub> is
   *   `num_output_levels[i] - 1`.
   *
   * @param lower_level
   *   The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
   *
   * @param upper_level
   *   The upper sample value bound (exclusive) for the highest histogram bin in each active
   * channel.
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
   * @param stream
   *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
   *
   * @param is_byte_sample
   *   type indicating whether or not SampleT is a 8b type
   */
  template <typename MaxPolicyT = typename ::cuda::std::_If<
              ::cuda::std::is_void_v<PolicyHub>,
              /* fallback_policy_hub */
              detail::histogram::policy_hub<SampleT, CounterT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, /* isEven */ 1>,
              PolicyHub>::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t DispatchEven(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    SampleIteratorT d_samples,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_levels,
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> lower_level,
    ::cuda::std::array<LevelT, NUM_ACTIVE_CHANNELS> upper_level,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    cudaStream_t stream,
    ::cuda::std::true_type /*is_byte_sample*/,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(launcher_factory.PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Use the pass-thru transform op for converting samples to privatized bins
      using PrivatizedDecodeOpT = typename TransformsT::PassThruTransform;

      // Use the scale transform op for converting privatized bins to output bins
      using OutputDecodeOpT = typename TransformsT::ScaleTransform;

      ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_levels;
      ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op{};
      ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op{};
      int max_levels = num_output_levels[0];

      for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
      {
        num_privatized_levels[channel] = 257;

        output_decode_op[channel].Init(num_output_levels[channel], upper_level[channel], lower_level[channel]);

        if (num_output_levels[channel] > max_levels)
        {
          max_levels = num_output_levels[channel];
        }
      }
      int max_num_output_bins = max_levels - 1;

      constexpr int PRIVATIZED_SMEM_BINS = 256;

      detail::histogram::dispatch_histogram<
        NUM_CHANNELS,
        NUM_ACTIVE_CHANNELS,
        PRIVATIZED_SMEM_BINS,
        SampleIteratorT,
        CounterT,
        PrivatizedDecodeOpT,
        OutputDecodeOpT,
        OffsetT,
        MaxPolicyT,
        KernelSource,
        KernelLauncherFactory>
        dispatch{
          d_temp_storage,
          temp_storage_bytes,
          d_samples,
          d_output_histograms,
          num_privatized_levels,
          privatized_decode_op,
          num_output_levels,
          output_decode_op,
          max_num_output_bins,
          num_row_pixels,
          num_rows,
          row_stride_samples,
          stream,
          kernel_source,
          launcher_factory};

      error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

CUB_NAMESPACE_END
