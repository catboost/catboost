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
 * cub::DeviceRadixSort provides device-wide, parallel operations for computing a radix sort across
 * a sequence of data items residing within device-accessible memory.
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

#include <cub/device/dispatch/kernels/radix_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_radix_sort.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/__algorithm_>
#include <cuda/std/type_traits>

// suppress warnings triggered by #pragma unroll:
// "warning: loop not unrolled: the optimizer was unable to perform the requested transformation; the transformation
// might be disabled or specified as part of an unsupported transformation ordering [-Wpass-failed=transform-warning]"
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wpass-failed")

CUB_NAMESPACE_BEGIN

namespace detail::radix_sort
{
template <typename MaxPolicyT, SortOrder Order, typename KeyT, typename ValueT, typename OffsetT, typename DecomposerT>
struct DeviceRadixSortKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(RadixSortSingleTileKernel,
                           DeviceRadixSortSingleTileKernel<MaxPolicyT, Order, KeyT, ValueT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortUpsweepKernel,
                           DeviceRadixSortUpsweepKernel<MaxPolicyT, false, Order, KeyT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortAltUpsweepKernel,
                           DeviceRadixSortUpsweepKernel<MaxPolicyT, true, Order, KeyT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(DeviceRadixSortScanBinsKernel, RadixSortScanBinsKernel<MaxPolicyT, OffsetT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortDownsweepKernel,
                           DeviceRadixSortDownsweepKernel<MaxPolicyT, false, Order, KeyT, ValueT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortAltDownsweepKernel,
                           DeviceRadixSortDownsweepKernel<MaxPolicyT, true, Order, KeyT, ValueT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortHistogramKernel,
                           DeviceRadixSortHistogramKernel<MaxPolicyT, Order, KeyT, OffsetT, DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(RadixSortExclusiveSumKernel, DeviceRadixSortExclusiveSumKernel<MaxPolicyT, OffsetT>);

  CUB_DEFINE_KERNEL_GETTER(
    RadixSortOnesweepKernel,
    DeviceRadixSortOnesweepKernel<MaxPolicyT, Order, KeyT, ValueT, OffsetT, int, int, DecomposerT>);

  CUB_RUNTIME_FUNCTION static constexpr size_t KeySize()
  {
    return sizeof(KeyT);
  }

  CUB_RUNTIME_FUNCTION static constexpr size_t ValueSize()
  {
    return sizeof(ValueT);
  }
};

template <typename MaxPolicyT,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename SegmentSizeT,
          typename DecomposerT>
struct DeviceSegmentedRadixSortKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    SegmentedRadixSortKernel,
    DeviceSegmentedRadixSortKernel<
      MaxPolicyT,
      false,
      Order,
      KeyT,
      ValueT,
      BeginOffsetIteratorT,
      EndOffsetIteratorT,
      SegmentSizeT,
      DecomposerT>);

  CUB_DEFINE_KERNEL_GETTER(
    AltSegmentedRadixSortKernel,
    DeviceSegmentedRadixSortKernel<
      MaxPolicyT,
      true,
      Order,
      KeyT,
      ValueT,
      BeginOffsetIteratorT,
      EndOffsetIteratorT,
      SegmentSizeT,
      DecomposerT>);

  CUB_RUNTIME_FUNCTION static constexpr size_t KeySize()
  {
    return sizeof(KeyT);
  }

  CUB_RUNTIME_FUNCTION static constexpr size_t ValueSize()
  {
    return sizeof(ValueT);
  }
};

} // namespace detail::radix_sort

/******************************************************************************
 * Single-problem dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for device-wide radix sort
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
 * @tparam DecomposerT
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename DecomposerT  = detail::identity_decomposer_t,
          typename PolicyHub    = detail::radix::policy_hub<KeyT, ValueT, OffsetT>,
          typename KernelSource = detail::radix_sort::
            DeviceRadixSortKernelSource<typename PolicyHub::MaxPolicy, Order, KeyT, ValueT, OffsetT, DecomposerT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchRadixSort
{
  //------------------------------------------------------------------------------
  // Constants
  //------------------------------------------------------------------------------

  // Whether this is a keys-only (or key-value) sort
  static constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  //------------------------------------------------------------------------------
  // Problem state
  //------------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage.
  //  When nullptr, the required allocation size is written to `temp_storage_bytes` and no work is
  //  done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Double-buffer whose current buffer contains the unsorted input keys and, upon return, is
  /// updated to point to the sorted output keys
  DoubleBuffer<KeyT>& d_keys;

  /// Double-buffer whose current buffer contains the unsorted input values and, upon return, is
  /// updated to point to the sorted output values
  DoubleBuffer<ValueT>& d_values;

  /// Number of items to sort
  OffsetT num_items;

  /// The beginning (least-significant) bit index needed for key comparison
  int begin_bit;

  /// The past-the-end (most-significant) bit index needed for key comparison
  int end_bit;

  /// CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
  cudaStream_t stream;

  /// PTX version
  int ptx_version;

  /// Whether is okay to overwrite source buffers
  bool is_overwrite_okay;

  DecomposerT decomposer;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  //------------------------------------------------------------------------------
  // Constructor
  //------------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchRadixSort(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    OffsetT num_items,
    int begin_bit,
    int end_bit,
    bool is_overwrite_okay,
    cudaStream_t stream,
    int ptx_version,
    DecomposerT decomposer                 = {},
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys(d_keys)
      , d_values(d_values)
      , num_items(num_items)
      , begin_bit(begin_bit)
      , end_bit(end_bit)
      , stream(stream)
      , ptx_version(ptx_version)
      , is_overwrite_okay(is_overwrite_okay)
      , decomposer(decomposer)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  //------------------------------------------------------------------------------
  // Small-problem (single tile) invocation
  //------------------------------------------------------------------------------

  /**
   * @brief Invoke a single block to sort in-core
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeviceRadixSortSingleTileKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of cub::DeviceRadixSortSingleTileKernel
   */
  template <typename ActivePolicyT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeSingleTile(SingleTileKernelT single_tile_kernel, ActivePolicyT policy = {})
  {
    cudaError error = cudaSuccess;
    do
    {
      // Return if the caller is simply requesting the size of the storage allocation
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
        break;
      }

// Log single_tile_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking single_tile_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy, current bit "
              "%d, bit_grain %d\n",
              1,
              policy.SingleTile().BlockThreads(),
              (long long) stream,
              policy.SingleTile().ItemsPerThread(),
              1,
              begin_bit,
              policy.RadixBits(policy.SingleTile()));
#endif

      // Invoke upsweep_kernel with same grid size as downsweep_kernel
      launcher_factory(1, policy.SingleTile().BlockThreads(), 0, stream)
        .doit(single_tile_kernel,
              d_keys.Current(),
              d_keys.Alternate(),
              d_values.Current(),
              d_values.Alternate(),
              num_items,
              begin_bit,
              end_bit,
              decomposer);

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

      // Update selector
      d_keys.selector ^= 1;
      d_values.selector ^= 1;
    } while (0);

    return error;
  }

  //------------------------------------------------------------------------------
  // Normal problem size invocation
  //------------------------------------------------------------------------------

  /**
   * Invoke a three-kernel sorting pass at the current bit.
   */
  template <typename PassConfigT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokePass(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    OffsetT* d_spine,
    int /*spine_length*/,
    int& current_bit,
    PassConfigT& pass_config)
  {
    cudaError error = cudaSuccess;
    do
    {
      int pass_bits = _CUDA_VSTD::min(pass_config.radix_bits, end_bit - current_bit);

// Log upsweep_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking upsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy, current bit %d, "
              "bit_grain %d\n",
              pass_config.even_share.grid_size,
              pass_config.upsweep_config.block_threads,
              (long long) stream,
              pass_config.upsweep_config.items_per_thread,
              pass_config.upsweep_config.sm_occupancy,
              current_bit,
              pass_bits);
#endif

      // Spine length written by the upsweep kernel in the current pass.
      int pass_spine_length = pass_config.even_share.grid_size * pass_config.radix_digits;

      // Invoke upsweep_kernel with same grid size as downsweep_kernel
      launcher_factory(pass_config.even_share.grid_size, pass_config.upsweep_config.block_threads, 0, stream)
        .doit(pass_config.upsweep_kernel,
              d_keys_in,
              d_spine,
              num_items,
              current_bit,
              pass_bits,
              pass_config.even_share,
              decomposer);

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

// Log scan_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking scan_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread\n",
              1,
              pass_config.scan_config.block_threads,
              (long long) stream,
              pass_config.scan_config.items_per_thread);
#endif

      // Invoke scan_kernel
      launcher_factory(1, pass_config.scan_config.block_threads, 0, stream)
        .doit(pass_config.scan_kernel, d_spine, pass_spine_length);

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

// Log downsweep_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking downsweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
              pass_config.even_share.grid_size,
              pass_config.downsweep_config.block_threads,
              (long long) stream,
              pass_config.downsweep_config.items_per_thread,
              pass_config.downsweep_config.sm_occupancy);
#endif

      // Invoke downsweep_kernel
      launcher_factory(pass_config.even_share.grid_size, pass_config.downsweep_config.block_threads, 0, stream)
        .doit(pass_config.downsweep_kernel,
              d_keys_in,
              d_keys_out,
              d_values_in,
              d_values_out,
              d_spine,
              num_items,
              current_bit,
              pass_bits,
              pass_config.even_share,
              decomposer);

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

      // Update current bit
      current_bit += pass_bits;
    } while (0);

    return error;
  }

  /// Pass configuration structure
  template <typename UpsweepKernelT, typename ScanKernelT, typename DownsweepKernelT>
  struct PassConfig
  {
    UpsweepKernelT upsweep_kernel;
    detail::KernelConfig upsweep_config;
    ScanKernelT scan_kernel;
    detail::KernelConfig scan_config;
    DownsweepKernelT downsweep_kernel;
    detail::KernelConfig downsweep_config;
    int radix_bits;
    int radix_digits;
    int max_downsweep_grid_size;
    GridEvenShare<OffsetT> even_share;

    /// Initialize pass configuration
    template <typename ActivePolicyT, typename UpsweepPolicyT, typename ScanPolicyT, typename DownsweepPolicyT>
    CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InitPassConfig(
      UpsweepKernelT upsweep_kernel,
      ScanKernelT scan_kernel,
      DownsweepKernelT downsweep_kernel,
      int /*ptx_version*/,
      int sm_count,
      OffsetT num_items,
      ActivePolicyT policy                   = {},
      UpsweepPolicyT upsweep_policy          = {},
      ScanPolicyT scan_policy                = {},
      DownsweepPolicyT downsweep_policy      = {},
      KernelLauncherFactory launcher_factory = {})
    {
      cudaError error = cudaSuccess;
      do
      {
        this->upsweep_kernel   = upsweep_kernel;
        this->scan_kernel      = scan_kernel;
        this->downsweep_kernel = downsweep_kernel;
        radix_bits             = policy.RadixBits(downsweep_policy);
        radix_digits           = 1 << radix_bits;

        error = CubDebug(upsweep_config.Init(upsweep_kernel, upsweep_policy, launcher_factory));
        if (cudaSuccess != error)
        {
          break;
        }

        error = CubDebug(scan_config.Init(scan_kernel, scan_policy, launcher_factory));
        if (cudaSuccess != error)
        {
          break;
        }

        error = CubDebug(downsweep_config.Init(downsweep_kernel, downsweep_policy, launcher_factory));
        if (cudaSuccess != error)
        {
          break;
        }

        max_downsweep_grid_size = (downsweep_config.sm_occupancy * sm_count) * detail::subscription_factor;

        even_share.DispatchInit(
          num_items, max_downsweep_grid_size, _CUDA_VSTD::max(downsweep_config.tile_size, upsweep_config.tile_size));

      } while (0);
      return error;
    }
  };

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokeOnesweep(ActivePolicyT policy = {})
  {
    // PortionOffsetT is used for offsets within a portion, and must be signed.
    using PortionOffsetT = int;
    using AtomicOffsetT  = PortionOffsetT;

    // compute temporary storage size
    const int RADIX_BITS                = policy.RadixBits(policy.Onesweep());
    const int RADIX_DIGITS              = 1 << RADIX_BITS;
    const int ONESWEEP_ITEMS_PER_THREAD = policy.Onesweep().ItemsPerThread();
    const int ONESWEEP_BLOCK_THREADS    = policy.Onesweep().BlockThreads();
    const int ONESWEEP_TILE_ITEMS       = ONESWEEP_ITEMS_PER_THREAD * ONESWEEP_BLOCK_THREADS;
    // portions handle inputs with >=2**30 elements, due to the way lookback works
    // for testing purposes, one portion is <= 2**28 elements
    const PortionOffsetT PORTION_SIZE = ((1 << 28) - 1) / ONESWEEP_TILE_ITEMS * ONESWEEP_TILE_ITEMS;
    int num_passes                    = ::cuda::ceil_div(end_bit - begin_bit, RADIX_BITS);
    OffsetT num_portions              = static_cast<OffsetT>(::cuda::ceil_div(num_items, PORTION_SIZE));
    PortionOffsetT max_num_blocks     = ::cuda::ceil_div(
      static_cast<int>(_CUDA_VSTD::min(num_items, static_cast<OffsetT>(PORTION_SIZE))), ONESWEEP_TILE_ITEMS);

    size_t value_size         = KEYS_ONLY ? 0 : kernel_source.ValueSize();
    size_t allocation_sizes[] = {
      // bins
      num_portions * num_passes * RADIX_DIGITS * sizeof(OffsetT),
      // lookback
      max_num_blocks * RADIX_DIGITS * sizeof(AtomicOffsetT),
      // extra key buffer
      is_overwrite_okay || num_passes <= 1 ? 0 : num_items * kernel_source.KeySize(),
      // extra value buffer
      is_overwrite_okay || num_passes <= 1 ? 0 : num_items * value_size,
      // counters
      num_portions * num_passes * sizeof(AtomicOffsetT),
    };
    constexpr int NUM_ALLOCATIONS      = sizeof(allocation_sizes) / sizeof(allocation_sizes[0]);
    void* allocations[NUM_ALLOCATIONS] = {};
    detail::AliasTemporaries<NUM_ALLOCATIONS>(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);

    // just return if no temporary storage is provided
    cudaError_t error = cudaSuccess;
    if (d_temp_storage == nullptr)
    {
      return error;
    }

    OffsetT* d_bins           = (OffsetT*) allocations[0];
    AtomicOffsetT* d_lookback = (AtomicOffsetT*) allocations[1];
    KeyT* d_keys_tmp2         = (KeyT*) allocations[2];
    ValueT* d_values_tmp2     = (ValueT*) allocations[3];
    AtomicOffsetT* d_ctrs     = (AtomicOffsetT*) allocations[4];

    do
    {
      // initialization
      error = CubDebug(cudaMemsetAsync(d_ctrs, 0, num_portions * num_passes * sizeof(AtomicOffsetT), stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // compute num_passes histograms with RADIX_DIGITS bins each
      error = CubDebug(cudaMemsetAsync(d_bins, 0, num_passes * RADIX_DIGITS * sizeof(OffsetT), stream));
      if (cudaSuccess != error)
      {
        break;
      }
      int device  = -1;
      int num_sms = 0;

      error = CubDebug(cudaGetDevice(&device));
      if (cudaSuccess != error)
      {
        break;
      }

      error = CubDebug(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
      if (cudaSuccess != error)
      {
        break;
      }

      const int HISTO_BLOCK_THREADS = policy.Histogram().BlockThreads();
      int histo_blocks_per_sm       = 1;
      auto histogram_kernel         = kernel_source.RadixSortHistogramKernel();

      error = CubDebug(launcher_factory.MaxSmOccupancy(histo_blocks_per_sm, histogram_kernel, HISTO_BLOCK_THREADS, 0));
      if (cudaSuccess != error)
      {
        break;
      }

// log histogram_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking histogram_kernel<<<%d, %d, 0, %lld>>>(), %d items per iteration, "
              "%d SM occupancy, bit_grain %d\n",
              histo_blocks_per_sm * num_sms,
              HISTO_BLOCK_THREADS,
              reinterpret_cast<long long>(stream),
              policy.Histogram().ItemsPerThread(),
              histo_blocks_per_sm,
              policy.RadixBits(policy.Histogram()));
#endif

      error = launcher_factory(histo_blocks_per_sm * num_sms, HISTO_BLOCK_THREADS, 0, stream)
                .doit(histogram_kernel, d_bins, d_keys.Current(), num_items, begin_bit, end_bit, decomposer);
      error = CubDebug(error);
      if (cudaSuccess != error)
      {
        break;
      }

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // exclusive sums to determine starts
      const int SCAN_BLOCK_THREADS = policy.BlockThreads(policy.ExclusiveSum());

// log exclusive_sum_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking exclusive_sum_kernel<<<%d, %d, 0, %lld>>>(), bit_grain %d\n",
              num_passes,
              SCAN_BLOCK_THREADS,
              reinterpret_cast<long long>(stream),
              policy.RadixBits(policy.ExclusiveSum()));
#endif

      error = launcher_factory(num_passes, SCAN_BLOCK_THREADS, 0, stream)
                .doit(kernel_source.RadixSortExclusiveSumKernel(), d_bins);
      error = CubDebug(error);
      if (cudaSuccess != error)
      {
        break;
      }

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // use the other buffer if no overwrite is allowed
      KeyT* d_keys_tmp     = d_keys.Alternate();
      ValueT* d_values_tmp = d_values.Alternate();
      if (!is_overwrite_okay && num_passes % 2 == 0)
      {
        d_keys.d_buffers[1]   = d_keys_tmp2;
        d_values.d_buffers[1] = d_values_tmp2;
      }

      for (int current_bit = begin_bit, pass = 0; current_bit < end_bit; current_bit += RADIX_BITS, ++pass)
      {
        int num_bits = _CUDA_VSTD::min(end_bit - current_bit, RADIX_BITS);
        for (OffsetT portion = 0; portion < num_portions; ++portion)
        {
          PortionOffsetT portion_num_items = static_cast<PortionOffsetT>(
            _CUDA_VSTD::min(num_items - portion * PORTION_SIZE, static_cast<OffsetT>(PORTION_SIZE)));

          PortionOffsetT num_blocks = ::cuda::ceil_div(portion_num_items, ONESWEEP_TILE_ITEMS);

          error = CubDebug(cudaMemsetAsync(d_lookback, 0, num_blocks * RADIX_DIGITS * sizeof(AtomicOffsetT), stream));
          if (cudaSuccess != error)
          {
            break;
          }

// log onesweep_kernel configuration
#ifdef CUB_DEBUG_LOG
          _CubLog("Invoking onesweep_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, "
                  "current bit %d, bit_grain %d, portion %d/%d\n",
                  num_blocks,
                  ONESWEEP_BLOCK_THREADS,
                  reinterpret_cast<long long>(stream),
                  policy.Onesweep().ItemsPerThread(),
                  current_bit,
                  num_bits,
                  static_cast<int>(portion),
                  static_cast<int>(num_portions));
#endif

          auto onesweep_kernel = kernel_source.RadixSortOnesweepKernel();

          error =
            launcher_factory(num_blocks, ONESWEEP_BLOCK_THREADS, 0, stream)
              .doit(onesweep_kernel,
                    d_lookback,
                    d_ctrs + portion * num_passes + pass,
                    portion < num_portions - 1 ? d_bins + ((portion + 1) * num_passes + pass) * RADIX_DIGITS : nullptr,
                    d_bins + (portion * num_passes + pass) * RADIX_DIGITS,
                    d_keys.Alternate(),
                    d_keys.Current() + portion * PORTION_SIZE,
                    d_values.Alternate(),
                    d_values.Current() + portion * PORTION_SIZE,
                    portion_num_items,
                    current_bit,
                    num_bits,
                    decomposer);
          error = CubDebug(error);
          if (cudaSuccess != error)
          {
            break;
          }

          error = CubDebug(detail::DebugSyncStream(stream));
          if (cudaSuccess != error)
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
          d_keys   = num_passes % 2 == 0 ? DoubleBuffer<KeyT>(d_keys_tmp, d_keys_tmp2)
                                         : DoubleBuffer<KeyT>(d_keys_tmp2, d_keys_tmp);
          d_values = num_passes % 2 == 0 ? DoubleBuffer<ValueT>(d_values_tmp, d_values_tmp2)
                                         : DoubleBuffer<ValueT>(d_values_tmp2, d_values_tmp);
        }
        d_keys.selector ^= 1;
        d_values.selector ^= 1;
      }
    } while (0);

    return error;
  }

  /**
   * @brief Invocation (run multiple digit passes)
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam UpsweepKernelT
   *   Function type of cub::DeviceRadixSortUpsweepKernel
   *
   * @tparam ScanKernelT
   *   Function type of cub::SpineScanKernel
   *
   * @tparam DownsweepKernelT
   *   Function type of cub::DeviceRadixSortDownsweepKernel
   *
   * @param[in] upsweep_kernel
   *   Kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
   *
   * @param[in] alt_upsweep_kernel
   *   Alternate kernel function pointer to parameterization of cub::DeviceRadixSortUpsweepKernel
   *
   * @param[in] scan_kernel
   *   Kernel function pointer to parameterization of cub::SpineScanKernel
   *
   * @param[in] downsweep_kernel
   *   Kernel function pointer to parameterization of cub::DeviceRadixSortDownsweepKernel
   *
   * @param[in] alt_downsweep_kernel
   *   Alternate kernel function pointer to parameterization of
   *   cub::DeviceRadixSortDownsweepKernel
   */
  template <typename ActivePolicyT, typename UpsweepKernelT, typename ScanKernelT, typename DownsweepKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InvokePasses(
    UpsweepKernelT upsweep_kernel,
    UpsweepKernelT alt_upsweep_kernel,
    ScanKernelT scan_kernel,
    DownsweepKernelT downsweep_kernel,
    DownsweepKernelT alt_downsweep_kernel,
    ActivePolicyT policy = {})
  {
    cudaError error = cudaSuccess;
    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get SM count
      int sm_count;
      error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Init regular and alternate-digit kernel configurations
      PassConfig<UpsweepKernelT, ScanKernelT, DownsweepKernelT> pass_config, alt_pass_config;
      error = pass_config.InitPassConfig(
        upsweep_kernel,
        scan_kernel,
        downsweep_kernel,
        ptx_version,
        sm_count,
        num_items,
        policy,
        policy.Upsweep(),
        policy.Scan(),
        policy.Downsweep(),
        launcher_factory);
      if (error)
      {
        break;
      }

      error = alt_pass_config.InitPassConfig(
        alt_upsweep_kernel,
        scan_kernel,
        alt_downsweep_kernel,
        ptx_version,
        sm_count,
        num_items,
        policy,
        policy.AltUpsweep(),
        policy.Scan(),
        policy.AltDownsweep(),
        launcher_factory);
      if (error)
      {
        break;
      }

      // Get maximum spine length
      int max_grid_size = _CUDA_VSTD::max(pass_config.max_downsweep_grid_size, alt_pass_config.max_downsweep_grid_size);
      int spine_length  = (max_grid_size * pass_config.radix_digits) + pass_config.scan_config.tile_size;

      // Temporary storage allocation requirements
      void* allocations[3]       = {};
      size_t allocation_sizes[3] = {
        // bytes needed for privatized block digit histograms
        spine_length * sizeof(OffsetT),

        // bytes needed for 3rd keys buffer
        (is_overwrite_okay) ? 0 : num_items * kernel_source.KeySize(),

        // bytes needed for 3rd values buffer
        (is_overwrite_okay || (KEYS_ONLY)) ? 0 : num_items * kernel_source.ValueSize(),
      };

      // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
      error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      // Return if the caller is simply requesting the size of the storage allocation
      if (d_temp_storage == nullptr)
      {
        return cudaSuccess;
      }

      // Pass planning.  Run passes of the alternate digit-size configuration until we have an even multiple of our
      // preferred digit size
      int num_bits           = end_bit - begin_bit;
      int num_passes         = ::cuda::ceil_div(num_bits, pass_config.radix_bits);
      bool is_num_passes_odd = num_passes & 1;
      int max_alt_passes     = (num_passes * pass_config.radix_bits) - num_bits;
      int alt_end_bit        = _CUDA_VSTD::min(end_bit, begin_bit + (max_alt_passes * alt_pass_config.radix_bits));

      // Alias the temporary storage allocations
      OffsetT* d_spine = static_cast<OffsetT*>(allocations[0]);

      DoubleBuffer<KeyT> d_keys_remaining_passes(
        (is_overwrite_okay || is_num_passes_odd) ? d_keys.Alternate() : static_cast<KeyT*>(allocations[1]),
        (is_overwrite_okay)   ? d_keys.Current()
        : (is_num_passes_odd) ? static_cast<KeyT*>(allocations[1])
                              : d_keys.Alternate());

      DoubleBuffer<ValueT> d_values_remaining_passes(
        (is_overwrite_okay || is_num_passes_odd) ? d_values.Alternate() : static_cast<ValueT*>(allocations[2]),
        (is_overwrite_okay)   ? d_values.Current()
        : (is_num_passes_odd) ? static_cast<ValueT*>(allocations[2])
                              : d_values.Alternate());

      // Run first pass, consuming from the input's current buffers
      int current_bit = begin_bit;
      error           = CubDebug(InvokePass(
        d_keys.Current(),
        d_keys_remaining_passes.Current(),
        d_values.Current(),
        d_values_remaining_passes.Current(),
        d_spine,
        spine_length,
        current_bit,
        (current_bit < alt_end_bit) ? alt_pass_config : pass_config));
      if (cudaSuccess != error)
      {
        break;
      }

      // Run remaining passes
      while (current_bit < end_bit)
      {
        error = CubDebug(InvokePass(
          d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
          d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
          d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
          d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
          d_spine,
          spine_length,
          current_bit,
          (current_bit < alt_end_bit) ? alt_pass_config : pass_config));

        if (cudaSuccess != error)
        {
          break;
        }

        // Invert selectors
        d_keys_remaining_passes.selector ^= 1;
        d_values_remaining_passes.selector ^= 1;
      }

      // Update selector
      if (!is_overwrite_okay)
      {
        num_passes = 1; // Sorted data always ends up in the other vector
      }

      d_keys.selector   = (d_keys.selector + num_passes) & 1;
      d_values.selector = (d_values.selector + num_passes) & 1;
    } while (0);

    return error;
  }

  //------------------------------------------------------------------------------
  // Chained policy invocation
  //------------------------------------------------------------------------------

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokeCopy()
  {
    // is_overwrite_okay == false here
    // Return the number of temporary bytes if requested
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

// Copy keys
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking async copy of %lld keys on stream %lld\n", (long long) num_items, (long long) stream);
#endif
    cudaError_t error = cudaSuccess;

    error = CubDebug(cudaMemcpyAsync(
      d_keys.Alternate(), d_keys.Current(), num_items * kernel_source.KeySize(), cudaMemcpyDefault, stream));
    if (cudaSuccess != error)
    {
      return error;
    }

    error = CubDebug(detail::DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }
    d_keys.selector ^= 1;

    // Copy values if necessary
    if (!KEYS_ONLY)
    {
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking async copy of %lld values on stream %lld\n", (long long) num_items, (long long) stream);
#endif
      error = CubDebug(cudaMemcpyAsync(
        d_values.Alternate(), d_values.Current(), num_items * kernel_source.ValueSize(), cudaMemcpyDefault, stream));
      if (cudaSuccess != error)
      {
        return error;
      }

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }
    d_values.selector ^= 1;

    return error;
  }

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT policy = {})
  {
    auto wrapped_policy = detail::radix::MakeRadixSortPolicyWrapper(policy);

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
      bool has_uva      = false;
      cudaError_t error = detail::HasUVA(has_uva);
      if (error != cudaSuccess)
      {
        return error;
      }
      if (has_uva)
      {
        return InvokeCopy();
      }
    }

    // Force kernel code-generation in all compiler passes
    if (num_items <= static_cast<OffsetT>(
          wrapped_policy.SingleTile().BlockThreads() * wrapped_policy.SingleTile().ItemsPerThread()))
    {
      // Small, single tile size
      return InvokeSingleTile(kernel_source.RadixSortSingleTileKernel(), wrapped_policy);
    }

    if CUB_DETAIL_CONSTEXPR_ISH (wrapped_policy.IsOnesweep())
    {
      return InvokeOnesweep(wrapped_policy);
    }
    else
    {
      return InvokePasses(
        kernel_source.RadixSortUpsweepKernel(),
        kernel_source.RadixSortAltUpsweepKernel(),
        kernel_source.DeviceRadixSortScanBinsKernel(),
        kernel_source.RadixSortDownsweepKernel(),
        kernel_source.RadixSortAltDownsweepKernel(),
        wrapped_policy);
    }
  }

  //------------------------------------------------------------------------------
  // Dispatch entrypoints
  //------------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When nullptr, the required
   *   allocation size is written to `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_keys
   *   Double-buffer whose current buffer contains the unsorted input keys and,
   *   upon return, is updated to point to the sorted output keys
   *
   * @param[in,out] d_values
   *   Double-buffer whose current buffer contains the unsorted input values and,
   *   upon return, is updated to point to the sorted output values
   *
   * @param[in] num_items
   *   Number of items to sort
   *
   * @param[in] begin_bit
   *   The beginning (least-significant) bit index needed for key comparison
   *
   * @param[in] end_bit
   *   The past-the-end (most-significant) bit index needed for key comparison
   *
   * @param[in] is_overwrite_okay
   *   Whether is okay to overwrite source buffers
   *
   * @param[in] stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    OffsetT num_items,
    int begin_bit,
    int end_bit,
    bool is_overwrite_okay,
    cudaStream_t stream,
    DecomposerT decomposer                 = {},
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    cudaError_t error;
    do
    {
      // Get PTX version
      int ptx_version = 0;

      error = CubDebug(launcher_factory.PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DispatchRadixSort dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_keys,
        d_values,
        num_items,
        begin_bit,
        end_bit,
        is_overwrite_okay,
        stream,
        ptx_version,
        decomposer,
        kernel_source,
        launcher_factory);

      // Dispatch to chained policy
      error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

/******************************************************************************
 * Segmented dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for segmented device-wide
 * radix sort
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
 * @tparam SegmentSizeT
 *   Integer type to index items within a segment
 */
template <SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename SegmentSizeT,
          typename PolicyHub    = detail::radix::policy_hub<KeyT, ValueT, SegmentSizeT>,
          typename DecomposerT  = detail::identity_decomposer_t,
          typename KernelSource = detail::radix_sort::DeviceSegmentedRadixSortKernelSource<
            typename PolicyHub::MaxPolicy,
            Order,
            KeyT,
            ValueT,
            BeginOffsetIteratorT,
            EndOffsetIteratorT,
            SegmentSizeT,
            DecomposerT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchSegmentedRadixSort
{
  //------------------------------------------------------------------------------
  // Constants
  //------------------------------------------------------------------------------

  // Whether this is a keys-only (or key-value) sort
  static constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  //------------------------------------------------------------------------------
  // Parameter members
  //------------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage.  When nullptr, the required allocation size
  /// is written to `temp_storage_bytes` and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Double-buffer whose current buffer contains the unsorted input keys and, upon return, is
  /// updated to point to the sorted output keys
  DoubleBuffer<KeyT>& d_keys;

  /// Double-buffer whose current buffer contains the unsorted input values and, upon return, is
  /// updated to point to the sorted output values
  DoubleBuffer<ValueT>& d_values;

  /// Number of items to sort
  ::cuda::std::int64_t num_items;

  /// The number of segments that comprise the sorting data
  ::cuda::std::int64_t num_segments;

  /// Random-access input iterator to the sequence of beginning offsets of length `num_segments`,
  /// such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup>
  /// data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
  BeginOffsetIteratorT d_begin_offsets;

  /// Random-access input iterator to the sequence of ending offsets of length `num_segments`,
  /// such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup>
  /// data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>. If <tt>d_end_offsets[i]-1</tt>
  /// <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
  EndOffsetIteratorT d_end_offsets;

  /// The beginning (least-significant) bit index needed for key comparison
  int begin_bit;

  /// The past-the-end (most-significant) bit index needed for key comparison
  int end_bit;

  /// CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
  cudaStream_t stream;

  /// PTX version
  int ptx_version;

  /// Whether is okay to overwrite source buffers
  bool is_overwrite_okay;

  DecomposerT decomposer;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  //------------------------------------------------------------------------------
  // Constructors
  //------------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchSegmentedRadixSort(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int begin_bit,
    int end_bit,
    bool is_overwrite_okay,
    cudaStream_t stream,
    int ptx_version,
    DecomposerT decomposer                 = {},
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys(d_keys)
      , d_values(d_values)
      , num_items(num_items)
      , num_segments(num_segments)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , begin_bit(begin_bit)
      , end_bit(end_bit)
      , stream(stream)
      , ptx_version(ptx_version)
      , is_overwrite_okay(is_overwrite_okay)
      , decomposer(decomposer)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  //------------------------------------------------------------------------------
  // Multi-segment invocation
  //------------------------------------------------------------------------------

  /// Invoke a three-kernel sorting pass at the current bit.
  template <typename PassConfigT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokePass(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    int& current_bit,
    PassConfigT& pass_config)
  {
    cudaError error = cudaSuccess;

    // The number of bits to process in this pass
    int pass_bits = _CUDA_VSTD::min(pass_config.radix_bits, (end_bit - current_bit));

    // The offset type (used to specialize the kernel template), large enough to index any segment within a single
    // invocation
    using per_invocation_segment_offset_t = ::cuda::std::int32_t;

    // The upper bound of segments that a single kernel invocation will process
    constexpr auto max_num_segments_per_invocation =
      static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<per_invocation_segment_offset_t>::max());

    // Number of radix sort invocations until all segments have been processed
    const auto num_invocations = ::cuda::ceil_div(num_segments, max_num_segments_per_invocation);

    BeginOffsetIteratorT begin_offsets_current_it = d_begin_offsets;
    EndOffsetIteratorT end_offsets_current_it     = d_end_offsets;

    // Iterate over chunks of segments
    for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
    {
      const auto current_segment_offset = invocation_index * max_num_segments_per_invocation;
      const auto num_current_segments =
        ::cuda::std::min(max_num_segments_per_invocation, num_segments - current_segment_offset);

// Log kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog(
        "Invoking segmented_kernels<<<%lld, %lld, 0, %lld>>>(), "
        "%lld items per thread, %lld SM occupancy, "
        "current segment offset %lld, current bit %d, bit_grain %d\n",
        (long long) num_current_segments,
        (long long) pass_config.segmented_config.block_threads,
        (long long) stream,
        (long long) pass_config.segmented_config.items_per_thread,
        (long long) pass_config.segmented_config.sm_occupancy,
        (long long) current_segment_offset,
        current_bit,
        pass_bits);
#endif

      launcher_factory(
        static_cast<unsigned int>(num_current_segments), pass_config.segmented_config.block_threads, 0, stream)
        .doit(pass_config.segmented_kernel,
              d_keys_in,
              d_keys_out,
              d_values_in,
              d_values_out,
              begin_offsets_current_it,
              end_offsets_current_it,
              current_bit,
              pass_bits,
              decomposer);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        return error;
      }

      if (invocation_index + 1 < num_invocations)
      {
        begin_offsets_current_it += num_current_segments;
        end_offsets_current_it += num_current_segments;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    // Update current bit once all segments have been processed for the current pass
    current_bit += pass_bits;

    return error;
  }

  /// PassConfig data structure
  template <typename SegmentedKernelT>
  struct PassConfig
  {
    SegmentedKernelT segmented_kernel;
    detail::KernelConfig segmented_config;
    int radix_bits;
    int radix_digits;

    /// Initialize pass configuration
    template <typename SegmentedPolicyT>
    CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t InitPassConfig(
      SegmentedKernelT segmented_kernel,
      int radix_bits,
      SegmentedPolicyT policy                = {},
      KernelLauncherFactory launcher_factory = {})
    {
      this->segmented_kernel = segmented_kernel;
      this->radix_bits       = radix_bits;
      this->radix_digits     = 1 << radix_bits;

      return CubDebug(segmented_config.Init(segmented_kernel, policy, launcher_factory));
    }
  };

  /**
   * @brief Invocation (run multiple digit passes)
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam SegmentedKernelT
   *   Function type of cub::DeviceSegmentedRadixSortKernel
   *
   * @param[in] segmented_kernel
   *   Kernel function pointer to parameterization of cub::DeviceSegmentedRadixSortKernel
   *
   * @param[in] alt_segmented_kernel
   *   Alternate kernel function pointer to parameterization of
   *   cub::DeviceSegmentedRadixSortKernel
   */
  template <typename ActivePolicyT, typename SegmentedKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokePasses(SegmentedKernelT segmented_kernel, SegmentedKernelT alt_segmented_kernel, ActivePolicyT policy = {})
  {
    cudaError error = cudaSuccess;
    do
    {
      // Init regular and alternate kernel configurations
      PassConfig<SegmentedKernelT> pass_config, alt_pass_config;
      if ((error = pass_config.InitPassConfig(
             segmented_kernel, policy.RadixBits(policy.Segmented()), policy.Segmented(), launcher_factory)))
      {
        break;
      }
      if ((error = alt_pass_config.InitPassConfig(
             alt_segmented_kernel, policy.RadixBits(policy.AltSegmented()), policy.AltSegmented(), launcher_factory)))
      {
        break;
      }

      // Temporary storage allocation requirements
      void* allocations[2]       = {};
      size_t allocation_sizes[2] = {
        // bytes needed for 3rd keys buffer
        (is_overwrite_okay) ? 0 : num_items * kernel_source.KeySize(),

        // bytes needed for 3rd values buffer
        (is_overwrite_okay || (KEYS_ONLY)) ? 0 : num_items * sizeof(ValueT),
      };

      // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
      error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      // Return if the caller is simply requesting the size of the storage allocation
      if (d_temp_storage == nullptr)
      {
        if (temp_storage_bytes == 0)
        {
          temp_storage_bytes = 1;
        }
        return cudaSuccess;
      }

      // Pass planning.  Run passes of the alternate digit-size configuration until we have an even multiple of our
      // preferred digit size
      int radix_bits         = policy.RadixBits(policy.Segmented());
      int alt_radix_bits     = policy.RadixBits(policy.AltSegmented());
      int num_bits           = end_bit - begin_bit;
      int num_passes         = _CUDA_VSTD::max(::cuda::ceil_div(num_bits, radix_bits), 1); // num_bits may be zero
      bool is_num_passes_odd = num_passes & 1;
      int max_alt_passes     = (num_passes * radix_bits) - num_bits;
      int alt_end_bit        = _CUDA_VSTD::min(end_bit, begin_bit + (max_alt_passes * alt_radix_bits));

      DoubleBuffer<KeyT> d_keys_remaining_passes(
        (is_overwrite_okay || is_num_passes_odd) ? d_keys.Alternate() : static_cast<KeyT*>(allocations[0]),
        (is_overwrite_okay)   ? d_keys.Current()
        : (is_num_passes_odd) ? static_cast<KeyT*>(allocations[0])
                              : d_keys.Alternate());

      DoubleBuffer<ValueT> d_values_remaining_passes(
        (is_overwrite_okay || is_num_passes_odd) ? d_values.Alternate() : static_cast<ValueT*>(allocations[1]),
        (is_overwrite_okay)   ? d_values.Current()
        : (is_num_passes_odd) ? static_cast<ValueT*>(allocations[1])
                              : d_values.Alternate());

      // Run first pass, consuming from the input's current buffers
      int current_bit = begin_bit;

      error = CubDebug(InvokePass(
        d_keys.Current(),
        d_keys_remaining_passes.Current(),
        d_values.Current(),
        d_values_remaining_passes.Current(),
        current_bit,
        (current_bit < alt_end_bit) ? alt_pass_config : pass_config));
      if (cudaSuccess != error)
      {
        break;
      }

      // Run remaining passes
      while (current_bit < end_bit)
      {
        error = CubDebug(InvokePass(
          d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
          d_keys_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
          d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector],
          d_values_remaining_passes.d_buffers[d_keys_remaining_passes.selector ^ 1],
          current_bit,
          (current_bit < alt_end_bit) ? alt_pass_config : pass_config));
        if (cudaSuccess != error)
        {
          break;
        }

        // Invert selectors and update current bit
        d_keys_remaining_passes.selector ^= 1;
        d_values_remaining_passes.selector ^= 1;
      }

      // Update selector
      if (!is_overwrite_okay)
      {
        num_passes = 1; // Sorted data always ends up in the other vector
      }

      d_keys.selector   = (d_keys.selector + num_passes) & 1;
      d_values.selector = (d_values.selector + num_passes) & 1;
    } while (0);

    return error;
  }

  //------------------------------------------------------------------------------
  // Chained policy invocation
  //------------------------------------------------------------------------------

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT policy = {})
  {
    // Return if empty problem, or if no bits to sort and double-buffering is used
    if (num_items == 0 || num_segments == 0 || (begin_bit == end_bit && is_overwrite_okay))
    {
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
      }
      return cudaSuccess;
    }

    // Force kernel code-generation in all compiler passes
    return InvokePasses(kernel_source.SegmentedRadixSortKernel(),
                        kernel_source.AltSegmentedRadixSortKernel(),
                        detail::radix::MakeRadixSortPolicyWrapper(policy));
  }

  //------------------------------------------------------------------------------
  // Dispatch entrypoints
  //------------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage.  When nullptr, the required allocation size
   *   is written to `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in,out] d_keys
   *   Double-buffer whose current buffer contains the unsorted input keys and, upon return, is
   * updated to point to the sorted output keys
   *
   * @param[in,out] d_values
   *   Double-buffer whose current buffer contains the unsorted input values and, upon return, is
   *   updated to point to the sorted output values
   *
   * @param[in] num_items
   *   Number of items to sort
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of length
   *   `num_segments`, such that <tt>d_begin_offsets[i]</tt> is the first element of the
   *   <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length `num_segments`,
   *   such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup>
   *   data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.
   *   If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>,
   *   the <em>i</em><sup>th</sup> is considered empty.
   *
   * @param[in] begin_bit
   *   The beginning (least-significant) bit index needed for key comparison
   *
   * @param[in] end_bit
   *   The past-the-end (most-significant) bit index needed for key comparison
   *
   * @param[in] is_overwrite_okay
   *   Whether is okay to overwrite source buffers
   *
   * @param[in] stream
   *   CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int begin_bit,
    int end_bit,
    bool is_overwrite_okay,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    cudaError_t error;
    do
    {
      // Get PTX version
      int ptx_version = 0;

      error = CubDebug(launcher_factory.PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DispatchSegmentedRadixSort dispatch(
        d_temp_storage,
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
        stream,
        ptx_version,
        {},
        kernel_source,
        launcher_factory);

      // Dispatch to chained policy
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

_CCCL_DIAG_POP
