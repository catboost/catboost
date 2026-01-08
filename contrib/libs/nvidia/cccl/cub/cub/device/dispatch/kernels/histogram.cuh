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

#include <cub/agent/agent_histogram.cuh>
#include <cub/grid/grid_queue.cuh>

CUB_NAMESPACE_BEGIN
namespace detail::histogram
{

template <typename LevelT, typename OffsetT, typename SampleT>
struct Transforms
{
  enum
  {
    // Maximum number of bins per channel for which we will use a privatized smem strategy
    MAX_PRIVATIZED_SMEM_BINS = 256
  };

  //---------------------------------------------------------------------
  // Transform functors for converting samples to bin-ids
  //---------------------------------------------------------------------

  // Searches for bin given a list of bin-boundary levels
  template <typename LevelIteratorT>
  struct SearchTransform
  {
    LevelIteratorT d_levels; // Pointer to levels array
    int num_output_levels; // Number of levels in array

    /**
     * @brief Initializer
     *
     * @param d_levels_ Pointer to levels array
     * @param num_output_levels_ Number of levels in array
     */
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void Init(LevelIteratorT d_levels_, int num_output_levels_)
    {
      this->d_levels          = d_levels_;
      this->num_output_levels = num_output_levels_;
    }

    // Method for converting samples to bin-ids
    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(_SampleT sample, int& bin, bool valid)
    {
      /// Level iterator wrapper type
      // Wrap the native input pointer with CacheModifiedInputIterator
      // or Directly use the supplied input iterator type
      using WrappedLevelIteratorT =
        ::cuda::std::_If<::cuda::std::is_pointer_v<LevelIteratorT>,
                         CacheModifiedInputIterator<LOAD_MODIFIER, LevelT, OffsetT>,
                         LevelIteratorT>;

      WrappedLevelIteratorT wrapped_levels(d_levels);

      int num_bins = num_output_levels - 1;
      if (valid)
      {
        bin = UpperBound(wrapped_levels, num_output_levels, (LevelT) sample) - 1;
        if (bin >= num_bins)
        {
          bin = -1;
        }
      }
    }
  };

  // Scales samples to evenly-spaced bins
  struct ScaleTransform
  {
  private:
    using CommonT = ::cuda::std::common_type_t<LevelT, SampleT>;
    static_assert(::cuda::std::is_convertible_v<CommonT, int>,
                  "The common type of `LevelT` and `SampleT` must be "
                  "convertible to `int`.");
    static_assert(::cuda::std::is_trivially_copyable_v<CommonT>,
                  "The common type of `LevelT` and `SampleT` must be "
                  "trivially copyable.");

// We currently don't have a way to get sizeof(SampleT) from c.parallel, so we just default to large sizes
#ifdef CCCL_C_EXPERIMENTAL
#  if _CCCL_HAS_INT128()
    using IntArithmeticT = __uint128_t;
#  else
    using IntArithmeticT = uint64_t;
#  endif
#else
    // An arithmetic type that's used for bin computation of integral types, guaranteed to not
    // overflow for (max_level - min_level) * scale.fraction.bins. Since we drop invalid samples
    // of less than min_level, (sample - min_level) is guaranteed to be non-negative. We use the
    // rule: 2^l * 2^r = 2^(l + r) to determine a sufficiently large type to hold the
    // multiplication result.
    // If CommonT used to be a 128-bit wide integral type already, we use CommonT's arithmetic
    using IntArithmeticT = ::cuda::std::_If< //
      sizeof(SampleT) + sizeof(CommonT) <= sizeof(uint32_t), //
      uint32_t, //
#  if _CCCL_HAS_INT128()
      ::cuda::std::_If< //
        (::cuda::std::is_same_v<CommonT, __int128_t> || //
         ::cuda::std::is_same_v<CommonT, __uint128_t>), //
        CommonT, //
        uint64_t> //
#  else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
      uint64_t
#  endif // !_CCCL_HAS_INT128()
      >;
#endif // CCCL_C_EXPERIMENTAL

    // Alias template that excludes __[u]int128 from the integral types
    template <typename T>
    using is_integral_excl_int128 =
#if _CCCL_HAS_INT128()
      ::cuda::std::_If<::cuda::std::is_same_v<T, __int128_t>&& ::cuda::std::is_same_v<T, __uint128_t>,
                       ::cuda::std::false_type,
                       ::cuda::std::is_integral<T>>;
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
      ::cuda::std::is_integral<T>;
#endif // !_CCCL_HAS_INT128()

    union ScaleT
    {
      // Used when CommonT is not floating-point to avoid intermediate
      // rounding errors (see NVIDIA/cub#489).
      struct FractionT
      {
        CommonT bins;
        CommonT range;
      } fraction;

      // Used when CommonT is floating-point as an optimization.
      CommonT reciprocal;
    };

    CommonT m_max; // Max sample level (exclusive)
    CommonT m_min; // Min sample level (inclusive)
    ScaleT m_scale; // Bin scaling

    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT
    ComputeScale(int num_levels, T max_level, T min_level, ::cuda::std::true_type /* is_fp */)
    {
      ScaleT result;
      result.reciprocal = static_cast<T>(static_cast<T>(num_levels - 1) / static_cast<T>(max_level - min_level));
      return result;
    }

    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT
    ComputeScale(int num_levels, T max_level, T min_level, ::cuda::std::false_type /* is_fp */)
    {
      ScaleT result;
      result.fraction.bins  = static_cast<T>(num_levels - 1);
      result.fraction.range = static_cast<T>(max_level - min_level);
      return result;
    }

    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT ComputeScale(int num_levels, T max_level, T min_level)
    {
      return this->ComputeScale(num_levels, max_level, min_level, ::cuda::std::is_floating_point<T>{});
    }

#if _CCCL_HAS_NVFP16()
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScaleT ComputeScale(int num_levels, __half max_level, __half min_level)
    {
      ScaleT result;
      NV_IF_TARGET(NV_PROVIDES_SM_53,
                   (result.reciprocal = __hdiv(__float2half(num_levels - 1), __hsub(max_level, min_level));),
                   (result.reciprocal = __float2half(
                      static_cast<float>(num_levels - 1) / (__half2float(max_level) - __half2float(min_level)));))
      return result;
    }
#endif // _CCCL_HAS_NVFP16()

    // All types but __half:
    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int SampleIsValid(T sample, T max_level, T min_level)
    {
      return sample >= min_level && sample < max_level;
    }

#if _CCCL_HAS_NVFP16()
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int SampleIsValid(__half sample, __half max_level, __half min_level)
    {
      NV_IF_TARGET(
        NV_PROVIDES_SM_53,
        (return __hge(sample, min_level) && __hlt(sample, max_level);),
        (return __half2float(sample) >= __half2float(min_level) && __half2float(sample) < __half2float(max_level);));
    }
#endif // _CCCL_HAS_NVFP16()

    /**
     * @brief Bin computation for floating point (and extended floating point) types
     */
    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int
    ComputeBin(T sample, T min_level, ScaleT scale, ::cuda::std::true_type /* is_fp */)
    {
      return static_cast<int>((sample - min_level) * scale.reciprocal);
    }

    /**
     * @brief Bin computation for custom types and __[u]int128
     */
    template <typename T>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int
    ComputeBin(T sample, T min_level, ScaleT scale, ::cuda::std::false_type /* is_fp */)
    {
      return static_cast<int>(((sample - min_level) * scale.fraction.bins) / scale.fraction.range);
    }

    /**
     * @brief Bin computation for integral types of up to 64-bit types
     */
    template <typename T, ::cuda::std::enable_if_t<is_integral_excl_int128<T>::value, int> = 0>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int ComputeBin(T sample, T min_level, ScaleT scale)
    {
      return static_cast<int>(
        (static_cast<IntArithmeticT>(sample - min_level) * static_cast<IntArithmeticT>(scale.fraction.bins))
        / static_cast<IntArithmeticT>(scale.fraction.range));
    }

    template <typename T, ::cuda::std::enable_if_t<!is_integral_excl_int128<T>::value, int> = 0>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int ComputeBin(T sample, T min_level, ScaleT scale)
    {
      return this->ComputeBin(sample, min_level, scale, ::cuda::std::is_floating_point<T>{});
    }

#if _CCCL_HAS_NVFP16()
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int ComputeBin(__half sample, __half min_level, ScaleT scale)
    {
      NV_IF_TARGET(
        NV_PROVIDES_SM_53,
        (return static_cast<int>(__hmul(__hsub(sample, min_level), scale.reciprocal));),
        (return static_cast<int>((__half2float(sample) - __half2float(min_level)) * __half2float(scale.reciprocal));));
    }
#endif // _CCCL_HAS_NVFP16()

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool
    MayOverflow(CommonT /* num_bins */, ::cuda::std::false_type /* is_integral */)
    {
      return false;
    }

    /**
     * @brief Returns true if the bin computation for a given combination of range `(max_level -
     * min_level)` and number of bins may overflow.
     */
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool MayOverflow(CommonT num_bins, ::cuda::std::true_type /* is_integral */)
    {
      return static_cast<IntArithmeticT>(m_max - m_min)
           > (::cuda::std::numeric_limits<IntArithmeticT>::max() / static_cast<IntArithmeticT>(num_bins));
    }

  public:
    /**
     * @brief Initializes the ScaleTransform for the given parameters
     * @return cudaErrorInvalidValue if the ScaleTransform for the given values may overflow,
     * cudaSuccess otherwise
     */
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE cudaError_t Init(int num_levels, LevelT max_level, LevelT min_level)
    {
      m_max = static_cast<CommonT>(max_level);
      m_min = static_cast<CommonT>(min_level);

      // Check whether accurate bin computation for an integral sample type may overflow
      if (MayOverflow(static_cast<CommonT>(num_levels - 1), ::cuda::std::is_integral<CommonT>{}))
      {
        return cudaErrorInvalidValue;
      }

      m_scale = this->ComputeScale(num_levels, m_max, m_min);
      return cudaSuccess;
    }

    // Method for converting samples to bin-ids
    template <CacheLoadModifier LOAD_MODIFIER>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(SampleT sample, int& bin, bool valid)
    {
      const CommonT common_sample = static_cast<CommonT>(sample);

      if (valid && this->SampleIsValid(common_sample, m_max, m_min))
      {
        bin = this->ComputeBin(common_sample, m_min, m_scale);
      }
    }
  };

  // Pass-through bin transform operator
  struct PassThruTransform
  {
// GCC 14 rightfully warns that when a value-initialized array of this struct is copied using memcpy, uninitialized
// bytes may be accessed. To avoid this, we add a dummy member, so value initialization actually initializes the memory.
#if _CCCL_COMPILER(GCC, >=, 13)
    char dummy;
#endif

    // Method for converting samples to bin-ids
    template <CacheLoadModifier LOAD_MODIFIER, typename _SampleT>
    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void BinSelect(_SampleT sample, int& bin, bool valid)
    {
      if (valid)
      {
        bin = (int) sample;
      }
    }
  };
};

/******************************************************************************
 * Histogram kernel entry points
 *****************************************************************************/

/**
 * Histogram initialization kernel entry point
 *
 * @tparam NUM_ACTIVE_CHANNELS
 *   Number of channels actively being histogrammed
 *
 * @tparam CounterT
 *   Integer type for counting sample occurrences per histogram bin
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param num_output_bins_wrapper
 *   Number of output histogram bins per channel
 *
 * @param d_output_histograms_wrapper
 *   Histogram counter data having logical dimensions
 *   `CounterT[NUM_ACTIVE_CHANNELS][num_bins.array[CHANNEL]]`
 *
 * @param tile_queue
 *   Drain queue descriptor for dynamically mapping tile data onto thread blocks
 */
template <int NUM_ACTIVE_CHANNELS, typename CounterT, typename OffsetT>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceHistogramInitKernel(
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_wrapper,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms_wrapper,
  GridQueue<int> tile_queue)
{
  if ((threadIdx.x == 0) && (blockIdx.x == 0))
  {
    tile_queue.ResetDrain();
  }

  int output_bin = (blockIdx.x * blockDim.x) + threadIdx.x;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
  {
    if (output_bin < num_output_bins_wrapper[CHANNEL])
    {
      d_output_histograms_wrapper[CHANNEL][output_bin] = 0;
    }
  }
}

/**
 * Histogram privatized sweep kernel entry point (multi-block).
 * Computes privatized histograms, one per thread block.
 *
 *
 * @tparam AgentHistogramPolicyT
 *   Parameterized AgentHistogramPolicy tuning policy type
 *
 * @tparam PRIVATIZED_SMEM_BINS
 *   Maximum number of histogram bins per channel (e.g., up to 256)
 *
 * @tparam NUM_CHANNELS
 *   Number of channels interleaved in the input data (may be greater than the number of channels
 *   being actively histogrammed)
 *
 * @tparam NUM_ACTIVE_CHANNELS
 *   Number of channels actively being histogrammed
 *
 * @tparam SampleIteratorT
 *   The input iterator type. @iterator.
 *
 * @tparam CounterT
 *   Integer type for counting sample occurrences per histogram bin
 *
 * @tparam PrivatizedDecodeOpT
 *   The transform operator type for determining privatized counter indices from samples,
 *   one for each channel
 *
 * @tparam OutputDecodeOpT
 *   The transform operator type for determining output bin-ids from privatized counter indices,
 *   one for each channel
 *
 * @tparam OffsetT
 *   integer type for global offsets
 *
 * @param d_samples
 *   Input data to reduce
 *
 * @param num_output_bins_wrapper
 *   The number bins per final output histogram
 *
 * @param num_privatized_bins_wrapper
 *   The number bins per privatized histogram
 *
 * @param d_output_histograms_wrapper
 *   Reference to final output histograms
 *
 * @param d_privatized_histograms_wrapper
 *   Reference to privatized histograms
 *
 * @param output_decode_op_wrapper
 *   The transform operator for determining output bin-ids from privatized counter indices,
 *   one for each channel
 *
 * @param privatized_decode_op_wrapper
 *   The transform operator for determining privatized counter indices from samples,
 *   one for each channel
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
 *   Drain queue descriptor for dynamically mapping tile data onto thread blocks
 */
template <typename ChainedPolicyT,
          int PRIVATIZED_SMEM_BINS,
          int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::AgentHistogramPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceHistogramSweepKernel(
    SampleIteratorT d_samples,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_wrapper,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_privatized_histograms_wrapper,
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op_wrapper,
    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op_wrapper,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    int tiles_per_row,
    GridQueue<int> tile_queue)
{
  // Thread block type for compositing input tiles
  using AgentHistogramPolicyT = typename ChainedPolicyT::ActivePolicy::AgentHistogramPolicyT;
  using AgentHistogramT =
    AgentHistogram<AgentHistogramPolicyT,
                   PRIVATIZED_SMEM_BINS,
                   NUM_CHANNELS,
                   NUM_ACTIVE_CHANNELS,
                   SampleIteratorT,
                   CounterT,
                   PrivatizedDecodeOpT,
                   OutputDecodeOpT,
                   OffsetT>;

  // Shared memory for AgentHistogram
  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  AgentHistogramT agent(
    temp_storage,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op_wrapper.data(),
    privatized_decode_op_wrapper.data());

  // Initialize counters
  agent.InitBinCounters();

  // Consume input tiles
  agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

  // Store output to global (if necessary)
  agent.StoreOutput();
}
} // namespace detail::histogram
CUB_NAMESPACE_END
