/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::DeviceHistogram provides device-wide parallel operations for 
 *       constructing histogram(s) from a sequence of samples data residing 
 *       within device-accessible memory.
 */

#pragma once
#pragma clang system_header


#include <stdio.h>
#include <iterator>
#include <limits>

#include <cub/config.cuh>
#include <cub/device/dispatch/dispatch_histogram.cuh>
#include <cub/util_deprecated.cuh>

CUB_NAMESPACE_BEGIN


/**
 * @brief DeviceHistogram provides device-wide parallel operations for 
 *        constructing histogram(s) from a sequence of samples data residing 
 *        within device-accessible memory. ![](histogram_logo.png)
 * @ingroup SingleModule
 *
 * @par Overview
 * A <a href="http://en.wikipedia.org/wiki/Histogram"><em>histogram</em></a>
 * counts the number of observations that fall into each of the disjoint categories (known as <em>bins</em>).
 *
 * @par Usage Considerations
 * @cdp_class{DeviceHistogram}
 *
 */
struct DeviceHistogram
{
  /******************************************************************//**
   * @name Evenly-segmented bin ranges
   *********************************************************************/
  //@{

  /**
   * @brief Computes an intensity histogram from a sequence of data samples 
   *        using equal-width bins.
   *
   * @par
   * - The number of histogram bins is (`num_levels - 1`)
   * - All bins comprise the same width of sample values: 
   *   `(upper_level - lower_level) / (num_levels - 1)`
   * - The ranges `[d_samples, d_samples + num_samples)` and 
   *   `[d_histogram, d_histogram + num_levels - 1)` shall not overlap 
   *   in any way.
   * - `cuda::std::common_type<LevelT, SampleT>` must be valid, and both LevelT
   *   and SampleT must be valid arithmetic types. The common type must be
   *   convertible to `int` and trivially copyable.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the computation of a six-bin histogram
   * from a sequence of float samples
   *
   * @par
   * @code
   * #include <cub/cub.cuh> // or equivalently <cub/device/device_histogram.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for 
   * // input samples and output histogram
   * int      num_samples;    // e.g., 10
   * float*   d_samples;      // e.g., [2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5]
   * int*     d_histogram;    // e.g., [ -, -, -, -, -, -]
   * int      num_levels;     // e.g., 7       (seven level boundaries for six bins)
   * float    lower_level;    // e.g., 0.0     (lower sample value boundary of lowest bin)
   * float    upper_level;    // e.g., 12.0    (upper sample value boundary of upper bin)
   * ...
   *
   * // Determine temporary device storage requirements
   * void*    d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceHistogram::HistogramEven(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, 
   *   lower_level, upper_level, num_samples);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Compute histograms
   * cub::DeviceHistogram::HistogramEven(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, 
   *   lower_level, upper_level, num_samples);
   *
   * // d_histogram   <-- [1, 5, 0, 3, 0, 0];
   * @endcode
   *
   * @tparam SampleIteratorT          
   *   **[inferred]** Random-access input iterator type for reading input 
   *   samples \iterator
   *
   * @tparam CounterT                 
   *   **[inferred]** Integer type for histogram bin counters
   *
   * @tparam LevelT                   
   *   **[inferred]** Type for specifying boundaries (levels)
   *
   * @tparam OffsetT                  
   *   **[inferred]** Signed integer type for sequence offsets, list lengths, 
   *   pointer differences, etc.  \offset_size1
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no 
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_samples 
   *   The pointer to the input sequence of data samples.
   *
   * @param[out] d_histogram 
   *   The pointer to the histogram counter output array of length 
   *   `num_levels - 1`.
   *
   * @param[in] num_levels 
   *   The number of boundaries (levels) for delineating histogram samples.  
   *   Implies that the number of bins is `num_levels - 1`.
   *
   * @param[in] lower_level 
   *   The lower sample value bound (inclusive) for the lowest histogram bin.
   *
   * @param[in] upper_level 
   *   The upper sample value bound (exclusive) for the highest histogram bin.
   *
   * @param[in] num_samples 
   *   The number of input samples (i.e., the length of `d_samples`)
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  HistogramEven(void *d_temp_storage,
                size_t &temp_storage_bytes,
                SampleIteratorT d_samples,
                CounterT *d_histogram,
                int num_levels,
                LevelT lower_level,
                LevelT upper_level,
                OffsetT num_samples,
                cudaStream_t stream    = 0)
  {
    /// The sample value type of the input iterator
    using SampleT = cub::detail::value_t<SampleIteratorT>;

    CounterT *d_histogram1[1] = {d_histogram};
    int num_levels1[1]        = {num_levels};
    LevelT lower_level1[1]    = {lower_level};
    LevelT upper_level1[1]    = {upper_level};

    return MultiHistogramEven<1, 1>(d_temp_storage,
                                    temp_storage_bytes,
                                    d_samples,
                                    d_histogram1,
                                    num_levels1,
                                    lower_level1,
                                    upper_level1,
                                    num_samples,
                                    static_cast<OffsetT>(1),
                                    sizeof(SampleT) * num_samples,
                                    stream);
  }

  template <typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  HistogramEven(void *d_temp_storage,
                size_t &temp_storage_bytes,
                SampleIteratorT d_samples,
                CounterT *d_histogram,
                int num_levels,
                LevelT lower_level,
                LevelT upper_level,
                OffsetT num_samples,
                cudaStream_t stream,
                bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return HistogramEven(d_temp_storage,
                         temp_storage_bytes,
                         d_samples,
                         d_histogram,
                         num_levels,
                         lower_level,
                         upper_level,
                         num_samples,
                         stream);
  }

  /**
   * @brief Computes an intensity histogram from a sequence of data samples 
   *        using equal-width bins.
   *
   * @par
   * - A two-dimensional *region of interest* within `d_samples` can be 
   *   specified using the `num_row_samples`, `num_rows`, and 
   *   `row_stride_bytes` parameters.
   * - The row stride must be a whole multiple of the sample data type
   *   size, i.e., `(row_stride_bytes % sizeof(SampleT)) == 0`.
   * - The number of histogram bins is (`num_levels - 1`)
   * - All bins comprise the same width of sample values:
   *   `(upper_level - lower_level) / (num_levels - 1)`
   * - For a given row `r` in `[0, num_rows)`, let 
   *   `row_begin = d_samples + r * row_stride_bytes / sizeof(SampleT)` and 
   *   `row_end = row_begin + num_row_samples`. The ranges
   *   `[row_begin, row_end)` and `[d_histogram, d_histogram + num_levels - 1)` 
   *   shall not overlap in any way.
   * - `cuda::std::common_type<LevelT, SampleT>` must be valid, and both LevelT
   *   and SampleT must be valid arithmetic types. The common type must be
   *   convertible to `int` and trivially copyable.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the computation of a six-bin histogram
   * from a 2x5 region of interest within a flattened 2x7 array of float samples.
   *
   * @par
   * @code
   * #include <cub/cub.cuh> // or equivalently <cub/device/device_histogram.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for 
   * // input samples and output histogram
   * int      num_row_samples;    // e.g., 5
   * int      num_rows;           // e.g., 2;
   * size_t   row_stride_bytes;   // e.g., 7 * sizeof(float)
   * float*   d_samples;          // e.g., [2.2, 6.1, 7.1, 2.9, 3.5,   -, -,
   *                              //        0.3, 2.9, 2.1, 6.1, 999.5, -, -]
   * int*     d_histogram;        // e.g., [ -, -, -, -, -, -]
   * int      num_levels;         // e.g., 7       (seven level boundaries for six bins)
   * float    lower_level;        // e.g., 0.0     (lower sample value boundary of lowest bin)
   * float    upper_level;        // e.g., 12.0    (upper sample value boundary of upper bin)
   * ...
   *
   * // Determine temporary device storage requirements
   * void*    d_temp_storage  = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceHistogram::HistogramEven(
   *     d_temp_storage, temp_storage_bytes,
   *     d_samples, d_histogram, num_levels, lower_level, upper_level,
   *     num_row_samples, num_rows, row_stride_bytes);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Compute histograms
   * cub::DeviceHistogram::HistogramEven(
   *     d_temp_storage, temp_storage_bytes, d_samples, d_histogram,
   *     d_samples, d_histogram, num_levels, lower_level, upper_level,
   *     num_row_samples, num_rows, row_stride_bytes);
   *
   * // d_histogram   <-- [1, 5, 0, 3, 0, 0];
   * @endcode
   *
   * @tparam SampleIteratorT          
   *   **[inferred]** Random-access input iterator type for reading 
   *   input samples. \iterator
   *
   * @tparam CounterT                 
   *   **[inferred]** Integer type for histogram bin counters
   *
   * @tparam LevelT                   
   *   **[inferred]** Type for specifying boundaries (levels)
   *
   * @tparam OffsetT                  
   *   **[inferred]** Signed integer type for sequence offsets, list lengths,
   *   pointer differences, etc. \offset_size1

   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no 
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_samples 
   *   The pointer to the input sequence of data samples.
   *
   * @param[out] d_histogram 
   *   The pointer to the histogram counter output array of 
   *   length `num_levels - 1`.
   *
   * @param[in] num_levels 
   *   The number of boundaries (levels) for delineating histogram samples.  
   *   Implies that the number of bins is `num_levels - 1`.
   *
   * @param[in] lower_level 
   *   The lower sample value bound (inclusive) for the lowest histogram bin.
   *
   * @param[in] upper_level 
   *   The upper sample value bound (exclusive) for the highest histogram bin.
   *
   * @param[in] num_row_samples 
   *   The number of data samples per row in the region of interest
   *
   * @param[in] num_rows 
   *   The number of rows in the region of interest
   *
   * @param[in] row_stride_bytes 
   *   The number of bytes between starts of consecutive rows in 
   *   the region of interest
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  HistogramEven(void *d_temp_storage,
                size_t &temp_storage_bytes,
                SampleIteratorT d_samples,
                CounterT *d_histogram,
                int num_levels,
                LevelT lower_level,
                LevelT upper_level,
                OffsetT num_row_samples,
                OffsetT num_rows,
                size_t row_stride_bytes,
                cudaStream_t stream    = 0)
  {
    CounterT *d_histogram1[1] = {d_histogram};
    int num_levels1[1]        = {num_levels};
    LevelT lower_level1[1]    = {lower_level};
    LevelT upper_level1[1]    = {upper_level};

    return MultiHistogramEven<1, 1>(d_temp_storage,
                                    temp_storage_bytes,
                                    d_samples,
                                    d_histogram1,
                                    num_levels1,
                                    lower_level1,
                                    upper_level1,
                                    num_row_samples,
                                    num_rows,
                                    row_stride_bytes,
                                    stream);
  }

  template <typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  HistogramEven(void *d_temp_storage,
                size_t &temp_storage_bytes,
                SampleIteratorT d_samples,
                CounterT *d_histogram,
                int num_levels,
                LevelT lower_level,
                LevelT upper_level,
                OffsetT num_row_samples,
                OffsetT num_rows,
                size_t row_stride_bytes,
                cudaStream_t stream,
                bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return HistogramEven(d_temp_storage,
                         temp_storage_bytes,
                         d_samples,
                         d_histogram,
                         num_levels,
                         lower_level,
                         upper_level,
                         num_row_samples,
                         num_rows,
                         row_stride_bytes,
                         stream);
  }

  /**
   * @brief Computes per-channel intensity histograms from a sequence of 
   *        multi-channel "pixel" data samples using equal-width bins.
   *
   * @par
   * - The input is a sequence of *pixel* structures, where each pixel comprises
   *   a record of `NUM_CHANNELS` consecutive data samples 
   *   (e.g., an *RGBA* pixel).
   * - Of the `NUM_CHANNELS` specified, the function will only compute 
   *   histograms for the first `NUM_ACTIVE_CHANNELS` 
   *   (e.g., only *RGB* histograms from *RGBA* pixel samples).
   * - The number of histogram bins for channel<sub><em>i</em></sub> is 
   *   `num_levels[i] - 1`.
   * - For channel<sub><em>i</em></sub>, the range of values for all histogram bins
   *   have the same width: 
   *   `(upper_level[i] - lower_level[i]) / (num_levels[i] - 1)`
   * - For a given channel `c` in `[0, NUM_ACTIVE_CHANNELS)`, the ranges 
   *   `[d_samples, d_samples + NUM_CHANNELS * num_pixels)` and 
   *   `[d_histogram[c], d_histogram[c] + num_levels[c] - 1)` shall not overlap 
   *   in any way.
   * - `cuda::std::common_type<LevelT, SampleT>` must be valid, and both LevelT
   *   and SampleT must be valid arithmetic types. The common type must be
   *   convertible to `int` and trivially copyable.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the computation of three 256-bin <em>RGB</em> histograms
   * from a quad-channel sequence of <em>RGBA</em> pixels (8 bits per channel per pixel)
   *
   * @par
   * @code
   * #include <cub/cub.cuh> // or equivalently <cub/device/device_histogram.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for 
   * // input samples and output histograms
   * int              num_pixels;         // e.g., 5
   * unsigned char*   d_samples;          // e.g., [(2, 6, 7, 5), (3, 0, 2, 1), (7, 0, 6, 2),
   *                                      //        (0, 6, 7, 5), (3, 0, 2, 6)]
   * int*             d_histogram[3];     // e.g., three device pointers to three device buffers,
   *                                      //       each allocated with 256 integer counters
   * int              num_levels[3];      // e.g., {257, 257, 257};
   * unsigned int     lower_level[3];     // e.g., {0, 0, 0};
   * unsigned int     upper_level[3];     // e.g., {256, 256, 256};
   * ...
   *
   * // Determine temporary device storage requirements
   * void*    d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceHistogram::MultiHistogramEven<4, 3>(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, 
   *   lower_level, upper_level, num_pixels);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Compute histograms
   * cub::DeviceHistogram::MultiHistogramEven<4, 3>(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, 
   *   lower_level, upper_level, num_pixels);
   *
   * // d_histogram   <-- [ [1, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, ..., 0],
   * //                     [0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, ..., 0],
   * //                     [0, 0, 2, 0, 0, 0, 1, 2, 0, 0, 0, ..., 0] ]
   * @endcode
   *
   * @tparam NUM_CHANNELS             
   *   Number of channels interleaved in the input data (may be greater than 
   *   the number of channels being actively histogrammed)
   *
   * @tparam NUM_ACTIVE_CHANNELS      
   *   **[inferred]** Number of channels actively being histogrammed
   *
   * @tparam SampleIteratorT          
   *   **[inferred]** Random-access input iterator type for reading 
   *   input samples. \iterator
   *
   * @tparam CounterT                 
   *   **[inferred]** Integer type for histogram bin counters
   *
   * @tparam LevelT                   
   *   **[inferred]** Type for specifying boundaries (levels)
   *
   * @tparam OffsetT                  
   *   **[inferred]** Signed integer type for sequence offsets, list lengths, 
   *   pointer differences, etc. \offset_size1
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no 
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_samples 
   *   The pointer to the multi-channel input sequence of data samples. 
   *   The samples from different channels are assumed to be interleaved 
   *   (e.g., an array of 32-bit pixels where each pixel consists of four 
   *   *RGBA* 8-bit samples).
   *
   * @param[out] d_histogram
   *   The pointers to the histogram counter output arrays, one for each active 
   *   channel. For channel<sub><em>i</em></sub>, the allocation length of 
   *   `d_histogram[i]` should be `num_levels[i] - 1`.
   *
   * @param[in] num_levels
   *   The number of boundaries (levels) for delineating histogram samples in 
   *   each active channel. Implies that the number of bins for 
   *   channel<sub><em>i</em></sub> is `num_levels[i] - 1`.
   *
   * @param[in] lower_level
   *   The lower sample value bound (inclusive) for the lowest histogram bin in 
   *   each active channel.
   *
   * @param[in] upper_level
   *   The upper sample value bound (exclusive) for the highest histogram bin 
   *   in each active channel.
   *
   * @param[in] num_pixels 
   *   The number of multi-channel pixels 
   *   (i.e., the length of `d_samples / NUM_CHANNELS`)
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <int NUM_CHANNELS,
            int NUM_ACTIVE_CHANNELS,
            typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  MultiHistogramEven(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     SampleIteratorT d_samples,
                     CounterT *d_histogram[NUM_ACTIVE_CHANNELS],
                     int num_levels[NUM_ACTIVE_CHANNELS],
                     LevelT lower_level[NUM_ACTIVE_CHANNELS],
                     LevelT upper_level[NUM_ACTIVE_CHANNELS],
                     OffsetT num_pixels,
                     cudaStream_t stream = 0)
  {
    /// The sample value type of the input iterator
    using SampleT = cub::detail::value_t<SampleIteratorT>;

    return MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_histogram,
      num_levels,
      lower_level,
      upper_level,
      num_pixels,
      static_cast<OffsetT>(1),
      sizeof(SampleT) * NUM_CHANNELS * num_pixels,
      stream);
  }

  template <int NUM_CHANNELS,
            int NUM_ACTIVE_CHANNELS,
            typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  MultiHistogramEven(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     SampleIteratorT d_samples,
                     CounterT *d_histogram[NUM_ACTIVE_CHANNELS],
                     int num_levels[NUM_ACTIVE_CHANNELS],
                     LevelT lower_level[NUM_ACTIVE_CHANNELS],
                     LevelT upper_level[NUM_ACTIVE_CHANNELS],
                     OffsetT num_pixels,
                     cudaStream_t stream,
                     bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return MultiHistogramEven(d_temp_storage,
                              temp_storage_bytes,
                              d_samples,
                              d_histogram,
                              num_levels,
                              lower_level,
                              upper_level,
                              num_pixels,
                              stream);
  }

  /**
   * @brief Computes per-channel intensity histograms from a sequence of 
   *        multi-channel "pixel" data samples using equal-width bins.
   *
   * @par
   * - The input is a sequence of *pixel* structures, where each pixel 
   *   comprises a record of `NUM_CHANNELS` consecutive data samples 
   *   (e.g., an *RGBA* pixel).
   * - Of the `NUM_CHANNELS` specified, the function will only compute 
   *   histograms for the first `NUM_ACTIVE_CHANNELS` (e.g., only *RGB* 
   *   histograms from *RGBA* pixel samples).
   * - A two-dimensional *region of interest* within `d_samples` can be 
   *   specified using the `num_row_samples`, `num_rows`, and 
   *   `row_stride_bytes` parameters.
   * - The row stride must be a whole multiple of the sample data type
   *   size, i.e., `(row_stride_bytes % sizeof(SampleT)) == 0`.
   * - The number of histogram bins for channel<sub><em>i</em></sub> is 
   *   `num_levels[i] - 1`.
   * - For channel<sub><em>i</em></sub>, the range of values for all histogram 
   *   bins have the same width: 
   *   `(upper_level[i] - lower_level[i]) / (num_levels[i] - 1)`
   * - For a given row `r` in `[0, num_rows)`, and sample `s` in 
   *   `[0, num_row_pixels)`, let 
   *   `row_begin = d_samples + r * row_stride_bytes / sizeof(SampleT)`, 
   *   `sample_begin = row_begin + s * NUM_CHANNELS`, and
   *   `sample_end = sample_begin + NUM_ACTIVE_CHANNELS`. For a given channel
   *    `c` in `[0, NUM_ACTIVE_CHANNELS)`, the ranges 
   *   `[sample_begin, sample_end)` and 
   *   `[d_histogram[c], d_histogram[c] + num_levels[c] - 1)` shall not overlap 
   *   in any way.
   * - `cuda::std::common_type<LevelT, SampleT>` must be valid, and both LevelT
   *   and SampleT must be valid arithmetic types. The common type must be
   *   convertible to `int` and trivially copyable.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the computation of three 256-bin 
   * *RGB* histograms from a 2x3 region of interest of within a flattened 2x4 
   * array of quad-channel *RGBA* pixels (8 bits per channel per pixel).
   *
   * @par
   * @code
   * #include <cub/cub.cuh> // or equivalently <cub/device/device_histogram.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for input 
   * // samples and output histograms
   * int              num_row_pixels;     // e.g., 3
   * int              num_rows;           // e.g., 2
   * size_t           row_stride_bytes;   // e.g., 4 * sizeof(unsigned char) * NUM_CHANNELS
   * unsigned char*   d_samples;          // e.g., [(2, 6, 7, 5), (3, 0, 2, 1), (7, 0, 6, 2), (-, -, -, -),
   *                                      //        (0, 6, 7, 5), (3, 0, 2, 6), (1, 1, 1, 1), (-, -, -, -)]
   * int*             d_histogram[3];     // e.g., three device pointers to three device buffers,
   *                                      //       each allocated with 256 integer counters
   * int              num_levels[3];      // e.g., {257, 257, 257};
   * unsigned int     lower_level[3];     // e.g., {0, 0, 0};
   * unsigned int     upper_level[3];     // e.g., {256, 256, 256};
   * ...
   *
   * // Determine temporary device storage requirements
   * void*    d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceHistogram::MultiHistogramEven<4, 3>(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, lower_level, upper_level,
   *   num_row_pixels, num_rows, row_stride_bytes);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Compute histograms
   * cub::DeviceHistogram::MultiHistogramEven<4, 3>(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, lower_level, upper_level,
   *   num_row_pixels, num_rows, row_stride_bytes);
   *
   * // d_histogram   <-- [ [1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0, ..., 0],
   * //                     [0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, ..., 0],
   * //                     [0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, ..., 0] ]
   * @endcode
   *
   * @tparam NUM_CHANNELS             
   *   Number of channels interleaved in the input data (may be greater than 
   *   the number of channels being actively histogrammed)
   *
   * @tparam NUM_ACTIVE_CHANNELS      
   *   **[inferred]** Number of channels actively being histogrammed
   *
   * @tparam SampleIteratorT          
   *   **[inferred]** Random-access input iterator type for reading input 
   *   samples. \iterator
   *
   * @tparam CounterT                 
   *   **[inferred]** Integer type for histogram bin counters
   *
   * @tparam LevelT                   
   *   **[inferred]** Type for specifying boundaries (levels)
   *
   * @tparam OffsetT                  
   *   **[inferred]** Signed integer type for sequence offsets, list lengths, 
   *   pointer differences, etc. \offset_size1
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no 
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_samples 
   *   The pointer to the multi-channel input sequence of data samples. The 
   *   samples from different channels are assumed to be interleaved (e.g., 
   *   an array of 32-bit pixels where each pixel consists of four 
   *   *RGBA* 8-bit samples).
   *
   * @param[out] d_histogram 
   *   The pointers to the histogram counter output arrays, one for each 
   *   active channel. For channel<sub><em>i</em></sub>, the allocation length 
   *   of `d_histogram[i]` should be `num_levels[i] - 1`.
   *
   * @param[in] num_levels 
   *   The number of boundaries (levels) for delineating histogram samples in 
   *   each active channel. Implies that the number of bins for 
   *   channel<sub><em>i</em></sub> is `num_levels[i] - 1`.
   *
   * @param[in] lower_level 
   *   The lower sample value bound (inclusive) for the lowest histogram bin in 
   *   each active channel.
   *
   * @param[in] upper_level 
   *   The upper sample value bound (exclusive) for the highest histogram bin 
   *   in each active channel.
   *
   * @param[in] num_row_pixels 
   *   The number of multi-channel pixels per row in the region of interest
   *
   * @param[in] num_rows 
   *   The number of rows in the region of interest
   *
   * @param[in] row_stride_bytes 
   *   The number of bytes between starts of consecutive rows in the region of 
   *   interest
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <int NUM_CHANNELS,
            int NUM_ACTIVE_CHANNELS,
            typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  MultiHistogramEven(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     SampleIteratorT d_samples,
                     CounterT *d_histogram[NUM_ACTIVE_CHANNELS],
                     int num_levels[NUM_ACTIVE_CHANNELS],
                     LevelT lower_level[NUM_ACTIVE_CHANNELS],
                     LevelT upper_level[NUM_ACTIVE_CHANNELS],
                     OffsetT num_row_pixels,
                     OffsetT num_rows,
                     size_t row_stride_bytes,
                     cudaStream_t stream = 0)
  {
    /// The sample value type of the input iterator
    using SampleT = cub::detail::value_t<SampleIteratorT>;
    Int2Type<sizeof(SampleT) == 1> is_byte_sample;

    if ((sizeof(OffsetT) > sizeof(int)) &&
        ((unsigned long long)(num_rows * row_stride_bytes) <
         (unsigned long long)INT_MAX))
    {
      // Down-convert OffsetT data type
      return DispatchHistogram<NUM_CHANNELS,
                               NUM_ACTIVE_CHANNELS,
                               SampleIteratorT,
                               CounterT,
                               LevelT,
                               int>::DispatchEven(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_samples,
                                                  d_histogram,
                                                  num_levels,
                                                  lower_level,
                                                  upper_level,
                                                  (int)num_row_pixels,
                                                  (int)num_rows,
                                                  (int)(row_stride_bytes /
                                                        sizeof(SampleT)),
                                                  stream,
                                                  is_byte_sample);
    }

    return DispatchHistogram<NUM_CHANNELS,
                             NUM_ACTIVE_CHANNELS,
                             SampleIteratorT,
                             CounterT,
                             LevelT,
                             OffsetT>::DispatchEven(d_temp_storage,
                                                    temp_storage_bytes,
                                                    d_samples,
                                                    d_histogram,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    num_row_pixels,
                                                    num_rows,
                                                    (OffsetT)(row_stride_bytes /
                                                              sizeof(SampleT)),
                                                    stream,
                                                    is_byte_sample);
  }

  template <int NUM_CHANNELS,
            int NUM_ACTIVE_CHANNELS,
            typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  MultiHistogramEven(void *d_temp_storage,
                     size_t &temp_storage_bytes,
                     SampleIteratorT d_samples,
                     CounterT *d_histogram[NUM_ACTIVE_CHANNELS],
                     int num_levels[NUM_ACTIVE_CHANNELS],
                     LevelT lower_level[NUM_ACTIVE_CHANNELS],
                     LevelT upper_level[NUM_ACTIVE_CHANNELS],
                     OffsetT num_row_pixels,
                     OffsetT num_rows,
                     size_t row_stride_bytes,
                     cudaStream_t stream,
                     bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return MultiHistogramEven(d_temp_storage,
                              temp_storage_bytes,
                              d_samples,
                              d_histogram,
                              num_levels,
                              lower_level,
                              upper_level,
                              num_row_pixels,
                              num_rows,
                              row_stride_bytes,
                              stream);
  }

  //@}  end member group
  /******************************************************************//**
   * @name Custom bin ranges
   *********************************************************************/
  //@{

  /**
   * @brief Computes an intensity histogram from a sequence of data samples 
   *        using the specified bin boundary levels.
   *
   * @par
   * - The number of histogram bins is (`num_levels - 1`)
   * - The value range for bin<sub><em>i</em></sub> is `[level[i], level[i+1])`
   * - The range `[d_histogram, d_histogram + num_levels - 1)` shall not 
   *   overlap `[d_samples, d_samples + num_samples)` nor 
   *   `[d_levels, d_levels + num_levels)` in any way. The ranges 
   *   `[d_levels, d_levels + num_levels)` and 
   *   `[d_samples, d_samples + num_samples)` may overlap.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the computation of an six-bin histogram
   * from a sequence of float samples
   *
   * @par
   * @code
   * #include <cub/cub.cuh> // or equivalently <cub/device/device_histogram.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for input 
   * // samples and output histogram
   * int      num_samples;    // e.g., 10
   * float*   d_samples;      // e.g., [2.2, 6.0, 7.1, 2.9, 3.5, 0.3, 2.9, 2.0, 6.1, 999.5]
   * int*     d_histogram;    // e.g., [ -, -, -, -, -, -]
   * int      num_levels      // e.g., 7 (seven level boundaries for six bins)
   * float*   d_levels;       // e.g., [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
   * ...
   *
   * // Determine temporary device storage requirements
   * void*    d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceHistogram::HistogramRange(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, d_levels, num_samples);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Compute histograms
   * cub::DeviceHistogram::HistogramRange(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, d_levels, num_samples);
   *
   * // d_histogram   <-- [1, 5, 0, 3, 0, 0];
   *
   * @endcode
   *
   * @tparam SampleIteratorT          
   *   **[inferred]** Random-access input iterator type for reading 
   *   input samples.\iterator
   *
   * @tparam CounterT                 
   *   **[inferred]** Integer type for histogram bin counters
   *
   * @tparam LevelT                   
   *   **[inferred]** Type for specifying boundaries (levels)
   *
   * @tparam OffsetT                  
   *   **[inferred]** Signed integer type for sequence offsets, list lengths, 
   *   pointer differences, etc. \offset_size1
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no work 
   *   is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_samples 
   *   The pointer to the input sequence of data samples.
   *
   * @param[out] d_histogram 
   *   The pointer to the histogram counter output array of length 
   *   `num_levels - 1`.
   *
   * @param[in] num_levels 
   *   The number of boundaries (levels) for delineating histogram samples.  
   *   Implies that the number of bins is `num_levels - 1`.
   *
   * @param[in] d_levels 
   *   The pointer to the array of boundaries (levels). Bin ranges are defined 
   *   by consecutive boundary pairings: lower sample value boundaries are 
   *   inclusive and upper sample value boundaries are exclusive.
   *
   * @param[in] num_samples 
   *   The number of data samples per row in the region of interest
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  HistogramRange(void *d_temp_storage,
                 size_t &temp_storage_bytes,
                 SampleIteratorT d_samples,
                 CounterT *d_histogram,
                 int num_levels,
                 LevelT *d_levels,
                 OffsetT num_samples,
                 cudaStream_t stream = 0)
  {
    /// The sample value type of the input iterator
    using SampleT = cub::detail::value_t<SampleIteratorT>;

    CounterT *d_histogram1[1] = {d_histogram};
    int num_levels1[1]        = {num_levels};
    LevelT *d_levels1[1]      = {d_levels};

    return MultiHistogramRange<1, 1>(d_temp_storage,
                                     temp_storage_bytes,
                                     d_samples,
                                     d_histogram1,
                                     num_levels1,
                                     d_levels1,
                                     num_samples,
                                     (OffsetT)1,
                                     (size_t)(sizeof(SampleT) * num_samples),
                                     stream);
  }

  template <typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  HistogramRange(void *d_temp_storage,
                 size_t &temp_storage_bytes,
                 SampleIteratorT d_samples,
                 CounterT *d_histogram,
                 int num_levels,
                 LevelT *d_levels,
                 OffsetT num_samples,
                 cudaStream_t stream,
                 bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return HistogramRange(d_temp_storage,
                          temp_storage_bytes,
                          d_samples,
                          d_histogram,
                          num_levels,
                          d_levels,
                          num_samples,
                          stream);
  }

  /**
   * @brief Computes an intensity histogram from a sequence of data samples 
   *        using the specified bin boundary levels.
   *
   * @par
   * - A two-dimensional *region of interest* within `d_samples` can be 
   *   specified using the `num_row_samples`, `num_rows`, and 
   *   `row_stride_bytes` parameters.
   * - The row stride must be a whole multiple of the sample data type
   *   size, i.e., `(row_stride_bytes % sizeof(SampleT)) == 0`.
   * - The number of histogram bins is (`num_levels - 1`)
   * - The value range for bin<sub><em>i</em></sub> is `[level[i], level[i+1])`
   * - For a given row `r` in `[0, num_rows)`, let 
   *   `row_begin = d_samples + r * row_stride_bytes / sizeof(SampleT)` and 
   *   `row_end = row_begin + num_row_samples`. The range
   *   `[d_histogram, d_histogram + num_levels - 1)` shall not overlap
   *   `[row_begin, row_end)` nor `[d_levels, d_levels + num_levels)`.
   *   The ranges `[d_levels, d_levels + num_levels)` and `[row_begin, row_end)`
   *   may overlap.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the computation of a six-bin histogram
   * from a 2x5 region of interest within a flattened 2x7 array of float samples.
   *
   * @par
   * @code
   * #include <cub/cub.cuh> // or equivalently <cub/device/device_histogram.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for input samples and
   * // output histogram
   * int      num_row_samples;    // e.g., 5
   * int      num_rows;           // e.g., 2;
   * int      row_stride_bytes;   // e.g., 7 * sizeof(float)
   * float*   d_samples;          // e.g., [2.2, 6.0, 7.1, 2.9, 3.5,   -, -,
   *                              //        0.3, 2.9, 2.0, 6.1, 999.5, -, -]
   * int*     d_histogram;        // e.g., [ -, -, -, -, -, -]
   * int      num_levels          // e.g., 7 (seven level boundaries for six bins)
   * float    *d_levels;          // e.g., [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
   * ...
   *
   * // Determine temporary device storage requirements
   * void*    d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceHistogram::HistogramRange(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, d_levels,
   *   num_row_samples, num_rows, row_stride_bytes);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Compute histograms
   * cub::DeviceHistogram::HistogramRange(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, d_levels,
   *   num_row_samples, num_rows, row_stride_bytes);
   *
   * // d_histogram   <-- [1, 5, 0, 3, 0, 0];
   * @endcode
   *
   * @tparam SampleIteratorT          
   *   **[inferred]** Random-access input iterator type for reading 
   *   input samples. \iterator
   * 
   * @tparam CounterT                 
   *   **[inferred]** Integer type for histogram bin counters
   * 
   * @tparam LevelT                   
   *   **[inferred]** Type for specifying boundaries (levels)
   * 
   * @tparam OffsetT                  
   *   **[inferred]** Signed integer type for sequence offsets, list lengths, 
   *   pointer differences, etc. \offset_size1
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no 
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_samples 
   *   The pointer to the input sequence of data samples.
   *
   * @param[out] d_histogram 
   *   The pointer to the histogram counter output array of length 
   *   `num_levels - 1`.
   *
   * @param[in] num_levels 
   *   The number of boundaries (levels) for delineating histogram samples.  
   *   Implies that the number of bins is `num_levels - 1`.
   *
   * @param[in] d_levels 
   *   The pointer to the array of boundaries (levels). Bin ranges are defined 
   *   by consecutive boundary pairings: lower sample value boundaries are 
   *   inclusive and upper sample value boundaries are exclusive.
   *
   * @param[in] num_row_samples 
   *   The number of data samples per row in the region of interest
   *
   * @param[in] num_rows 
   *   The number of rows in the region of interest
   *
   * @param[in] row_stride_bytes 
   *   The number of bytes between starts of consecutive rows in the region 
   *   of interest
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  template <typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  HistogramRange(void *d_temp_storage,
                 size_t &temp_storage_bytes,
                 SampleIteratorT d_samples,
                 CounterT *d_histogram,
                 int num_levels,
                 LevelT *d_levels,
                 OffsetT num_row_samples,
                 OffsetT num_rows,
                 size_t row_stride_bytes,
                 cudaStream_t stream = 0)
  {
    CounterT *d_histogram1[1] = {d_histogram};
    int num_levels1[1]        = {num_levels};
    LevelT *d_levels1[1]      = {d_levels};

    return MultiHistogramRange<1, 1>(d_temp_storage,
                                     temp_storage_bytes,
                                     d_samples,
                                     d_histogram1,
                                     num_levels1,
                                     d_levels1,
                                     num_row_samples,
                                     num_rows,
                                     row_stride_bytes,
                                     stream);
  }

  template <typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  HistogramRange(void *d_temp_storage,
                 size_t &temp_storage_bytes,
                 SampleIteratorT d_samples,
                 CounterT *d_histogram,
                 int num_levels,
                 LevelT *d_levels,
                 OffsetT num_row_samples,
                 OffsetT num_rows,
                 size_t row_stride_bytes,
                 cudaStream_t stream,
                 bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return HistogramRange(d_temp_storage,
                          temp_storage_bytes,
                          d_samples,
                          d_histogram,
                          num_levels,
                          d_levels,
                          num_row_samples,
                          num_rows,
                          row_stride_bytes,
                          stream);
  }

  /**
   * @brief Computes per-channel intensity histograms from a sequence of 
   *        multi-channel "pixel" data samples using the specified bin 
   *        boundary levels.
   *
   * @par
   * - The input is a sequence of *pixel* structures, where each pixel 
   *   comprises a record of `NUM_CHANNELS` consecutive data samples 
   *   (e.g., an *RGBA* pixel).
   * - Of the `NUM_CHANNELS` specified, the function will only compute 
   *   histograms for the first `NUM_ACTIVE_CHANNELS` (e.g., *RGB* histograms 
   *   from *RGBA* pixel samples).
   * - The number of histogram bins for channel<sub><em>i</em></sub> is 
   *   `num_levels[i] - 1`.
   * - For channel<sub><em>i</em></sub>, the range of values for all histogram 
   *   bins have the same width: 
   *   `(upper_level[i] - lower_level[i]) / (num_levels[i] - 1)`
   * - For given channels `c1` and `c2` in `[0, NUM_ACTIVE_CHANNELS)`, the 
   *   range `[d_histogram[c1], d_histogram[c1] + num_levels[c1] - 1)` shall 
   *   not overlap `[d_samples, d_samples + NUM_CHANNELS * num_pixels)` nor
   *   `[d_levels[c2], d_levels[c2] + num_levels[c2])` in any way.
   *   The ranges `[d_levels[c2], d_levels[c2] + num_levels[c2])` and
   *   `[d_samples, d_samples + NUM_CHANNELS * num_pixels)` may overlap.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the computation of three 4-bin *RGB* 
   * histograms from a quad-channel sequence of *RGBA* pixels 
   * (8 bits per channel per pixel)
   *
   * @par
   * @code
   * #include <cub/cub.cuh> // or equivalently <cub/device/device_histogram.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for 
   * // input samples and output histograms
   * int            num_pixels;       // e.g., 5
   * unsigned char  *d_samples;       // e.g., [(2, 6, 7, 5),(3, 0, 2, 1),(7, 0, 6, 2),
   *                                  //        (0, 6, 7, 5),(3, 0, 2, 6)]
   * unsigned int   *d_histogram[3];  // e.g., [[ -, -, -, -],[ -, -, -, -],[ -, -, -, -]];
   * int            num_levels[3];    // e.g., {5, 5, 5};
   * unsigned int   *d_levels[3];     // e.g., [ [0, 2, 4, 6, 8],
   *                                  //         [0, 2, 4, 6, 8],
   *                                  //         [0, 2, 4, 6, 8] ];
   * ...
   *
   * // Determine temporary device storage requirements
   * void*    d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceHistogram::MultiHistogramRange<4, 3>(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, d_levels, num_pixels);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Compute histograms
   * cub::DeviceHistogram::MultiHistogramRange<4, 3>(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, d_levels, num_pixels);
   *
   * // d_histogram   <-- [ [1, 3, 0, 1],
   * //                     [3, 0, 0, 2],
   * //                     [0, 2, 0, 3] ]
   *
   * @endcode
   *
   * @tparam NUM_CHANNELS             
   *   Number of channels interleaved in the input data (may be greater than 
   *   the number of channels being actively histogrammed)
   * 
   * @tparam NUM_ACTIVE_CHANNELS      
   *   **[inferred]** Number of channels actively being histogrammed
   * 
   * @tparam SampleIteratorT          
   *   **[inferred]** Random-access input iterator type for reading 
   *   input samples. \iterator
   * 
   * @tparam CounterT                 
   *   **[inferred]** Integer type for histogram bin counters
   * 
   * @tparam LevelT                   
   *   **[inferred]** Type for specifying boundaries (levels)
   * 
   * @tparam OffsetT                  
   *   **[inferred]** Signed integer type for sequence offsets, list lengths, 
   *   pointer differences, etc. \offset_size1
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to `temp_storage_bytes` and no 
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_samples 
   *   The pointer to the multi-channel input sequence of data samples. 
   *   The samples from different channels are assumed to be interleaved (e.g., 
   *   an array of 32-bit pixels where each pixel consists of four *RGBA* 
   *   8-bit samples).
   *
   * @param[out] d_histogram 
   *   The pointers to the histogram counter output arrays, one for each active 
   *   channel. For channel<sub><em>i</em></sub>, the allocation length of 
   *   `d_histogram[i]` should be `num_levels[i] - 1`.
   *
   * @param[in] num_levels 
   *   The number of boundaries (levels) for delineating histogram samples in 
   *   each active channel. Implies that the number of bins for 
   *   channel<sub><em>i</em></sub> is `num_levels[i] - 1`.
   *
   * @param[in] d_levels 
   *   The pointers to the arrays of boundaries (levels), one for each active 
   *   channel. Bin ranges are defined by consecutive boundary pairings: lower 
   *   sample value boundaries are inclusive and upper sample value boundaries 
   *   are exclusive.
   *
   * @param[in] num_pixels 
   *   The number of multi-channel pixels 
   *   (i.e., the length of `d_samples / NUM_CHANNELS`)
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <int NUM_CHANNELS,
            int NUM_ACTIVE_CHANNELS,
            typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  MultiHistogramRange(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      SampleIteratorT d_samples,
                      CounterT *d_histogram[NUM_ACTIVE_CHANNELS],
                      int num_levels[NUM_ACTIVE_CHANNELS],
                      LevelT *d_levels[NUM_ACTIVE_CHANNELS],
                      OffsetT num_pixels,
                      cudaStream_t stream = 0)
  {
    /// The sample value type of the input iterator
    using SampleT = cub::detail::value_t<SampleIteratorT>;

    return MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
      d_temp_storage,
      temp_storage_bytes,
      d_samples,
      d_histogram,
      num_levels,
      d_levels,
      num_pixels,
      (OffsetT)1,
      (size_t)(sizeof(SampleT) * NUM_CHANNELS * num_pixels),
      stream);
  }

  template <int NUM_CHANNELS,
            int NUM_ACTIVE_CHANNELS,
            typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  MultiHistogramRange(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      SampleIteratorT d_samples,
                      CounterT *d_histogram[NUM_ACTIVE_CHANNELS],
                      int num_levels[NUM_ACTIVE_CHANNELS],
                      LevelT *d_levels[NUM_ACTIVE_CHANNELS],
                      OffsetT num_pixels,
                      cudaStream_t stream,
                      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return MultiHistogramRange(d_temp_storage,
                               temp_storage_bytes,
                               d_samples,
                               d_histogram,
                               num_levels,
                               d_levels,
                               num_pixels,
                               stream);
  }

  /**
   * @brief Computes per-channel intensity histograms from a sequence of 
   *        multi-channel "pixel" data samples using the specified bin boundary 
   *        levels.
   *
   * @par
   * - The input is a sequence of *pixel* structures, where each pixel comprises
   *   a record of `NUM_CHANNELS` consecutive data samples 
   *   (e.g., an *RGBA* pixel).
   * - Of the `NUM_CHANNELS` specified, the function will only compute 
   *   histograms for the first `NUM_ACTIVE_CHANNELS` (e.g., *RGB* histograms 
   *   from *RGBA* pixel samples).
   * - A two-dimensional *region of interest* within `d_samples` can be 
   *   specified using the `num_row_samples`, `num_rows`, and `row_stride_bytes` 
   *   parameters.
   * - The row stride must be a whole multiple of the sample data type
   *   size, i.e., `(row_stride_bytes % sizeof(SampleT)) == 0`.
   * - The number of histogram bins for channel<sub><em>i</em></sub> is 
   *   `num_levels[i] - 1`.
   * - For channel<sub><em>i</em></sub>, the range of values for all histogram 
   *   bins have the same width: 
   *   `(upper_level[i] - lower_level[i]) / (num_levels[i] - 1)`
   * - For a given row `r` in `[0, num_rows)`, and sample `s` in 
   *   `[0, num_row_pixels)`, let 
   *   `row_begin = d_samples + r * row_stride_bytes / sizeof(SampleT)`, 
   *   `sample_begin = row_begin + s * NUM_CHANNELS`, and
   *   `sample_end = sample_begin + NUM_ACTIVE_CHANNELS`. For given channels
   *    `c1` and `c2` in `[0, NUM_ACTIVE_CHANNELS)`, the range
   *   `[d_histogram[c1], d_histogram[c1] + num_levels[c1] - 1)` shall not 
   *   overlap `[sample_begin, sample_end)` nor
   *   `[d_levels[c2], d_levels[c2] + num_levels[c2])` in any way. The ranges
   *   `[d_levels[c2], d_levels[c2] + num_levels[c2])` and 
   *   `[sample_begin, sample_end)` may overlap.
   * - @devicestorage
   *
   * @par Snippet
   * The code snippet below illustrates the computation of three 4-bin *RGB* 
   * histograms from a 2x3 region of interest of within a flattened 2x4 array 
   * of quad-channel *RGBA* pixels (8 bits per channel per pixel).
   *
   * @par
   * @code
   * #include <cub/cub.cuh> // or equivalently <cub/device/device_histogram.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for input 
   * // samples and output histograms
   * int              num_row_pixels;     // e.g., 3
   * int              num_rows;           // e.g., 2
   * size_t           row_stride_bytes;   // e.g., 4 * sizeof(unsigned char) * NUM_CHANNELS
   * unsigned char*   d_samples;          // e.g., [(2, 6, 7, 5),(3, 0, 2, 1),(1, 1, 1, 1),(-, -, -, -),
   *                                      //        (7, 0, 6, 2),(0, 6, 7, 5),(3, 0, 2, 6),(-, -, -, -)]
   * int*             d_histogram[3];     // e.g., [[ -, -, -, -],[ -, -, -, -],[ -, -, -, -]];
   * int              num_levels[3];      // e.g., {5, 5, 5};
   * unsigned int*    d_levels[3];        // e.g., [ [0, 2, 4, 6, 8],
   *                                      //         [0, 2, 4, 6, 8],
   *                                      //         [0, 2, 4, 6, 8] ];
   * ...
   *
   * // Determine temporary device storage requirements
   * void*    d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceHistogram::MultiHistogramRange<4, 3>(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, d_levels, 
   *   num_row_pixels, num_rows, row_stride_bytes);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Compute histograms
   * cub::DeviceHistogram::MultiHistogramRange<4, 3>(
   *   d_temp_storage, temp_storage_bytes,
   *   d_samples, d_histogram, num_levels, 
   *   d_levels, num_row_pixels, num_rows, row_stride_bytes);
   *
   * // d_histogram   <-- [ [2, 3, 0, 1],
   * //                     [3, 0, 0, 2],
   * //                     [1, 2, 0, 3] ]
   *
   * @endcode
   *
   * @tparam NUM_CHANNELS             
   *   Number of channels interleaved in the input data (may be greater than 
   *   the number of channels being actively histogrammed)
   * 
   * @tparam NUM_ACTIVE_CHANNELS      
   *   **[inferred]** Number of channels actively being histogrammed
   * 
   * @tparam SampleIteratorT          
   *   **[inferred]** Random-access input iterator type for reading input 
   *   samples. \iterator
   * 
   * @tparam CounterT                 
   *   **[inferred]** Integer type for histogram bin counters
   * 
   * @tparam LevelT                   
   *   **[inferred]** Type for specifying boundaries (levels)
   * 
   * @tparam OffsetT                  
   *   **[inferred]** Signed integer type for sequence offsets, list lengths, 
   *   pointer differences, etc.  \offset_size1
   *
   * @param[in] d_temp_storage 
   *   Device-accessible allocation of temporary storage. When `nullptr`, the 
   *   required allocation size is written to \p temp_storage_bytes and no work is done.
   *
   * @param[in,out] temp_storage_bytes 
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_samples 
   *   The pointer to the multi-channel input sequence of data samples. The 
   *   samples from different channels are assumed to be interleaved (e.g., an 
   *   array of 32-bit pixels where each pixel consists of four 
   *   *RGBA* 8-bit samples).
   *
   * @param[out] d_histogram 
   *   The pointers to the histogram counter output arrays, one for each active 
   *   channel. For channel<sub><em>i</em></sub>, the allocation length of 
   *   `d_histogram[i]` should be `num_levels[i] - 1`.
   *
   * @param[in] num_levels 
   *   The number of boundaries (levels) for delineating histogram samples in 
   *   each active channel. Implies that the number of bins for 
   *   channel<sub><em>i</em></sub> is `num_levels[i] - 1`.
   *
   * @param[in] d_levels 
   *   The pointers to the arrays of boundaries (levels), one for each active 
   *   channel. Bin ranges are defined by consecutive boundary pairings: lower 
   *   sample value boundaries are inclusive and upper sample value boundaries 
   *   are exclusive.
   *
   * @param[in] num_row_pixels 
   *   The number of multi-channel pixels per row in the region of interest
   *
   * @param[in] num_rows 
   *   The number of rows in the region of interest
   *
   * @param[in] row_stride_bytes 
   *   The number of bytes between starts of consecutive rows in the 
   *   region of interest
   *
   * @param[in] stream 
   *   **[optional]** CUDA stream to launch kernels within.  
   *   Default is stream<sub>0</sub>.
   */
  template <int NUM_CHANNELS,
            int NUM_ACTIVE_CHANNELS,
            typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  MultiHistogramRange(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      SampleIteratorT d_samples,
                      CounterT *d_histogram[NUM_ACTIVE_CHANNELS],
                      int num_levels[NUM_ACTIVE_CHANNELS],
                      LevelT *d_levels[NUM_ACTIVE_CHANNELS],
                      OffsetT num_row_pixels,
                      OffsetT num_rows,
                      size_t row_stride_bytes,
                      cudaStream_t stream = 0)
  {
    /// The sample value type of the input iterator
    using SampleT = cub::detail::value_t<SampleIteratorT>;
    Int2Type<sizeof(SampleT) == 1> is_byte_sample;

    if ((sizeof(OffsetT) > sizeof(int)) &&
        ((unsigned long long)(num_rows * row_stride_bytes) <
         (unsigned long long)INT_MAX))
    {
      // Down-convert OffsetT data type
      return DispatchHistogram<NUM_CHANNELS,
                               NUM_ACTIVE_CHANNELS,
                               SampleIteratorT,
                               CounterT,
                               LevelT,
                               int>::DispatchRange(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_samples,
                                                   d_histogram,
                                                   num_levels,
                                                   d_levels,
                                                   (int)num_row_pixels,
                                                   (int)num_rows,
                                                   (int)(row_stride_bytes /
                                                         sizeof(SampleT)),
                                                   stream,
                                                   is_byte_sample);
    }

    return DispatchHistogram<NUM_CHANNELS,
                             NUM_ACTIVE_CHANNELS,
                             SampleIteratorT,
                             CounterT,
                             LevelT,
                             OffsetT>::DispatchRange(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_samples,
                                                     d_histogram,
                                                     num_levels,
                                                     d_levels,
                                                     num_row_pixels,
                                                     num_rows,
                                                     (OffsetT)(row_stride_bytes /
                                                               sizeof(SampleT)),
                                                     stream,
                                                     is_byte_sample);
  }

  template <int NUM_CHANNELS,
            int NUM_ACTIVE_CHANNELS,
            typename SampleIteratorT,
            typename CounterT,
            typename LevelT,
            typename OffsetT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  MultiHistogramRange(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      SampleIteratorT d_samples,
                      CounterT *d_histogram[NUM_ACTIVE_CHANNELS],
                      int num_levels[NUM_ACTIVE_CHANNELS],
                      LevelT *d_levels[NUM_ACTIVE_CHANNELS],
                      OffsetT num_row_pixels,
                      OffsetT num_rows,
                      size_t row_stride_bytes,
                      cudaStream_t stream,
                      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return MultiHistogramRange(d_temp_storage,
                               temp_storage_bytes,
                               d_samples,
                               d_histogram,
                               num_levels,
                               d_levels,
                               num_row_pixels,
                               num_rows,
                               row_stride_bytes,
                               stream);
  }

  //@}  end member group
};

CUB_NAMESPACE_END
