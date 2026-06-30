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
 * cub::WarpScanSmem provides smem-based variants of parallel prefix scan of items partitioned
 * across a CUDA thread warp.
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

#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_type.cuh>

#include <cuda/ptx>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief WarpScanSmem provides smem-based variants of parallel prefix scan of items partitioned
 *        across a CUDA thread warp.
 *
 * @tparam T
 *   Data type being scanned
 *
 * @tparam LOGICAL_WARP_THREADS
 *   Number of threads per logical warp
 */
template <typename T, int LOGICAL_WARP_THREADS>
struct WarpScanSmem
{
  /******************************************************************************
   * Constants and type definitions
   ******************************************************************************/

  /// Whether the logical warp size and the PTX warp size coincide
  static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == warp_threads);

  /// The number of warp scan steps
  static constexpr int STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE;

  /// The number of threads in half a warp
  static constexpr int HALF_WARP_THREADS = 1 << (STEPS - 1);

  /// The number of shared memory elements per warp
  static constexpr int WARP_SMEM_ELEMENTS = LOGICAL_WARP_THREADS + HALF_WARP_THREADS;

  /// Storage cell type (workaround for SM1x compiler bugs with custom-ops like Max() on signed chars)
  using CellT = T;

  /// Shared memory storage layout type (1.5 warps-worth of elements for each warp)
  using _TempStorage = CellT[WARP_SMEM_ELEMENTS];

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  /******************************************************************************
   * Thread fields
   ******************************************************************************/

  _TempStorage& temp_storage;
  unsigned int lane_id;
  unsigned int member_mask;

  /******************************************************************************
   * Construction
   ******************************************************************************/

  /// Constructor
  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpScanSmem(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      ,

      lane_id(IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : ::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS)
      ,

      member_mask(WarpMask<LOGICAL_WARP_THREADS>(::cuda::ptx::get_sreg_laneid() / LOGICAL_WARP_THREADS))
  {}

  /******************************************************************************
   * Utility methods
   ******************************************************************************/

  /// Basic inclusive scan iteration (template unrolled, inductive-case specialization)
  template <bool HAS_IDENTITY, int STEP, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanStep(T& partial, ScanOp scan_op, constant_t<STEP> /*step*/)
  {
    constexpr int OFFSET = 1 << STEP;

    // Share partial into buffer
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], (CellT) partial);

    __syncwarp(member_mask);

    // Update partial if addend is in range
    if (HAS_IDENTITY || (lane_id >= OFFSET))
    {
      T addend = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - OFFSET]);
      partial  = scan_op(addend, partial);
    }
    __syncwarp(member_mask);

    ScanStep<HAS_IDENTITY>(partial, scan_op, constant_v<STEP + 1>);
  }

  /// Basic inclusive scan iteration(template unrolled, base-case specialization)
  template <bool HAS_IDENTITY, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanStep(T& /*partial*/, ScanOp /*scan_op*/, constant_t<STEPS> /*step*/)
  {}

  /**
   * @brief Inclusive prefix scan (specialized for summation across primitive types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in]
   *   Marker type indicating whether T is primitive type
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScan(T input, T& output, ::cuda::std::plus<> scan_op, ::cuda::std::true_type /*is_primitive*/)
  {
    T identity = 0;
    ThreadStore<STORE_VOLATILE>(&temp_storage[lane_id], (CellT) identity);

    __syncwarp(member_mask);

    // Iterate scan steps
    output = input;
    ScanStep<true>(output, scan_op, constant_v<0>);
  }

  /**
   * @brief Inclusive prefix scan
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] is_primitive
   *   Marker type indicating whether T is primitive type
   */
  template <typename ScanOp, bool IS_PRIMITIVE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScan(T input, T& output, ScanOp scan_op, ::cuda::std::bool_constant<IS_PRIMITIVE> /*is_primitive*/)
  {
    // Iterate scan steps
    output = input;
    ScanStep<false>(output, scan_op, constant_v<0>);
  }

  /******************************************************************************
   * Interface
   ******************************************************************************/

  //---------------------------------------------------------------------
  // Broadcast
  //---------------------------------------------------------------------

  /**
   * @brief Broadcast
   *
   * @param[in] input
   *   The value to broadcast
   *
   * @param[in] src_lane
   *   Which warp lane is to do the broadcasting
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE T Broadcast(T input, unsigned int src_lane)
  {
    if (lane_id == src_lane)
    {
      ThreadStore<STORE_VOLATILE>(temp_storage, (CellT) input);
    }

    __syncwarp(member_mask);

    return (T) ThreadLoad<LOAD_VOLATILE>(temp_storage);
  }

  //---------------------------------------------------------------------
  // Inclusive operations
  //---------------------------------------------------------------------

  /**
   * @brief Inclusive scan
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op)
  {
    InclusiveScan(input, inclusive_output, scan_op, bool_constant_v<is_primitive<T>::value>);
  }

  /**
   * @brief Inclusive scan with aggregate
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[out] warp_aggregate
   *   Warp-wide aggregate reduction of input items.
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op, T& warp_aggregate)
  {
    InclusiveScan(input, inclusive_output, scan_op);

    // Retrieve aggregate
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], (CellT) inclusive_output);

    __syncwarp(member_mask);

    warp_aggregate = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[WARP_SMEM_ELEMENTS - 1]);

    __syncwarp(member_mask);
  }

  //---------------------------------------------------------------------
  // Get exclusive from inclusive
  //---------------------------------------------------------------------

  /**
   * @brief Update inclusive and exclusive using input and inclusive
   *
   * @param[in] input
   *
   * @param[in, out] inclusive
   *
   * @param[out] exclusive
   *
   * @param[in] scan_op
   *
   * @param[in] is_integer
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Update(T /*input*/, T& inclusive, T& exclusive, ScanOpT /*scan_op*/, IsIntegerT /*is_integer*/)
  {
    // initial value unknown
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], (CellT) inclusive);

    __syncwarp(member_mask);

    exclusive = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 1]);
  }

  /**
   * @brief Update inclusive and exclusive using input and inclusive (specialized for summation of
   *        integer types)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Update(T input, T& inclusive, T& exclusive, ::cuda::std::plus<> /*scan_op*/, ::cuda::std::true_type /*is_integer*/)
  {
    // initial value presumed 0
    exclusive = inclusive - input;
  }

  /**
   * @brief Update inclusive and exclusive using initial value using input, inclusive, and initial
   *        value
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Update(T /*input*/, T& inclusive, T& exclusive, ScanOpT scan_op, T initial_value, IsIntegerT /*is_integer*/)
  {
    inclusive = scan_op(initial_value, inclusive);
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], (CellT) inclusive);

    __syncwarp(member_mask);

    exclusive = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 1]);
    if (lane_id == 0)
    {
      exclusive = initial_value;
    }
  }

  /**
   * @brief Update inclusive and exclusive using initial value using input and inclusive
   *        (specialized for summation of integer types)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void Update(
    T input,
    T& inclusive,
    T& exclusive,
    ::cuda::std::plus<> scan_op,
    T initial_value,
    ::cuda::std::true_type /*is_integer*/)
  {
    inclusive = scan_op(initial_value, inclusive);
    exclusive = inclusive - input;
  }

  /**
   * @brief Update inclusive, exclusive, and warp aggregate using input and inclusive
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Update(T /*input*/, T& inclusive, T& exclusive, T& warp_aggregate, ScanOpT /*scan_op*/, IsIntegerT /*is_integer*/)
  {
    // Initial value presumed to be unknown or identity (either way our padding is correct)
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], (CellT) inclusive);

    __syncwarp(member_mask);

    exclusive      = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 1]);
    warp_aggregate = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[WARP_SMEM_ELEMENTS - 1]);
  }

  /**
   * @brief Update inclusive, exclusive, and warp aggregate using input and inclusive (specialized
   *        for summation of integer types)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void Update(
    T input,
    T& inclusive,
    T& exclusive,
    T& warp_aggregate,
    ::cuda::std::plus<> /*scan_o*/,
    ::cuda::std::true_type /*is_integer*/)
  {
    // Initial value presumed to be unknown or identity (either way our padding is correct)
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], (CellT) inclusive);

    __syncwarp(member_mask);

    warp_aggregate = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[WARP_SMEM_ELEMENTS - 1]);
    exclusive      = inclusive - input;
  }

  /**
   * @brief Update inclusive, exclusive, and warp aggregate using input, inclusive, and initial
   *        value
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Update(
    T /*input*/,
    T& inclusive,
    T& exclusive,
    T& warp_aggregate,
    ScanOpT scan_op,
    T initial_value,
    IsIntegerT /*is_integer*/)
  {
    // Broadcast warp aggregate
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], (CellT) inclusive);

    __syncwarp(member_mask);

    warp_aggregate = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[WARP_SMEM_ELEMENTS - 1]);

    __syncwarp(member_mask);

    // Update inclusive with initial value
    inclusive = scan_op(initial_value, inclusive);

    // Get exclusive from exclusive
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 1], (CellT) inclusive);

    __syncwarp(member_mask);

    exclusive = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 2]);

    if (lane_id == 0)
    {
      exclusive = initial_value;
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
