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
 * cub::WarpScanShfl provides SHFL-based variants of parallel prefix scan of items partitioned
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

#include <cub/thread/thread_operators.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/warp>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief WarpScanShfl provides SHFL-based variants of parallel prefix scan of items partitioned
 *        across a CUDA thread warp.
 *
 * @tparam T
 *   Data type being scanned
 *
 * @tparam LOGICAL_WARP_THREADS
 *   Number of threads per logical warp (must be a power-of-two)
 */
template <typename T, int LOGICAL_WARP_THREADS>
struct WarpScanShfl
{
  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  enum
  {
    /// Whether the logical warp size and the PTX warp size coincide
    IS_ARCH_WARP = (LOGICAL_WARP_THREADS == warp_threads),

    /// The number of warp scan steps
    STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    SHFL_C = (warp_threads - LOGICAL_WARP_THREADS) << 8
  };

  template <typename S>
  struct IntegerTraits
  {
    enum
    {
      /// Whether the data type is a small (32b or less) integer for which we can use a single SFHL instruction per
      /// exchange
      IS_SMALL_UNSIGNED =
        ::cuda::std::is_integral_v<S> && ::cuda::std::is_unsigned_v<S> && (sizeof(S) <= sizeof(unsigned int)),
    };
  };

  /// Shared memory storage layout type
  struct TempStorage
  {};

  //---------------------------------------------------------------------
  // Thread fields
  //---------------------------------------------------------------------

  /// Lane index in logical warp
  unsigned int lane_id;

  /// Logical warp index in 32-thread physical warp
  unsigned int warp_id;

  /// 32-thread physical warp member mask of logical warp
  unsigned int member_mask;

  //---------------------------------------------------------------------
  // Construction
  //---------------------------------------------------------------------

  /// Constructor
  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpScanShfl(TempStorage& /*temp_storage*/)
      : lane_id(::cuda::ptx::get_sreg_laneid())
      , warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {
    if (!IS_ARCH_WARP)
    {
      lane_id = lane_id % LOGICAL_WARP_THREADS;
    }
  }

  //---------------------------------------------------------------------
  // Inclusive scan steps
  //---------------------------------------------------------------------

  /**
   * @brief Inclusive prefix scan step (specialized for summation across int32 types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE int
  InclusiveScanStep(int input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    int output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .s32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.s32 r0, r0, %4;"
      "  mov.s32 %0, r0;"
      "}"
      : "=r"(output)
      : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across uint32 types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int
  InclusiveScanStep(unsigned int input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    unsigned int output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.u32 r0, r0, %4;"
      "  mov.u32 %0, r0;"
      "}"
      : "=r"(output)
      : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across fp32 types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE float
  InclusiveScanStep(float input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    float output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .f32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.f32 r0, r0, %4;"
      "  mov.f32 %0, r0;"
      "}"
      : "=f"(output)
      : "f"(input), "r"(offset), "r"(shfl_c), "f"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across unsigned long long types)
   *
   * @param[in]  input
   *   Calling thread's input item
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned long long
  InclusiveScanStep(unsigned long long input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    unsigned long long output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u64 r0;"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
      "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
      "  mov.b64 r0, {lo, hi};"
      "  @p add.u64 r0, r0, %4;"
      "  mov.u64 %0, r0;"
      "}"
      : "=l"(output)
      : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across long long types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE long long
  InclusiveScanStep(long long input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    long long output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .s64 r0;"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
      "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
      "  mov.b64 r0, {lo, hi};"
      "  @p add.s64 r0, r0, %4;"
      "  mov.s64 %0, r0;"
      "}"
      : "=l"(output)
      : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across fp64 types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE double
  InclusiveScanStep(double input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    double output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  .reg .f64 r0;"
      "  mov.b64 %0, %1;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.up.b32 lo|p, lo, %2, %3, %4;"
      "  shfl.sync.up.b32 hi|p, hi, %2, %3, %4;"
      "  mov.b64 r0, {lo, hi};"
      "  @p add.f64 %0, %0, r0;"
      "}"
      : "=d"(output)
      : "d"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));

    return output;
  }

  /*
  /// Inclusive prefix scan (specialized for ReduceBySegmentOp<::cuda::std::plus<>> across KeyValuePair<OffsetT, Value>
  /// types)
  template <typename Value, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE KeyValuePair<OffsetT, Value> InclusiveScanStep(
    KeyValuePair<OffsetT, Value> input, ///< [in] Calling thread's input item.
    ReduceBySegmentOp<::cuda::std::plus<>> scan_op, ///< [in] Binary scan operator
    int first_lane, ///< [in] Index of first lane in segment
    int offset) ///< [in] Up-offset to pull from
  {
    KeyValuePair<OffsetT, Value> output;
    output.value = InclusiveScanStep(
      input.value, ::cuda::std::plus<>{}, first_lane, offset, Int2Type<IntegerTraits<Value>::IS_SMALL_UNSIGNED>());
    output.key = InclusiveScanStep(
      input.key, ::cuda::std::plus<>{}, first_lane, offset, Int2Type<IntegerTraits<OffsetT>::IS_SMALL_UNSIGNED>());

    if (input.key > 0)
      output.value = input.value;

    return output;
  }
  */

  /**
   * @brief Inclusive prefix scan step (generic)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename _T, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE _T InclusiveScanStep(_T input, ScanOpT scan_op, int first_lane, int offset)
  {
    _T temp = ShuffleUp<LOGICAL_WARP_THREADS>(input, offset, first_lane, member_mask);

    // Perform scan op if from a valid peer
    _T output = scan_op(temp, input);
    if (static_cast<int>(lane_id) < first_lane + offset)
    {
      output = input;
    }

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for small integers size 32b or less)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   *
   * @param[in] is_small_unsigned
   *   Marker type indicating whether T is a small integer
   */
  template <typename _T, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE _T
  InclusiveScanStep(_T input, ScanOpT scan_op, int first_lane, int offset, ::cuda::std::true_type /*is_small_unsigned*/)
  {
    return InclusiveScanStep(input, scan_op, first_lane, offset);
  }

  /**
   * @brief Inclusive prefix scan step (specialized for types other than small integers size
   *        32b or less)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   *
   * @param[in] is_small_unsigned
   *   Marker type indicating whether T is a small integer
   */
  template <typename _T, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE _T InclusiveScanStep(
    _T input, ScanOpT scan_op, int first_lane, int offset, ::cuda::std::false_type /*is_small_unsigned*/)
  {
    return InclusiveScanStep(input, scan_op, first_lane, offset);
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
  _CCCL_DEVICE _CCCL_FORCEINLINE T Broadcast(T input, int src_lane)
  {
    return ShuffleIndex<LOGICAL_WARP_THREADS>(input, src_lane, member_mask);
  }

  //---------------------------------------------------------------------
  // Inclusive operations
  //---------------------------------------------------------------------

  /**
   * @brief Inclusive scan
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   */
  template <typename _T, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(_T input, _T& inclusive_output, ScanOpT scan_op)
  {
    inclusive_output = input;

    // Iterate scan steps
    int segment_first_lane = 0;

    // Iterate scan steps
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int STEP = 0; STEP < STEPS; STEP++)
    {
      inclusive_output = InclusiveScanStep(
        inclusive_output,
        scan_op,
        segment_first_lane,
        (1 << STEP),
        bool_constant_v<IntegerTraits<T>::IS_SMALL_UNSIGNED>);
    }
  }

  /**
   * @brief Inclusive scan, specialized for reduce-value-by-key
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   */
  template <typename KeyT, typename ValueT, typename ReductionOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(
    KeyValuePair<KeyT, ValueT> input, KeyValuePair<KeyT, ValueT>& inclusive_output, ReduceByKeyOp<ReductionOpT> scan_op)
  {
    inclusive_output = input;

    KeyT pred_key = ShuffleUp<LOGICAL_WARP_THREADS>(inclusive_output.key, 1, 0, member_mask);

    unsigned int ballot = __ballot_sync(member_mask, (pred_key != inclusive_output.key));

    // Mask away all lanes greater than ours
    ballot = ballot & ::cuda::ptx::get_sreg_lanemask_le();

    // Find index of first set bit
    int segment_first_lane = ::cuda::std::__bit_log2(ballot);

    // Iterate scan steps
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int STEP = 0; STEP < STEPS; STEP++)
    {
      inclusive_output.value = InclusiveScanStep(
        inclusive_output.value,
        scan_op.op,
        segment_first_lane,
        (1 << STEP),
        bool_constant_v<IntegerTraits<T>::IS_SMALL_UNSIGNED>);
    }
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
   *   Warp-wide aggregate reduction of input items
   */
  template <typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOpT scan_op, T& warp_aggregate)
  {
    InclusiveScan(input, inclusive_output, scan_op);

    // Grab aggregate from last warp lane
    warp_aggregate = ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive_output, LOGICAL_WARP_THREADS - 1, member_mask);
  }

  //---------------------------------------------------------------------
  // Get exclusive from inclusive
  //---------------------------------------------------------------------

  /**
   * @brief Update inclusive and exclusive using input and inclusive
   *
   * @param[in] input
   *
   * @param[out] inclusive
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
    exclusive = ShuffleUp<LOGICAL_WARP_THREADS>(inclusive, 1, 0, member_mask);
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
    exclusive = ShuffleUp<LOGICAL_WARP_THREADS>(inclusive, 1, 0, member_mask);

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
  Update(T input, T& inclusive, T& exclusive, T& warp_aggregate, ScanOpT scan_op, IsIntegerT is_integer)
  {
    warp_aggregate = ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive, LOGICAL_WARP_THREADS - 1, member_mask);
    Update(input, inclusive, exclusive, scan_op, is_integer);
  }

  /**
   * @brief Update inclusive, exclusive, and warp aggregate using input, inclusive, and initial
   *        value
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Update(
    T input, T& inclusive, T& exclusive, T& warp_aggregate, ScanOpT scan_op, T initial_value, IsIntegerT is_integer)
  {
    warp_aggregate = ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive, LOGICAL_WARP_THREADS - 1, member_mask);
    Update(input, inclusive, exclusive, scan_op, initial_value, is_integer);
  }
};
} // namespace detail

CUB_NAMESPACE_END
