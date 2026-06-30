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
 * PTX intrinsics
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

#include <cub/util_debug.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Inlined PTX intrinsics
 ******************************************************************************/

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/**
 * Bitfield-extract.
 */
template <typename UnsignedBits, int BYTE_LEN>
//! deprecated [Since 3.0]
CCCL_DEPRECATED_BECAUSE("Use cuda::bitfield_extract()") _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int BFE(
  UnsignedBits source, unsigned int bit_start, unsigned int num_bits, detail::constant_t<BYTE_LEN> /*byte_len*/)
{
  unsigned int bits;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
  return bits;
}

/**
 * Bitfield-extract for 64-bit types.
 */
template <typename UnsignedBits>
//! deprecated [Since 3.0]
CCCL_DEPRECATED_BECAUSE("Use cuda::bitfield_extract()") _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int BFE(
  UnsignedBits source, unsigned int bit_start, unsigned int num_bits, detail::constant_t<8> /*byte_len*/)
{
  const unsigned long long MASK = (1ull << num_bits) - 1;
  return (source >> bit_start) & MASK;
}

#  if _CCCL_HAS_INT128()
/**
 * Bitfield-extract for 128-bit types.
 */
template <typename UnsignedBits>
//! deprecated [Since 3.0]
CCCL_DEPRECATED_BECAUSE("Use cuda::bitfield_extract()") _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int BFE(
  UnsignedBits source, unsigned int bit_start, unsigned int num_bits, detail::constant_t<16> /*byte_len*/)
{
  const __uint128_t MASK = (__uint128_t{1} << num_bits) - 1;
  return (source >> bit_start) & MASK;
}
#  endif

#endif // _CCCL_DOXYGEN_INVOKED

/**
 * \brief Bitfield-extract.  Extracts \p num_bits from \p source starting at bit-offset \p bit_start.  The input \p
 * source may be an 8b, 16b, 32b, or 64b unsigned integer type.
 */
template <typename UnsignedBits>
//! deprecated [Since 3.0]
CCCL_DEPRECATED_BECAUSE("Use cuda::bitfield_extract()") _CCCL_DEVICE
_CCCL_FORCEINLINE unsigned int BFE(UnsignedBits source, unsigned int bit_start, unsigned int num_bits)
{
  return BFE(source, bit_start, num_bits, detail::constant_v<int{sizeof(UnsignedBits)}>);
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/**
 * Warp synchronous shfl_up
 */
_CCCL_DEVICE _CCCL_FORCEINLINE unsigned int
SHFL_UP_SYNC(unsigned int word, int src_offset, int flags, unsigned int member_mask)
{
  asm volatile("shfl.sync.up.b32 %0, %1, %2, %3, %4;"
               : "=r"(word)
               : "r"(word), "r"(src_offset), "r"(flags), "r"(member_mask));
  return word;
}

/**
 * Warp synchronous shfl_down
 */
_CCCL_DEVICE _CCCL_FORCEINLINE unsigned int
SHFL_DOWN_SYNC(unsigned int word, int src_offset, int flags, unsigned int member_mask)
{
  asm volatile("shfl.sync.down.b32 %0, %1, %2, %3, %4;"
               : "=r"(word)
               : "r"(word), "r"(src_offset), "r"(flags), "r"(member_mask));
  return word;
}

#endif // _CCCL_DOXYGEN_INVOKED

/**
 * \brief Terminates the calling thread
 */
_CCCL_DEVICE _CCCL_FORCEINLINE void ThreadExit()
{
  asm volatile("exit;");
}

/**
 * \brief Returns the row-major linear thread identifier for a multidimensional thread block
 */
_CCCL_DEVICE _CCCL_FORCEINLINE int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z)
{
  return ((block_dim_z == 1) ? 0 : (threadIdx.z * block_dim_x * block_dim_y))
       + ((block_dim_y == 1) ? 0 : (threadIdx.y * block_dim_x)) + threadIdx.x;
}

/**
 * @brief Returns the warp mask for a warp of @p LOGICAL_WARP_THREADS threads
 *
 * @par
 * If the number of threads assigned to the virtual warp is not a power of two,
 * it's assumed that only one virtual warp exists.
 *
 * @tparam LOGICAL_WARP_THREADS <b>[optional]</b> The number of threads per
 *                              "logical" warp (may be less than the number of
 *                              hardware warp threads).
 * @param warp_id Id of virtual warp within architectural warp
 */
template <int LOGICAL_WARP_THREADS>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE unsigned int WarpMask([[maybe_unused]] unsigned int warp_id)
{
  constexpr bool is_pow_of_two = PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE;
  constexpr bool is_arch_warp  = LOGICAL_WARP_THREADS == detail::warp_threads;

  unsigned int member_mask = 0xFFFFFFFFu >> (detail::warp_threads - LOGICAL_WARP_THREADS);

  if constexpr (is_pow_of_two && !is_arch_warp)
  {
    member_mask <<= warp_id * LOGICAL_WARP_THREADS;
  }

  return member_mask;
}

/**
 * @brief Shuffle-up for any data type.
 *        Each <em>warp-lane<sub>i</sub></em> obtains the value @p input contributed by
 *        <em>warp-lane</em><sub><em>i</em>-<tt>src_offset</tt></sub>.
 *        For thread lanes @e i < src_offset, the thread's own @p input is returned to the thread.
 *        ![](shfl_up_logo.png)
 *
 * @tparam LOGICAL_WARP_THREADS
 *   The number of threads per "logical" warp. Must be a power-of-two <= 32.
 *
 * @tparam T
 *   <b>[inferred]</b> The input/output element type
 *
 * @par
 * - Available only for SM3.0 or newer
 *
 * @par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from the
 * predecessor of its predecessor.
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from two ranks below
 *     double peer_data = ShuffleUp<32>(thread_data, 2, 0, 0xffffffff);
 *
 * @endcode
 * @par
 * Suppose the set of input @p thread_data across the first warp of threads is
 * <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>. The corresponding output @p peer_data will be
 * <tt>{1.0, 2.0, 1.0, 2.0, 3.0, ..., 30.0}</tt>.
 *
 * @param[in] input
 *   The value to broadcast
 *
 * @param[in] src_offset
 *   The relative down-offset of the peer to read from
 *
 * @param[in] first_thread
 *   Index of first lane in logical warp (typically 0)
 *
 * @param[in] member_mask
 *   32-bit mask of participating warp lanes
 */
template <int LOGICAL_WARP_THREADS, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE T ShuffleUp(T input, int src_offset, int first_thread, unsigned int member_mask)
{
  /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
  enum
  {
    SHFL_C = (32 - LOGICAL_WARP_THREADS) << 8
  };

  using ShuffleWord = typename UnitWord<T>::ShuffleWord;

  constexpr int WORDS = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

  T output;
  ShuffleWord* output_alias = reinterpret_cast<ShuffleWord*>(&output);
  ShuffleWord* input_alias  = reinterpret_cast<ShuffleWord*>(&input);

  unsigned int shuffle_word;
  shuffle_word    = SHFL_UP_SYNC((unsigned int) input_alias[0], src_offset, first_thread | SHFL_C, member_mask);
  output_alias[0] = shuffle_word;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int WORD = 1; WORD < WORDS; ++WORD)
  {
    shuffle_word       = SHFL_UP_SYNC((unsigned int) input_alias[WORD], src_offset, first_thread | SHFL_C, member_mask);
    output_alias[WORD] = shuffle_word;
  }

  return output;
}

/**
 * @brief Shuffle-down for any data type.
 *        Each <em>warp-lane<sub>i</sub></em> obtains the value @p input contributed by
 *        <em>warp-lane</em><sub><em>i</em>+<tt>src_offset</tt></sub>.
 *        For thread lanes @e i >= WARP_THREADS, the thread's own @p input is returned to the
 *        thread. ![](shfl_down_logo.png)
 *
 * @tparam LOGICAL_WARP_THREADS
 *   The number of threads per "logical" warp.  Must be a power-of-two <= 32.
 *
 * @tparam T
 *   <b>[inferred]</b> The input/output element type
 *
 * @par
 * - Available only for SM3.0 or newer
 *
 * @par Snippet
 * The code snippet below illustrates each thread obtaining a @p double value from the
 * successor of its successor.
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from two ranks below
 *     double peer_data = ShuffleDown<32>(thread_data, 2, 31, 0xffffffff);
 *
 * @endcode
 * @par
 * Suppose the set of input @p thread_data across the first warp of threads is
 * <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output @p peer_data will be
 * <tt>{3.0, 4.0, 5.0, 6.0, 7.0, ..., 32.0}</tt>.
 *
 * @param[in] input
 *   The value to broadcast
 *
 * @param[in] src_offset
 *   The relative up-offset of the peer to read from
 *
 * @param[in] last_thread
 *   Index of last thread in logical warp (typically 31 for a 32-thread warp)
 *
 * @param[in] member_mask
 *   32-bit mask of participating warp lanes
 */
template <int LOGICAL_WARP_THREADS, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE T ShuffleDown(T input, int src_offset, int last_thread, unsigned int member_mask)
{
  /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
  enum
  {
    SHFL_C = (32 - LOGICAL_WARP_THREADS) << 8
  };

  using ShuffleWord = typename UnitWord<T>::ShuffleWord;

  constexpr int WORDS = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

  T output;
  ShuffleWord* output_alias = reinterpret_cast<ShuffleWord*>(&output);
  ShuffleWord* input_alias  = reinterpret_cast<ShuffleWord*>(&input);

  unsigned int shuffle_word;
  shuffle_word    = SHFL_DOWN_SYNC((unsigned int) input_alias[0], src_offset, last_thread | SHFL_C, member_mask);
  output_alias[0] = shuffle_word;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int WORD = 1; WORD < WORDS; ++WORD)
  {
    shuffle_word = SHFL_DOWN_SYNC((unsigned int) input_alias[WORD], src_offset, last_thread | SHFL_C, member_mask);
    output_alias[WORD] = shuffle_word;
  }

  return output;
}

/**
 * @brief Shuffle-broadcast for any data type.
 *        Each <em>warp-lane<sub>i</sub></em> obtains the value @p input
 *        contributed by <em>warp-lane</em><sub><tt>src_lane</tt></sub>.
 *        For @p src_lane < 0 or @p src_lane >= WARP_THREADS,
 *        then the thread's own @p input is returned to the thread.
 *        ![](shfl_broadcast_logo.png)
 *
 * @tparam LOGICAL_WARP_THREADS
 *   The number of threads per "logical" warp.  Must be a power-of-two <= 32.
 *
 * @tparam T
 *   <b>[inferred]</b> The input/output element type
 *
 * @par
 * - Available only for SM3.0 or newer
 *
 * @par Snippet
 * The code snippet below illustrates each thread obtaining a @p double value from
 * <em>warp-lane</em><sub>0</sub>.
 *
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from thread 0
 *     double peer_data = ShuffleIndex<32>(thread_data, 0, 0xffffffff);
 *
 * @endcode
 * @par
 * Suppose the set of input @p thread_data across the first warp of threads is
 * <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output @p peer_data will be
 * <tt>{1.0, 1.0, 1.0, 1.0, 1.0, ..., 1.0}</tt>.
 *
 * @param[in] input
 *   The value to broadcast
 *
 * @param[in] src_lane
 *   Which warp lane is to do the broadcasting
 *
 * @param[in] member_mask
 *   32-bit mask of participating warp lanes
 */
template <int LOGICAL_WARP_THREADS, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE T ShuffleIndex(T input, int src_lane, unsigned int member_mask)
{
  using ShuffleWord = typename UnitWord<T>::ShuffleWord;

  constexpr int WORDS = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

  T output;
  ShuffleWord* output_alias = reinterpret_cast<ShuffleWord*>(&output);
  ShuffleWord* input_alias  = reinterpret_cast<ShuffleWord*>(&input);

  unsigned int shuffle_word;
  shuffle_word    = __shfl_sync(member_mask, (unsigned int) input_alias[0], src_lane, LOGICAL_WARP_THREADS);
  output_alias[0] = shuffle_word;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int WORD = 1; WORD < WORDS; ++WORD)
  {
    shuffle_word       = __shfl_sync(member_mask, (unsigned int) input_alias[WORD], src_lane, LOGICAL_WARP_THREADS);
    output_alias[WORD] = shuffle_word;
  }
  return output;
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{

/**
 * Implementation detail for `MatchAny`. It provides specializations for full and partial warps.
 * For partial warps, inactive threads must be masked out. This is done in the partial warp
 * specialization below.
 * Usage:
 * ```
 * // returns a mask of threads with the same 4 least-significant bits of `label`
 * // in a warp with 16 active threads
 * warp_matcher_t<4, 16>::match_any(label);
 *
 * // returns a mask of threads with the same 4 least-significant bits of `label`
 * // in a warp with 32 active threads (no extra work is done)
 * warp_matcher_t<4, 32>::match_any(label);
 * ```
 */
template <int LABEL_BITS, int WARP_ACTIVE_THREADS>
struct warp_matcher_t
{
  static _CCCL_DEVICE unsigned int match_any(unsigned int label)
  {
    return warp_matcher_t<LABEL_BITS, 32>::match_any(label) & ~(~0 << WARP_ACTIVE_THREADS);
  }
};

template <int LABEL_BITS>
struct warp_matcher_t<LABEL_BITS, warp_threads>
{
  // match.any.sync.b32 is slower when matching a few bits
  // using a ballot loop instead
  static _CCCL_DEVICE unsigned int match_any(unsigned int label)
  {
    unsigned int retval;

    // Extract masks of common threads for each bit
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int BIT = 0; BIT < LABEL_BITS; ++BIT)
    {
      unsigned int mask;
      unsigned int current_bit = 1 << BIT;
      asm("{\n"
          "    .reg .pred p;\n"
          "    and.b32 %0, %1, %2;"
          "    setp.ne.u32 p, %0, 0;\n"
          "    vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
          "    @!p not.b32 %0, %0;\n"
          "}\n"
          : "=r"(mask)
          : "r"(label), "r"(current_bit));

      // Remove peers who differ
      retval = (BIT == 0) ? mask : retval & mask;
    }

    return retval;
  }
};

/**
 * @brief Shifts @p val left by the amount specified by unsigned 32-bit value in @p num_bits. If @p
 * num_bits is larger than 32 bits, @p num_bits is clamped to 32.
 */
_CCCL_DEVICE _CCCL_FORCEINLINE uint32_t LogicShiftLeft(uint32_t val, uint32_t num_bits)
{
  uint32_t ret{};
  asm("shl.b32 %0, %1, %2;" : "=r"(ret) : "r"(val), "r"(num_bits));
  return ret;
}

/**
 * @brief Shifts @p val right by the amount specified by unsigned 32-bit value in @p num_bits. If @p
 * num_bits is larger than 32 bits, @p num_bits is clamped to 32.
 */
_CCCL_DEVICE _CCCL_FORCEINLINE uint32_t LogicShiftRight(uint32_t val, uint32_t num_bits)
{
  uint32_t ret{};
  asm("shr.b32 %0, %1, %2;" : "=r"(ret) : "r"(val), "r"(num_bits));
  return ret;
}

} // namespace detail
#endif // _CCCL_DOXYGEN_INVOKED

/**
 * Compute a 32b mask of threads having the same least-significant
 * LABEL_BITS of \p label as the calling thread.
 */
template <int LABEL_BITS, int WARP_ACTIVE_THREADS = detail::warp_threads>
inline _CCCL_DEVICE unsigned int MatchAny(unsigned int label)
{
  return detail::warp_matcher_t<LABEL_BITS, WARP_ACTIVE_THREADS>::match_any(label);
}

CUB_NAMESPACE_END
