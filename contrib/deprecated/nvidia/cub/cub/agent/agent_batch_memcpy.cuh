/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
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
 * cub::AgentBatchMemcpy implements device-wide copying of a batch of device-accessible
 * source-buffers to device-accessible destination-buffers.
 */

#pragma once
#pragma clang system_header


#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_run_length_decode.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <bool PTR_IS_FOUR_BYTE_ALIGNED>
__forceinline__ __device__ void LoadVectorAndFunnelShiftR(uint32_t const *aligned_ptr,
                                                          uint32_t bit_shift,
                                                          uint4 &data_out)
{
  data_out = {aligned_ptr[0], aligned_ptr[1], aligned_ptr[2], aligned_ptr[3]};

  if (!PTR_IS_FOUR_BYTE_ALIGNED)
  {
    uint32_t tail = aligned_ptr[4];
    data_out.x    = __funnelshift_r(data_out.x, data_out.y, bit_shift);
    data_out.y    = __funnelshift_r(data_out.y, data_out.z, bit_shift);
    data_out.z    = __funnelshift_r(data_out.z, data_out.w, bit_shift);
    data_out.w    = __funnelshift_r(data_out.w, tail, bit_shift);
  }
}

template <bool PTR_IS_FOUR_BYTE_ALIGNED>
__forceinline__ __device__ void LoadVectorAndFunnelShiftR(uint32_t const *aligned_ptr,
                                                          uint32_t bit_shift,
                                                          uint2 &data_out)
{
  data_out = {aligned_ptr[0], aligned_ptr[1]};

  if (!PTR_IS_FOUR_BYTE_ALIGNED)
  {
    uint32_t tail = aligned_ptr[2];
    data_out.x    = __funnelshift_r(data_out.x, data_out.y, bit_shift);
    data_out.y    = __funnelshift_r(data_out.y, tail, bit_shift);
  }
}

template <bool PTR_IS_FOUR_BYTE_ALIGNED>
__forceinline__ __device__ void LoadVectorAndFunnelShiftR(uint32_t const *aligned_ptr,
                                                          uint32_t bit_shift,
                                                          uint32_t &data_out)
{
  data_out = aligned_ptr[0];

  if (!PTR_IS_FOUR_BYTE_ALIGNED)
  {
    uint32_t tail = aligned_ptr[1];
    data_out      = __funnelshift_r(data_out, tail, bit_shift);
  }
}

/**
 * @brief Loads data from \p ptr into \p data_out without requiring \p ptr to be aligned.
 * @note If \p ptr isn't aligned to four bytes, the bytes from the last four-byte aligned address up
 * to \p ptr are loaded too (but dropped) and, hence, need to be device-accessible. Similarly, if
 * \p ptr isn't aligned to four bytes, the bytes from `(ptr + sizeof(VectorT))` up to the following
 * four-byte aligned address are loaded too (but dropped), and, hence, need to be device-accessible.
 *
 * @tparam VectorT The vector type used for vectorized stores (i.e., one of uint4, uint2, uint32_t)
 * @param ptr The pointer from which the data is supposed to be loaded
 * @param data_out The vector type that stores the data loaded from \p ptr
 */
template <typename VectorT>
__forceinline__ __device__ void LoadVector(const char *ptr, VectorT &data_out)
{
  const uint32_t offset            = reinterpret_cast<std::uintptr_t>(ptr) % 4U;
  const uint32_t *aligned_ptr      = reinterpret_cast<uint32_t const *>(ptr - offset);
  constexpr uint32_t bits_per_byte = 8U;
  const uint32_t bit_shift         = offset * bits_per_byte;

  // If `ptr` is aligned to four bytes, we can perform a simple uint32_t-aliased load
  if (offset == 0)
  {
    LoadVectorAndFunnelShiftR<true>(aligned_ptr, bit_shift, data_out);
  }
  // Otherwise, we need to load extra bytes and perform funnel-shifting
  else
  {
    LoadVectorAndFunnelShiftR<false>(aligned_ptr, bit_shift, data_out);
  }
}

/**
 * @brief Helper data structure to hold information on the byte range for which we can safely
 * perform vectorized copies.
 *
 * @tparam VectorT The vector type used for vectorized stores (i.e., one of uint4, uint2, uint32_t)
 */
template <typename VectorT>
struct PointerRange
{
  VectorT *out_begin;
  VectorT *out_end;
  const char *in_begin;
  const char *in_end;
};

/**
 * @brief Both `out_start_aligned` and `out_end_aligned` are indices into `out_ptr`.
 * `out_start_aligned` is the first VectorT-aligned memory location after `out_ptr + 3`.
 * `out_end_aligned` is the last VectorT-aligned memory location before `out_end - 4`, where out_end
 * corresponds to one past the last byte to be copied. Bytes between `[out_start_aligned,
 * out_end_aligned)` will be copied using VectorT. `out_ptr + 3` and `out_end - 4` are used instead
 * of `out_ptr` and `out_end` to avoid `LoadVector` reading beyond data boundaries.
 *
 * @tparam VectorT The vector type used for vectorized stores (i.e., one of uint4, uint2, uint32_t)
 * @tparam ByteOffsetT Type used to index the bytes within the buffers
 * @param in_begin Pointer to the beginning of the byte range that shall be copied
 * @param out_begin Pointer to the beginning of the byte range that shall be copied
 * @param num_bytes Number of bytes that shall be copied
 * @return The byte range that can safely be copied using vectorized stores of type VectorT
 */
template <typename VectorT, typename ByteOffsetT>
__device__ __forceinline__ PointerRange<VectorT> GetAlignedPtrs(const void *in_begin,
                                                                void *out_begin,
                                                                ByteOffsetT num_bytes)
{
  // Data type size used for vectorized stores
  constexpr size_t out_datatype_size = sizeof(VectorT);
  // Data type size used for type-aliased loads
  constexpr size_t in_datatype_size = sizeof(uint32_t);

  // char-aliased ptrs to simplify pointer arithmetic
  char *out_ptr      = reinterpret_cast<char *>(out_begin);
  const char *in_ptr = reinterpret_cast<const char *>(in_begin);

  // Number of bytes between the first VectorT-aligned address at or before out_begin and out_begin
  const uint32_t alignment_offset = reinterpret_cast<std::uintptr_t>(out_ptr) % out_datatype_size;

  // The first VectorT-aligned address before (or at) out_begin
  char *out_chars_aligned = reinterpret_cast<char *>(out_ptr - alignment_offset);

  // The number of extra bytes preceding `in_ptr` that are loaded but dropped
  uint32_t in_extra_bytes = reinterpret_cast<std::uintptr_t>(in_ptr) % in_datatype_size;

  // The offset required by `LoadVector`:
  // If the input pointer is not aligned, we load data from the last aligned address preceding the
  // pointer. That is, loading up to (in_datatype_size-1) bytes before `in_ptr`
  uint32_t in_offset_req = in_extra_bytes;

  // Bytes after `out_chars_aligned` to the first VectorT-aligned address at or after `out_begin`
  uint32_t out_start_aligned =
    CUB_QUOTIENT_CEILING(in_offset_req + alignment_offset, out_datatype_size) * out_datatype_size;

  // Compute the beginning of the aligned ranges (output and input pointers)
  VectorT *out_aligned_begin   = reinterpret_cast<VectorT *>(out_chars_aligned + out_start_aligned);
  const char *in_aligned_begin = in_ptr + (reinterpret_cast<char *>(out_aligned_begin) - out_ptr);

  // If the aligned range is not aligned for the input pointer, we load up to (in_datatype_size-1)
  // bytes after the last byte that is copied. That is, we always load four bytes up to the next
  // aligned input address at a time. E.g., if the last byte loaded is one byte past the last
  // aligned address we'll also load the three bytes after that byte.
  uint32_t in_extra_bytes_from_aligned =
    (reinterpret_cast<std::uintptr_t>(in_aligned_begin) % in_datatype_size);
  uint32_t in_end_padding_req = (in_datatype_size - in_extra_bytes_from_aligned) % in_datatype_size;

  // Bytes after `out_chars_aligned` to the last VectorT-aligned
  // address at (or before) `out_begin` + `num_bytes`
  uint32_t out_end_aligned{};
  if (in_end_padding_req + alignment_offset > num_bytes)
  {
    out_end_aligned = out_start_aligned;
  }
  else
  {
    out_end_aligned = (num_bytes - in_end_padding_req + alignment_offset) / out_datatype_size *
                      out_datatype_size;
  }

  VectorT *out_aligned_end   = reinterpret_cast<VectorT *>(out_chars_aligned + out_end_aligned);
  const char *in_aligned_end = in_ptr + (reinterpret_cast<char *>(out_aligned_end) - out_ptr);

  return {out_aligned_begin, out_aligned_end, in_aligned_begin, in_aligned_end};
}

/**
 * @brief Cooperatively copies \p num_bytes from \p src to \p dest using vectorized stores of type
 * \p VectorT for addresses within [dest, dest + num_bytes) that are aligned to \p VectorT. A
 * byte-wise copy is used for byte-ranges that are not aligned to \p VectorT.
 *
 * @tparam LOGICAL_WARP_SIZE The number of threads cooperaing to copy the data; all threads within
 * [0,  `LOGICAL_WARP_SIZE`) must invoke this method with the same arguments
 * @tparam VectorT The vector type used for vectorized stores (i.e., one of uint4, uint2, uint32_t)
 * @tparam ByteOffsetT Type used to index the bytes within the buffers
 * @param thread_rank The thread rank within the group that cooperates to copy the data must be
 * within [0, `LOGICAL_WARP_SIZE`)
 * @param dest Pointer to the memory location to copy to
 * @param num_bytes Number of bytes to copy
 * @param src Pointer to the memory location to copy from
 */
template <int LOGICAL_WARP_SIZE, typename VectorT, typename ByteOffsetT>
__device__ __forceinline__ void
VectorizedCopy(int32_t thread_rank, void *dest, ByteOffsetT num_bytes, const void *src)
{
  char *out_ptr      = reinterpret_cast<char *>(dest);
  const char *in_ptr = reinterpret_cast<const char *>(src);

  // Gets the byte range that can safely be copied using vectorized stores of type VectorT
  auto aligned_range = GetAlignedPtrs<VectorT>(src, dest, num_bytes);

  // If byte range for which we can use vectorized copies is empty -> use byte-wise copies
  if (aligned_range.out_end <= aligned_range.out_begin)
  {
    for (ByteOffsetT ichar = thread_rank; ichar < num_bytes; ichar += LOGICAL_WARP_SIZE)
    {
      out_ptr[ichar] = in_ptr[ichar];
    }
  }
  else
  {
    // Copy bytes in range `[dest, aligned_range.out_begin)`
    out_ptr += thread_rank;
    in_ptr += thread_rank;
    while (out_ptr < reinterpret_cast<char *>(aligned_range.out_begin))
    {
      *out_ptr = *in_ptr;
      out_ptr += LOGICAL_WARP_SIZE;
      in_ptr += LOGICAL_WARP_SIZE;
    }

    // Copy bytes in range `[aligned_range.out_begin, aligned_range.out_end)`
    VectorT *aligned_range_begin = aligned_range.out_begin + thread_rank;
    const char *in_aligned_begin = aligned_range.in_begin + thread_rank * sizeof(VectorT);
    while (aligned_range_begin < aligned_range.out_end)
    {
      VectorT data_in;
      LoadVector(in_aligned_begin, data_in);
      *aligned_range_begin = data_in;
      in_aligned_begin += sizeof(VectorT) * LOGICAL_WARP_SIZE;
      aligned_range_begin += LOGICAL_WARP_SIZE;
    }

    // Copy bytes in range `[aligned_range.out_end, dest + num_bytes)`.
    out_ptr = reinterpret_cast<char *>(aligned_range.out_end) + thread_rank;
    in_ptr  = aligned_range.in_end + thread_rank;
    while (out_ptr < reinterpret_cast<char *>(dest) + num_bytes)
    {
      *out_ptr = *in_ptr;
      out_ptr += LOGICAL_WARP_SIZE;
      in_ptr += LOGICAL_WARP_SIZE;
    }
  }
}

/**
 * @brief A helper class that allows threads to maintain multiple counters, where the counter that
 * shall be incremented can be addressed dynamically without incurring register spillage.
 *
 * @tparam NUM_ITEMS The number of counters to allocate
 * @tparam MAX_ITEM_VALUE The maximum count that must be supported.
 * @tparam PREFER_POW2_BITS Whether the number of bits to dedicate to each counter should be a
 * power-of-two. If enabled, this allows replacing integer multiplication with a bit-shift in
 * exchange for higher register pressure.
 * @tparam BackingUnitT The data type that is used to provide the bits of all the counters that
 * shall be allocated.
 */
template <uint32_t NUM_ITEMS,
          uint32_t MAX_ITEM_VALUE,
          bool PREFER_POW2_BITS,
          typename BackingUnitT = uint32_t>
class BitPackedCounter
{
private:
  /// The minimum number of bits required to represent all values from [0, MAX_ITEM_VALUE]
  static constexpr uint32_t MIN_BITS_PER_ITEM =
    (MAX_ITEM_VALUE == 0U) ? 1U : cub::Log2<static_cast<int32_t>(MAX_ITEM_VALUE + 1U)>::VALUE;

  /// The number of bits allocated for each item. For pre-Volta, we prefer a power-of-2 here to
  /// have the compiler replace costly integer multiplication with bit-shifting.
  static constexpr uint32_t BITS_PER_ITEM =
    PREFER_POW2_BITS ? (0x01ULL << (cub::Log2<static_cast<int32_t>(MIN_BITS_PER_ITEM)>::VALUE))
                     : MIN_BITS_PER_ITEM;

  /// The number of bits that each backing data type can store
  static constexpr uint32_t NUM_BITS_PER_UNIT = sizeof(BackingUnitT) * 8;

  /// The number of items that each backing data type can store
  static constexpr uint32_t ITEMS_PER_UNIT = NUM_BITS_PER_UNIT / BITS_PER_ITEM;

  /// The number of bits the backing data type is actually making use of
  static constexpr uint32_t USED_BITS_PER_UNIT = ITEMS_PER_UNIT * BITS_PER_ITEM;

  /// The number of backing data types required to store the given number of items
  static constexpr uint32_t NUM_TOTAL_UNITS = CUB_QUOTIENT_CEILING(NUM_ITEMS, ITEMS_PER_UNIT);

  /// This is the net number of bit-storage provided by each unit (remainder bits are unused)
  static constexpr uint32_t UNIT_MASK = (USED_BITS_PER_UNIT >= (8U * sizeof(uint32_t)))
                                          ? 0xFFFFFFFF
                                          : (0x01U << USED_BITS_PER_UNIT) - 1;
  /// This is the bit-mask for each item
  static constexpr uint32_t ITEM_MASK = (BITS_PER_ITEM >= (8U * sizeof(uint32_t)))
                                          ? 0xFFFFFFFF
                                          : (0x01U << BITS_PER_ITEM) - 1;

  //------------------------------------------------------------------------------
  // ACCESSORS
  //------------------------------------------------------------------------------
public:
  __device__ __forceinline__ uint32_t Get(uint32_t index) const
  {
    const uint32_t target_offset = index * BITS_PER_ITEM;
    uint32_t val                 = 0;

#pragma unroll
    for (uint32_t i = 0; i < NUM_TOTAL_UNITS; ++i)
    {
      // In case the bit-offset of the counter at <index> is larger than the bit range of the
      // current unit, the bit_shift amount will be larger than the bits provided by this unit. As
      // C++'s bit-shift has undefined behaviour if the bits being shifted exceed the operand width,
      // we use the PTX instruction `shr` to make sure behaviour is well-defined.
      // Negative bit-shift amounts wrap around in unsigned integer math and are ultimately clamped.
      const uint32_t bit_shift = target_offset - i * USED_BITS_PER_UNIT;
      val |= detail::LogicShiftRight(data[i], bit_shift) & ITEM_MASK;
    }
    return val;
  }

  __device__ __forceinline__ void Add(uint32_t index, uint32_t value)
  {
    const uint32_t target_offset = index * BITS_PER_ITEM;

#pragma unroll
    for (uint32_t i = 0; i < NUM_TOTAL_UNITS; ++i)
    {
      // In case the bit-offset of the counter at <index> is larger than the bit range of the
      // current unit, the bit_shift amount will be larger than the bits provided by this unit. As
      // C++'s bit-shift has undefined behaviour if the bits being shifted exceed the operand width,
      // we use the PTX instruction `shl` to make sure behaviour is well-defined.
      // Negative bit-shift amounts wrap around in unsigned integer math and are ultimately clamped.
      const uint32_t bit_shift = target_offset - i * USED_BITS_PER_UNIT;
      data[i] += detail::LogicShiftLeft(value, bit_shift) & UNIT_MASK;
    }
  }

  __device__ BitPackedCounter operator+(const BitPackedCounter &rhs) const
  {
    BitPackedCounter result;
#pragma unroll
    for (uint32_t i = 0; i < NUM_TOTAL_UNITS; ++i)
    {
      result.data[i] = data[i] + rhs.data[i];
    }
    return result;
  }

  //------------------------------------------------------------------------------
  // MEMBER VARIABLES
  //------------------------------------------------------------------------------
private:
  BackingUnitT data[NUM_TOTAL_UNITS] = {};
};

/**
 * Parameterizable tuning policy type for AgentBatchMemcpy
 */
template <uint32_t _BLOCK_THREADS,
          uint32_t _BUFFERS_PER_THREAD,
          uint32_t _TLEV_BYTES_PER_THREAD,
          bool _PREFER_POW2_BITS,
          uint32_t _BLOCK_LEVEL_TILE_SIZE>
struct AgentBatchMemcpyPolicy
{
  /// Threads per thread block
  static constexpr uint32_t BLOCK_THREADS = _BLOCK_THREADS;
  /// Items per thread (per tile of input)
  static constexpr uint32_t BUFFERS_PER_THREAD = _BUFFERS_PER_THREAD;
  /// The number of bytes that each thread will work on with each iteration of reading in bytes
  /// from one or more
  // source-buffers and writing them out to the respective destination-buffers.
  static constexpr uint32_t TLEV_BYTES_PER_THREAD = _TLEV_BYTES_PER_THREAD;
  /// Whether the BitPackedCounter should prefer allocating a power-of-2 number of bits per
  /// counter
  static constexpr uint32_t PREFER_POW2_BITS = _PREFER_POW2_BITS;
  /// BLEV tile size granularity
  static constexpr uint32_t BLOCK_LEVEL_TILE_SIZE = _BLOCK_LEVEL_TILE_SIZE;
};

template <typename AgentMemcpySmallBuffersPolicyT,
          typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename BufferOffsetT,
          typename BlevBufferSrcsOutItT,
          typename BlevBufferDstsOutItT,
          typename BlevBufferSizesOutItT,
          typename BlevBufferTileOffsetsOutItT,
          typename BlockOffsetT,
          typename BLevBufferOffsetTileState,
          typename BLevBlockOffsetTileState>
class AgentBatchMemcpy
{
private:
  //---------------------------------------------------------------------
  // CONFIGS / CONSTANTS
  //---------------------------------------------------------------------
  // Tuning policy-based configurations
  static constexpr uint32_t BLOCK_THREADS      = AgentMemcpySmallBuffersPolicyT::BLOCK_THREADS;
  static constexpr uint32_t BUFFERS_PER_THREAD = AgentMemcpySmallBuffersPolicyT::BUFFERS_PER_THREAD;
  static constexpr uint32_t TLEV_BYTES_PER_THREAD =
    AgentMemcpySmallBuffersPolicyT::TLEV_BYTES_PER_THREAD;
  static constexpr bool PREFER_POW2_BITS = AgentMemcpySmallBuffersPolicyT::PREFER_POW2_BITS;
  static constexpr uint32_t BLOCK_LEVEL_TILE_SIZE =
    AgentMemcpySmallBuffersPolicyT::BLOCK_LEVEL_TILE_SIZE;

  // Derived configs
  static constexpr uint32_t BUFFERS_PER_BLOCK       = BUFFERS_PER_THREAD * BLOCK_THREADS;
  static constexpr uint32_t TLEV_BUFFERS_PER_THREAD = BUFFERS_PER_THREAD;
  static constexpr uint32_t BLEV_BUFFERS_PER_THREAD = BUFFERS_PER_THREAD;

  static constexpr uint32_t WARP_LEVEL_THRESHOLD  = 128;
  static constexpr uint32_t BLOCK_LEVEL_THRESHOLD = 8 * 1024;

  static constexpr uint32_t BUFFER_STABLE_PARTITION = false;

  // Constants
  enum : uint32_t
  {
    TLEV_SIZE_CLASS = 0,
    WLEV_SIZE_CLASS,
    BLEV_SIZE_CLASS,
    NUM_SIZE_CLASSES,
  };

  //---------------------------------------------------------------------
  // TYPE DECLARATIONS
  //---------------------------------------------------------------------
  /// Internal load/store type. For byte-wise memcpy, a single-byte type
  using AliasT = char;

  /// Type that has to be sufficiently large to hold any of the buffers' sizes.
  /// The BufferSizeIteratorT's value type must be convertible to this type.
  using BufferSizeT = cub::detail::value_t<BufferSizeIteratorT>;

  /// Type used to index into the tile of buffers that this thread block is assigned to.
  using BlockBufferOffsetT = uint16_t;

  /// Internal type used to index into the bytes of and represent size of a TLEV buffer
  using TLevBufferSizeT = uint16_t;

  /**
   * @brief Helper struct to simplify BlockExchange within a single four-byte word
   */
  struct ZippedTLevByteAssignment
  {
    // The buffer id within this tile
    BlockBufferOffsetT tile_buffer_id;

    // Byte-offset within that buffer
    TLevBufferSizeT buffer_byte_offset;
  };

  /**
   * POD to keep track of <buffer_id, buffer_size> pairs after having partitioned this tile's
   * buffers by their size.
   */
  struct BufferTuple
  {
    // Size is only valid (and relevant) for buffers that are use thread-level collaboration
    TLevBufferSizeT size;

    // The buffer id relativ to this tile (i.e., the buffer id within this tile)
    BlockBufferOffsetT buffer_id;
  };

  // Load buffers in a striped arrangement if we do not want to performa a stable partitioning into
  // small, medium, and large buffers, otherwise load them in a blocked arrangement
  using BufferLoadT =
    BlockLoad<BufferSizeT,
              static_cast<int32_t>(BLOCK_THREADS),
              static_cast<int32_t>(BUFFERS_PER_THREAD),
              BUFFER_STABLE_PARTITION ? BLOCK_LOAD_WARP_TRANSPOSE : BLOCK_LOAD_STRIPED>;

  // A vectorized counter that will count the number of buffers that fall into each of the
  // size-classes. Where the size class representes the collaboration level that is required to
  // process a buffer. The collaboration level being either:
  //-> (1) TLEV (thread-level collaboration), requiring one or multiple threads but not a FULL warp
  // to collaborate
  //-> (2) WLEV (warp-level collaboration), requiring a full warp to collaborate on a buffer
  //-> (3) BLEV (block-level collaboration), requiring one or multiple thread blocks to collaborate
  // on a buffer */
  using VectorizedSizeClassCounterT =
    BitPackedCounter<NUM_SIZE_CLASSES, BUFFERS_PER_BLOCK, PREFER_POW2_BITS>;

  // Block-level scan used to compute the write offsets
  using BlockSizeClassScanT =
    cub::BlockScan<VectorizedSizeClassCounterT, static_cast<int32_t>(BLOCK_THREADS)>;

  //
  using BlockBLevTileCountScanT = cub::BlockScan<BlockOffsetT, static_cast<int32_t>(BLOCK_THREADS)>;

  // Block-level run-length decode algorithm to evenly distribute work of all buffers requiring
  // thread-level collaboration
  using BlockRunLengthDecodeT =
    cub::BlockRunLengthDecode<BlockBufferOffsetT,
                              static_cast<int32_t>(BLOCK_THREADS),
                              static_cast<int32_t>(TLEV_BUFFERS_PER_THREAD),
                              static_cast<int32_t>(TLEV_BYTES_PER_THREAD)>;

  using BlockExchangeTLevT = cub::BlockExchange<ZippedTLevByteAssignment,
                                                static_cast<int32_t>(BLOCK_THREADS),
                                                static_cast<int32_t>(TLEV_BYTES_PER_THREAD)>;

  using BLevBuffScanPrefixCallbackOpT =
    TilePrefixCallbackOp<BufferOffsetT, Sum, BLevBufferOffsetTileState>;
  using BLevBlockScanPrefixCallbackOpT =
    TilePrefixCallbackOp<BlockOffsetT, Sum, BLevBlockOffsetTileState>;

  //-----------------------------------------------------------------------------
  // SHARED MEMORY DECLARATIONS
  //-----------------------------------------------------------------------------
  struct _TempStorage
  {
    union
    {
      typename BufferLoadT::TempStorage load_storage;

      // Stage 1: histogram over the size classes in preparation for partitioning buffers by size
      typename BlockSizeClassScanT::TempStorage size_scan_storage;

      // Stage 2: Communicate the number ofer buffers requiring block-level collaboration
      typename BLevBuffScanPrefixCallbackOpT::TempStorage buffer_scan_callback;

      // Stage 3; batch memcpy buffers that require only thread-level collaboration
      struct
      {
        BufferTuple buffers_by_size_class[BUFFERS_PER_BLOCK];

        // Stage 3.1: Write buffers requiring block-level collaboration to queue
        union
        {
          struct
          {
            typename BLevBlockScanPrefixCallbackOpT::TempStorage block_scan_callback;
            typename BlockBLevTileCountScanT::TempStorage block_scan_storage;
          } blev;

          // Stage 3.3: run-length decode & block exchange for tlev
          // rld_state needs to be persistent across loop iterations (RunLengthDecode calls) and,
          // hence, cannot alias block_exchange_storage
          struct
          {
            typename BlockRunLengthDecodeT::TempStorage rld_state;
            typename BlockExchangeTLevT::TempStorage block_exchange_storage;
          } tlev;
        };
      } staged;
    };
    BufferOffsetT blev_buffer_offset;
  };

  //-----------------------------------------------------------------------------
  // PUBLIC TYPE MEMBERS
  //-----------------------------------------------------------------------------
public:
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //-----------------------------------------------------------------------------
  // PRIVATE MEMBER FUNCTIONS
  //-----------------------------------------------------------------------------
private:
  /// Shared storage reference
  _TempStorage &temp_storage;

  /**
   * @brief Loads this tile's buffers' sizes, without any guards (i.e., out-of-bounds checks)
   */
  __device__ __forceinline__ void
  LoadBufferSizesFullTile(BufferSizeIteratorT tile_buffer_sizes_it,
                          BufferSizeT (&buffer_sizes)[BUFFERS_PER_THREAD])
  {
    BufferLoadT(temp_storage.load_storage).Load(tile_buffer_sizes_it, buffer_sizes);
  }

  /**
   * @brief Loads this tile's buffers' sizes, making sure to read at most \p num_valid items.
   */
  __device__ __forceinline__ void
  LoadBufferSizesPartialTile(BufferSizeIteratorT tile_buffer_sizes_it,
                             BufferSizeT (&buffer_sizes)[BUFFERS_PER_THREAD],
                             BufferOffsetT num_valid)
  {
    // Out-of-bounds buffer items are initialized to '0', so those buffers will simply be ignored
    // later on
    constexpr BufferSizeT OOB_DEFAULT_BUFFER_SIZE = 0U;

    BufferLoadT(temp_storage.load_storage)
      .Load(tile_buffer_sizes_it, buffer_sizes, num_valid, OOB_DEFAULT_BUFFER_SIZE);
  }

  /**
   * @brief Computes the histogram over the number of buffers belonging to each of the three
   * size-classes (TLEV, WLEV, BLEV).
   */
  __device__ __forceinline__ VectorizedSizeClassCounterT
  GetBufferSizeClassHistogram(const BufferSizeT (&buffer_sizes)[BUFFERS_PER_THREAD])
  {
    VectorizedSizeClassCounterT vectorized_counters{};
#pragma unroll
    for (uint32_t i = 0; i < BUFFERS_PER_THREAD; i++)
    {
      // Whether to increment ANY of the buffer size classes at all
      const uint32_t increment = buffer_sizes[i] > 0 ? 1U : 0U;
      // Identify the buffer's size class
      uint32_t buffer_size_class = 0;
      buffer_size_class += buffer_sizes[i] > WARP_LEVEL_THRESHOLD ? 1U : 0U;
      buffer_size_class += buffer_sizes[i] > BLOCK_LEVEL_THRESHOLD ? 1U : 0U;

      // Increment the count of the respective size class
      vectorized_counters.Add(buffer_size_class, increment);
    }
    return vectorized_counters;
  }

  /**
   * @brief Scatters the buffers into the respective buffer's size-class partition.
   */
  __device__ __forceinline__ void
  PartitionBuffersBySize(const BufferSizeT (&buffer_sizes)[BUFFERS_PER_THREAD],
                         VectorizedSizeClassCounterT &vectorized_offsets,
                         BufferTuple (&buffers_by_size_class)[BUFFERS_PER_BLOCK])
  {
    // If we intend to perform a stable partitioning, the thread's buffer are in a blocked
    // arrangement, otherwise they are in a striped arrangement
    BlockBufferOffsetT buffer_id = BUFFER_STABLE_PARTITION ? (BUFFERS_PER_THREAD * threadIdx.x)
                                                           : (threadIdx.x);
    constexpr BlockBufferOffsetT BUFFER_STRIDE = BUFFER_STABLE_PARTITION
                                                   ? static_cast<BlockBufferOffsetT>(1)
                                                   : static_cast<BlockBufferOffsetT>(BLOCK_THREADS);

#pragma unroll
    for (uint32_t i = 0; i < BUFFERS_PER_THREAD; i++)
    {
      if (buffer_sizes[i] > 0)
      {
        uint32_t buffer_size_class = 0;
        buffer_size_class += buffer_sizes[i] > WARP_LEVEL_THRESHOLD ? 1U : 0U;
        buffer_size_class += buffer_sizes[i] > BLOCK_LEVEL_THRESHOLD ? 1U : 0U;
        const uint32_t write_offset         = vectorized_offsets.Get(buffer_size_class);
        buffers_by_size_class[write_offset] = {static_cast<TLevBufferSizeT>(buffer_sizes[i]),
                                               buffer_id};
        vectorized_offsets.Add(buffer_size_class, 1U);
      }
      buffer_id += BUFFER_STRIDE;
    }
  }

  /**
   * @brief Read in all the buffers that require block-level collaboration and put them to a queue
   * that will get picked up in a separate, subsequent kernel.
   */
  __device__ __forceinline__ void EnqueueBLEVBuffers(BufferTuple *buffers_by_size_class,
                                                     InputBufferIt tile_buffer_srcs,
                                                     OutputBufferIt tile_buffer_dsts,
                                                     BufferSizeIteratorT tile_buffer_sizes,
                                                     BlockBufferOffsetT num_blev_buffers,
                                                     BufferOffsetT tile_buffer_offset,
                                                     BufferOffsetT tile_id)
  {
    BlockOffsetT block_offset[BLEV_BUFFERS_PER_THREAD];
    // Read in the BLEV buffer partition (i.e., the buffers that require block-level collaboration)
    uint32_t blev_buffer_offset = threadIdx.x * BLEV_BUFFERS_PER_THREAD;
#pragma unroll
    for (uint32_t i = 0; i < BLEV_BUFFERS_PER_THREAD; i++)
    {
      if (blev_buffer_offset < num_blev_buffers)
      {
        BlockBufferOffsetT tile_buffer_id = buffers_by_size_class[blev_buffer_offset].buffer_id;
        block_offset[i]                   = CUB_QUOTIENT_CEILING(tile_buffer_sizes[tile_buffer_id],
                                               BLOCK_LEVEL_TILE_SIZE);
      }
      else
      {
        // Out-of-bounds buffers are assigned a tile count of '0'
        block_offset[i] = 0U;
      }
      blev_buffer_offset++;
    }

    if (tile_id == 0)
    {
      BlockOffsetT block_aggregate;
      BlockBLevTileCountScanT(temp_storage.staged.blev.block_scan_storage)
        .ExclusiveSum(block_offset, block_offset, block_aggregate);
      if (threadIdx.x == 0)
      {
        blev_block_scan_state.SetInclusive(0, block_aggregate);
      }
    }
    else
    {
      BLevBlockScanPrefixCallbackOpT blev_tile_prefix_op(
        blev_block_scan_state,
        temp_storage.staged.blev.block_scan_callback,
        Sum(),
        tile_id);
      BlockBLevTileCountScanT(temp_storage.staged.blev.block_scan_storage)
        .ExclusiveSum(block_offset, block_offset, blev_tile_prefix_op);
    }
    CTA_SYNC();

    // Read in the BLEV buffer partition (i.e., the buffers that require block-level collaboration)
    blev_buffer_offset = threadIdx.x * BLEV_BUFFERS_PER_THREAD;
#pragma unroll
    for (uint32_t i = 0; i < BLEV_BUFFERS_PER_THREAD; i++)
    {
      if (blev_buffer_offset < num_blev_buffers)
      {
        BlockBufferOffsetT tile_buffer_id = buffers_by_size_class[blev_buffer_offset].buffer_id;
        blev_buffer_srcs[tile_buffer_offset + blev_buffer_offset] =
          tile_buffer_srcs[tile_buffer_id];
        blev_buffer_dsts[tile_buffer_offset + blev_buffer_offset] =
          tile_buffer_dsts[tile_buffer_id];
        blev_buffer_sizes[tile_buffer_offset + blev_buffer_offset] =
          tile_buffer_sizes[tile_buffer_id];
        blev_buffer_tile_offsets[tile_buffer_offset + blev_buffer_offset] = block_offset[i];
        blev_buffer_offset++;
      }
    }
  }

  /**
   * @brief Read in all the buffers of this tile that require warp-level collaboration and copy
   * their bytes to the corresponding destination buffer
   */
  __device__ __forceinline__ void BatchMemcpyWLEVBuffers(BufferTuple *buffers_by_size_class,
                                                         InputBufferIt tile_buffer_srcs,
                                                         OutputBufferIt tile_buffer_dsts,
                                                         BufferSizeIteratorT tile_buffer_sizes,
                                                         BlockBufferOffsetT num_wlev_buffers)
  {
    const int32_t warp_id              = threadIdx.x / CUB_PTX_WARP_THREADS;
    const int32_t warp_lane            = threadIdx.x % CUB_PTX_WARP_THREADS;
    constexpr uint32_t WARPS_PER_BLOCK = BLOCK_THREADS / CUB_PTX_WARP_THREADS;

    for (BlockBufferOffsetT buffer_offset = warp_id; buffer_offset < num_wlev_buffers;
         buffer_offset += WARPS_PER_BLOCK)
    {
      const auto buffer_id = buffers_by_size_class[buffer_offset].buffer_id;
      detail::VectorizedCopy<CUB_PTX_WARP_THREADS, uint4>(warp_lane,
                                                          tile_buffer_dsts[buffer_id],
                                                          tile_buffer_sizes[buffer_id],
                                                          tile_buffer_srcs[buffer_id]);
    }
  }

  /**
   * @brief Read in all the buffers of this tile that require thread-level collaboration and copy
   * their bytes to the corresponding destination buffer
   */
  __device__ __forceinline__ void BatchMemcpyTLEVBuffers(BufferTuple *buffers_by_size_class,
                                                         InputBufferIt tile_buffer_srcs,
                                                         OutputBufferIt tile_buffer_dsts,
                                                         BlockBufferOffsetT num_tlev_buffers)
  {
    // Read in the buffers' ids that require thread-level collaboration (where buffer id is the
    // buffer within this tile)
    BlockBufferOffsetT tlev_buffer_ids[TLEV_BUFFERS_PER_THREAD];
    TLevBufferSizeT tlev_buffer_sizes[TLEV_BUFFERS_PER_THREAD];
    // Currently we do not go over the TLEV buffers in multiple iterations, so we need to make sure
    // we are able to be covered for the case that all our buffers are TLEV buffers
    static_assert(TLEV_BUFFERS_PER_THREAD >= BUFFERS_PER_THREAD,
                  "Unsupported confiugraiton: The number of 'thread-level buffers' must be at "
                  "least as large as the number of overall buffers being processed by each "
                  "thread.");

    // Read in the TLEV buffer partition (i.e., the buffers that require thread-level collaboration)
    uint32_t tlev_buffer_offset = threadIdx.x * TLEV_BUFFERS_PER_THREAD;

    // Pre-populate the buffer sizes to 0 (i.e. zero-padding towards the end) to ensure
    // out-of-bounds TLEV buffers will not be considered
#pragma unroll
    for (uint32_t i = 0; i < TLEV_BUFFERS_PER_THREAD; i++)
    {
      tlev_buffer_sizes[i] = 0;
    }

    // Assign TLEV buffers in a blocked arrangement (each thread is assigned consecutive TLEV
    // buffers)
#pragma unroll
    for (uint32_t i = 0; i < TLEV_BUFFERS_PER_THREAD; i++)
    {
      if (tlev_buffer_offset < num_tlev_buffers)
      {
        tlev_buffer_ids[i]   = buffers_by_size_class[tlev_buffer_offset].buffer_id;
        tlev_buffer_sizes[i] = buffers_by_size_class[tlev_buffer_offset].size;
      }
      tlev_buffer_offset++;
    }

    // Evenly distribute all the bytes that have to be copied from all the buffers that require
    // thread-level collaboration using BlockRunLengthDecode
    uint32_t num_total_tlev_bytes = 0U;
    BlockRunLengthDecodeT block_run_length_decode(temp_storage.staged.tlev.rld_state,
                                                  tlev_buffer_ids,
                                                  tlev_buffer_sizes,
                                                  num_total_tlev_bytes);

    // Run-length decode the buffers' sizes into a window buffer of limited size. This is repeated
    // until we were able to cover all the bytes of TLEV buffers
    uint32_t decoded_window_offset = 0U;
    while (decoded_window_offset < num_total_tlev_bytes)
    {
      BlockBufferOffsetT buffer_id[TLEV_BYTES_PER_THREAD];
      TLevBufferSizeT buffer_byte_offset[TLEV_BYTES_PER_THREAD];

      // Now we have a balanced assignment: buffer_id[i] will hold the tile's buffer id and
      // buffer_byte_offset[i] that buffer's byte that this thread supposed to copy
      block_run_length_decode.RunLengthDecode(buffer_id, buffer_byte_offset, decoded_window_offset);

      // Zip from SoA to AoS
      ZippedTLevByteAssignment zipped_byte_assignment[TLEV_BYTES_PER_THREAD];
#pragma unroll
      for (int32_t i = 0; i < TLEV_BYTES_PER_THREAD; i++)
      {
        zipped_byte_assignment[i] = {buffer_id[i], buffer_byte_offset[i]};
      }

      // Exchange from blocked to striped arrangement for coalesced memory reads and writes
      BlockExchangeTLevT(temp_storage.staged.tlev.block_exchange_storage)
        .BlockedToStriped(zipped_byte_assignment, zipped_byte_assignment);

      // Read in the bytes that this thread is assigned to
      constexpr uint32_t WINDOW_SIZE = (TLEV_BYTES_PER_THREAD * BLOCK_THREADS);
      const bool is_full_window      = decoded_window_offset + WINDOW_SIZE < num_total_tlev_bytes;
      if (is_full_window)
      {
        uint32_t absolute_tlev_byte_offset = decoded_window_offset + threadIdx.x;
        AliasT src_byte[TLEV_BYTES_PER_THREAD];
#pragma unroll
        for (int32_t i = 0; i < TLEV_BYTES_PER_THREAD; i++)
        {
          src_byte[i] = reinterpret_cast<const AliasT *>(
            tile_buffer_srcs[zipped_byte_assignment[i].tile_buffer_id])[zipped_byte_assignment[i]
                                                                          .buffer_byte_offset];
          absolute_tlev_byte_offset += BLOCK_THREADS;
        }
#pragma unroll
        for (int32_t i = 0; i < TLEV_BYTES_PER_THREAD; i++)
        {
          reinterpret_cast<AliasT *>(
            tile_buffer_dsts[zipped_byte_assignment[i].tile_buffer_id])[zipped_byte_assignment[i]
                                                                          .buffer_byte_offset] =
            src_byte[i];
        }
      }
      else
      {
        uint32_t absolute_tlev_byte_offset = decoded_window_offset + threadIdx.x;
#pragma unroll
        for (int32_t i = 0; i < TLEV_BYTES_PER_THREAD; i++)
        {
          if (absolute_tlev_byte_offset < num_total_tlev_bytes)
          {
            const AliasT src_byte = reinterpret_cast<const AliasT *>(
              tile_buffer_srcs[zipped_byte_assignment[i].tile_buffer_id])[zipped_byte_assignment[i]
                                                                            .buffer_byte_offset];
            reinterpret_cast<AliasT *>(
              tile_buffer_dsts[zipped_byte_assignment[i].tile_buffer_id])[zipped_byte_assignment[i]
                                                                            .buffer_byte_offset] =
              src_byte;
          }
          absolute_tlev_byte_offset += BLOCK_THREADS;
        }
      }

      decoded_window_offset += WINDOW_SIZE;

      // Ensure all threads finished collaborative BlockExchange so temporary storage can be reused
      // with next iteration
      CTA_SYNC();
    }
  }

  //-----------------------------------------------------------------------------
  // PUBLIC MEMBER FUNCTIONS
  //-----------------------------------------------------------------------------
public:
  __device__ __forceinline__ void ConsumeTile(BufferOffsetT tile_id)
  {
    // Offset into this tile's buffers
    BufferOffsetT buffer_offset = tile_id * BUFFERS_PER_BLOCK;

    // Indicates whether all of this tiles items are within bounds
    bool is_full_tile = buffer_offset + BUFFERS_PER_BLOCK < num_buffers;

    // Load the buffer sizes of this tile's buffers
    BufferSizeIteratorT tile_buffer_sizes_it = buffer_sizes_it + buffer_offset;
    BufferSizeT buffer_sizes[BUFFERS_PER_THREAD];
    if (is_full_tile)
    {
      LoadBufferSizesFullTile(tile_buffer_sizes_it, buffer_sizes);
    }
    else
    {
      LoadBufferSizesPartialTile(tile_buffer_sizes_it, buffer_sizes, num_buffers - buffer_offset);
    }

    // Ensure we can repurpose the BlockLoad's temporary storage
    CTA_SYNC();

    // Count how many buffers fall into each size-class
    VectorizedSizeClassCounterT size_class_histogram = GetBufferSizeClassHistogram(buffer_sizes);

    // Compute the prefix sum over the histogram
    VectorizedSizeClassCounterT size_class_agg = {};
    BlockSizeClassScanT(temp_storage.size_scan_storage)
      .ExclusiveSum(size_class_histogram, size_class_histogram, size_class_agg);

    // Ensure we can repurpose the scan's temporary storage for scattering the buffer ids
    CTA_SYNC();

    // Factor in the per-size-class counts / offsets
    // That is, WLEV buffer offset has to be offset by the TLEV buffer count and BLEV buffer offset
    // has to be offset by the TLEV+WLEV buffer count
    uint32_t buffer_count = 0U;
    for (uint32_t i = 0; i < NUM_SIZE_CLASSES; i++)
    {
      size_class_histogram.Add(i, buffer_count);
      buffer_count += size_class_agg.Get(i);
    }

    // Signal the number of BLEV buffers we're planning to write out
    BufferOffsetT buffer_exclusive_prefix = 0;
    if (tile_id == 0)
    {
      if (threadIdx.x == 0)
      {
        blev_buffer_scan_state.SetInclusive(tile_id, size_class_agg.Get(BLEV_SIZE_CLASS));
      }
      buffer_exclusive_prefix = 0;
    }
    else
    {
      BLevBuffScanPrefixCallbackOpT blev_buffer_prefix_op(blev_buffer_scan_state,
                                                          temp_storage.buffer_scan_callback,
                                                          Sum(),
                                                          tile_id);

      // Signal our partial prefix and wait for the inclusive prefix of previous tiles
      if (threadIdx.x < CUB_PTX_WARP_THREADS)
      {
        buffer_exclusive_prefix = blev_buffer_prefix_op(size_class_agg.Get(BLEV_SIZE_CLASS));
      }
    }
    if (threadIdx.x == 0)
    {
      temp_storage.blev_buffer_offset = buffer_exclusive_prefix;
    }

    // Ensure the prefix callback has finished using its temporary storage and that it can be reused
    // in the next stage
    CTA_SYNC();

    // Scatter the buffers into one of the three partitions (TLEV, WLEV, BLEV) depending on their
    // size
    PartitionBuffersBySize(buffer_sizes,
                           size_class_histogram,
                           temp_storage.staged.buffers_by_size_class);

    // Ensure all buffers have been partitioned by their size class AND
    // ensure that blev_buffer_offset has been written to shared memory
    CTA_SYNC();

    // TODO: think about prefetching tile_buffer_{srcs,dsts} into shmem
    InputBufferIt tile_buffer_srcs        = input_buffer_it + buffer_offset;
    OutputBufferIt tile_buffer_dsts       = output_buffer_it + buffer_offset;
    BufferSizeIteratorT tile_buffer_sizes = buffer_sizes_it + buffer_offset;

    // Copy block-level buffers
    EnqueueBLEVBuffers(
      &temp_storage.staged.buffers_by_size_class[size_class_agg.Get(TLEV_SIZE_CLASS) +
                                                 size_class_agg.Get(WLEV_SIZE_CLASS)],
      tile_buffer_srcs,
      tile_buffer_dsts,
      tile_buffer_sizes,
      size_class_agg.Get(BLEV_SIZE_CLASS),
      temp_storage.blev_buffer_offset,
      tile_id);

    // Ensure we can repurpose the temporary storage required by EnqueueBLEVBuffers
    CTA_SYNC();

    // Copy warp-level buffers
    BatchMemcpyWLEVBuffers(
      &temp_storage.staged.buffers_by_size_class[size_class_agg.Get(TLEV_SIZE_CLASS)],
      tile_buffer_srcs,
      tile_buffer_dsts,
      tile_buffer_sizes,
      size_class_agg.Get(WLEV_SIZE_CLASS));

    // Perform batch memcpy for all the buffers that require thread-level collaboration
    uint32_t num_tlev_buffers = size_class_agg.Get(TLEV_SIZE_CLASS);
    BatchMemcpyTLEVBuffers(temp_storage.staged.buffers_by_size_class,
                           tile_buffer_srcs,
                           tile_buffer_dsts,
                           num_tlev_buffers);
  }

  //-----------------------------------------------------------------------------
  // CONSTRUCTOR
  //-----------------------------------------------------------------------------
  __device__ __forceinline__ AgentBatchMemcpy(TempStorage &temp_storage,
                                              InputBufferIt input_buffer_it,
                                              OutputBufferIt output_buffer_it,
                                              BufferSizeIteratorT buffer_sizes_it,
                                              BufferOffsetT num_buffers,
                                              BlevBufferSrcsOutItT blev_buffer_srcs,
                                              BlevBufferDstsOutItT blev_buffer_dsts,
                                              BlevBufferSizesOutItT blev_buffer_sizes,
                                              BlevBufferTileOffsetsOutItT blev_buffer_tile_offsets,
                                              BLevBufferOffsetTileState blev_buffer_scan_state,
                                              BLevBlockOffsetTileState blev_block_scan_state)
      : temp_storage(temp_storage.Alias())
      , input_buffer_it(input_buffer_it)
      , output_buffer_it(output_buffer_it)
      , buffer_sizes_it(buffer_sizes_it)
      , num_buffers(num_buffers)
      , blev_buffer_srcs(blev_buffer_srcs)
      , blev_buffer_dsts(blev_buffer_dsts)
      , blev_buffer_sizes(blev_buffer_sizes)
      , blev_buffer_tile_offsets(blev_buffer_tile_offsets)
      , blev_buffer_scan_state(blev_buffer_scan_state)
      , blev_block_scan_state(blev_block_scan_state)
  {}

private:
  // Iterator providing the pointers to the source memory buffers
  InputBufferIt input_buffer_it;
  // Iterator providing the pointers to the destination memory buffers
  OutputBufferIt output_buffer_it;
  // Iterator providing the number of bytes to be copied for each pair of buffers
  BufferSizeIteratorT buffer_sizes_it;
  // The total number of buffer pairs
  BufferOffsetT num_buffers;
  // Output iterator to which the source pointers of the BLEV buffers are written
  BlevBufferSrcsOutItT blev_buffer_srcs;
  // Output iterator to which the destination pointers of the BLEV buffers are written
  BlevBufferDstsOutItT blev_buffer_dsts;
  // Output iterator to which the number of bytes to be copied of the BLEV buffers are written
  BlevBufferSizesOutItT blev_buffer_sizes;
  // Output iterator to which the mapping of tiles to BLEV buffers is written
  BlevBufferTileOffsetsOutItT blev_buffer_tile_offsets;
  // The single-pass prefix scan's tile state used for tracking the prefix sum over the number of
  // BLEV buffers
  BLevBufferOffsetTileState blev_buffer_scan_state;
  // The single-pass prefix scan's tile state used for tracking the prefix sum over tiles of BLEV
  // buffers
  BLevBlockOffsetTileState blev_block_scan_state;
};

} // namespace detail

CUB_NAMESPACE_END
