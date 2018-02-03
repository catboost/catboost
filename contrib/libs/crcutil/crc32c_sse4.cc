// Copyright 2010 Google Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Implements CRC32C using Intel's SSE4 crc32 instruction.
// Uses _mm_crc32_u64/32/8 intrinsics if CRCUTIL_USE_MM_CRC32 is not zero,
// emilates intrinsics via CRC_WORD/CRC_BYTE otherwise.

#include "crc32c_sse4.h"

#include <util/system/compiler.h>

#if HAVE_I386 || HAVE_AMD64

namespace crcutil {

#define UPDATE_STRIPE_CRCS(index, block_size, num_stripes) do { \
  CRC_UPDATE_WORD(crc0, \
      reinterpret_cast<const size_t *>(src + \
          0 * CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes))[index]); \
  CRC_UPDATE_WORD(crc1, \
      reinterpret_cast<const size_t *>(src + \
          1 * CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes))[index]); \
  CRC_UPDATE_WORD(crc2, \
      reinterpret_cast<const size_t *>(src + \
          2 * CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes))[index]); \
  if (num_stripes > 3) { \
    CRC_UPDATE_WORD(crc3, \
        reinterpret_cast<const size_t *>(src + \
            3 * CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes))[index]); \
  } \
} while (0)

// Multiplies "crc" by "x**(8 *  STRIPE_SIZE(block_size)"
// using appropriate multiplication table(s).
//
#if 0

// This variant is for illustration purposes only.
// Actual implementation below:
// 1. Splits the computation into 2 data-independent paths
//    by independently multiplying lower and upper halves
//    of "crc0" in interleaved manner, and combining the
//    results in the end.
// 2. Removing redundant "crc0 = 0" etc. in the beginning.
// 3. Removing redundant shifts of "tmp0" and "tmp1" in the last round.
#define MULTIPLY_CRC(crc0, block_size, num_stripes) do { \
  size_t tmp0 = crc0; \
  crc0 = 0; \
  for (size_t i = 0; i < kNumTables; ++i) { \
    crc0 ^= CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
            [i][tmp0 & (kTableEntries - 1)]; \
    tmp0 >>= kTableEntryBits; \
  } \
} while (0)

#else

#define MULTIPLY_CRC(crc0, block_size, num_stripes) do { \
  size_t tmp0 = crc0; \
  size_t tmp1 = crc0 >> (kTableEntryBits * kNumTablesHalfHi); \
  crc0 = CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
         [0][tmp0 & (kTableEntries - 1)]; \
  tmp0 >>= kTableEntryBits; \
  size_t crc1 = CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
                [kNumTablesHalfHi][tmp1 & (kTableEntries - 1)]; \
  tmp1 >>= kTableEntryBits; \
  for (size_t i = 1; i < kNumTablesHalfLo - 1; ++i) { \
    crc0 ^= CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
            [i][tmp0 & (kTableEntries - 1)]; \
    tmp0 >>= kTableEntryBits; \
    crc1 ^= CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
            [i + kNumTablesHalfHi][tmp1 & (kTableEntries - 1)]; \
    tmp1 >>= kTableEntryBits; \
  } \
  crc0 ^= CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
          [kNumTablesHalfLo - 1][tmp0 & (kTableEntries - 1)]; \
  if (kNumTables & 1) { \
    tmp0 >>= kTableEntryBits; \
  } \
  crc1 ^= CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
          [kNumTables - 1][tmp1]; \
  if (kNumTables & 1) { \
    crc0 ^= CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
            [kNumTablesHalfLo][tmp0 & (kTableEntries - 1)]; \
  } \
  crc0 ^= crc1; \
} while (0)

#endif

// Given CRCs (crc0, crc1, etc.) of consequitive
// stripes of STRIPE_SIZE(block_size) bytes each,
// produces CRC of concatenated stripes.
#define COMBINE_STRIPE_CRCS(block_size, num_stripes) do { \
  MULTIPLY_CRC(crc0, block_size, num_stripes); \
  crc0 ^= crc1; \
  MULTIPLY_CRC(crc0, block_size, num_stripes); \
  crc0 ^= crc2; \
  if (num_stripes > 3) { \
    MULTIPLY_CRC(crc0, block_size, num_stripes); \
    crc0 ^= crc3; \
  } \
} while (0)

// Processes input BLOCK_SIZE(block) bytes per iteration
// by splitting a block of BLOCK_SIZE(block) bytes into N
// equally-sized stripes of STRIPE_SIZE(block_size) each,
// computing CRC of each stripe, and concatenating stripe CRCs.
#define PROCESS_BLOCK(block_size, num_stripes) do { \
  while (bytes >= CRC32C_SSE4_BLOCK_SIZE(block_size, num_stripes)) { \
    Crc crc1 = 0; \
    Crc crc2 = 0; \
    Crc crc3; \
    if (num_stripes > 3) crc3 = 0; \
    { \
      const uint8 *stripe_end = src + \
          (CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes) / \
              kUnrolledLoopBytes) * kUnrolledLoopBytes; \
      do { \
        UPDATE_STRIPE_CRCS(0, block_size, num_stripes); \
        UPDATE_STRIPE_CRCS(1, block_size, num_stripes); \
        UPDATE_STRIPE_CRCS(2, block_size, num_stripes); \
        UPDATE_STRIPE_CRCS(3, block_size, num_stripes); \
        UPDATE_STRIPE_CRCS(4, block_size, num_stripes); \
        UPDATE_STRIPE_CRCS(5, block_size, num_stripes); \
        UPDATE_STRIPE_CRCS(6, block_size, num_stripes); \
        UPDATE_STRIPE_CRCS(7, block_size, num_stripes); \
        src += kUnrolledLoopBytes; \
      } while (src < stripe_end); \
      if ((CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes) % \
          kUnrolledLoopBytes) != 0) { \
        stripe_end += \
            CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes) % \
                kUnrolledLoopBytes; \
        do { \
          UPDATE_STRIPE_CRCS(0, block_size, num_stripes); \
          src += sizeof(size_t); \
        } while (src < stripe_end); \
      } \
    } \
    COMBINE_STRIPE_CRCS(block_size, num_stripes); \
    src += CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes) * \
           ((num_stripes) - 1); \
    bytes = static_cast<size_t>(end - src); \
  } \
 no_more_##block_size##_##num_stripes:; \
} while (0)

Y_NO_SANITIZE("undefined")
size_t Crc32cSSE4::Crc32c(const void *data, size_t bytes, Crc crc0) const {
  const uint8 *src = static_cast<const uint8 *>(data);
  const uint8 *end = src + bytes;
  crc0 ^= Base().Canonize();

  // If we don't have too much data to process,
  // do not waste time trying to align input etc.
  // Noticeably improves performance on small inputs.
  if (bytes < 4 * sizeof(size_t)) goto less_than_4_size_t;
  if (bytes < 8 * sizeof(size_t)) goto less_than_8_size_t;
  if (bytes < 16 * sizeof(size_t)) goto less_than_16_size_t;

#define PROCESS_TAIL_IF_SMALL(block_size, num_stripes) do { \
  if (bytes < CRC32C_SSE4_BLOCK_SIZE(block_size, num_stripes)) { \
    goto no_more_##block_size##_##num_stripes; \
  } \
} while (0)
#define NOOP(block_size, num_stripes)

  CRC32C_SSE4_ENUMERATE_ALL_BLOCKS_ASCENDING(PROCESS_TAIL_IF_SMALL,
                                             NOOP,
                                             NOOP);

#undef PROCESS_TAIL_IF_SMALL


  // Do not use ALIGN_ON_WORD_BOUNDARY_IF_NEEDED() here because:
  // 1. It uses CRC_BYTE() which won't work.
  // 2. Its threshold may be incorrect becuase Crc32 that uses
  //    native CPU crc32 instruction is much faster than
  //    generic table-based CRC computation.
  //
  // In case of X5550 CPU, break even point is at 2KB -- exactly.
  if (bytes >= 2 * 1024) {
    while ((reinterpret_cast<size_t>(src) & (sizeof(Word) - 1)) != 0) {
      if (src >= end) {
        return (crc0 ^ Base().Canonize());
      }
      CRC_UPDATE_BYTE(crc0, src[0]);
      src += 1;
    }
    bytes = static_cast<size_t>(end - src);
  }
  if (src >= end) {
    return (crc0 ^ Base().Canonize());
  }

  // Quickly skip processing of too large blocks
  // Noticeably improves performance on small inputs.
#define SKIP_BLOCK_IF_NEEDED(block_size, num_stripes) do { \
  if (bytes < CRC32C_SSE4_BLOCK_SIZE(block_size, num_stripes)) { \
    goto no_more_##block_size##_##num_stripes; \
  } \
} while (0)

  CRC32C_SSE4_ENUMERATE_ALL_BLOCKS_ASCENDING(NOOP,
                                             SKIP_BLOCK_IF_NEEDED,
                                             SKIP_BLOCK_IF_NEEDED);

#undef SKIP_BLOCK_IF_NEEDED

  // Process data in all blocks.
  CRC32C_SSE4_ENUMERATE_ALL_BLOCKS_DESCENDING(PROCESS_BLOCK,
                                              PROCESS_BLOCK,
                                              PROCESS_BLOCK);

  // Finish the tail word-by-word and then byte-by-byte.
#define CRC_UPDATE_WORD_4(index) do { \
  CRC_UPDATE_WORD(crc0, reinterpret_cast<const size_t *>(src)[index]); \
  CRC_UPDATE_WORD(crc0, reinterpret_cast<const size_t *>(src)[index + 1]); \
  CRC_UPDATE_WORD(crc0, reinterpret_cast<const size_t *>(src)[index + 2]); \
  CRC_UPDATE_WORD(crc0, reinterpret_cast<const size_t *>(src)[index + 3]); \
} while (0)

  if (bytes >= 4 * 4 * sizeof(size_t)) {
    end -= 4 * 4 * sizeof(size_t);
    do {
      CRC_UPDATE_WORD_4(4 * 0);
      CRC_UPDATE_WORD_4(4 * 1);
      CRC_UPDATE_WORD_4(4 * 2);
      CRC_UPDATE_WORD_4(4 * 3);
      src += 4 * 4 * sizeof(size_t);
    } while (src <= end);
    end += 4 * 4 * sizeof(size_t);
    bytes = static_cast<size_t>(end - src);
  }
 less_than_16_size_t:

  if (bytes >= 4 * 2 * sizeof(size_t)) {
    CRC_UPDATE_WORD_4(4 * 0);
    CRC_UPDATE_WORD_4(4 * 1);
    src += 4 * 2 * sizeof(size_t);
    bytes -= 4 * 2 * sizeof(size_t);
  }
 less_than_8_size_t:

  if (bytes >= 4 * sizeof(size_t)) {
    CRC_UPDATE_WORD_4(0);
    src += 4 * sizeof(size_t);
    bytes -= 4 * sizeof(size_t);
  }
 less_than_4_size_t:

  if (bytes >= 1 * sizeof(size_t)) {
    end -= 1 * sizeof(size_t);
    do {
      CRC_UPDATE_WORD(crc0, reinterpret_cast<const size_t *>(src)[0]);
      src += 1 * sizeof(size_t);
    } while (src <= end);
    end += 1 * sizeof(size_t);
  }

  while (src < end) {
    CRC_UPDATE_BYTE(crc0, src[0]);
    src += 1;
  }

  return (crc0 ^ Base().Canonize());
}


void Crc32cSSE4::Init(bool constant) {
  base_.Init(FixedGeneratingPolynomial(), FixedDegree(), constant);

#define INIT_MUL_TABLE(block_size, num_stripes) do { \
  size_t multiplier = \
      Base().Xpow8N(CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes)); \
  for (size_t table = 0; table < kNumTables; ++table) { \
    for (size_t entry = 0; entry < kTableEntries; ++entry) { \
      size_t value = static_cast<uint32>(entry << (kTableEntryBits * table)); \
      CRC32C_SSE4_MUL_TABLE(block_size, num_stripes)[table][entry] = \
            static_cast<Entry>(Base().Multiply(value, multiplier)); \
    } \
  } \
} while (0)

  CRC32C_SSE4_ENUMERATE_ALL_BLOCKS(INIT_MUL_TABLE);

#undef INIT_MUL_TABLE

#if !CRCUTIL_USE_MM_CRC32
  for (size_t j = 0; j < sizeof(Word); ++j) {
    Crc k = Base().XpowN((sizeof(Word) - 1 - j) * 8 + 32);
    for (size_t i = 0; i < 256; ++i) {
      crc_word_[j][i] = Base().MultiplyUnnormalized(i, 8, k);
    }
  }
#endif  // !CRCUTIL_USE_MM_CRC32
}


bool Crc32cSSE4::IsSSE42Available() {
#if defined(_MSC_VER)
  int cpu_info[4];
  __cpuid(cpu_info, 1);
  return ((cpu_info[2] & (1 << 20)) != 0);
#elif defined(__GNUC__) && (HAVE_AMD64 || HAVE_I386)
  // Not using "cpuid.h" intentionally: it is missing from
  // too many installations.
  uint32 eax;
  uint32 ecx;
  uint32 edx;
  __asm__ volatile(
#if HAVE_I386 && defined(__PIC__)
    "push %%ebx\n"
    "cpuid\n"
    "pop %%ebx\n"
#else
    "cpuid\n"
#endif  // HAVE_I386 && defined(__PIC__)
    : "=a" (eax), "=c" (ecx), "=d" (edx)
    : "a" (1), "2" (0)
    : "%ebx"
  );
  return ((ecx & (1 << 20)) != 0);
#else
  return false;
#endif
}


void RollingCrc32cSSE4::Init(const Crc32cSSE4 &crc,
                             size_t roll_window_bytes,
                             const Crc &start_value) {
  crc_ = &crc;
  roll_window_bytes_ = roll_window_bytes;
  start_value_ = start_value;

  Crc add = crc.Base().Canonize() ^ start_value;
  add = crc.Base().Multiply(add, crc.Base().Xpow8N(roll_window_bytes));
  add ^= crc.Base().Canonize();
  Crc mul = crc.Base().One() ^ crc.Base().Xpow8N(1);
  add = crc.Base().Multiply(add, mul);

  mul = crc.Base().XpowN(8 * roll_window_bytes + crc.Base().Degree());
  for (size_t i = 0; i < 256; ++i) {
    out_[i] = static_cast<Entry>(
                  crc.Base().MultiplyUnnormalized(
                      static_cast<Crc>(i), 8, mul) ^ add);
  }

#if !CRCUTIL_USE_MM_CRC32
  memcpy(crc_word_, crc_->crc_word_, sizeof(crc_word_));
#endif  // !CRCUTIL_USE_MM_CRC32
}

}  // namespace crcutil

#endif  // HAVE_I386 || HAVE_AMD64
