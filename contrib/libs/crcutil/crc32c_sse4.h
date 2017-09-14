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

#ifndef CRCUTIL_CRC32C_SSE4_H_
#define CRCUTIL_CRC32C_SSE4_H_

#include "gf_util.h"              // base types, gf_util class, etc.
#include "crc32c_sse4_intrin.h"   // _mm_crc32_u* intrinsics

#if HAVE_I386 || HAVE_AMD64

#if CRCUTIL_USE_MM_CRC32

#if HAVE_I386
#define CRC_UPDATE_WORD(crc, value) (crc = _mm_crc32_u32(crc, (value)))
#else
#define CRC_UPDATE_WORD(crc, value) (crc = _mm_crc32_u64(crc, (value)))
#endif  // HAVE_I386

#define CRC_UPDATE_BYTE(crc, value) \
    (crc = _mm_crc32_u8(static_cast<uint32>(crc), static_cast<uint8>(value)))

#else

#include "generic_crc.h"

#define CRC_UPDATE_WORD(crc, value) do { \
  size_t buf = (value); \
  CRC_WORD(this, crc, buf); \
} while (0)
#define CRC_UPDATE_BYTE(crc, value) do { \
  CRC_BYTE(this, crc, (value)); \
} while (0)

#endif  // CRCUTIL_USE_MM_CRC32

namespace crcutil {

#pragma pack(push, 16)

// Since the same pieces should be parameterized in many different places
// and we do not want to introduce a mistake which is rather hard to find,
// use a macro to enumerate all block sizes.
//
// Block sizes and number of stripes were tuned for best performance.
//
// All constants should be literal constants (too lazy to fix the macro).
//
// The use of different "macro_first", "macro", and "macro_last"
// allows generation of different code for smallest, in between,
// and largest block sizes.
//
// This macro shall be kept in sync with
// CRC32C_SSE4_ENUMERATE_ALL_BLOCKS_DESCENDING.
// Failure to do so will cause compile-time error.
#define CRC32C_SSE4_ENUMERATE_ALL_BLOCKS_ASCENDING( \
    macro_smallest, macro, macro_largest) \
  macro_smallest(512, 3); \
  macro(1024, 3); \
  macro(4096, 3); \
  macro_largest(32768, 3)

// This macro shall be kept in sync with
// CRC32C_SSE4_ENUMERATE_ALL_BLOCKS_ASCENDING.
// Failure to do so will cause compile-time error.
#define CRC32C_SSE4_ENUMERATE_ALL_BLOCKS_DESCENDING( \
    macro_smallest, macro, macro_largest) \
  macro_largest(32768, 3); \
  macro(4096, 3); \
  macro(1024, 3); \
  macro_smallest(512, 3)

// Enumerates all block sizes.
#define CRC32C_SSE4_ENUMERATE_ALL_BLOCKS(macro) \
  CRC32C_SSE4_ENUMERATE_ALL_BLOCKS_ASCENDING(macro, macro, macro)

#define CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes) \
  (((block_size) / (num_stripes)) & ~(sizeof(size_t) - 1))

#define CRC32C_SSE4_BLOCK_SIZE(block_size, num_stripes) \
  (CRC32C_SSE4_STRIPE_SIZE(block_size, num_stripes) * (num_stripes))

#define CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
  mul_table_##block_size##_##num_blocks##_

class RollingCrc32cSSE4;

class Crc32cSSE4 {
 public:
  // Exports Crc, TableEntry, and Word (needed by RollingCrc).
  typedef size_t Crc;
  typedef Crc Word;
  typedef Crc TableEntry;

  Crc32cSSE4() {}

  // Initializes the tables given generating polynomial of degree (degree).
  // If "canonical" is true, crc value will be XOR'ed with (-1) before and
  // after actual CRC computation.
  explicit Crc32cSSE4(bool canonical) {
    Init(canonical);
  }
  void Init(bool canonical);

  // Initializes the tables given generating polynomial of degree.
  // If "canonical" is true, crc value will be XOR'ed with (-1) before and
  // after actual CRC computation.
  // Provided for compatibility with GenericCrc.
  Crc32cSSE4(const Crc &generating_polynomial,
            size_t degree,
            bool canonical) {
    Init(generating_polynomial, degree, canonical);
  }
  void Init(const Crc &generating_polynomial,
            size_t degree,
            bool canonical) {
    if (generating_polynomial == FixedGeneratingPolynomial() &&
        degree == FixedDegree()) {
      Init(canonical);
    }
  }

  // Returns fixed generating polymonial the class implements.
  static Crc FixedGeneratingPolynomial() {
    return 0x82f63b78;
  }

  // Returns degree of fixed generating polymonial the class implements.
  static Crc FixedDegree() {
    return 32;
  }

  // Returns base class.
  const GfUtil<Crc> &Base() const { return base_; }

  // Computes CRC32.
  size_t CrcDefault(const void *data, size_t bytes, const Crc &crc) const {
    return Crc32c(data, bytes, crc);
  }

  // Returns true iff crc32 instruction is available.
  static bool IsSSE42Available();

 protected:
  // Actual implementation.
  size_t Crc32c(const void *data, size_t bytes, Crc crc) const;

  enum {
    kTableEntryBits = 8,
    kTableEntries = 1 << kTableEntryBits,
    kNumTables = (32 + kTableEntryBits - 1) / kTableEntryBits,
    kNumTablesHalfLo = kNumTables / 2,
    kNumTablesHalfHi = (kNumTables + 1) / 2,

    kUnrolledLoopCount = 8,
    kUnrolledLoopBytes = kUnrolledLoopCount * sizeof(size_t),
  };

  // May be set to size_t or uint32, whichever is faster.
  typedef uint32 Entry;

#define DECLARE_MUL_TABLE(block_size, num_stripes) \
  Entry CRC32C_SSE4_MUL_TABLE(block_size, num_stripes) \
      [kNumTables][kTableEntries]

  CRC32C_SSE4_ENUMERATE_ALL_BLOCKS(DECLARE_MUL_TABLE);

#undef DECLARE_MUL_TABLE

  GfUtil<Crc> base_;

#if !CRCUTIL_USE_MM_CRC32
  TableEntry crc_word_[sizeof(Word)][256];
  friend class RollingCrc32cSSE4;
#endif  // !CRCUTIL_USE_MM_CRC32
} GCC_ALIGN_ATTRIBUTE(16);

class RollingCrc32cSSE4 {
 public:
  typedef Crc32cSSE4::Crc Crc;
  typedef Crc32cSSE4::TableEntry TableEntry;
  typedef Crc32cSSE4::Word Word;

  RollingCrc32cSSE4() {}

  // Initializes internal data structures.
  // Retains reference to "crc" instance -- it is used by Start().
  RollingCrc32cSSE4(const Crc32cSSE4 &crc,
            size_t roll_window_bytes,
            const Crc &start_value) {
    Init(crc, roll_window_bytes, start_value);
  }
  void Init(const Crc32cSSE4 &crc,
            size_t roll_window_bytes,
            const Crc &start_value);

  // Computes crc of "roll_window_bytes" using
  // "start_value" of "crc" (see Init()).
  Crc Start(const void *data) const {
    return crc_->CrcDefault(data, roll_window_bytes_, start_value_);
  }

  // Computes CRC of "roll_window_bytes" starting in next position.
  Crc Roll(const Crc &old_crc, size_t byte_out, size_t byte_in) const {
    Crc crc = old_crc;
    CRC_UPDATE_BYTE(crc, byte_in);
    crc ^= out_[byte_out];
    return crc;
  }

  // Returns start value.
  Crc StartValue() const { return start_value_; }

  // Returns length of roll window.
  size_t WindowBytes() const { return roll_window_bytes_; }

 protected:
  typedef Crc Entry;
  Entry out_[256];

  // Used only by Start().
  Crc start_value_;
  const Crc32cSSE4 *crc_;
  size_t roll_window_bytes_;

#if !CRCUTIL_USE_MM_CRC32
  TableEntry crc_word_[sizeof(Word)][256];
#endif  // !CRCUTIL_USE_MM_CRC32
} GCC_ALIGN_ATTRIBUTE(16);

#pragma pack(pop)

}  // namespace crcutil

#endif  // HAVE_I386 || HAVE_AMD64

#endif  // CRCUTIL_CRC32C_SSE4_H_
