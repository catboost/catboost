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

// Defines GenericCrc class which implements arbitrary CRCs.
//
// Please read crc.pdf to understand how it all works.

#ifndef CRCUTIL_GENERIC_CRC_H_
#define CRCUTIL_GENERIC_CRC_H_

#include "base_types.h"     // uint8
#include "crc_casts.h"      // TO_BYTE(), Downcast<>.
#include "gf_util.h"        // GfUtil<Crc> class.
#include "platform.h"       // GCC_ALIGN_ATTRIBUTE(16)
#include "uint128_sse2.h"   // uint128_sse2 type (if necessary)

namespace crcutil {

#pragma pack(push, 16)

// Extends CRC by one byte.
// Technically, if degree of a polynomial does not exceed 8,
// right shift by 8 bits is not required, but who cares about CRC-8?
#define CRC_BYTE(table, crc, byte) do { \
  crc = ((sizeof(crc) > 1) ? SHIFT_RIGHT_SAFE(crc, 8) : 0) ^ \
        table->crc_word_[sizeof(Word) - 1][TO_BYTE(crc) ^ (byte)]; \
} while (0)

#define TABLE_ENTRY(table, byte, buf) \
  table[byte][Downcast<Word, uint8>(buf)]

#define TABLE_ENTRY_LAST(table, buf) \
  table[sizeof(Word) - 1][buf]

// Extends CRC by one word.
#define CRC_WORD(table, crc, buf) do { \
  buf ^= Downcast<Crc, Word>(crc); \
  if (sizeof(crc) > sizeof(buf)) { \
    crc = SHIFT_RIGHT_SAFE(crc, sizeof(buf) * 8); \
    crc ^= TABLE_ENTRY(table->crc_word_, 0, buf); \
  } else { \
    crc = TABLE_ENTRY(table->crc_word_, 0, buf); \
  } \
  buf >>= 8; \
  for (size_t byte = 1; byte < sizeof(buf) - 1; ++byte) { \
    crc ^= TABLE_ENTRY(table->crc_word_, byte, buf); \
    buf >>= 8; \
  } \
  crc ^= TABLE_ENTRY_LAST(table->crc_word_, buf); \
} while (0)

// Process beginning of data block byte by byte until source pointer
// becomes perfectly aligned on Word boundary.
#define ALIGN_ON_WORD_BOUNDARY(table, src, end, crc, Word) do { \
  while ((reinterpret_cast<size_t>(src) & (sizeof(Word) - 1)) != 0) { \
    if (src >= end) { \
      return (crc ^ table->Base().Canonize()); \
    } \
    CRC_BYTE(table, crc, *src); \
    src += 1; \
  } \
} while (0)


// On amd64, enforcing alignment is 2-4% slower on small (<= 64 bytes) blocks
// but 6-10% faster on larger blocks (>= 2KB).
// Break-even point (+-1%) is around 1KB (Q9650, E6600).
//
#define ALIGN_ON_WORD_BOUNDARY_IF_NEEDED(bytes, table, src, end, crc, Word) \
do { \
  if (sizeof(Word) > 8 || (bytes) > CRCUTIL_MIN_ALIGN_SIZE) { \
    ALIGN_ON_WORD_BOUNDARY(table, src, end, crc, Word); \
  } \
} while (0)

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127)  // conditional expression is constant
#endif  // defined(_MSC_VER)

// Forward declarations.
template<typename CrcImplementation> class RollingCrc;

// Crc        is the type used internally and to return values of N-bit CRC.
//            It should be at least as large as "TableEntry" and "Word" but
//            may be larger (e.g. for 16-bit CRC, TableEntry and Word may be
//            set to uint16 but Crc may be set to uint32).
//
// TableEntry is the type of values stored in the tables.
//            To implement N-bit CRC, TableEntry should be large enough
//            to store N bits.
//
// Word       is the type used to read data sizeof(Word) at a time.
//            Ideally, it shoulde be "most suitable for given architecture"
//            integer type -- typically "size_t".
//
// kStride    is the number of words processed in interleaved manner by
//            CrcMultiword() and CrcWordblock(). Shall be either 3 or 4.
//            Optimal value depends on hardware architecture (AMD64, ARM, etc).
//
template<typename _Crc, typename _TableEntry, typename _Word, int kStride>
    class GenericCrc {
 public:
  // Make Crc, TableEntry, and Word types visible (used by RollingCrc etc.)
  typedef _Crc Crc;
  typedef _TableEntry TableEntry;
  typedef _Word Word;

  GenericCrc() {}

  // Initializes the tables given generating polynomial of degree.
  // If "canonical" is true, crc value will be XOR'ed with (-1) before and
  // after actual CRC computation.
  GenericCrc(const Crc &generating_polynomial, size_t degree, bool canonical) {
    Init(generating_polynomial, degree, canonical);
  }
  void Init(const Crc &generating_polynomial, size_t degree, bool canonical) {
    base_.Init(generating_polynomial, degree, canonical);

    // Instead of computing
    //    table[j][i] = MultiplyUnnormalized(i, 8, k),
    // for all i = 0...255, we may notice that
    // if i = 2**n then for all m = 1...(i-1)
    // MultiplyUnnormalized(i + m, 8, k) =
    //    MultiplyUnnormalized(i ^ m, 8, k) =
    //    MultiplyUnnormalized(i, 8, k) ^ MultiplyUnnormalized(m, 8, k) =
    //    MultiplyUnnormalized(i, 8, k) ^ crc_word_interleaved[j][m] =
    //    table[i] ^ table[m].
#if 0
    for (size_t j = 0; j < sizeof(Word); ++j) {
      Crc k = Base().XpowN((sizeof(Word) * kStride - 1 - j) * 8 + degree);
      for (size_t i = 0; i < 256; ++i) {
        Crc temp = Base().MultiplyUnnormalized(static_cast<Crc>(i), 8, k);
        this->crc_word_interleaved_[j][i] = Downcast<Crc, TableEntry>(temp);
      }
    }
#else
    for (size_t j = 0; j < sizeof(Word); ++j) {
      Crc k = Base().XpowN((sizeof(Word) * kStride - 1 - j) * 8 + degree);
      TableEntry *table = this->crc_word_interleaved_[j];
      table[0] = 0;  // Init 0s entry -- multiply 0 by anything yields 0.
      for (size_t i = 1; i < 256; i <<= 1) {
        TableEntry value = Downcast<Crc, TableEntry>(
            Base().MultiplyUnnormalized(static_cast<Crc>(i), 8, k));
        table[i] = value;
        for (size_t m = 1; m < i; ++m) {
          table[i + m] = value ^ table[m];
        }
      }
    }
#endif

#if 0
    for (size_t j = 0; j < sizeof(Word); ++j) {
      Crc k = Base().XpowN((sizeof(Word) - 1 - j) * 8 + degree);
      for (size_t i = 0; i < 256; ++i) {
        Crc temp = Base().MultiplyUnnormalized(static_cast<Crc>(i), 8, k);
        this->crc_word_[j][i] = Downcast<Crc, TableEntry>(temp);
      }
    }
#else
    for (size_t j = 0; j < sizeof(Word); ++j) {
      Crc k = Base().XpowN((sizeof(Word) - 1 - j) * 8 + degree);
      TableEntry *table = this->crc_word_[j];
      table[0] = 0;  // Init 0s entry -- multiply 0 by anything yields 0.
      for (size_t i = 1; i < 256; i <<= 1) {
        TableEntry value = Downcast<Crc, TableEntry>(
            Base().MultiplyUnnormalized(static_cast<Crc>(i), 8, k));
        table[i] = value;
        for (size_t m = 1; m < i; ++m) {
          table[i + m] = value ^ table[m];
        }
      }
    }
#endif
  }

  // Default CRC implementation
  Crc CrcDefault(const void *data, size_t bytes, const Crc &start) const {
#if HAVE_AMD64 || HAVE_I386 || defined(__aarch64__)
    return CrcMultiword(data, bytes, start);
#else
    // Very few CPUs have multiple ALUs and speculative execution
    // (Itanium is an exception) so sophisticated algorithms will
    // not perform better than good old Sarwate algorithm.
    return CrcByteUnrolled(data, bytes, start);
#endif  // HAVE_AMD64 || HAVE_I386
  }

  // Returns base class.
  const GfUtil<Crc> &Base() const { return base_; }

 protected:
  // Canonical, byte-by-byte CRC computation.
  Crc CrcByte(const void *data, size_t bytes, const Crc &start) const {
    const uint8 *src = static_cast<const uint8 *>(data);
    Crc crc = start ^ Base().Canonize();
    for (const uint8 *end = src + bytes; src < end; ++src) {
      CRC_BYTE(this, crc, *src);
    }
    return (crc ^ Base().Canonize());
  }

  // Byte-by-byte CRC with main loop unrolled.
  Crc CrcByteUnrolled(const void *data, size_t bytes, const Crc &start) const {
    if (bytes == 0) {
      return start;
    }

    const uint8 *src = static_cast<const uint8 *>(data);
    const uint8 *end = src + bytes;
    Crc crc = start ^ Base().Canonize();

    // Unroll loop 4 times.
    end -= 3;
    for (; src < end; src += 4) {
      PREFETCH(src);
      CRC_BYTE(this, crc, src[0]);
      CRC_BYTE(this, crc, src[1]);
      CRC_BYTE(this, crc, src[2]);
      CRC_BYTE(this, crc, src[3]);
    }
    end += 3;

    // Compute CRC of remaining bytes.
    for (; src < end; ++src) {
      CRC_BYTE(this, crc, *src);
    }

    return (crc ^ Base().Canonize());
  }

  // Canonical, byte-by-byte CRC computation.
  Crc CrcByteWord(const void *data, size_t bytes, const Crc &start) const {
    const uint8 *src = static_cast<const uint8 *>(data);
    const uint8 *end = src + bytes;
    Crc crc0 = start ^ Base().Canonize();

    ALIGN_ON_WORD_BOUNDARY_IF_NEEDED(bytes, this, src, end, crc0, Crc);
    if (src >= end) {
      return (crc0 ^ Base().Canonize());
    }

    // Process 4*sizeof(Crc) bytes at a time.
    end -= 4 * sizeof(Crc) - 1;
    for (; src < end; src += 4 * sizeof(Crc)) {
      for (size_t i = 0; i < 4; ++i) {
        crc0 ^= reinterpret_cast<const Crc *>(src)[i];
        if (i == 0) {
          PREFETCH(src);
        }
        for (size_t byte = 0; byte < sizeof(crc0); ++byte) {
          CRC_BYTE(this, crc0, 0);
        }
      }
    }
    end += 4 * sizeof(Crc) - 1;

    // Process sizeof(Crc) bytes at a time.
    end -= sizeof(Crc) - 1;
    for (; src < end; src += sizeof(Crc)) {
      crc0 ^= reinterpret_cast<const Crc *>(src)[0];
      for (size_t byte = 0; byte < sizeof(crc0); ++byte) {
        CRC_BYTE(this, crc0, 0);
      }
    }
    end += sizeof(Crc) - 1;

    // Compute CRC of remaining bytes.
    for (;src < end; ++src) {
      CRC_BYTE(this, crc0, *src);
    }

    return (crc0 ^ Base().Canonize());
  }

  // Faster, word-by-word CRC.
  Crc CrcWord(const void *data, size_t bytes, const Crc &start) const {
    const uint8 *src = static_cast<const uint8 *>(data);
    const uint8 *end = src + bytes;
    Crc crc0 = start ^ Base().Canonize();

    ALIGN_ON_WORD_BOUNDARY_IF_NEEDED(bytes, this, src, end, crc0, Word);
    if (src >= end) {
      return (crc0 ^ Base().Canonize());
    }

    // Process 4 sizeof(Word) bytes at once.
    end -= 4 * sizeof(Word) - 1;
    for (; src < end; src += 4 * sizeof(Word)) {
      Word buf0 = reinterpret_cast<const Word *>(src)[0];
      PREFETCH(src);
      CRC_WORD(this, crc0, buf0);
      buf0 = reinterpret_cast<const Word *>(src)[1];
      CRC_WORD(this, crc0, buf0);
      buf0 = reinterpret_cast<const Word *>(src)[2];
      CRC_WORD(this, crc0, buf0);
      buf0 = reinterpret_cast<const Word *>(src)[3];
      CRC_WORD(this, crc0, buf0);
    }
    end += 4 * sizeof(Word) - 1;

    // Process sizeof(Word) bytes at a time.
    end -= sizeof(Word) - 1;
    for (; src < end; src += sizeof(Word)) {
      Word buf0 = reinterpret_cast<const Word *>(src)[0];
      CRC_WORD(this, crc0, buf0);
    }
    end += sizeof(Word) - 1;

    // Compute CRC of remaining bytes.
    for (;src < end; ++src) {
      CRC_BYTE(this, crc0, *src);
    }

    return (crc0 ^ Base().Canonize());
  }

#define REPEAT_FROM_1(macro) \
  macro(1); \
  macro(2); \
  macro(3); \
  macro(4); \
  macro(5); \
  macro(6); \
  macro(7);

#define REPEAT_FROM_0(macro) \
  macro(0); \
  REPEAT_FROM_1(macro)

  // Faster, process adjusent blocks in parallel and concatenate CRCs.
  Crc CrcBlockword(const void *data, size_t bytes, const Crc &start) const {
    if (kStride < 2 || kStride > 8) {
      // Unsupported configuration;
      // fall back to something sensible.
      return CrcWord(data, bytes, start);
    }

    const uint8 *src = static_cast<const uint8 *>(data);
    const uint8 *end = src + bytes;
    Crc crc0 = start ^ Base().Canonize();
    enum {
      // Add 16 to avoid false L1 cache collisions.
      kStripe = (15*1024 + 16) & ~(sizeof(Word) - 1),
    };

    ALIGN_ON_WORD_BOUNDARY_IF_NEEDED(bytes, this, src, end, crc0, Word);
    if (src >= end) {
      return (crc0 ^ Base().Canonize());
    }

    end -= kStride * kStripe - 1;
    if (src < end) {
      Crc x_pow_8kStripe = Base().Xpow8N(kStripe);
      do {
        const uint8 *stripe_end = src + kStripe;

#define INIT_CRC(reg) \
        Crc crc##reg; \
        if (kStride >= reg) { \
          crc##reg = 0; \
        }
        REPEAT_FROM_1(INIT_CRC);
#undef INIT_CRC

        do {
#define FIRST(reg) \
          Word buf##reg; \
          if (kStride > reg) { \
            buf##reg = reinterpret_cast<const Word *>(src + reg * kStripe)[0]; \
            buf##reg ^= Downcast<Crc, Word>(crc##reg); \
            if (sizeof(crc##reg) > sizeof(buf##reg)) { \
              crc##reg = SHIFT_RIGHT_SAFE(crc##reg, sizeof(buf##reg) * 8); \
              crc##reg ^= TABLE_ENTRY(this->crc_word_, 0, buf##reg); \
            } else { \
              crc##reg = TABLE_ENTRY(this->crc_word_, 0, buf##reg); \
            } \
            buf##reg >>= 8; \
          }
          REPEAT_FROM_0(FIRST);
#undef FIRST

          for (size_t byte = 1; byte < sizeof(buf0) - 1; ++byte) {
#define NEXT(reg) do { \
            if (kStride > reg) { \
              crc##reg ^= TABLE_ENTRY(this->crc_word_, byte, buf##reg); \
              buf##reg >>= 8; \
            } \
} while (0)
            REPEAT_FROM_0(NEXT);
#undef NEXT
          }

#define LAST(reg) do { \
          if (kStride > reg) { \
            crc##reg ^= TABLE_ENTRY_LAST(this->crc_word_, buf##reg); \
          } \
} while (0)
          REPEAT_FROM_0(LAST);
#undef LAST

          src += sizeof(Word);
        } while (src < stripe_end);

#if 0
// The code is left for illustrational purposes only.
#define COMBINE(reg) do { \
        if (reg > 0 && kStride > reg) { \
          crc0 = Base().ChangeStartValue(crc##reg, kStripe, 0, crc0); \
        } \
} while (0)
#else
#define COMBINE(reg) do { \
        if (reg > 0 && kStride > reg) { \
          crc0 = crc##reg ^ Base().Multiply(crc0, x_pow_8kStripe); \
        } \
} while (0)
#endif
        REPEAT_FROM_0(COMBINE);
#undef COMBINE

        src += (kStride - 1) * kStripe;
      }
      while (src < end);
    }
    end += kStride * kStripe - 1;

    // Process sizeof(Word) bytes at a time.
    end -= sizeof(Word) - 1;
    for (; src < end; src += sizeof(Word)) {
      Word buf0 = reinterpret_cast<const Word *>(src)[0];
      CRC_WORD(this, crc0, buf0);
    }
    end += sizeof(Word) - 1;

    // Compute CRC of remaining bytes.
    for (;src < end; ++src) {
      CRC_BYTE(this, crc0, *src);
    }

    return (crc0 ^ Base().Canonize());
  }

  // Fastest, interleaved multi-byte CRC.
  Crc CrcMultiword(const void *data, size_t bytes, const Crc &start) const {
    if (kStride < 2 || kStride > 8) {
      // Unsupported configuration;
      // fall back to something sensible.
      return CrcWord(data, bytes, start);
    }

    const uint8 *src = static_cast<const uint8 *>(data);
    const uint8 *end = src + bytes;
    Crc crc0 = start ^ Base().Canonize();

    ALIGN_ON_WORD_BOUNDARY_IF_NEEDED(bytes, this, src, end, crc0, Word);
    if (src >= end) {
      return (crc0 ^ Base().Canonize());
    }

    // Process kStride Word registers at once;
    // should have have at least 2*kInterleaveBytes of data to start.
    end -= 2*kInterleaveBytes - 1;
    if (src < end) {
      Crc crc_carryover;
      if (sizeof(Crc) > sizeof(Word)) {
        // crc_carryover is used if and only if Crc is wider than Word.
        crc_carryover = 0;
      }
#define INIT_CRC(reg) \
      Crc crc##reg; \
      if (reg > 0 && kStride > reg) { \
        crc##reg = 0; \
      }
      REPEAT_FROM_1(INIT_CRC);
#undef INIT_CRC

#define INIT_BUF(reg) \
      Word buf##reg; \
      if (kStride > reg) { \
        buf##reg = reinterpret_cast<const Word *>(src)[reg]; \
      }
      REPEAT_FROM_0(INIT_BUF);
#undef INIT_BUF

      do {
        PREFETCH(src);
        src += kInterleaveBytes;

        if (sizeof(Crc) > sizeof(Word)) {
          crc0 ^= crc_carryover;
        }

#define FIRST(reg, next_reg) do { \
        if (kStride > reg) { \
          buf##reg ^= Downcast<Crc, Word>(crc##reg); \
          if (sizeof(Crc) > sizeof(Word)) { \
            if (reg < kStride - 1) { \
              crc##next_reg ^= SHIFT_RIGHT_SAFE(crc##reg, 8 * sizeof(buf0)); \
            } else { \
              crc_carryover = SHIFT_RIGHT_SAFE(crc##reg, 8 * sizeof(buf0)); \
            } \
          } \
          crc##reg = TABLE_ENTRY(this->crc_word_interleaved_, 0, buf##reg); \
          buf##reg >>= 8; \
        } \
} while (0)
        FIRST(0, 1);
        FIRST(1, 2);
        FIRST(2, 3);
        FIRST(3, 4);
        FIRST(4, 5);
        FIRST(5, 6);
        FIRST(6, 7);
        FIRST(7, 0);
#undef FIRST

        for (size_t byte = 1; byte < sizeof(Word) - 1; ++byte) {
#define NEXT(reg) do { \
          if (kStride > reg) { \
            crc##reg ^= \
                TABLE_ENTRY(this->crc_word_interleaved_, byte, buf##reg); \
            buf##reg >>= 8; \
          } \
} while(0)
          REPEAT_FROM_0(NEXT);
#undef NEXT
        }

#define LAST(reg) do { \
        if (kStride > reg) { \
          crc##reg ^= TABLE_ENTRY_LAST(this->crc_word_interleaved_, buf##reg); \
          buf##reg = reinterpret_cast<const Word *>(src)[reg]; \
        } \
} while(0)
        REPEAT_FROM_0(LAST);
#undef LAST
      }
      while (src < end);

      if (sizeof(Crc) > sizeof(Word)) {
        crc0 ^= crc_carryover;
      }

#define COMBINE(reg) do { \
      if (kStride > reg) { \
        if (reg != 0) { \
          crc0 ^= crc##reg; \
        } \
        CRC_WORD(this, crc0, buf##reg); \
      } \
} while (0)
      REPEAT_FROM_0(COMBINE);
#undef COMBINE

      src += kInterleaveBytes;
    }
    end += 2*kInterleaveBytes - 1;

    // Process sizeof(Word) bytes at once.
    end -= sizeof(Word) - 1;
    for (; src < end; src += sizeof(Word)) {
      Word buf0 = reinterpret_cast<const Word *>(src)[0];
      CRC_WORD(this, crc0, buf0);
    }
    end += sizeof(Word) - 1;

    // Compute CRC of remaining bytes.
    for (;src < end; ++src) {
      CRC_BYTE(this, crc0, *src);
    }

    return (crc0 ^ Base().Canonize());
  }

 protected:
  enum {
    kInterleaveBytes = sizeof(Word) * kStride,
  };

  // Multiplication tables used by CRCs.
  TableEntry crc_word_interleaved_[sizeof(Word)][256];
  TableEntry crc_word_[sizeof(Word)][256];

  // Base class stored after CRC tables so that the most frequently
  // used table is at offset 0 and may be accessed faster.
  GfUtil<Crc> base_;

  friend class RollingCrc< GenericCrc<Crc, TableEntry, Word, kStride> >;

 private:
  // CrcMultiword on amd64 may run at 1.2 CPU cycles per byte which is
  // noticeably faster than CrcWord (2.2-2.6 cycles/byte depending on
  // hardware and compiler). However, there are problems with compilers.
  //
  // Test system: P45 chipset, Intel Q9650 CPU, 800MHz 4-4-4-12 memory.
  //
  // 64-bit compiler, <= 64-bit CRC, 64-bit tables, 64-bit reads:
  // CL 15.00.307291.1  C++   >1.2< CPU cycles/byte
  // ICL 11.1.051 -O3   C++    1.5  CPU cycles/byte
  // GCC 4.5 -O3        C++    2.0  CPU cycles/byte
  // GCC 4.x -O3        ASM   >1.2< CPU cycles/byte
  //
  // 32-bit compiler, MMX used, <= 64-bit CRC, 64-bit tables, 64-bit reads
  // CL 15.00.307291.1  C++   2.0  CPU cycles/byte
  // GCC 4.5 -O3        C++   1.9  CPU cycles/byte
  // ICL 11.1.051 -S    C++   1.6  CPU cycles/byte
  // GCC 4.x -O3        ASM  >1.3< CPU cycles/byte
  //
  // So, use inline ASM code for GCC for both i386 and amd64.

  Crc CrcMultiwordI386Mmx(
          const void *data, size_t bytes, const Crc &start) const;
  Crc CrcMultiwordGccAmd64(
          const void *data, size_t bytes, const Crc &start) const;
  Crc CrcMultiwordGccAmd64Sse2(
          const uint8 *src, const uint8 *end, const Crc &start) const;
} GCC_ALIGN_ATTRIBUTE(16);

#undef REPEAT_FROM_0
#undef REPEAT_FROM_1


// Specialized variants.
#if CRCUTIL_USE_ASM

#if (defined(__GNUC__) && (HAVE_AMD64 || (HAVE_I386 && HAVE_MMX)))

// Declare specialized functions.
template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiword(
    const void *data, size_t bytes, const uint64 &start) const;

#if HAVE_AMD64 && HAVE_SSE2
template<>
uint128_sse2
GenericCrc<uint128_sse2, uint128_sse2, uint64, 4>::CrcMultiword(
    const void *data, size_t bytes, const uint128_sse2 &start) const;
#endif  // HAVE_AMD64 && HAVE_SSE2

#elif defined(_MSC_FULL_VER) && _MSC_FULL_VER <= 150030729 && \
      (HAVE_I386 && HAVE_MMX)

// Work around bug in MSC (present at least in v. 15.00.30729.1)
template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiwordI386Mmx(
    const void *data,
    size_t bytes,
    const uint64 &start) const;
template<> __forceinline
uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiword(
    const void *data,
    size_t bytes,
    const uint64 &start) const {
  typedef uint64 Word;
  typedef uint64 Crc;
  if (bytes <= 12) {
    const uint8 *src = static_cast<const uint8 *>(data);
    uint64 crc = start ^ Base().Canonize();
    for (const uint8 *end = src + bytes; src < end; ++src) {
      CRC_BYTE(this, crc, *src);
    }
    return (crc ^ Base().Canonize());
  }
  return CrcMultiwordI386Mmx(data, bytes, start);
}

#endif  // (defined(__GNUC__) && (HAVE_AMD64 || (HAVE_I386 && HAVE_MMX)))

#endif  // CRCUTIL_USE_ASM


#pragma pack(pop)

}  // namespace crcutil

#endif  // CRCUTIL_GENERIC_CRC_H_
