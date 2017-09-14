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

// Implements multiword CRC for GCC on i386.
//
// Small comment: the trick described in
// http://software.intel.com/en-us/articles/fast-simd-integer-move-for-the-intel-pentiumr-4-processor
// (replace "movdqa dst, src" with "pshufd $0xE4, src, dst")
// did not work: execution time increased from
// 1.8 CPU cycles/byte to 2.1 CPU cycles/byte.
// So it may be good idea on P4 but it's not on newer CPUs.
//
// movaps/xorps vs. movdqa/pxor did not make any difference.

#include "generic_crc.h"
#include "uint128_sse2.h"

#if defined(__GNUC__) && CRCUTIL_USE_ASM && HAVE_AMD64 && HAVE_SSE2

namespace crcutil {

template<> uint128_sse2
GenericCrc<uint128_sse2, uint128_sse2, uint64, 4>::CrcMultiwordGccAmd64Sse2(
    const uint8 *src, const uint8 *end, const uint128_sse2 &start) const;

template<>
uint128_sse2 GenericCrc<uint128_sse2, uint128_sse2, uint64, 4>::CrcMultiword(
    const void *data, size_t bytes, const uint128_sse2 &start) const {
  const uint8 *src = static_cast<const uint8 *>(data);
  uint128_sse2 crc = start ^ this->Base().Canonize();
  const uint8 *end = src + bytes;
  if (bytes <= 7) {
    for (; src < end; ++src) {
      CRC_BYTE(this, crc, *src);
    }
    return (crc ^ this->Base().Canonize());
  }

  ALIGN_ON_WORD_BOUNDARY_IF_NEEDED(bytes, this, src, end, crc, uint64);
  if (src >= end) {
    return (crc ^ this->Base().Canonize());
  }

  return CrcMultiwordGccAmd64Sse2(src, end, crc);
}

#define CRC_WORD_ASM() \
    SSE2_MOVQ " %[crc0], %[tmp0]\n" \
    "xorq %[tmp0], %[buf0]\n" \
    "psrldq $8, %[crc0]\n" \
    "movzbq %b[buf0], %[tmp0]\n" \
    "shrq $8, %[buf0]\n" \
    "addq %[tmp0], %[tmp0]\n" \
    "pxor (%[table_word], %[tmp0], 8), %[crc0]\n" \
    "movzbq %b[buf0], %[tmp1]\n" \
    "shrq $8, %[buf0]\n" \
    "addq %[tmp1], %[tmp1]\n" \
    "pxor 1*256*16(%[table_word], %[tmp1], 8), %[crc0]\n" \
    "movzbq %b[buf0], %[tmp0]\n" \
    "shrq $8, %[buf0]\n" \
    "addq %[tmp0], %[tmp0]\n" \
    "pxor 2*256*16(%[table_word], %[tmp0], 8), %[crc0]\n" \
    "movzbq %b[buf0], %[tmp1]\n" \
    "shrq $8, %[buf0]\n" \
    "addq %[tmp1], %[tmp1]\n" \
    "pxor 3*256*16(%[table_word], %[tmp1], 8), %[crc0]\n" \
    "movzbq %b[buf0], %[tmp0]\n" \
    "shrq $8, %[buf0]\n" \
    "addq %[tmp0], %[tmp0]\n" \
    "pxor 4*256*16(%[table_word], %[tmp0], 8), %[crc0]\n" \
    "movzbq %b[buf0], %[tmp1]\n" \
    "shrq $8, %[buf0]\n" \
    "addq %[tmp1], %[tmp1]\n" \
    "pxor 5*256*16(%[table_word], %[tmp1], 8), %[crc0]\n" \
    "movzbq %b[buf0], %[tmp0]\n" \
    "shrq $8, %[buf0]\n" \
    "addq %[tmp0], %[tmp0]\n" \
    "pxor 6*256*16(%[table_word], %[tmp0], 8), %[crc0]\n" \
    "addq %[buf0], %[buf0]\n" \
    "pxor 7*256*16(%[table_word], %[buf0], 8), %[crc0]\n"

template<> uint128_sse2
GenericCrc<uint128_sse2, uint128_sse2, uint64, 4>::CrcMultiwordGccAmd64Sse2(
    const uint8 *src, const uint8 *end, const uint128_sse2 &start) const {
  __m128i crc0 = start;
  __m128i crc1;
  __m128i crc2;
  __m128i crc3;
  __m128i crc_carryover;

  uint64 buf0;
  uint64 buf1;
  uint64 buf2;
  uint64 buf3;

  uint64 tmp0;
  uint64 tmp1;

  asm(
    "sub $2*4*8 - 1, %[end]\n"
    "cmpq  %[src], %[end]\n"
    "jbe 2f\n"

    "pxor %[crc1], %[crc1]\n"
    "pxor %[crc2], %[crc2]\n"
    "pxor %[crc3], %[crc3]\n"
    "pxor %[crc_carryover], %[crc_carryover]\n"
    "movq (%[src]), %[buf0]\n"
    "movq 1*8(%[src]), %[buf1]\n"
    "movq 2*8(%[src]), %[buf2]\n"
    "movq 3*8(%[src]), %[buf3]\n"

    "1:\n"
#if HAVE_SSE && CRCUTIL_PREFETCH_WIDTH > 0
    "prefetcht0 " TO_STRING(CRCUTIL_PREFETCH_WIDTH) "(%[src])\n"
#endif
#if GCC_VERSION_AVAILABLE(4, 5)
    // Bug in GCC 4.2.4?
    "add $4*8, %[src]\n"
#else
    "lea 4*8(%[src]), %[src]\n"
#endif
    "pxor %[crc_carryover], %[crc0]\n"

    SSE2_MOVQ " %[crc0], %[tmp0]\n"
    "psrldq $8, %[crc0]\n"
    "xorq %[tmp0], %[buf0]\n"
    "movzbq %b[buf0], %[tmp0]\n"
    "pxor %[crc0], %[crc1]\n"
    "addq %[tmp0], %[tmp0]\n"
    "shrq $8, %[buf0]\n"
    "movdqa (%[table], %[tmp0], 8), %[crc0]\n"

    SSE2_MOVQ " %[crc1], %[tmp1]\n"
    "psrldq $8, %[crc1]\n"
    "xorq %[tmp1], %[buf1]\n"
    "movzbq %b[buf1], %[tmp1]\n"
    "pxor %[crc1], %[crc2]\n"
    "addq %[tmp1], %[tmp1]\n"
    "shrq $8, %[buf1]\n"
    "movdqa (%[table], %[tmp1], 8), %[crc1]\n"

    SSE2_MOVQ " %[crc2], %[tmp0]\n"
    "psrldq $8, %[crc2]\n"
    "xorq %[tmp0], %[buf2]\n"
    "movzbq %b[buf2], %[tmp0]\n"
    "pxor %[crc2], %[crc3]\n"
    "addq %[tmp0], %[tmp0]\n"
    "shrq $8, %[buf2]\n"
    "movdqa (%[table], %[tmp0], 8), %[crc2]\n"

    SSE2_MOVQ " %[crc3], %[tmp1]\n"
    "psrldq $8, %[crc3]\n"
    "xorq %[tmp1], %[buf3]\n"
    "movzbq %b[buf3], %[tmp1]\n"
    "movdqa %[crc3], %[crc_carryover]\n"
    "addq %[tmp1], %[tmp1]\n"
    "shrq $8, %[buf3]\n"
    "movdqa (%[table], %[tmp1], 8), %[crc3]\n"

#define XOR(byte) \
    "movzbq %b[buf0], %[tmp0]\n" \
    "shrq $8, %[buf0]\n" \
    "addq %[tmp0], %[tmp0]\n" \
    "pxor " #byte "*256*16(%[table], %[tmp0], 8), %[crc0]\n" \
    "movzbq %b[buf1], %[tmp1]\n" \
    "shrq $8, %[buf1]\n" \
    "addq %[tmp1], %[tmp1]\n" \
    "pxor " #byte "*256*16(%[table], %[tmp1], 8), %[crc1]\n" \
    "movzbq %b[buf2], %[tmp0]\n" \
    "shrq $8, %[buf2]\n" \
    "addq %[tmp0], %[tmp0]\n" \
    "pxor " #byte "*256*16(%[table], %[tmp0], 8), %[crc2]\n" \
    "movzbq %b[buf3], %[tmp1]\n" \
    "shrq $8, %[buf3]\n" \
    "addq %[tmp1], %[tmp1]\n" \
    "pxor " #byte "*256*16(%[table], %[tmp1], 8), %[crc3]\n"

    XOR(1)
    XOR(2)
    XOR(3)
    XOR(4)
    XOR(5)
    XOR(6)
#undef XOR

    "addq %[buf0], %[buf0]\n"
    "pxor 7*256*16(%[table], %[buf0], 8), %[crc0]\n"
    "movq (%[src]), %[buf0]\n"

    "addq %[buf1], %[buf1]\n"
    "pxor 7*256*16(%[table], %[buf1], 8), %[crc1]\n"
    "movq 1*8(%[src]), %[buf1]\n"

    "addq %[buf2], %[buf2]\n"
    "pxor 7*256*16(%[table], %[buf2], 8), %[crc2]\n"
    "movq 2*8(%[src]), %[buf2]\n"

    "addq %[buf3], %[buf3]\n"
    "pxor 7*256*16(%[table], %[buf3], 8), %[crc3]\n"
    "movq 3*8(%[src]), %[buf3]\n"

    "cmpq %[src], %[end]\n"
    "ja 1b\n"

    "pxor %[crc_carryover], %[crc0]\n"
    CRC_WORD_ASM()

    "pxor %[crc1], %[crc0]\n"
    "movq %[buf1], %[buf0]\n"
    CRC_WORD_ASM()

    "pxor %[crc2], %[crc0]\n"
    "movq %[buf2], %[buf0]\n"
    CRC_WORD_ASM()

    "pxor %[crc3], %[crc0]\n"
    "movq %[buf3], %[buf0]\n"
    CRC_WORD_ASM()

    "add $4*8, %[src]\n"
    "2:\n"

    "add $2*4*8 - 8, %[end]\n"

    "cmpq %[src], %[end]\n"
    "jbe 4f\n"
    "3:\n"
    "movq (%[src]), %[buf0]\n"
    "addq $8, %[src]\n"
    CRC_WORD_ASM()
    "cmpq %[src], %[end]\n"
    "ja 3b\n"

    "4:\n"
    "add $7, %[end]\n"

    "cmpq %[src], %[end]\n"
    "jbe 6f\n"

    "5:\n"
    "movzbq (%[src]), %[buf0]\n"
    "add $1, %[src]\n"
    SSE2_MOVQ " %[crc0], %[tmp0]\n"
    "movzx %b[tmp0], %[tmp0]\n"
    "psrldq $1, %[crc0]\n"
    "xor %[buf0], %[tmp0]\n"
    "addq %[tmp0], %[tmp0]\n"
    "pxor 7*256*16(%[table_word], %[tmp0], 8), %[crc0]\n"

    "cmpq %[src], %[end]\n"
    "ja 5b\n"

    "6:\n"

    :   // outputs
      [src] "+r" (src),
      [end] "+r" (end),
      [crc0] "+x" (crc0),
      [crc1] "=&x" (crc1),
      [crc2] "=&x" (crc2),
      [crc3] "=&x" (crc3),
      [crc_carryover] "=&x" (crc_carryover),
      [buf0] "=&r" (buf0),
      [buf1] "=&r" (buf1),
      [buf2] "=&r" (buf2),
      [buf3] "=&r" (buf3),
      [tmp0] "=&r" (tmp0),
      [tmp1] "=&r" (tmp1)

    :   // inputs
      [table_word] "r" (this->crc_word_),
      [table] "r" (this->crc_word_interleaved_));

  return (this->Base().Canonize() ^ crc0);
}

}  // namespace crcutil

#endif  // defined(__GNUC__) && CRCUTIL_USE_ASM && HAVE_AMD64 && HAVE_SSE2
