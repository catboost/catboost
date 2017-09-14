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

#include "generic_crc.h"

#if defined(__GNUC__) && CRCUTIL_USE_ASM && HAVE_I386 && HAVE_MMX

namespace crcutil {

template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiwordI386Mmx(
    const void *data, size_t bytes, const uint64 &start)
        const GCC_OMIT_FRAME_POINTER;

template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiword(
    const void *data, size_t bytes, const uint64 &start) const {
  if (bytes <= 7) {
    const uint8 *src = static_cast<const uint8 *>(data);
    uint64 crc = start ^ this->Base().Canonize();
    for (const uint8 *end = src + bytes; src < end; ++src) {
      CRC_BYTE(this, crc, *src);
    }
    return (crc ^ this->Base().Canonize());
  }
  return CrcMultiwordI386Mmx(data, bytes, start);
}

#define CRC_WORD_MMX() \
    "pxor %[crc0], %[buf0]\n" \
    "movd %[buf0], %[tmp0]\n" \
    "psrlq $32, %[buf0]\n" \
    "movzbl %b[tmp0], %[temp]\n" \
    "shrl $8, %[tmp0]\n" \
    "movq (%[table], %[temp], 8), %[crc0]\n" \
    "movzbl %b[tmp0], %[temp]\n" \
    "shrl $8, %[tmp0]\n" \
    "pxor 1*256*8(%[table], %[temp], 8), %[crc0]\n" \
    "movzbl %b[tmp0], %[temp]\n" \
    "shrl $8, %[tmp0]\n" \
    "pxor 2*256*8(%[table], %[temp], 8), %[crc0]\n" \
    "pxor 3*256*8(%[table], %[tmp0], 8), %[crc0]\n" \
    "movd %[buf0], %[tmp0]\n" \
    "movzbl %b[tmp0], %[temp]\n" \
    "shrl $8, %[tmp0]\n" \
    "pxor 4*256*8(%[table], %[temp], 8), %[crc0]\n" \
    "movzbl %b[tmp0], %[temp]\n" \
    "shrl $8, %[tmp0]\n" \
    "pxor 5*256*8(%[table], %[temp], 8), %[crc0]\n" \
    "movzbl %b[tmp0], %[temp]\n" \
    "shrl $8, %[tmp0]\n" \
    "pxor 6*256*8(%[table], %[temp], 8), %[crc0]\n" \
    "pxor 7*256*8(%[table], %[tmp0], 8), %[crc0]\n"

template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiwordI386Mmx(
    const void *data, size_t bytes, const uint64 &start) const {
  const uint8 *src = static_cast<const uint8 *>(data);
  const uint8 *end = src + bytes;
  uint64 crc0 = start ^ this->Base().Canonize();

  ALIGN_ON_WORD_BOUNDARY_IF_NEEDED(bytes, this, src, end, crc0, uint64);
  if (src >= end) {
    return (crc0 ^ this->Base().Canonize());
  }

  uint64 crc1;
  uint64 crc2;
  uint64 crc3;

  uint64 buf0;
  uint64 buf1;
  uint64 buf2;
  uint64 buf3;

  uint32 tmp0;
  uint32 tmp1;
  uint32 tmp2;
  uint32 tmp3;

  uint32 temp;

  void *table_ptr;
  const uint64 *table_interleaved = &this->crc_word_interleaved_[0][0];
  const uint64 *table_word = &this->crc_word_[0][0];

  asm(
    "sub $2*4*8 - 1, %[end]\n"
    "cmpl  %[src], %[end]\n"
    "jbe 2f\n"

    "pxor %[crc1], %[crc1]\n"
    "pxor %[crc2], %[crc2]\n"
    "pxor %[crc3], %[crc3]\n"
    "movq (%[src]), %[buf0]\n"
    "movq 1*8(%[src]), %[buf1]\n"
    "movq 2*8(%[src]), %[buf2]\n"
    "movq 3*8(%[src]), %[buf3]\n"

    "movl %[table_interleaved], %[table]\n"
    "1:\n"
#if HAVE_SSE && CRCUTIL_PREFETCH_WIDTH > 0
    "prefetcht0 " TO_STRING(CRCUTIL_PREFETCH_WIDTH) "(%[src])\n"
#endif
    "addl $0x20, %[src]\n"
    "pxor %[crc0], %[buf0]\n"
    "pxor %[crc1], %[buf1]\n"
    "pxor %[crc2], %[buf2]\n"
    "pxor %[crc3], %[buf3]\n"

    "movd %[buf0], %[tmp0]\n"
    "psrlq $32, %[buf0]\n"
    "movd %[buf1], %[tmp1]\n"
    "psrlq $32, %[buf1]\n"
    "movd %[buf2], %[tmp2]\n"
    "psrlq $32, %[buf2]\n"
    "movd %[buf3], %[tmp3]\n"
    "psrlq $32, %[buf3]\n"

    "movzbl %b[tmp0], %[temp]\n"
    "shrl $8, %[tmp0]\n"
    "movq (%[table], %[temp], 8), %[crc0]\n"
    "movzbl %b[tmp1], %[temp]\n"
    "shrl $8, %[tmp1]\n"
    "movq (%[table], %[temp], 8), %[crc1]\n"
    "movzbl %b[tmp2], %[temp]\n"
    "shrl $8, %[tmp2]\n"
    "movq (%[table], %[temp], 8), %[crc2]\n"
    "movzbl %b[tmp3], %[temp]\n"
    "shrl $8, %[tmp3]\n"
    "movq (%[table], %[temp], 8), %[crc3]\n"

#define XOR(byte) \
    "movzbl %b[tmp0], %[temp]\n" \
    "shrl $8, %[tmp0]\n" \
    "pxor " #byte "*256*8(%[table], %[temp], 8), %[crc0]\n" \
    "movzbl %b[tmp1], %[temp]\n" \
    "shrl $8, %[tmp1]\n" \
    "pxor " #byte "*256*8(%[table], %[temp], 8), %[crc1]\n" \
    "movzbl %b[tmp2], %[temp]\n" \
    "shrl $8, %[tmp2]\n" \
    "pxor " #byte "*256*8(%[table], %[temp], 8), %[crc2]\n" \
    "movzbl %b[tmp3], %[temp]\n" \
    "shrl $8, %[tmp3]\n" \
    "pxor " #byte "*256*8(%[table], %[temp], 8), %[crc3]\n"

    XOR(1)
    XOR(2)

    "pxor 3*256*8(%[table], %[tmp0], 8), %[crc0]\n"
    "movd %[buf0], %[tmp0]\n"
    "pxor 3*256*8(%[table], %[tmp1], 8), %[crc1]\n"
    "movd %[buf1], %[tmp1]\n"
    "pxor 3*256*8(%[table], %[tmp2], 8), %[crc2]\n"
    "movd %[buf2], %[tmp2]\n"
    "pxor 3*256*8(%[table], %[tmp3], 8), %[crc3]\n"
    "movd %[buf3], %[tmp3]\n"

    XOR(4)
    XOR(5)
    XOR(6)

    "pxor 7*256*8(%[table], %[tmp0], 8), %[crc0]\n"
    "movq (%[src]), %[buf0]\n"
    "pxor 7*256*8(%[table], %[tmp1], 8), %[crc1]\n"
    "movq 1*8(%[src]), %[buf1]\n"
    "pxor 7*256*8(%[table], %[tmp2], 8), %[crc2]\n"
    "movq 2*8(%[src]), %[buf2]\n"
    "pxor 7*256*8(%[table], %[tmp3], 8), %[crc3]\n"
    "movq 3*8(%[src]), %[buf3]\n"
    "cmpl %[src], %[end]\n"
    "ja 1b\n"
#undef XOR

    "movl %[table_word], %[table]\n"
    CRC_WORD_MMX()

    "pxor %[crc1], %[buf1]\n"
    "movq %[buf1], %[buf0]\n"
    CRC_WORD_MMX()

    "pxor %[crc2], %[buf2]\n"
    "movq %[buf2], %[buf0]\n"
    CRC_WORD_MMX()

    "pxor %[crc3], %[buf3]\n"
    "movq %[buf3], %[buf0]\n"
    CRC_WORD_MMX()

    "add $4*8, %[src]\n"
    "2:\n"
    "movl %[table_word], %[table]\n"

    "add $2*4*8 - 8, %[end]\n"
    "cmpl %[src], %[end]\n"
    "jbe 4f\n"
    "3:\n"
    "movq (%[src]), %[buf0]\n"
    "addl $0x8, %[src]\n"
    CRC_WORD_MMX()
    "cmpl %[src], %[end]\n"
    "ja 3b\n"
    "4:\n"
    "add $7, %[end]\n"

    "cmpl %[src], %[end]\n"
    "jbe 6f\n"

    "5:\n"
    "movd %[crc0], %[tmp0]\n"
    "movzbl (%[src]), %[temp]\n"
    "movzbl %b[tmp0], %[tmp0]\n"
    "psrlq $8, %[crc0]\n"
    "xorl %[tmp0], %[temp]\n"
    "add $1, %[src]\n"
    "pxor 7*256*8(%[table], %[temp], 8), %[crc0]\n"
    "cmpl %[src], %[end]\n"
    "ja 5b\n"

    "6:\n"

    :   // outputs
      [src] "+r" (src),
      [end] "+m" (end),
      [crc0] "+y" (crc0),
      [crc1] "=&y" (crc1),
      [crc2] "=&y" (crc2),
      [crc3] "=&y" (crc3),
      [buf0] "=&y" (buf0),
      [buf1] "=&y" (buf1),
      [buf2] "=&y" (buf2),
      [buf3] "=&y" (buf3),
      [tmp0] "=&q" (tmp0),
      [tmp1] "=&q" (tmp1),
      [tmp2] "=&q" (tmp2),
      [tmp3] "=&q" (tmp3),
      [temp] "=&r" (temp),
      [table] "=&r" (table_ptr)

    :   // inputs
      [table_interleaved] "m" (table_interleaved),
      [table_word] "m" (table_word));

  asm volatile("emms");

  return (crc0 ^ this->Base().Canonize());
}

}  // namespace crcutil

#endif  // defined(__GNUC__) && HAVE_AMD64 && CRCUTIL_USE_ASM
