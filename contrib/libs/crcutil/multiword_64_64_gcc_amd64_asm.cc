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

// Implements multiword CRC for GCC on AMD64.
//
// Accoding to "Software Optimization Guide for AMD Family 10h Processors"
// http://www.amd.com/us-en/assets/content_type/white_papers_and_tech_docs/40546.pdf
// instead of
//     movzbq %al, %rsi
//     shrq $8, %rax
//     [use %rsi]
//     movzbq %al, %rsi
//     shrq $8, %rax
//     [use %rsi]
// it is better to use 32-bit registers
// (high 32 bits will be cleared on assignment), i.e.
//     movzbl %al, %esi
//     [use %rsi]
//     movzbl %ah, %esi
//     shrq $16, %rax
//     [use %rsi]
// Makes instructions shorter and removes one shift
// (the latter is not such a big deal as it's execution time
// is nicely masked by [use %rsi] instruction).
//
// Performance difference:
// About 10% degradation on bytes = 8 .. 16
// (clobbering registers that should be saved)
// Break even at 32 bytes.
// 3% improvement starting from 64 bytes.

#include "generic_crc.h"

#if defined(__GNUC__) && CRCUTIL_USE_ASM && HAVE_AMD64

namespace crcutil {

template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiwordGccAmd64(
    const void *data, size_t bytes, const uint64 &start) const;

template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiword(
    const void *data,
    size_t bytes,
    const uint64 &start) const {
  if (bytes <= 6 * sizeof(Word) - 1) {
    const uint8 *src = static_cast<const uint8 *>(data);
    uint64 crc = start ^ this->Base().Canonize();
    const uint8 *end = src + bytes;
#define PROCESS_ONE_WORD() do { \
    Word buf = reinterpret_cast<const Word *>(src)[0]; \
    CRC_WORD(this, crc, buf); \
    src += sizeof(Word); \
} while (0)
    if (bytes >= 1 * sizeof(Word)) {
      PROCESS_ONE_WORD();
      if (bytes >= 2 * sizeof(Word)) {
        PROCESS_ONE_WORD();
        if (bytes >= 3 * sizeof(Word)) {
          PROCESS_ONE_WORD();
          if (bytes >= 4 * sizeof(Word)) {
            PROCESS_ONE_WORD();
            if (bytes >= 5 * sizeof(Word)) {
              PROCESS_ONE_WORD();
            }
          }
        }
      }
    }
    for (; src < end; ++src) {
      CRC_BYTE(this, crc, *src);
    }
    return (crc ^ this->Base().Canonize());
  }
  return this->CrcMultiwordGccAmd64(data, bytes, start);
}

#define TMP0  "%%rsi"
#define TMP0W "%%esi"

#define BUF0  "%%rax"
#define BUF0L "%%al"
#define BUF0H "%%ah"

#define BUF1  "%%rbx"
#define BUF1L "%%bl"
#define BUF1H "%%bh"

#define BUF2  "%%rcx"
#define BUF2L "%%cl"
#define BUF2H "%%ch"

#define BUF3  "%%rdx"
#define BUF3L "%%dl"
#define BUF3H "%%dh"

#define CRC_WORD_ASM() \
    "xorq %[crc0], " BUF0 "\n" \
    "movzbq " BUF0L ", " TMP0 "\n" \
    "movq (%[table_word], " TMP0 ", 8), %[crc0]\n" \
    "movzbl " BUF0H ", " TMP0W "\n" \
    "shrq $16, " BUF0 "\n" \
    "xorq 1*256*8(%[table_word], " TMP0 ", 8), %[crc0]\n" \
    "movzbq " BUF0L ", " TMP0 "\n" \
    "xorq 2*256*8(%[table_word], " TMP0 ", 8), %[crc0]\n" \
    "movzbl " BUF0H ", " TMP0W "\n" \
    "shrq $16, " BUF0 "\n" \
    "xorq 3*256*8(%[table_word], " TMP0 ", 8), %[crc0]\n" \
    "movzbq " BUF0L ", " TMP0 "\n" \
    "xorq 4*256*8(%[table_word], " TMP0 ", 8), %[crc0]\n" \
    "movzbl " BUF0H ", " TMP0W "\n" \
    "shrq $16, " BUF0 "\n" \
    "xorq 5*256*8(%[table_word], " TMP0 ", 8), %[crc0]\n" \
    "movzbq " BUF0L ", " TMP0 "\n" \
    "xorq 6*256*8(%[table_word], " TMP0 ", 8), %[crc0]\n" \
    "movzbl " BUF0H ", " TMP0W "\n" \
    "xorq 7*256*8(%[table_word], " TMP0 ", 8), %[crc0]\n"

template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiwordGccAmd64(
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

  asm(
    "sub $2*4*8 - 1, %[end]\n"
    "cmpq  %[src], %[end]\n"
    "jbe 2f\n"
    "xorq %[crc1], %[crc1]\n"
    "movq (%[src]), " BUF0 "\n"
    "movq 1*8(%[src]), " BUF1 "\n"
    "movq 2*8(%[src]), " BUF2 "\n"
    "movq 3*8(%[src]), " BUF3 "\n"
    "movq %[crc1], %[crc2]\n"
    "movq %[crc1], %[crc3]\n"

    "1:\n"
#if HAVE_SSE && CRCUTIL_PREFETCH_WIDTH > 0
    "prefetcht0 " TO_STRING(CRCUTIL_PREFETCH_WIDTH) "(%[src])\n"
#endif  // HAVE_SSE
    "add $4*8, %[src]\n"

    // Set buffer data.
    "xorq %[crc0], " BUF0 "\n"
    "xorq %[crc1], " BUF1 "\n"
    "xorq %[crc2], " BUF2 "\n"
    "xorq %[crc3], " BUF3 "\n"

    // LOAD crc of byte 0 and shift buffers.
    "movzbl " BUF0L ", " TMP0W "\n"
    "movq (%[table], " TMP0 ", 8), %[crc0]\n"
    "movzbl " BUF1L ", " TMP0W "\n"
    "movq (%[table], " TMP0 ", 8), %[crc1]\n"
    "movzbl " BUF2L ", " TMP0W "\n"
    "movq (%[table], " TMP0 ", 8), %[crc2]\n"
    "movzbl " BUF3L ", " TMP0W "\n"
    "movq (%[table], " TMP0 ", 8), %[crc3]\n"

#define XOR1(byte1) \
    "movzbl " BUF0L ", " TMP0W "\n" \
    "xorq " #byte1 "*256*8(%[table], " TMP0 ", 8), %[crc0]\n" \
    "movzbl " BUF1L ", " TMP0W "\n" \
    "xorq " #byte1 "*256*8(%[table], " TMP0 ", 8), %[crc1]\n" \
    "movzbl " BUF2L ", " TMP0W "\n" \
    "xorq " #byte1 "*256*8(%[table], " TMP0 ", 8), %[crc2]\n" \
    "movzbl " BUF3L ", " TMP0W "\n" \
    "xorq " #byte1 "*256*8(%[table], " TMP0 ", 8), %[crc3]\n"

#define XOR2(byte2) \
    "movzbl " BUF0H ", " TMP0W "\n" \
    "shrq $16, " BUF0 "\n" \
    "xorq " #byte2 "*256*8(%[table], " TMP0 ", 8), %[crc0]\n" \
    "movzbl " BUF1H ", " TMP0W "\n" \
    "shrq $16, " BUF1 "\n" \
    "xorq " #byte2 "*256*8(%[table], " TMP0 ", 8), %[crc1]\n" \
    "movzbl " BUF2H ", " TMP0W "\n" \
    "shrq $16, " BUF2 "\n" \
    "xorq " #byte2 "*256*8(%[table], " TMP0 ", 8), %[crc2]\n" \
    "movzbl " BUF3H ", " TMP0W "\n" \
    "shrq $16, " BUF3 "\n" \
    "xorq " #byte2 "*256*8(%[table], " TMP0 ", 8), %[crc3]\n"

    XOR2(1)
    XOR1(2)
    XOR2(3)
    XOR1(4)
    XOR2(5)
    XOR1(6)

    // Update CRC registers and load buffers.
    "movzbl " BUF0H ", " TMP0W "\n"
    "xorq 7*256*8(%[table], " TMP0 ", 8), %[crc0]\n"
    "movq (%[src]), " BUF0 "\n"
    "movzbl " BUF1H ", " TMP0W "\n"
    "xorq 7*256*8(%[table], " TMP0 ", 8), %[crc1]\n"
    "movq 1*8(%[src]), " BUF1 "\n"
    "movzbl " BUF2H ", " TMP0W "\n"
    "xorq 7*256*8(%[table], " TMP0 ", 8), %[crc2]\n"
    "movq 2*8(%[src]), " BUF2 "\n"
    "movzbl " BUF3H ", " TMP0W "\n"
    "xorq 7*256*8(%[table], " TMP0 ", 8), %[crc3]\n"
    "movq 3*8(%[src]), " BUF3 "\n"

    "cmpq  %[src], %[end]\n"
    "ja 1b\n"

    CRC_WORD_ASM()

    "xorq %[crc1], " BUF1 "\n"
    "movq " BUF1 ", " BUF0 "\n"
    CRC_WORD_ASM()

    "xorq %[crc2], " BUF2 "\n"
    "movq " BUF2 ", " BUF0 "\n"
    CRC_WORD_ASM()

    "xorq %[crc3], " BUF3 "\n"
    "movq " BUF3 ", " BUF0 "\n"
    CRC_WORD_ASM()

    "add $4*8, %[src]\n"

    "2:\n"
    "add $2*4*8 - 8, %[end]\n"
    "cmpq %[src], %[end]\n"
    "jbe 4f\n"

    "3:\n"
    "movq (%[src]), " BUF0 "\n"
    "add $8, %[src]\n"
    CRC_WORD_ASM()
    "cmpq %[src], %[end]\n"
    "ja 3b\n"

    "4:\n"
    "add $7, %[end]\n"

    "cmpq %[src], %[end]\n"
    "jbe 6f\n"

    "5:\n"
    "movzbq (%[src]), " BUF0 "\n"
    "movzbq %b[crc0], " TMP0 "\n"
    "shrq  $8, %[crc0]\n"
    "xorq " BUF0 ", " TMP0 "\n"
    "add $1, %[src]\n"
    "xorq 7*256*8(%[table_word], " TMP0 ", 8), %[crc0]\n"
    "cmpq %[src], %[end]\n"
    "ja 5b\n"

    "6:\n"


    :   // outputs
      [src] "+r" (src),
      [end] "+r" (end),
      [crc0] "+r" (crc0),
      [crc1] "=&r" (crc1),
      [crc2] "=&r" (crc2),
      [crc3] "=&r" (crc3)

    :   // inputs
      [table] "r" (&this->crc_word_interleaved_[0][0]),
      [table_word] "r" (&this->crc_word_[0][0])

    :   // clobbers
      "%rax",   // BUF0
      "%rbx",   // BUF1
      "%rcx",   // BUF2
      "%rdx",   // BUF3
      "%rsi"    // TMP0
    );

  return (crc0 ^ this->Base().Canonize());
}

}  // namespace crcutil

#endif  // defined(__GNUC__) && HAVE_AMD64 && CRCUTIL_USE_ASM
