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

// Implements 64-bit multiword CRC for Microsoft and Intel compilers
// using MMX instructions (i386).

#include "generic_crc.h"

#if CRCUTIL_USE_ASM && HAVE_I386 && HAVE_MMX && defined(_MSC_VER)

namespace crcutil {

#define CRC_WORD_MMX() \
    __asm   pxor  BUF0, CRC0 \
    __asm   movd  TMP0, BUF0 \
    __asm   psrlq BUF0, 32 \
    __asm   movzx TEMP, TMP0L \
    __asm   shr   TMP0, 8 \
    __asm   movq  CRC0, [TABLE + TEMP * 8] \
    __asm   movzx TEMP, TMP0L \
    __asm   shr   TMP0, 8 \
    __asm   pxor  CRC0, [TABLE + TEMP * 8 + 1 * 256 * 8] \
    __asm   movzx TEMP, TMP0L \
    __asm   shr   TMP0, 8 \
    __asm   pxor  CRC0, [TABLE + TEMP * 8 + 2 * 256 * 8] \
    __asm   pxor  CRC0, [TABLE + TMP0 * 8 + 3 * 256 * 8] \
    __asm   movd  TMP0, BUF0 \
    __asm   movzx TEMP, TMP0L \
    __asm   shr   TMP0, 8 \
    __asm   pxor  CRC0, [TABLE + TEMP * 8 + 4 * 256 * 8] \
    __asm   movzx TEMP, TMP0L \
    __asm   shr   TMP0, 8 \
    __asm   pxor  CRC0, [TABLE + TEMP * 8 + 5 * 256 * 8] \
    __asm   movzx TEMP, TMP0L \
    __asm   shr   TMP0, 8 \
    __asm   pxor  CRC0, [TABLE + TEMP * 8 + 6 * 256 * 8] \
    __asm   pxor  CRC0, [TABLE + TMP0 * 8 + 7 * 256 * 8]

// frame pointer register 'ebp' modified by inline assembly code
#pragma warning(disable: 4731)

template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiwordI386Mmx(
    const void *data,
    size_t bytes,
    const uint64 &start) const {
  const uint8 *src = static_cast<const uint8 *>(data);
  const uint8 *end = src + bytes;
  uint64 crc0 = start ^ this->Base().Canonize();

  ALIGN_ON_WORD_BOUNDARY_IF_NEEDED(bytes, this, src, end, crc0, uint64);
  if (src >= end) {
    return (crc0 ^ this->Base().Canonize());
  }

#define CRC0  mm0
#define CRC1  mm1
#define CRC2  mm2
#define CRC3  mm3
#define BUF0  mm4
#define BUF1  mm5
#define BUF2  mm6
#define BUF3  mm7
#define TMP0  eax
#define TMP0L  al
#define TMP0H  ah
#define TMP1  ebx
#define TMP1L  bl
#define TMP1H  bh
#define TMP2  ecx
#define TMP2L  cl
#define TMP2H  ch
#define TMP3  edx
#define TMP3L  dl
#define TMP3H  dh
#define TEMP  edi
#define SRC   esi
#define END   [esp]
#define TABLE ebp


  const uint64 *interleaved_table_address =
                    &this->crc_word_interleaved_[0][0];
  const uint64 *word_table_address = &this->crc_word_[0][0];

  __asm {
    push  ebp

    mov   TMP0, interleaved_table_address

    movq  CRC0, crc0
    mov   SRC, src
    mov   TMP1, end
    sub   TMP1, 2*4*8 - 1
    cmp   SRC, TMP1
    mov   TABLE, word_table_address
    jae   end_main_loop

    push  TABLE
    mov   TABLE, TMP0
    push  TMP1

    pxor  CRC1, CRC1
    pxor  CRC2, CRC2
    pxor  CRC3, CRC3

    movq  BUF0, [SRC]
    movq  BUF1, [SRC + 1 * 8]
    movq  BUF2, [SRC + 2 * 8]
    movq  BUF3, [SRC + 3 * 8]

 main_loop:
#if HAVE_SSE && CRCUTIL_PREFETCH_WIDTH > 0
    prefetcht0 [SRC + CRCUTIL_PREFETCH_WIDTH]
#endif
    add   SRC, 32
    pxor  BUF0, CRC0
    pxor  BUF1, CRC1
    pxor  BUF2, CRC2
    pxor  BUF3, CRC3

    movd  TMP0, BUF0
    psrlq BUF0, 32
    movd  TMP1, BUF1
    psrlq BUF1, 32
    movd  TMP2, BUF2
    psrlq BUF2, 32
    movd  TMP3, BUF3
    psrlq BUF3, 32

    movzx TEMP, TMP0L
    movq  CRC0, [TABLE + TEMP * 8]
    movzx TEMP, TMP1L
    movq  CRC1, [TABLE + TEMP * 8]
    movzx TEMP, TMP2L
    movq  CRC2, [TABLE + TEMP * 8]
    movzx TEMP, TMP3L
    movq  CRC3, [TABLE + TEMP * 8]

    movzx TEMP, TMP0H
    shr TMP0, 16
    pxor CRC0, [TABLE + TEMP * 8 + 1 * 256 * 8]
    movzx TEMP, TMP1H
    shr TMP1, 16
    pxor CRC1, [TABLE + TEMP * 8 + 1 * 256 * 8]
    movzx TEMP, TMP2H
    shr TMP2, 16
    pxor CRC2, [TABLE + TEMP * 8 + 1 * 256 * 8]
    movzx TEMP, TMP3H
    shr TMP3, 16
    pxor CRC3, [TABLE + TEMP * 8 + 1 * 256 * 8]

    movzx TEMP, TMP0L
    shr TMP0, 8
    pxor CRC0, [TABLE + TEMP * 8 + 2 * 256 * 8]
    movzx TEMP, TMP1L
    shr TMP1, 8
    pxor CRC1, [TABLE + TEMP * 8 + 2 * 256 * 8]
    movzx TEMP, TMP2L
    shr TMP2, 8
    pxor CRC2, [TABLE + TEMP * 8 + 2 * 256 * 8]
    movzx TEMP, TMP3L
    shr TMP3, 8
    pxor CRC3, [TABLE + TEMP * 8 + 2 * 256 * 8]

    pxor  CRC0, [TABLE + TMP0 * 8 + 3 * 256 * 8]
    movd  TMP0, BUF0
    pxor  CRC1, [TABLE + TMP1 * 8 + 3 * 256 * 8]
    movd  TMP1, BUF1
    pxor  CRC2, [TABLE + TMP2 * 8 + 3 * 256 * 8]
    movd  TMP2, BUF2
    pxor  CRC3, [TABLE + TMP3 * 8 + 3 * 256 * 8]
    movd  TMP3, BUF3

    movzx TEMP, TMP0L
    pxor CRC0, [TABLE + TEMP * 8 + 4 * 256 * 8]
    movzx TEMP, TMP1L
    pxor CRC1, [TABLE + TEMP * 8 + 4 * 256 * 8]
    movzx TEMP, TMP2L
    pxor CRC2, [TABLE + TEMP * 8 + 4 * 256 * 8]
    movzx TEMP, TMP3L
    pxor CRC3, [TABLE + TEMP * 8 + 4 * 256 * 8]

    movzx TEMP, TMP0H
    shr TMP0, 16
    pxor CRC0, [TABLE + TEMP * 8 + 5 * 256 * 8]
    movzx TEMP, TMP1H
    shr TMP1, 16
    pxor CRC1, [TABLE + TEMP * 8 + 5 * 256 * 8]
    movzx TEMP, TMP2H
    shr TMP2, 16
    pxor CRC2, [TABLE + TEMP * 8 + 5 * 256 * 8]
    movzx TEMP, TMP3H
    shr TMP3, 16
    pxor CRC3, [TABLE + TEMP * 8 + 5 * 256 * 8]

    movzx TEMP, TMP0L
    shr TMP0, 8
    pxor CRC0, [TABLE + TEMP * 8 + 6 * 256 * 8]
    movzx TEMP, TMP1L
    shr TMP1, 8
    pxor CRC1, [TABLE + TEMP * 8 + 6 * 256 * 8]
    movzx TEMP, TMP2L
    shr TMP2, 8
    pxor CRC2, [TABLE + TEMP * 8 + 6 * 256 * 8]
    movzx TEMP, TMP3L
    shr TMP3, 8
    pxor CRC3, [TABLE + TEMP * 8 + 6 * 256 * 8]

    pxor  CRC0, [TABLE + TMP0 * 8 + 7 * 256 * 8]
    movq  BUF0, [SRC]
    pxor  CRC1, [TABLE + TMP1 * 8 + 7 * 256 * 8]
    movq  BUF1, [SRC + 1 * 8]
    pxor  CRC2, [TABLE + TMP2 * 8 + 7 * 256 * 8]
    movq  BUF2, [SRC + 2 * 8]
    pxor  CRC3, [TABLE + TMP3 * 8 + 7 * 256 * 8]
    movq  BUF3, [SRC + 3 * 8]

    cmp   END, SRC
    ja    main_loop

#undef END
#define END TMP1
    pop   END
    pop   TABLE
    add   SRC, 32

    CRC_WORD_MMX()

    pxor  BUF1, CRC1
    movq  BUF0, BUF1
    CRC_WORD_MMX()

    pxor  BUF2, CRC2
    movq  BUF0, BUF2
    CRC_WORD_MMX()

    pxor  BUF3, CRC3
    movq  BUF0, BUF3
    CRC_WORD_MMX()

 end_main_loop:
    add   END, 2*4*8 - 8
    cmp   SRC, END
    jae   end_word_loop

 word_loop:
    movq  BUF0, [SRC]
    add   SRC, 8
    CRC_WORD_MMX()
    cmp   END, SRC
    ja    word_loop
 end_word_loop:

#if 0   // Plain C version is faster?
    add   END, 7
    cmp   SRC, END
    jae   end_byte_loop

 byte_loop:
    movd  TMP0, CRC0
    movzx TEMP, byte ptr [SRC]
    movzx TMP0, TMP0L
    psrlq CRC0, 8
    xor   TEMP, TMP0
    add   SRC, 1
    pxor  CRC0, [TABLE + TEMP*8 + 7*256*8]
    cmp   END, SRC
    ja    byte_loop
 end_byte_loop:
#endif

    pop   ebp

    mov   src, SRC
    movq  crc0, CRC0

    emms
  }

#if 1
  // Compute CRC of remaining bytes.
  for (;src < end; ++src) {
    CRC_BYTE(this, crc0, *src);
  }
#endif

  return (crc0 ^ this->Base().Canonize());
}


}  // namespace crcutil

#endif  // CRCUTIL_USE_ASM && HAVE_I386 && HAVE_MMX && defined(_MSC_VER)
