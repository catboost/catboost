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

// Implements 64-bit multiword CRC using MMX built-in functions.

#include "generic_crc.h"

#if CRCUTIL_USE_ASM && HAVE_I386 && HAVE_MMX

namespace crcutil {

template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiwordI386Mmx(
    const void *data, size_t bytes, const uint64 &start)
        const GCC_OMIT_FRAME_POINTER;

#if !defined(_MSC_VER)
template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiword(
    const void *data,
    size_t bytes,
    const uint64 &start) const {
  if (bytes <= 7) {
    const uint8 *src = static_cast<const uint8 *>(data);
    uint64 crc = start ^ Base().Canonize();
    for (const uint8 *end = src + bytes; src < end; ++src) {
      CRC_BYTE(this, crc, *src);
    }
    return (crc ^ Base().Canonize());
  }
  return CrcMultiwordI386Mmx(data, bytes, start);
}
#else
#pragma warning(push)
// CL: uninitialized local variable 'crc1' used
// Wrong: crc1 = XOR(crc1, crc1) sets it to 0.
#pragma warning(disable: 4700)

#pragma warning(disable: 4619)  // there is no warning number '592'

// ICL: variable "crc1" is used before its value is set
// Wrong: crc1 = XOR(crc1, crc1) sets it to 0.
#pragma warning(disable: 592)
#endif  // !defined(_MSC_VER)

#define MM64(adr) reinterpret_cast<const __m64 *>(adr)
#define MM64_TABLE(byte) MM64(crc_word_interleaved_[byte])

#define CRC_WORD_MMX(this, crc, buf) do { \
  buf = _mm_xor_si64(buf, crc); \
  uint32 tmp = static_cast<uint32>(_mm_cvtsi64_si32(buf)); \
  buf = _mm_srli_si64(buf, 32); \
  crc = MM64(crc_word_[0])[TO_BYTE(tmp)]; \
  tmp >>= 8; \
  crc = _mm_xor_si64(crc, MM64(crc_word_[1])[TO_BYTE(tmp)]); \
  tmp >>= 8; \
  crc = _mm_xor_si64(crc, MM64(crc_word_[2])[TO_BYTE(tmp)]); \
  tmp >>= 8; \
  crc = _mm_xor_si64(crc, MM64(crc_word_[3])[tmp]); \
  tmp = static_cast<uint32>(_mm_cvtsi64_si32(buf)); \
  crc = _mm_xor_si64(crc, MM64(crc_word_[4])[TO_BYTE(tmp)]); \
  tmp >>= 8; \
  crc = _mm_xor_si64(crc, MM64(crc_word_[5])[TO_BYTE(tmp)]); \
  tmp >>= 8; \
  crc = _mm_xor_si64(crc, MM64(crc_word_[6])[TO_BYTE(tmp)]); \
  tmp >>= 8; \
  crc = _mm_xor_si64(crc, MM64(crc_word_[7])[tmp]); \
} while (0)

template<> uint64 GenericCrc<uint64, uint64, uint64, 4>::CrcMultiwordI386Mmx(
    const void *data, size_t bytes, const uint64 &start) const {
  const uint8 *src = static_cast<const uint8 *>(data);
  const uint8 *end = src + bytes;
  uint64 crc = start ^ Base().Canonize();

  ALIGN_ON_WORD_BOUNDARY_IF_NEEDED(bytes, this, src, end, crc, uint64);
  if (src >= end) {
    return (crc ^ Base().Canonize());
  }

  // Process 4 registers of sizeof(uint64) bytes at once.
  bytes = static_cast<size_t>(end - src) & ~(4*8 - 1);
  if (bytes > 4*8) {
    const uint8 *stop = src + bytes - 4*8;
    union {
      __m64 m64;
      uint64 u64;
    } temp;
    __m64 crc0;
    __m64 crc1;
    __m64 crc2;
    __m64 crc3;
    __m64 buf0 = MM64(src)[0];
    __m64 buf1 = MM64(src)[1];
    __m64 buf2 = MM64(src)[2];
    __m64 buf3 = MM64(src)[3];

    temp.u64 = crc;
    crc0 = temp.m64;
#if defined(__GNUC__) && !GCC_VERSION_AVAILABLE(4, 4)
    // There is no way to suppress a warning in GCC;
    // generate extra assignments.
    temp.u64 = 0;
    crc1 = temp.m64;
    crc2 = temp.m64;
    crc3 = temp.m64;
#else
    crc1 = _mm_xor_si64(crc1, crc1);
    crc2 = _mm_xor_si64(crc2, crc2);
    crc3 = _mm_xor_si64(crc3, crc3);
#endif  // defined(__GNUC__) && !GCC_VERSION_AVAILABLE(4, 4)

    do {
      PREFETCH(src);
      src += 4*8;

      buf0 = _mm_xor_si64(buf0, crc0);
      buf1 = _mm_xor_si64(buf1, crc1);
      buf2 = _mm_xor_si64(buf2, crc2);
      buf3 = _mm_xor_si64(buf3, crc3);

      uint32 tmp0 = static_cast<uint32>(_mm_cvtsi64_si32(buf0));
      uint32 tmp1 = static_cast<uint32>(_mm_cvtsi64_si32(buf1));
      uint32 tmp2 = static_cast<uint32>(_mm_cvtsi64_si32(buf2));
      uint32 tmp3 = static_cast<uint32>(_mm_cvtsi64_si32(buf3));

      buf0 = _mm_srli_si64(buf0, 32);
      buf1 = _mm_srli_si64(buf1, 32);
      buf2 = _mm_srli_si64(buf2, 32);
      buf3 = _mm_srli_si64(buf3, 32);

      crc0 = MM64_TABLE(0)[TO_BYTE(tmp0)];
      tmp0 >>= 8;
      crc1 = MM64_TABLE(0)[TO_BYTE(tmp1)];
      tmp1 >>= 8;
      crc2 = MM64_TABLE(0)[TO_BYTE(tmp2)];
      tmp2 >>= 8;
      crc3 = MM64_TABLE(0)[TO_BYTE(tmp3)];
      tmp3 >>= 8;

#define XOR(byte) do { \
      crc0 = _mm_xor_si64(crc0, MM64_TABLE(byte)[TO_BYTE(tmp0)]); \
      tmp0 >>= 8; \
      crc1 = _mm_xor_si64(crc1, MM64_TABLE(byte)[TO_BYTE(tmp1)]); \
      tmp1 >>= 8; \
      crc2 = _mm_xor_si64(crc2, MM64_TABLE(byte)[TO_BYTE(tmp2)]); \
      tmp2 >>= 8; \
      crc3 = _mm_xor_si64(crc3, MM64_TABLE(byte)[TO_BYTE(tmp3)]); \
      tmp3 >>= 8; \
} while (0)

      XOR(1);
      XOR(2);

      crc0 = _mm_xor_si64(crc0, MM64_TABLE(3)[tmp0]);
      tmp0 = static_cast<uint32>(_mm_cvtsi64_si32(buf0));
      crc1 = _mm_xor_si64(crc1, MM64_TABLE(3)[tmp1]);
      tmp1 = static_cast<uint32>(_mm_cvtsi64_si32(buf1));
      crc2 = _mm_xor_si64(crc2, MM64_TABLE(3)[tmp2]);
      tmp2 = static_cast<uint32>(_mm_cvtsi64_si32(buf2));
      crc3 = _mm_xor_si64(crc3, MM64_TABLE(3)[tmp3]);
      tmp3 = static_cast<uint32>(_mm_cvtsi64_si32(buf3));

      XOR(4);
      XOR(5);
      XOR(6);

#undef XOR

      crc0 = _mm_xor_si64(crc0, MM64_TABLE(sizeof(uint64) - 1)[tmp0]);
      buf0 = MM64(src)[0];
      crc1 = _mm_xor_si64(crc1, MM64_TABLE(sizeof(uint64) - 1)[tmp1]);
      buf1 = MM64(src)[1];
      crc2 = _mm_xor_si64(crc2, MM64_TABLE(sizeof(uint64) - 1)[tmp2]);
      buf2 = MM64(src)[2];
      crc3 = _mm_xor_si64(crc3, MM64_TABLE(sizeof(uint64) - 1)[tmp3]);
      buf3 = MM64(src)[3];
    }
    while (src < stop);

    CRC_WORD_MMX(this, crc0, buf0);
    buf1 = _mm_xor_si64(buf1, crc1);
    CRC_WORD_MMX(this, crc0, buf1);
    buf2 = _mm_xor_si64(buf2, crc2);
    CRC_WORD_MMX(this, crc0, buf2);
    buf3 = _mm_xor_si64(buf3, crc3);
    CRC_WORD_MMX(this, crc0, buf3);

    temp.m64 = crc0;
    crc = temp.u64;

    _mm_empty();

    src += 4*8;
  }

  // Process sizeof(uint64) bytes at once.
  bytes = static_cast<size_t>(end - src) & ~(sizeof(uint64) - 1);
  if (bytes > 0) {
    union {
      __m64 m64;
      uint64 u64;
    } temp;
    __m64 crc0;

    temp.u64 = crc;
    crc0 = temp.m64;

    for (const uint8 *stop = src + bytes; src < stop; src += sizeof(uint64)) {
      __m64 buf0 = MM64(src)[0];
      CRC_WORD_MMX(this, crc0, buf0);
    }

    temp.m64 = crc0;
    crc = temp.u64;

    _mm_empty();
  }

  // Compute CRC of remaining bytes.
  for (;src < end; ++src) {
    CRC_BYTE(this, crc, *src);
  }

  return (crc ^ Base().Canonize());
}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif  // defined(_MSC_VER)

}  // namespace crcutil

#endif  // CRCUTIL_USE_ASM && HAVE_I386 && HAVE_MMX
