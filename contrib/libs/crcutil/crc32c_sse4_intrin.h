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

// Provides _mm_crc32_u64/32/8 intrinsics.

#ifndef CRCUTIL_CRC32C_SSE4_INTRIN_H_
#define CRCUTIL_CRC32C_SSE4_INTRIN_H_

#include "platform.h"
#include "base_types.h"

#if CRCUTIL_USE_MM_CRC32 && (HAVE_I386 || HAVE_AMD64)

#if defined(_MSC_VER) || defined(__SSE4_2__)

#if defined(_MSC_VER)
#pragma warning(push)
// '_M_IA64' is not defined as a preprocessor macro
#pragma warning(disable: 4668)
#endif  // defined(_MSC_VER)

#include <nmmintrin.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif  // defined(_MSC_VER)

#elif GCC_VERSION_AVAILABLE(4, 5) && !defined(CRCUTIL_FORCE_ASM_CRC32C)
// Allow the use of _mm_crc32_u* intrinsic when CRCUTIL_USE_MM_CRC32
// is set irrespective of "-msse*" settings. This way, the sources
// may be compiled with "-msse2 -mcrc32" and work on older CPUs,
// while taking full advantage of "crc32" instruction on newer
// CPUs (requires dynamic CPU detection). See "interface.cc".
//
// If neither -msse4 or -mcrc32 is provided and CRCUTIL_USE_MM_CRC32 is set
// and CRCUTIL_FORCE_ASM_CRC32 is not set, compile-time error will happen.
// Why? Becuase GCC disables __builtin_ia32_crc32* intrinsics when compiled
// without -msse4 or -mcrc32. -msse4 could be detected at run time by checking
// whether __SSE4_2__ is defined, but there is no way to tell whether the
// sources are compiled with -mcrc32.

extern __inline unsigned int __attribute__((
    __gnu_inline__, __always_inline__, __artificial__))
_mm_crc32_u8(unsigned int __C, unsigned char __V) {
  return __builtin_ia32_crc32qi(__C, __V);
}
#ifdef __x86_64__
extern __inline unsigned long long __attribute__((
    __gnu_inline__, __always_inline__, __artificial__))
_mm_crc32_u64(unsigned long long __C, unsigned long long __V) {
  return __builtin_ia32_crc32di(__C, __V);
}
#else
extern __inline unsigned int __attribute__((
    __gnu_inline__, __always_inline__, __artificial__))
_mm_crc32_u32(unsigned int __C, unsigned int __V) {
  return __builtin_ia32_crc32si (__C, __V);
}
#endif  // __x86_64__

#else

// GCC 4.4.x and earlier: use inline asm.

namespace crcutil {

__forceinline uint64 _mm_crc32_u64(uint64 crc, uint64 value) {
  asm("crc32q %[value], %[crc]\n" : [crc] "+r" (crc) : [value] "rm" (value));
  return crc;
}

__forceinline uint32 _mm_crc32_u32(uint32 crc, uint64 value) {
  asm("crc32l %[value], %[crc]\n" : [crc] "+r" (crc) : [value] "rm" (value));
  return crc;
}

__forceinline uint32 _mm_crc32_u8(uint32 crc, uint8 value) {
  asm("crc32b %[value], %[crc]\n" : [crc] "+r" (crc) : [value] "rm" (value));
  return crc;
}

}  // namespace crcutil

#endif

#endif  // CRCUTIL_USE_MM_CRC32 && (HAVE_I386 || HAVE_AMD64)

#endif  // CRCUTIL_CRC32C_SSE4_INTRIN_H_
