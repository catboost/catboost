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

// Detects configuration and defines compiler-specific macros.
// Also, sets user-defined CRUTIL_USE_* macros to default values.

#ifndef CRCUTIL_PLATFORM_H_
#define CRCUTIL_PLATFORM_H_

// Permanently disable some annoying warnings generated
// by Microsoft CL when compiling Microsoft's headers.
#include "std_headers.h"

// Use inline asm version of the code?
#if !defined(CRCUTIL_USE_ASM)
#define CRCUTIL_USE_ASM 1
#endif  // !defined(CRCUTIL_USE_ASM)


#if !defined(HAVE_I386)
#if defined(__i386__) || defined(_M_IX86)
#define HAVE_I386 1
#else
#define HAVE_I386 0
#endif  // defined(__i386__) || defined(_M_IX86)
#endif  // defined(HAVE_I386)


#if !defined(HAVE_AMD64)
#if defined(__amd64__) || defined(_M_AMD64)
#define HAVE_AMD64 1
#else
#define HAVE_AMD64 0
#endif  // defined(__amd64__) || defined(_M_AMD64)
#endif  // defined(HAVE_AMD64)


#if HAVE_AMD64 || HAVE_I386
#if defined(_MSC_VER)
#pragma warning(push)
// '_M_IX86' is not defined as a preprocessor macro
#pragma warning(disable: 4668)
#include <intrin.h>
#pragma warning(pop)
#endif  // defined(_MSC_VER)


#if !defined(HAVE_MMX)
#if defined(_MSC_VER) || (defined(__GNUC__) && defined(__MMX__))
#define HAVE_MMX 1
#else
#define HAVE_MMX 0
#endif  // defined(_MSC_VER) || (defined(__GNUC__) && defined(__MMX__))
#endif  // !defined(HAVE_MMX)


#if !defined(HAVE_SSE)
#if defined(_MSC_VER) || (defined(__GNUC__) && defined(__SSE__))
#include <xmmintrin.h>
#define HAVE_SSE 1
#else
#define HAVE_SSE 0
#endif  // defined(_MSC_VER) || (defined(__GNUC__) && defined(__SSE__))
#endif  // !defined(HAVE_SSE)


#if !defined(HAVE_SSE2)
#if defined(_MSC_VER) || (defined(__GNUC__) && defined(__SSE2__))
#include <emmintrin.h>
#define HAVE_SSE2 1
#else
#define HAVE_SSE2 0
#endif  // defined(_MSC_VER) || (defined(__GNUC__) && defined(__SSE2__))
#endif  // !defined(HAVE_SSE2)

#else

#if !defined(HAVE_MMX)
#define HAVE_MMX 0
#endif  // !defined(HAVE_MMX)

#if !defined(HAVE_SSE)
#define HAVE_SSE 0
#endif  // !defined(HAVE_SSE)

#if !defined(HAVE_SSE2)
#define HAVE_SSE2 0
#endif  // !defined(HAVE_SSE2)

#endif  // HAVE_AMD64 || HAVE_I386

// Error checking
#if HAVE_SSE && !HAVE_MMX
#error SSE is available but not MMX?
#endif  // HAVE_SSE && !HAVE_MMX

#if HAVE_SSE2 && (!HAVE_SSE || !HAVE_MMX)
#error SSE2 is available but not SSE or MMX?
#endif  // HAVE_SSE2 && (!HAVE_SSE || !HAVE_MMX)


#if !defined(CRCUTIL_PREFETCH_WIDTH)
// On newer X5550 CPU, heavily optimized CrcMultiword is 3% faster without
// prefetch for inputs smaller than 8MB and less than 1% slower for 8MB and
// larger blocks. On older Q9650 CPU, the code is 2-3% faster for inputs
// smaller than 8MB, 4-5% slower when length >= 8MB.
// Tested with prefetch length 256, 512, and 4096.
//
// At this moment there is no compelling reason to use prefetching.
//
#define CRCUTIL_PREFETCH_WIDTH 0
#endif  // !defined(CRCUTIL_PREFETCH_WIDTH)


#if HAVE_SSE && CRCUTIL_PREFETCH_WIDTH > 0
#define PREFETCH(src) \
  _mm_prefetch(reinterpret_cast<const char *>(src) + CRCUTIL_PREFETCH_WIDTH, \
               _MM_HINT_T0)
#else
#define PREFETCH(src)
#endif  // HAVE_SSE && CRCUTIL_PREFETCH_WIDTH > 0


// If block size exceeds CRCUTIL_MIN_ALIGN_SIZE, align the data
// before accessing it at word boundary. See generic_crc.cc,
// ALIGN_ON_WORD_BOUNDARY_IF_NEEDED() macro.
#if !defined(CRCUTIL_MIN_ALIGN_SIZE)
#if HAVE_AMD64 || HAVE_I386
#define CRCUTIL_MIN_ALIGN_SIZE (1024)
#else
#define CRCUTIL_MIN_ALIGN_SIZE 0
#endif  // HAVE_AMD64 || HAVE_I386
#endif  // !defined(CRCUTIL_MIN_ALIGN_SIZE)


// Use _mm_crc32_u64/32/8 intrinics?
// If not, they will be implemented in software.
#if !HAVE_I386 && !HAVE_AMD64

#undef CRCUTIL_USE_MM_CRC32
#define CRCUTIL_USE_MM_CRC32 0

#else

#if !defined(CRCUTIL_USE_MM_CRC32)
#if defined(_MSC_VER) || defined(__GNUC__)
#define CRCUTIL_USE_MM_CRC32 1
#else
#define CRCUTIL_USE_MM_CRC32 0
#endif  // defined(_MSC_VER) || defined(__GNUC__)
#endif  // !defined(CRCUTIL_USE_MM_CRC32)

#endif  // !HAVE_I386 && !HAVE_AMD64


// Stringize -- always handy.
#define TO_STRING_VALUE(arg) #arg
#define TO_STRING(arg) TO_STRING_VALUE(arg)


// Compilers give "right shift count >= width of type" warning even
// though the shift happens only under appropriate "if".
#define SHIFT_RIGHT_NO_WARNING(value, bits) \
  ((value) >> (((bits) < (8 * sizeof(value))) ? (bits) : 0))
#define SHIFT_RIGHT_SAFE(value, bits) \
  ((bits) < (8 * sizeof(value)) ? SHIFT_RIGHT_NO_WARNING(value, bits) : 0)

// The same for left shifts.
#define SHIFT_LEFT_NO_WARNING(value, bits) \
  ((value) << (((bits) < (8 * sizeof(value))) ? (bits) : 0))
#define SHIFT_LEFT_SAFE(value, bits) \
  ((bits) < (8 * sizeof(value)) ? SHIFT_LEFT_NO_WARNING(value, bits) : 0)

// GCC-specific macros.
//
#define GCC_VERSION_AVAILABLE(major, minor) \
    (defined(__GNUC__) && \
        (__GNUC__ > (major) || \
            (__GNUC__ == (major) && __GNUC_MINOR__ >= (minor))))


#if defined(__GNUC__)

// The GenericCrc tables must be properly aligned.
// Penalty for misalignment? 50% performance degradation.
// For 128-bit SSE2, the penalty is access violation.
#define GCC_ALIGN_ATTRIBUTE(n) __attribute__((aligned(n)))

#if GCC_VERSION_AVAILABLE(4, 4)
// If not marked as "omit frame pointer",
// GCC won't be able to find enough registers.
#define GCC_OMIT_FRAME_POINTER \
    __attribute__((__optimize__(2, "omit-frame-pointer")))
#endif  // GCC_VERSION_AVAILABLE(4, 4)

#if !defined(__forceinline)
#define __forceinline __attribute__((__always_inline__)) inline
#endif  // !defined(__forceinline)

#if defined(__APPLE_CC__)
// The version of GCC used by Max OS X xCode v 5664 does not understand
// "movq xmm, r64" instruction and requires the use of "movd" (probably
// because of the bug in GCC which treats "movq/movd xmm,r64 or r64,xmm"
// the same).
//
// Leaving common sense aside, let's peek into Intel's instruction
// reference manual. That's what description of MOVD command says:
// MOVD xmm, r/m32 (opcode 66 0F 6E /r)
// MOVD r/m32, xmm (opcode 66 0F 7E /r)
// MOVQ xmm, r/m64 (opcode 66 REX.W 0F 6E /r)
// MOVQ r/m64, xmm (opcode 66 REX.W 0F 7E /r)
#define SSE2_MOVQ "movd"
#else
#define SSE2_MOVQ "movq"
#endif  // defined(__APPLE_CC__)

#endif  // defined(__GNUC__)


// Define compiler-specific macros that were not set yet.
#if !defined(_MSC_VER) && !defined(__forceinline)
#define __forceinline inline
#endif  // !defined(_MSC_VER) && !defined(__forceinline)

#if !defined(GCC_OMIT_FRAME_POINTER)
#define GCC_OMIT_FRAME_POINTER
#endif  // !defined(GCC_OMIT_FRAME_POINTER)

#if !defined(GCC_ALIGN_ATTRIBUTE)
#define GCC_ALIGN_ATTRIBUTE(n)
#endif  // !defined(GCC_ALIGN_ATTRIBUTE)


#endif  // CRCUTIL_PLATFORM_H_
