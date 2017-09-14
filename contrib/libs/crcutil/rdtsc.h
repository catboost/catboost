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

// Reads CPU cycle counter on AMD64 and I386 (for performance measurements).
// Thanks to __rdtsc() intrinsic, it's easy with Microsoft and Intel
// compilers, but real pain with GCC.

#ifndef CRCUTIL_RDTSC_H_
#define CRCUTIL_RDTSC_H_

#include "platform.h"

namespace crcutil {

struct Rdtsc {
  static inline uint64 Get() {
#if defined(_MSC_VER) && (HAVE_AMD64 || HAVE_I386)
    return __rdtsc();
#elif defined(__GNUC__) && HAVE_AMD64
    int64 result;
    __asm__ volatile(
        "rdtsc\n"
        : "=a" (result));
    return result;
#elif defined(__GNUC__) && HAVE_I386
    // If "low" and "high" are defined as "uint64" to
    // avoid explicit cast to uint64, GCC 4.5.0 in "-m32" mode
    // fails with "impossible register constraint" error
    // (no, it is not because one cannot use 64-bit value as argument
    // for 32-bit register, but because its register allocator
    // could not resolve a conflict under high register pressure).
    uint32 low;
    uint32 high;
    __asm__ volatile(
        "rdtsc\n"
        : "=a" (low), "=d" (high));
    return ((static_cast<uint64>(high) << 32) | low);
#else
    // It is hard to find low overhead timer with
    // sub-millisecond resolution and granularity.
    return 0;
#endif
  }
};

}  // namespace crcutil

#endif  // CRCUTIL_RDTSC_H_
