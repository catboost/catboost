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

// Implements a limited set of 128-bit arithmetic operations
// (the ones that are used by CRC) using SSE2 intrinsics.

#ifndef CRCUTIL_UINT128_SSE2_H_
#define CRCUTIL_UINT128_SSE2_H_

#include "base_types.h"
#include "crc_casts.h"      // Downcast, CrcFromUint64, Uint64FromCrc
#include "platform.h"

#if HAVE_SSE2

namespace crcutil {

// Specialized functions handling __m128i.
template<> __forceinline uint64 Downcast(const __m128i &value) {
#if HAVE_AMD64 && defined(__GNUC__)
    // GCC 4.4.x is too smart and, instead of MOVQ, generates SSE4 PEXTRQ
    // instruction when the code is compiled with -mmsse4.
    // Fixed in 4.5 which generates conversion through memory (why?).
    // And -- yes, it makes quite measurable difference.
    uint64 temp;
    asm(SSE2_MOVQ " %[i128], %[u64]\n" : [u64] "=r" (temp) : [i128] "x" (value));
    return temp;
#elif HAVE_AMD64 && (!defined(_MSC_FULL_VER) || _MSC_FULL_VER > 150030729)
    return static_cast<uint64>(_mm_cvtsi128_si64(value));
#else
    // 64-bit CL 15.00.30729.1 -O2 generates incorrect code (tests fail).
    // _mm_cvtsi128_si64() is not available on i386.
    uint64 temp;
    _mm_storel_epi64(reinterpret_cast<__m128i *>(&temp), value);
    return temp;
#endif
}


class uint128_sse2 {
 public:
  uint128_sse2() {}
  ~uint128_sse2() {}

  // Default casts to uint128_sse2 and assignment operator.
  __forceinline void operator =(uint64 value) {
#if HAVE_AMD64 && defined(__GNUC__) && !GCC_VERSION_AVAILABLE(4, 5)
    // Prevent generation of SSE4 pinsrq insruction when
    // compiling with GCC 4.4.x with -msse4 flag.
    asm(SSE2_MOVQ " %[u64], %[i128]\n" : [i128] "=x" (x_) : [u64] "r" (value));
#elif HAVE_AMD64
    x_ = _mm_cvtsi64_si128(static_cast<int64>(value));
#else
    x_ = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&value));
#endif
  }
  __forceinline uint128_sse2(uint64 x) {
    *this = x;
  }
  __forceinline uint128_sse2(const __m128i x) : x_(x) {
  }
  __forceinline operator __m128i() const {
    return x_;
  }
  __forceinline void operator =(const uint128_sse2 &x) {
    x_ = x.x_;
  }

  // Extracts 64 less significant bits.
  __forceinline uint64 to_uint64() const {
    return Downcast<__m128i, uint64>(x_);
  }

  // Comparisons.
  __forceinline bool operator ==(const uint128_sse2 &y) const {
    union {
      __m128i i128;
      uint64  u64[2];
    } t;
    t.i128 = _mm_xor_si128(x_, y.x_);
    return (t.u64[0] | t.u64[1]) == 0;
  }
  __forceinline bool operator ==(uint64 value) const {
    union {
      __m128i i128;
      uint64  u64[2];
    } t;
    t.i128 = x_;
    return (t.u64[0] == value && t.u64[1] == 0);
  }
  __forceinline bool operator !=(const uint128_sse2 &y) const {
    union {
      __m128i i128;
      uint64  u64[2];
    } t;
    t.i128 = _mm_xor_si128(x_, y.x_);
    return (t.u64[0] | t.u64[1]) != 0;
  }
  __forceinline bool operator !=(uint64 value) const {
    union {
      __m128i i128;
      uint64  u64[2];
    } t;
    t.i128 = x_;
    return (t.u64[0] != value || t.u64[1] != 0);
  }

  __forceinline bool operator <(const uint128_sse2 &y) const {
    union {
      __m128i i128;
      uint64  u64[2];
    } xx, yy;
    xx.i128 = x_;
    yy.i128 = y.x_;
    return (xx.u64[0] < yy.u64[0] ||
           (xx.u64[0] == yy.u64[0] && xx.u64[1] < yy.u64[1]));
  }

  // Bitwise logic operators.
  __forceinline uint128_sse2 operator ^(const uint128_sse2 &y) const {
    return _mm_xor_si128(x_, y.x_);
  }
  __forceinline uint128_sse2 operator &(const uint128_sse2 &y) const {
    return _mm_and_si128(x_, y.x_);
  }
  __forceinline uint128_sse2 operator |(const uint128_sse2 &y) const {
    return _mm_or_si128(x_, y.x_);
  }

  __forceinline void operator ^=(const uint128_sse2 &y) {
    *this = *this ^ y.x_;
  }
  __forceinline void operator &=(const uint128_sse2 &y) {
    *this = *this & y.x_;
  }
  __forceinline void operator |=(const uint128_sse2 &y) {
    *this = *this | y.x_;
  }

  // Arithmetic operators.
  __forceinline uint128_sse2 operator +(uint64 y) const {
    union {
      __m128i i128;
      uint64  u64[2];
    } temp;
    temp.i128 = x_;
    // a + b >= 2**64 iff
    // a + b > (2**64 - 1) iff
    // a > (2**64 - 1) - b iff
    // a > ~b
    if (temp.u64[0] > ~y) {
      temp.u64[1] += 1;
    }
    temp.u64[0] += y;
    return temp.i128;
  }
  __forceinline void operator +=(uint64 x) {
    *this = *this + x;
  }
  __forceinline uint128_sse2 operator -(uint64 y) const {
    union {
      __m128i i128;
      uint64  u64[2];
    } temp;
    temp.i128 = x_;
    if (temp.u64[0] < y) {
      temp.u64[1] -= 1;
    }
    temp.u64[0] -= y;
    return temp.i128;
  }
  __forceinline void operator -=(uint64 x) {
    *this = *this - x;
  }

  // Bitwise logical shifts.
  __forceinline uint128_sse2 operator >>(const int bits) const {
    if (bits == 8) {
      return _mm_srli_si128(x_, 1);
    } else if (bits == 16) {
      return _mm_srli_si128(x_, 2);
    } else if (bits == 32) {
      return _mm_srli_si128(x_, 4);
    } else if (bits == 64) {
      return _mm_srli_si128(x_, 8);
    } else {
      return long_shift_right(bits);
    }
  }
  __forceinline uint128_sse2 operator >>(const size_t bits) const {
    return *this >> static_cast<int>(bits);
  }
  __forceinline void operator >>=(const int bits) {
    *this = *this >> bits;
  }
  __forceinline void operator >>=(const size_t bits) {
    *this = *this >> static_cast<int>(bits);
  }

  __forceinline uint128_sse2 operator <<(int bits) const {
    if (bits == 8) {
      return _mm_slli_si128(x_, 1);
    } else if (bits == 16) {
      return _mm_slli_si128(x_, 2);
    } else if (bits == 32) {
      return _mm_slli_si128(x_, 4);
    } else if (bits == 64) {
      return _mm_slli_si128(x_, 8);
    } else {
      return long_shift_left(bits);
    }
  }
  __forceinline uint128_sse2 operator <<(size_t bits) const {
    return *this << static_cast<int>(bits);
  }
  __forceinline void operator <<=(int bits) {
    *this = *this << bits;
  }
  __forceinline void operator <<=(size_t bits) {
    *this = *this << static_cast<int>(bits);
  }

 protected:
  __forceinline uint128_sse2 long_shift_right(int bits) const {
    union {
      __m128i i128;
      uint64 u64[2];
    } x;
    x.i128 = x_;
    for (; bits > 0; --bits) {
      x.u64[0] >>= 1;
      if (x.u64[1] & 1) {
        x.u64[0] |= static_cast<uint64>(1) << 63;
      }
      x.u64[1] >>= 1;
    }
    return x.i128;
  }

  __forceinline uint128_sse2 long_shift_left(int bits) const {
    union {
      __m128i i128;
      int64   i64[2];
    } x;
    x.i128 = x_;
    for (; bits > 0; --bits) {
      x.i64[1] <<= 1;
      if (x.i64[0] < 0) {
        x.i64[1] |= 1;
      }
      x.i64[0] <<= 1;
    }
    return x.i128;
  }

  __m128i x_;
} GCC_ALIGN_ATTRIBUTE(16);


// Specialized versions.
template<> __forceinline uint64 Downcast(const uint128_sse2 &x) {
  return x.to_uint64();
}
template<> __forceinline uint32 Downcast(const uint128_sse2 &x) {
  return static_cast<uint32>(x.to_uint64());
}
template<> __forceinline uint16 Downcast(const uint128_sse2 &x) {
  return static_cast<uint16>(x.to_uint64());
}
template<> __forceinline uint8 Downcast(const uint128_sse2 &x) {
  return static_cast<uint8>(x.to_uint64());
}

template<> __forceinline uint128_sse2 CrcFromUint64(uint64 lo, uint64 hi) {
  union {
    __m128i i128;
    uint64  u64[2];
  } temp;
  temp.u64[0] = lo;
  temp.u64[1] = hi;
  return temp.i128;
}

template<> __forceinline void Uint64FromCrc(const uint128_sse2 &crc,
                              uint64 *lo, uint64 *hi) {
  union {
    __m128i i128;
    uint64  u64[2];
  } temp;
  temp.i128 = crc;
  *lo = temp.u64[0];
  *hi = temp.u64[1];
}

}  // namespace crcutil

#endif  // HAVE_SSE2

#endif  // CRCUTIL_UINT128_SSE2_H_
