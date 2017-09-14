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

// Casting between integers and compound CRC types.

#ifndef CRCUTIL_CRC_CASTS_H_
#define CRCUTIL_CRC_CASTS_H_

#include "base_types.h"   // uint8, uint64
#include "platform.h"     // __forceinline

namespace crcutil {

// Downcasts a value of (oftentimes larger) Crc type to (smaller base integer)
// Result type, enabling specialized downcasts implemented by "large integer"
// classes (e.g. uint128_sse2).
template<typename Crc, typename Result>
__forceinline Result Downcast(const Crc &x) {
  return static_cast<Result>(x);
}

// Extracts 8 least significant bits from a value of Crc type.
#define TO_BYTE(x) Downcast<Crc, uint8>(x)

// Converts a pair of uint64 bit values into single value of CRC type.
// It is caller's responsibility to ensure that the input is correct.
template<typename Crc>
__forceinline Crc CrcFromUint64(uint64 lo, uint64 hi = 0) {
  if (sizeof(Crc) <= sizeof(lo)) {
    return static_cast<Crc>(lo);
  } else {
    // static_cast to keep compiler happy.
    Crc result = static_cast<Crc>(hi);
    result = SHIFT_LEFT_SAFE(result, 8 * sizeof(lo));
    result ^= lo;
    return result;
  }
}

// Converts Crc value to a pair of uint64 values.
template<typename Crc>
__forceinline void Uint64FromCrc(const Crc &crc,
                                 uint64 *lo, uint64 *hi = NULL) {
  if (sizeof(*lo) >= sizeof(crc)) {
    *lo = Downcast<Crc, uint64>(crc);
    if (hi != NULL) {
      *hi = 0;
    }
  } else {
    *lo = Downcast<Crc, uint64>(crc);
    *hi = Downcast<Crc, uint64>(SHIFT_RIGHT_SAFE(crc, 8 * sizeof(lo)));
  }
}

}  // namespace crcutil

#endif  // CRCUTIL_CRC_CASTS_H_
