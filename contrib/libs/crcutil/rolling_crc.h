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

// Implements rolling CRC (e.g. for Rabin fingerprinting).

#ifndef CRCUTIL_ROLLING_CRC_H_
#define CRCUTIL_ROLLING_CRC_H_

#include "base_types.h"   // size_t, uint8
#include "crc_casts.h"    // TO_BYTE

namespace crcutil {

#pragma pack(push, 16)

// CrcImplementation should provide:
// - typename Crc
// - typename TableEntry
// - typename Word
// - Crc CrcDefault(const void *data, size_t bytes, const Crc &start)
// - const GfUtil<Crc> &Base() const
template<typename CrcImplementation> class RollingCrc {
 public:
  typedef typename CrcImplementation::Crc Crc;
  typedef typename CrcImplementation::TableEntry TableEntry;
  typedef typename CrcImplementation::Word Word;

  RollingCrc() {}

  // Initializes internal data structures.
  // Retains reference to "crc" instance -- it is used by Start().
  RollingCrc(const CrcImplementation &crc,
             size_t roll_window_bytes,
             const Crc &start_value) {
    Init(crc, roll_window_bytes, start_value);
  }

  // Computes crc of "roll_window_bytes" using
  // "start_value" of "crc" (see Init()).
  Crc Start(const void *data) const {
    return crc_->CrcDefault(data, roll_window_bytes_, start_value_);
  }

  // Computes CRC of "roll_window_bytes" starting in next position.
  Crc Roll(const Crc &old_crc, size_t byte_out, size_t byte_in) const {
    return (old_crc >> 8) ^ in_[TO_BYTE(old_crc) ^ byte_in] ^ out_[byte_out];
  }

  // Initializes internal data structures.
  // Retains reference to "crc" instance -- it is used by Start().
  void Init(const CrcImplementation &crc,
            size_t roll_window_bytes,
            const Crc &start_value) {
    crc_ = &crc;
    roll_window_bytes_ = roll_window_bytes;
    start_value_ = start_value;

    Crc add = crc.Base().Canonize() ^ start_value;
    add = crc.Base().Multiply(add, crc.Base().Xpow8N(roll_window_bytes));
    add ^= crc.Base().Canonize();
    Crc mul = crc.Base().One() ^ crc.Base().Xpow8N(1);
    add = crc.Base().Multiply(add, mul);

    mul = crc.Base().XpowN(8 * roll_window_bytes + crc.Base().Degree());
    for (size_t i = 0; i < 256; ++i) {
      out_[i] = static_cast<TableEntry>(
                    crc.Base().MultiplyUnnormalized(
                        static_cast<Crc>(i), 8, mul) ^ add);
    }
    for (size_t i = 0; i < 256; ++i) {
      in_[i] = crc.crc_word_[sizeof(Word) - 1][i];
    }
  }

  // Returns start value.
  Crc StartValue() const { return start_value_; }

  // Returns length of roll window.
  size_t WindowBytes() const { return roll_window_bytes_; }

 protected:
  TableEntry in_[256];
  TableEntry out_[256];

  // Used only by Start().
  Crc start_value_;
  const CrcImplementation *crc_;
  size_t roll_window_bytes_;
} GCC_ALIGN_ATTRIBUTE(16);

#pragma pack(pop)

}  // namespace crcutil

#endif  // CRCUTIL_ROLLING_CRC_H_
