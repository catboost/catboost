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

// Protects CRC tables with its own CRC.
// CRC tables get corrupted too, and if corruption is
// not caught, data poisoning becomes a reality.

#ifndef CRCUTIL_PROTECTED_CRC_H_
#define CRCUTIL_PROTECTED_CRC_H_

namespace crcutil {

#pragma pack(push, 16)

// Class CrcImplementation should not have virtual functions:
// vptr is stored as the very first field, vptr value is defined
// at runtime, so it is impossible to CRC(*this) once and
// guarantee that this value will not change from run to run.
//
template<typename CrcImplementation> class ProtectedCrc
    : public CrcImplementation {
 public:
  typedef typename CrcImplementation::Crc Crc;

  // Returns check value that the caller should compare
  // against pre-computed, trusted constant.
  //
  // Computing SelfCheckValue() after CRC initialization,
  // storing it in memory, and periodically checking against
  // stored value may not work: if CRC tables were initialized
  // incorrectly and/or had been corrupted during initialization,
  // CheckValue() will return garbage. Garbage in, garbage out.
  // Consequitive checks will not detect a problem, the application
  // will happily produce and save the data with corrupt CRC.
  //
  // The application should call SelfCheckValue() regularly:
  // 1. First and foremost, on every CRC mismatch.
  // 2. After CRC'ing the  data but before sending it out or writing it.
  // 3. Worst case, every Nth CRC'ed byte or every Nth call to CRC.
  //
  Crc SelfCheckValue() const {
    return CrcDefault(this, sizeof(*this), 0);
  }
} GCC_ALIGN_ATTRIBUTE(16);

#pragma pack(pop)

}  // namespace crcutil

#endif  // CRCUTIL_PROTECTED_CRC_H_
