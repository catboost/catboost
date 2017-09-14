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

// Example how to use CRC implementation via the interface which
// hides details of implementation.
//
// The raw implementation is not indended to be used in a project
// directly because:
// - Implementation lives in the header files because that is the
//   only way to use templates efficiently.
// - Header files are quite "dirty" -- they define and use a
//   lot of macros. Bringing these macros to all files in
//   a project is not particularly good idea.
// - The code takes forever to compile with GCC (e.g. GCC
//   4.4.3 and 4.5.0 compile the unittest for about 30 seconds).
//
// Solution:
// - Create your own, clean interface.
// - Do not expose interface internals in a header file.
// - Proxy all calls to your interface to CRC implementation.
// - Keep only one copy of actual implementation.

#ifndef CRCUTIL_INTERFACE_H_
#define CRCUTIL_INTERFACE_H_

#include "std_headers.h"    // size_t

namespace crcutil_interface {

// Many projects define their own uint64. Do it here.
typedef unsigned long long UINT64;

class CRC {
 public:
  // Creates new instance of CRC class.
  // If arguments are illegal (e.g. provided generating polynomial
  // has more bits than provided degree), returns NULL.
  //
  // poly_* - generating polynomial (reversed bit format).
  // degree - degree of generating polynomial.
  // canonical - if true, input CRC value will be XOR'ed with
  //             (inverted) before and after CRC computation.
  // roll_start_value - starting value of rolling CRC.
  // roll_window_bytes - length of rolling CRC window in bytes.
  //                     If roll_length is 0, roll_start_value
  //                     shall be 0.
  // use_sse4_2 - if true, use SSE4.2 crc32 instruction to compute
  //              CRC when generating polynomial is CRC32C (Castagnoli)
  // allocated_memory - optional (may be NULL) address of a variable
  //                    to store the address of actually allocated memory.
  static CRC *Create(UINT64 poly_lo,
                     UINT64 poly_hi,
                     size_t degree,
                     bool canonical,
                     UINT64 roll_start_value_lo,
                     UINT64 roll_start_value_hi,
                     size_t roll_window_bytes,
                     bool use_sse4_2,
                     const void **allocated_memory);

  // Deletes the instance of CRC class.
  virtual void Delete() = 0;

  // Returns true if SSE4.2 is available.
  static bool IsSSE42Available();

  // Returns generating polynomial.
  virtual void GeneratingPolynomial(/* OUT */ UINT64 *lo,
                                    /* OUT */ UINT64 *hi = NULL) const = 0;

  // Returns degree of generating polynomial.
  virtual size_t Degree() const = 0;

  // Returns canonization constant used to XOR crc value
  // before and after CRC computation.
  virtual void CanonizeValue(/* OUT */ UINT64 *lo,
                             /* OUT */ UINT64 *hi = NULL) const = 0;

  // Returns rolling CRC starting value.
  virtual void RollStartValue(/* OUT */ UINT64 *lo,
                              /* OUT */ UINT64 *hi = NULL) const = 0;

  // Returns length of rolling CRC window.
  virtual size_t RollWindowBytes() const = 0;

  // Returns CRC of CRC tables to enable verification
  // of integrity of CRC function itself by comparing
  // the result with pre-computed value.
  virtual void SelfCheckValue(/* OUT */ UINT64 *lo,
                              /* OUT */ UINT64 *hi = NULL) const = 0;

  // Given CRC value of previous chunk of data,
  // extends it to new chunk, retuning the result in-place.
  //
  // If degree of CRC polynomial is 64 or less,
  // (*hi) will not be touched.
  virtual void Compute(const void *data,
                       size_t bytes,
                       /* INOUT */ UINT64 *lo,
                       /* INOUT */ UINT64 *hi = NULL) const = 0;

  // Starts rolling CRC by computing CRC of first
  // "roll_length" bytes of "data", using "roll_start_value"
  // as starting value (see Create()).
  // Should not be called if the value of "roll_value" was 0.
  virtual void RollStart(const void *data,
                         /* OUT */ UINT64 *lo,
                         /* OUT */ UINT64 *hi = NULL) const = 0;

  // Rolls CRC by 1 byte, given the bytes leaving and
  // entering the window of "roll_length" bytes.
  // RollStart() should be called before "Roll".
  // Should not be called if the value of "roll_value" was 0.
  virtual void Roll(size_t byte_out,
                    size_t byte_in,
                    /* INOUT */ UINT64 *lo,
                    /* INOUT */ UINT64 *hi = NULL) const = 0;

  // Computes CRC of sequence of zeroes -- without touching the data.
  virtual void CrcOfZeroes(UINT64 bytes,
                           /* INOUT */ UINT64 *lo,
                           /* INOUT */ UINT64 *hi = NULL) const = 0;

  // Computes value of CRC(A, bytes, start_new) given known
  // crc=CRC(A, bytes, start_old) -- without touching the data.
  virtual void ChangeStartValue(
      UINT64 start_old_lo, UINT64 start_old_hi,
      UINT64 start_new_lo, UINT64 start_new_hi,
      UINT64 bytes,
      /* INOUT */ UINT64 *lo,
      /* INOUT */ UINT64 *hi = NULL) const = 0;

  // Returns CRC of concatenation of blocks A and B when CRCs
  // of blocks A and B are known -- without touching the data.
  //
  // To be precise, given CRC(A, |A|, startA) and CRC(B, |B|, 0),
  // returns CRC(AB, |AB|, startA).
  virtual void Concatenate(UINT64 crcB_lo, UINT64 crcB_hi,
                           UINT64 bytes_B,
                           /* INOUT */ UINT64* crcA_lo,
                           /* INOUT */ UINT64* crcA_hi = NULL) const = 0;

  // Given CRC of a message, stores extra (degree + 7)/8 bytes after
  // the message so that CRC(message+extra, start) = result.
  // Does not change CRC start value (use ChangeStartValue for that).
  // Returns number of stored bytes.
  virtual size_t StoreComplementaryCrc(
      void *dst,
      UINT64 message_crc_lo, UINT64 message_crc_hi,
      UINT64 result_crc_lo, UINT64 result_crc_hi = 0) const = 0;

  // Stores given CRC of a message as (degree + 7)/8 bytes filled
  // with 0s to the right. Returns number of stored bytes.
  // CRC of the message and stored CRC is a constant value returned
  // by CrcOfCrc() -- it does not depend on contents of the message.
  virtual size_t StoreCrc(/* OUT */ void *dst,
                          UINT64 lo,
                          UINT64 hi = 0) const = 0;

  // Computes expected CRC value of CRC(Message,CRC(Message))
  // when CRC is stored after the message. This value is fixed
  // and does not depend on the message or CRC start value.
  virtual void CrcOfCrc(/* OUT */ UINT64 *lo,
                        /* OUT */ UINT64 *hi = NULL) const = 0;

 protected:
  // CRC instance should be created only once (most of the time):
  // - Creation and initializion is relatively expensive.
  // - CRC is fully defined by its generating polynomials
  //   (well, and few more parameters).
  // - CRC instances are pure constants. There is no
  //   reason to have 2 instances of the same CRC.
  // - There are not too many generating polynomials that are
  //   used on practice. It is hard to imagine a project
  //   which uses 50 different generating polynomials.
  //   Thus, a handful of CRC instances is sufficient
  //   to cover the needs of even very large project.
  // - Finally and most importantly, CRC tables should be
  //   aligned properly. No, the instances of CRC class
  //   are not created by blind "new" -- they use placement
  //   "new" and, in absense of placement "delete",
  //   should be deleted by calling explicit Delete() method.
  virtual ~CRC();

  // Cannot instantiate the class -- instances may be created
  // by CRC::Create() only.
  CRC();
};

}  // namespace crcutil_interface


#endif  // CRCUTIL_INTERFACE_H_
