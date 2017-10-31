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

// This is the only file where all details of CRC implementation are buried.

#include "interface.h"

#include "aligned_alloc.h"
#include "crc32c_sse4.h"
#include "generic_crc.h"
#include "protected_crc.h"
#include "rolling_crc.h"

// Align all CRC tables on kAlign boundary.
// Shall be exact power of 2.
static size_t kAlign = 4 * 1024;

using namespace crcutil;

#if (!defined(__clang__) && defined(__GNUC__))
// Suppress 'invalid access to non-static data member ...  of NULL object'
#undef offsetof
#define offsetof(TYPE, MEMBER) (reinterpret_cast <size_t> \
    ((&reinterpret_cast <const char &>( \
        reinterpret_cast <const TYPE *>(1)->MEMBER))) - 1)
#endif  // defined(__GNUC__)

namespace crcutil_interface {

template<typename CrcImplementation, typename RollingCrcImplementation>
    class Implementation : public CRC {
 public:
  typedef typename CrcImplementation::Crc Crc;
  typedef Implementation<CrcImplementation, RollingCrcImplementation> Self;

  Implementation(const Crc &poly,
                 size_t degree,
                 bool canonical,
                 const Crc &roll_start_value,
                 size_t roll_length)
    : crc_(poly, degree, canonical),
      rolling_crc_(crc_, roll_length, roll_start_value) {
  }

  static Self *Create(const Crc &poly,
                      size_t degree,
                      bool canonical,
                      const Crc &roll_start_value,
                      size_t roll_length,
                      const void **allocated_memory) {
    void *memory = AlignedAlloc(sizeof(Self),
                                offsetof(Self, crc_),
                                kAlign,
                                allocated_memory);
    return new(memory) Self(poly,
                            degree,
                            canonical,
                            roll_start_value,
                            roll_length);
  }

  virtual void Delete() {
    AlignedFree(this);
  }

  void *operator new(size_t, void *p) {
    return p;
  }

  virtual void GeneratingPolynomial(/* OUT */ UINT64 *lo,
                                    /* OUT */ UINT64 *hi = NULL) const {
    SetValue(crc_.Base().GeneratingPolynomial(), lo, hi);
  }

  virtual size_t Degree() const {
    return crc_.Base().Degree();
  }

  virtual void CanonizeValue(/* OUT */ UINT64 *lo,
                             /* OUT */ UINT64 *hi = NULL) const {
    SetValue(crc_.Base().Canonize(), lo, hi);
  }

  virtual void RollStartValue(/* OUT */ UINT64 *lo,
                              /* OUT */ UINT64 *hi = NULL) const {
    SetValue(rolling_crc_.StartValue(), lo, hi);
  }

  virtual size_t RollWindowBytes() const {
    return rolling_crc_.WindowBytes();
  }

  virtual void SelfCheckValue(/* OUT */ UINT64 *lo,
                              /* OUT */ UINT64 *hi = NULL) const {
    Crc crc = crc_.CrcDefault(&crc_, sizeof(crc_), 0);
    crc = crc_.CrcDefault(&rolling_crc_, sizeof(rolling_crc_), crc);
    SetValue(crc, lo, hi);
  }

  virtual void Compute(const void *data,
                       size_t bytes,
                       /* INOUT */ UINT64 *lo,
                       /* INOUT */ UINT64 *hi = NULL) const {
    SetValue(crc_.CrcDefault(data, bytes, GetValue(lo, hi)), lo, hi);
  }

  virtual void RollStart(const void *data,
                         /* INOUT */ UINT64 *lo,
                         /* INOUT */ UINT64 *hi = NULL) const {
    SetValue(rolling_crc_.Start(data), lo, hi);
  }

  virtual void Roll(size_t byte_out,
                    size_t byte_in,
                    /* INOUT */ UINT64 *lo,
                    /* INOUT */ UINT64 *hi = NULL) const {
    SetValue(rolling_crc_.Roll(GetValue(lo, hi), byte_out, byte_in), lo, hi);
  }

  virtual void CrcOfZeroes(UINT64 bytes,
                           /* INOUT */ UINT64 *lo,
                           /* INOUT */ UINT64 *hi = NULL) const {
    SetValue(crc_.Base().CrcOfZeroes(bytes, GetValue(lo, hi)), lo, hi);
  }

  virtual void ChangeStartValue(
      UINT64 start_old_lo, UINT64 start_old_hi,
      UINT64 start_new_lo, UINT64 start_new_hi,
      UINT64 bytes,
      /* INOUT */ UINT64 *lo,
      /* INOUT */ UINT64 *hi = NULL) const {
    SetValue(crc_.Base().ChangeStartValue(
                    GetValue(lo, hi),
                    bytes,
                    GetValue(start_old_lo, start_old_hi),
                    GetValue(start_new_lo, start_new_hi)),
             lo,
             hi);
  }

  virtual void Concatenate(UINT64 crcB_lo, UINT64 crcB_hi,
                           UINT64 bytes_B,
                           /* INOUT */ UINT64* crcA_lo,
                           /* INOUT */ UINT64* crcA_hi = NULL) const {
    SetValue(crc_.Base().Concatenate(GetValue(crcA_lo, crcA_hi),
                                     GetValue(crcB_lo, crcB_hi),
                                     bytes_B),
             crcA_lo,
             crcA_hi);
  }

  virtual size_t StoreComplementaryCrc(
      void *dst,
      UINT64 message_crc_lo, UINT64 message_crc_hi,
      UINT64 result_crc_lo, UINT64 result_crc_hi = 0) const {
    return crc_.Base().StoreComplementaryCrc(
        dst,
        GetValue(message_crc_lo, message_crc_hi),
        GetValue(result_crc_lo, result_crc_hi));
  }

  virtual size_t StoreCrc(void *dst,
                          UINT64 lo,
                          UINT64 hi = 0) const {
    return crc_.Base().StoreCrc(dst, GetValue(lo, hi));
  }

  virtual void CrcOfCrc(/* OUT */ UINT64 *lo,
                        /* OUT */ UINT64 *hi = NULL) const {
    SetValue(crc_.Base().CrcOfCrc(), lo, hi);
  }

 private:
  static Crc GetValue(UINT64 *lo, UINT64 *hi) {
    if (sizeof(Crc) <= sizeof(*lo)) {
      return CrcFromUint64<Crc>(*lo);
    } else {
      return CrcFromUint64<Crc>(*lo, *hi);
    }
  }

  static Crc GetValue(UINT64 lo, UINT64 hi) {
    return CrcFromUint64<Crc>(lo, hi);
  }

  static void SetValue(const Crc &crc, UINT64 *lo, UINT64 *hi) {
    Uint64FromCrc<Crc>(crc,
                       reinterpret_cast<crcutil::uint64 *>(lo),
                       reinterpret_cast<crcutil::uint64 *>(hi));
  }

  const CrcImplementation crc_;
  const RollingCrcImplementation rolling_crc_;

  const Self &operator =(const Self &) {}
};

#if defined(_MSC_VER)
// 'use_sse4_2' : unreferenced formal parameter
#pragma warning(disable: 4100)
#endif  // defined(_MSC_VER)

bool CRC::IsSSE42Available() {
#if HAVE_AMD64 || HAVE_I386
  return Crc32cSSE4::IsSSE42Available();
#else
  return false;
#endif  // HAVE_AMD64 || HAVE_I386
}

CRC::~CRC() {}
CRC::CRC() {}

CRC *CRC::Create(UINT64 poly_lo,
                 UINT64 poly_hi,
                 size_t degree,
                 bool canonical,
                 UINT64 roll_start_value_lo,
                 UINT64 roll_start_value_hi,
                 size_t roll_length,
                 bool use_sse4_2,
                 const void **allocated_memory) {
  if (degree == 0) {
    return NULL;
  }

  if (degree > 64) {
#if !HAVE_SSE2
    return NULL;
#else
    if (degree > 128) {
      return NULL;
    }
    uint128_sse2 poly = CrcFromUint64<uint128_sse2>(poly_lo, poly_hi);
    if (degree != 128 && (poly >> degree) != 0) {
      return NULL;
    }
    uint128_sse2 roll_start_value =
        CrcFromUint64<uint128_sse2>(roll_start_value_lo, roll_start_value_hi);
    if (degree != 128 && (roll_start_value >> degree) != 0) {
      return NULL;
    }
#if HAVE_I386
    typedef GenericCrc<uint128_sse2, uint128_sse2, crcutil::uint32, 3> Crc128;
#elif defined(__GNUC__) && GCC_VERSION_AVAILABLE(4, 5)
    typedef GenericCrc<uint128_sse2, uint128_sse2, crcutil::uint64, 6> Crc128;
#else
    typedef GenericCrc<uint128_sse2, uint128_sse2, crcutil::uint64, 4> Crc128;
#endif  // HAVE_I386
    return Implementation<Crc128, RollingCrc<Crc128> >::Create(
        poly,
        degree,
        canonical,
        roll_start_value,
        roll_length,
        allocated_memory);
#endif  // !HAVE_SSE2
  }

#if CRCUTIL_USE_MM_CRC32 && (HAVE_I386 || HAVE_AMD64)
  if (use_sse4_2 &&
      degree == Crc32cSSE4::FixedDegree() &&
      poly_lo == Crc32cSSE4::FixedGeneratingPolynomial() &&
      poly_hi == 0) {
      if (roll_start_value_hi != 0 || (roll_start_value_lo >> 32) != 0) {
        return NULL;
      }
    return Implementation<Crc32cSSE4, RollingCrc32cSSE4>::Create(
        static_cast<size_t>(poly_lo),
        degree,
        canonical,
        static_cast<size_t>(roll_start_value_lo),
        static_cast<size_t>(roll_length),
        allocated_memory);
  }
#endif  // CRCUTIL_USE_MM_CRC32 && (HAVE_I386 || HAVE_AMD64)

  if (poly_hi != 0 || (degree != 64 && (poly_lo >> degree) != 0)) {
    return NULL;
  }
  if (roll_start_value_hi != 0 ||
      (degree != 64 && (roll_start_value_lo >> degree) != 0)) {
    return NULL;
  }
  typedef GenericCrc<crcutil::uint64, crcutil::uint64, crcutil::uint64, 4>
      Crc64;
  return Implementation<Crc64, RollingCrc<Crc64> >::Create(
      poly_lo,
      degree,
      canonical,
      roll_start_value_lo,
      roll_length,
      allocated_memory);
}

}  // namespace crcutil_interface
