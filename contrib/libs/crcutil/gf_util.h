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

// Defines GfUtil template class which implements
// 1. some useful operations in GF(2^n),
// 2. CRC helper function (e.g. concatenation of CRCs) which are
//    not affected by specific implemenation of CRC computation per se.
//
// Please read crc.pdf to understand how it all works.

#ifndef CRCUTIL_GF_UTIL_H_
#define CRCUTIL_GF_UTIL_H_

#include "base_types.h"   // uint8, uint64
#include "crc_casts.h"    // TO_BYTE()
#include "platform.h"     // GCC_ALIGN_ATTRIBUTE(16), SHIFT_*_SAFE

namespace crcutil {

#pragma pack(push, 16)

// "Crc" is the type used internally and to return values of N-bit CRC.
template<typename Crc> class GfUtil {
 public:
  // Initializes the tables given generating polynomial of degree (degree).
  // If "canonical" is true, starting CRC value and computed CRC value will be
  // XOR-ed with 111...111.
  GfUtil() {}
  GfUtil(const Crc &generating_polynomial, size_t degree, bool canonical) {
    Init(generating_polynomial, degree, canonical);
  }
  void Init(const Crc &generating_polynomial, size_t degree, bool canonical) {
    Crc one = 1;
    one <<= degree - 1;
    this->generating_polynomial_ = generating_polynomial;
    this->crc_bytes_ = (degree + 7) >> 3;
    this->degree_ = degree;
    this->one_ = one;
    if (canonical) {
      this->canonize_ = one | (one - 1);
    } else {
      this->canonize_ = 0;
    }
    this->normalize_[0] = 0;
    this->normalize_[1] = generating_polynomial;

    Crc k = one >> 1;
    for (size_t i = 0; i < sizeof(uint64) * 8; ++i) {
      this->x_pow_2n_[i] = k;
      k = Multiply(k, k);
    }

    this->crc_of_crc_ = Multiply(this->canonize_,
                                 this->one_ ^ Xpow8N((degree + 7) >> 3));

    FindLCD(Xpow8N(this->crc_bytes_), &this->x_pow_minus_W_);
  }

  // Returns generating polynomial.
  Crc GeneratingPolynomial() const {
    return this->generating_polynomial_;
  }

  // Returns number of bits in CRC (degree of generating polynomial).
  size_t Degree() const {
    return this->degree_;
  }

  // Returns start/finish adjustment constant.
  Crc Canonize() const {
    return this->canonize_;
  }

  // Returns normalized value of 1.
  Crc One() const {
    return this->one_;
  }

  // Returns value of CRC(A, |A|, start_new) given known
  // crc=CRC(A, |A|, start_old) -- without touching the data.
  Crc ChangeStartValue(const Crc &crc, uint64 bytes,
                       const Crc &start_old,
                       const Crc &start_new) const {
    return (crc ^ Multiply(start_new ^ start_old, Xpow8N(bytes)));
  }

  // Returns CRC of concatenation of blocks A and B when CRCs
  // of blocks A and B are known -- without touching the data.
  //
  // To be precise, given CRC(A, |A|, startA) and CRC(B, |B|, 0),
  // returns CRC(AB, |AB|, startA).
  Crc Concatenate(const Crc &crc_A, const Crc &crc_B, uint64 bytes_B) const {
    return ChangeStartValue(crc_B, bytes_B, 0 /* start_B */, crc_A);
  }

  // Returns CRC of sequence of zeroes -- without touching the data.
  Crc CrcOfZeroes(uint64 bytes, const Crc &start) const {
    Crc tmp = Multiply(start ^ this->canonize_, Xpow8N(bytes));
    return (tmp ^ this->canonize_);
  }

  // Given CRC of a message, stores extra (degree + 7)/8 bytes after
  // the message so that CRC(message+extra, start) = result.
  // Does not change CRC start value (use ChangeStartValue for that).
  // Returns number of stored bytes.
  size_t StoreComplementaryCrc(void *dst,
                               const Crc &message_crc,
                               const Crc &result) const {
    Crc crc0 = Multiply(result ^ this->canonize_, this->x_pow_minus_W_);
    crc0 ^= message_crc ^ this->canonize_;
    uint8 *d = reinterpret_cast<uint8 *>(dst);
    for (size_t i = 0; i < this->crc_bytes_; ++i) {
      d[i] = TO_BYTE(crc0);
      crc0 >>= 8;
    }
    return this->crc_bytes_;
  }

  // Stores given CRC of a message as (degree + 7)/8 bytes filled
  // with 0s to the right. Returns number of stored bytes.
  // CRC of the message and stored CRC is a constant value returned
  // by CrcOfCrc() -- it does not depend on contents of the message.
  size_t StoreCrc(void *dst, const Crc &crc) const {
    uint8 *d = reinterpret_cast<uint8 *>(dst);
    Crc crc0 = crc;
    for (size_t i = 0; i < this->crc_bytes_; ++i) {
      d[i] = TO_BYTE(crc0);
      crc0 >>= 8;
    }
    return this->crc_bytes_;
  }

  // Returns expected CRC value of CRC(Message,CRC(Message))
  // when CRC is stored after the message. This value is fixed
  // and does not depend on the message or CRC start value.
  Crc CrcOfCrc() const {
    return this->crc_of_crc_;
  }

  // Returns ((a * b) mod P) where "a" and "b" are of degree <= (D-1).
  Crc Multiply(const Crc &aa, const Crc &bb) const {
    Crc a = aa;
    Crc b = bb;
    if ((a ^ (a - 1)) < (b ^ (b - 1))) {
      Crc temp = a;
      a = b;
      b = temp;
    }

    if (a == 0) {
      return a;
    }

    Crc product = 0;
    Crc one = this->one_;
    for (; a != 0; a <<= 1) {
      if ((a & one) != 0) {
        product ^= b;
        a ^= one;
      }
      b = (b >> 1) ^ this->normalize_[Downcast<Crc, size_t>(b & 1)];
    }

    return product;
  }

  // Returns ((unnorm * m) mod P) where degree of m is <= (D-1)
  // and degree of value "unnorm" is provided explicitly.
  Crc MultiplyUnnormalized(const Crc &unnorm, size_t degree,
                           const Crc &m) const {
    Crc v = unnorm;
    Crc result = 0;
    while (degree > this->degree_) {
      degree -= this->degree_;
      Crc value = v & (this->one_ | (this->one_ - 1));
      result ^= Multiply(value, Multiply(m, XpowN(degree)));
      v >>= this->degree_;
    }
    result ^= Multiply(v << (this->degree_ - degree), m);
    return result;
  }

  // returns ((x ** n) mod P).
  Crc XpowN(uint64 n) const {
    Crc one = this->one_;
    Crc result = one;

    for (size_t i = 0; n != 0; ++i, n >>= 1) {
      if (n & 1) {
        result = Multiply(result, this->x_pow_2n_[i]);
      }
    }

    return result;
  }

  // Returns (x ** (8 * n) mod P).
  Crc Xpow8N(uint64 n) const {
    return XpowN(n << 3);
  }

  // Returns remainder (A mod B) and sets *q = (A/B) of division
  // of two polynomials:
  //    A = dividend + dividend_x_pow_D_coef * x**degree,
  //    B = divisor.
  Crc Divide(const Crc &dividend0, int dividend_x_pow_D_coef,
             const Crc &divisor0, Crc *q) const {
    Crc divisor = divisor0;
    Crc dividend = dividend0;
    Crc quotient = 0;
    Crc coef = this->one_;

    while ((divisor & 1) == 0) {
      divisor >>= 1;
      coef >>= 1;
    }

    if (dividend_x_pow_D_coef) {
      quotient = coef >> 1;
      dividend ^= divisor >> 1;
    }

    Crc x_pow_degree_b = 1;
    for (;;) {
      if ((dividend & x_pow_degree_b) != 0) {
        dividend ^= divisor;
        quotient ^= coef;
      }
      if (coef == this->one_) {
        break;
      }
      coef <<= 1;
      x_pow_degree_b <<= 1;
      divisor <<= 1;
    }

    *q = quotient;
    return dividend;
  }

  // Extended Euclid's algorith -- for given A finds LCD(A, P) and
  // value B such that (A * B) mod P = LCD(A, P).
  Crc FindLCD(const Crc &A, Crc *B) const {
    if (A == 0 || A == this->one_) {
      *B = A;
      return A;
    }

    // Actually, generating polynomial is
    // (generating_polynomial_ + x**degree).
    int r0_x_pow_D_coef = 1;
    Crc r0 = this->generating_polynomial_;
    Crc b0 = 0;
    Crc r1 = A;
    Crc b1 = this->one_;

    for (;;) {
      Crc q;
      Crc r = Divide(r0, r0_x_pow_D_coef, r1, &q);
      if (r == 0) {
        break;
      }
      r0_x_pow_D_coef = 0;

      r0 = r1;
      r1 = r;

      Crc b = b0 ^ Multiply(q, b1);
      b0 = b1;
      b1 = b;
    }

    *B = b1;
    return r1;
  }

 protected:
  Crc canonize_;
  Crc x_pow_2n_[sizeof(uint64) * 8];
  Crc generating_polynomial_;
  Crc one_;
  Crc x_pow_minus_W_;
  Crc crc_of_crc_;
  Crc normalize_[2];
  size_t crc_bytes_;
  size_t degree_;
} GCC_ALIGN_ATTRIBUTE(16);

#pragma pack(pop)

}  // namespace crcutil

#endif  // CRCUTIL_GF_UTIL_H_
