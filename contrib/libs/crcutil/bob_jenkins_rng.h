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

// Glorified C++ version of Bob Jenkins' random number generator.
// See http://burtleburtle.net/bob/rand/smallprng.html for more details.

#ifndef CRCUTIL_BOB_JENKINS_RNG_H_
#define CRCUTIL_BOB_JENKINS_RNG_H_

#include "base_types.h"

#if !defined(_MSC_VER)
#define _rotl(value, bits) \
  static_cast<uint32>(((value) << (bits)) + ((value) >> (32 - (bits))))
#define _rotl64(value, bits) \
  static_cast<uint64>(((value) << (bits)) + ((value) >> (64 - (bits))))
#endif  // !defined(_MSC_VER)

namespace crcutil {

#pragma pack(push, 8)

template<typename T> class BobJenkinsRng;

template<> class BobJenkinsRng<uint32> {
 public:
  typedef uint32 value;

  value Get() {
    value e = a_ - _rotl(b_, 23);
    a_ = b_ ^ _rotl(c_, 16);
    b_ = c_ + _rotl(d_, 11);
    c_ = d_ + e;
    d_ = e + a_;
    return (d_);
  }

  void Init(value seed) {
    a_ = 0xf1ea5eed;
    b_ = seed;
    c_ = seed;
    d_ = seed;
    for (size_t i = 0; i < 20; ++i) {
      (void) Get();
    }
  }

  explicit BobJenkinsRng(value seed) {
    Init(seed);
  }

  BobJenkinsRng() {
    Init(0x1234567);
  }

 private:
  value a_;
  value b_;
  value c_;
  value d_;
};


#if HAVE_UINT64

template<> class BobJenkinsRng<uint64> {
 public:
  typedef uint64 value;

  value Get() {
    value e = a_ - _rotl64(b_, 7);
    a_ = b_ ^ _rotl64(c_, 13);
    b_ = c_ + _rotl64(d_, 37);
    c_ = d_ + e;
    d_ = e + a_;
    return d_;
  }

  void Init(value seed) {
    a_ = 0xf1ea5eed;
    b_ = seed;
    c_ = seed;
    d_ = seed;
    for (size_t i = 0; i < 20; ++i) {
      (void) Get();
    }
  }

  explicit BobJenkinsRng(value seed) {
    Init(seed);
  }

  BobJenkinsRng() {
    Init(0x1234567);
  }

 private:
  value a_;
  value b_;
  value c_;
  value d_;
};

#endif  // HAVE_UINT64

#if !defined(_MSC_VER)
#undef _rotl
#undef _rotl64
#endif  // !defined(_MSC_VER)

#pragma pack(pop)

}  // namespace crcutil

#endif  // CRCUTIL_BOB_JENKINS_RNG_H_
