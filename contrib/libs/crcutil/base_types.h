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

// Defines 8/16/32/64-bit integer types.
//
// Either uint64 or uint32 will map to size_t.
// This way, specialized variants of CRC implementation
// parameterized by "size_t" will be reused when
// parameterized by "uint64" or "uint32".
// In their turn, specialized verisons are parameterized
// by "size_t" so that one version of the code is optimal
// both on 32-bit and 64-bit platforms.

#ifndef CRCUTIL_BASE_TYPES_H_
#define CRCUTIL_BASE_TYPES_H_

#include "std_headers.h"    // size_t, ptrdiff_t

namespace crcutil {

template<typename A, typename B> class ChooseFirstIfSame {
 public:
  template<bool same_size, typename AA, typename BB> class ChooseFirstIfTrue {
   public:
    typedef AA Type;
  };
  template<typename AA, typename BB> class ChooseFirstIfTrue<false, AA, BB> {
   public:
    typedef BB Type;
  };

  typedef typename ChooseFirstIfTrue<sizeof(A) == sizeof(B), A, B>::Type Type;
};

typedef unsigned char uint8;
typedef signed char int8;

typedef unsigned short uint16;
typedef short int16;

typedef ChooseFirstIfSame<size_t, unsigned int>::Type uint32;
typedef ChooseFirstIfSame<ptrdiff_t, int>::Type int32;

#if defined(_MSC_VER)
typedef ChooseFirstIfSame<size_t, unsigned __int64>::Type uint64;
typedef ChooseFirstIfSame<ptrdiff_t, __int64>::Type int64;
#define HAVE_UINT64 1
#elif defined(__GNUC__)
typedef ChooseFirstIfSame<size_t, unsigned long long>::Type uint64;
typedef ChooseFirstIfSame<ptrdiff_t, long long>::Type int64;
#define HAVE_UINT64 1
#else
// TODO: ensure that everything compiles and works when HAVE_UINT64 is false.
// TODO: remove HAVE_UINT64 and use sizeof(uint64) instead?
#define HAVE_UINT64 0
typedef uint32 uint64;
typedef int32 int64;
#endif

}  // namespace crcutil

#endif  // CRCUTIL_BASE_TYPES_H_
