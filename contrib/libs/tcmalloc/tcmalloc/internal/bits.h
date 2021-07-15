// Copyright 2019 The TCMalloc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TCMALLOC_INTERNAL_BITS_H_
#define TCMALLOC_INTERNAL_BITS_H_

#include <cstdint>
#include <type_traits>

#include "tcmalloc/internal/logging.h"

namespace tcmalloc {
namespace tcmalloc_internal {

class Bits {
 public:
  // Returns true if a value is zero or a power of two.
  template <typename T>
  static constexpr
      typename std::enable_if<std::is_unsigned<T>::value, bool>::type
      IsZeroOrPow2(T n) {
    return (n & (n - 1)) == 0;
  }

  // Returns true if a value is a power of two.
  template <typename T>
  static constexpr
      typename std::enable_if<std::is_unsigned<T>::value, bool>::type
      IsPow2(T n) {
    return n != 0 && (n & (n - 1)) == 0;
  }

  template <typename T>
  static constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type
  Log2Floor(T n) {
    if (n == 0) {
      return -1;
    }

    if (sizeof(T) <= sizeof(unsigned int)) {
      return std::numeric_limits<T>::digits - 1 - __builtin_clz(n);
    } else if (sizeof(T) <= sizeof(unsigned long)) {
      return std::numeric_limits<T>::digits - 1 - __builtin_clzl(n);
    } else {
      static_assert(sizeof(T) <= sizeof(unsigned long long));
      return std::numeric_limits<T>::digits - 1 - __builtin_clzll(n);
    }
  }

  template <typename T>
  static constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type
  Log2Ceiling(T n) {
    T floor = Log2Floor(n);
    if (IsZeroOrPow2(n))
      return floor;
    else
      return floor + 1;
  }

  template <typename T>
  static constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type
  RoundUpToPow2(T n) {
    if (n == 0) return 1;
    return T{1} << Log2Ceiling(n);
  }
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc

#endif  // TCMALLOC_INTERNAL_BITS_H_
