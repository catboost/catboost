// Copyright 2018 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Define the default Hash and Eq functions for SwissTable containers.
//
// std::hash<T> and std::equal_to<T> are not appropriate hash and equal
// functions for SwissTable containers. There are two reasons for this.
//
// SwissTable containers are power of 2 sized containers:
//
// This means they use the lower bits of the hash value to find the slot for
// each entry. The typical hash function for integral types is the identity.
// This is a very weak hash function for SwissTable and any power of 2 sized
// hashtable implementation which will lead to excessive collisions. For
// SwissTable we use murmur3 style mixing to reduce collisions to a minimum.
//
// SwissTable containers support heterogeneous lookup:
//
// In order to make heterogeneous lookup work, hash and equal functions must be
// polymorphic. At the same time they have to satisfy the same requirements the
// C++ standard imposes on hash functions and equality operators. That is:
//
//   if hash_default_eq<T>(a, b) returns true for any a and b of type T, then
//   hash_default_hash<T>(a) must equal hash_default_hash<T>(b)
//
// For SwissTable containers this requirement is relaxed to allow a and b of
// any and possibly different types. Note that like the standard the hash and
// equal functions are still bound to T. This is important because some type U
// can be hashed by/tested for equality differently depending on T. A notable
// example is `const char*`. `const char*` is treated as a c-style string when
// the hash function is hash<TString> but as a pointer when the hash
// function is hash<void*>.
//
#ifndef Y_ABSL_CONTAINER_INTERNAL_HASH_FUNCTION_DEFAULTS_H_
#define Y_ABSL_CONTAINER_INTERNAL_HASH_FUNCTION_DEFAULTS_H_

#include <stdint.h>
#include <cstddef>
#include <memory>
#include <util/generic/string.h>
#include <type_traits>

#include "y_absl/base/config.h"
#include "y_absl/hash/hash.h"
#include "y_absl/strings/cord.h"
#include "y_absl/strings/string_view.h"

#ifdef Y_ABSL_HAVE_STD_STRING_VIEW
#include <string_view>
#endif

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace container_internal {

// The hash of an object of type T is computed by using y_absl::Hash.
template <class T, class E = void>
struct HashEq {
  using Hash = y_absl::Hash<T>;
  using Eq = std::equal_to<T>;
};

struct StringHash {
  using is_transparent = void;

  size_t operator()(y_absl::string_view v) const {
    return y_absl::Hash<y_absl::string_view>{}(v);
  }
  size_t operator()(const y_absl::Cord& v) const {
    return y_absl::Hash<y_absl::Cord>{}(v);
  }
};

struct StringEq {
  using is_transparent = void;
  bool operator()(y_absl::string_view lhs, y_absl::string_view rhs) const {
    return lhs == rhs;
  }
  bool operator()(const y_absl::Cord& lhs, const y_absl::Cord& rhs) const {
    return lhs == rhs;
  }
  bool operator()(const y_absl::Cord& lhs, y_absl::string_view rhs) const {
    return lhs == rhs;
  }
  bool operator()(y_absl::string_view lhs, const y_absl::Cord& rhs) const {
    return lhs == rhs;
  }
};

// Supports heterogeneous lookup for string-like elements.
struct StringHashEq {
  using Hash = StringHash;
  using Eq = StringEq;
};

template <>
struct HashEq<TString> : StringHashEq {};
template <>
struct HashEq<y_absl::string_view> : StringHashEq {};
template <>
struct HashEq<y_absl::Cord> : StringHashEq {};

#ifdef Y_ABSL_HAVE_STD_STRING_VIEW

template <typename TChar>
struct BasicStringHash {
  using is_transparent = void;

  size_t operator()(std::basic_string_view<TChar> v) const {
    return y_absl::Hash<std::basic_string_view<TChar>>{}(v);
  }
};

template <typename TChar>
struct BasicStringEq {
  using is_transparent = void;
  bool operator()(std::basic_string_view<TChar> lhs,
                  std::basic_string_view<TChar> rhs) const {
    return lhs == rhs;
  }
};

// Supports heterogeneous lookup for w/u16/u32 string + string_view + char*.
template <typename TChar>
struct BasicStringHashEq {
  using Hash = BasicStringHash<TChar>;
  using Eq = BasicStringEq<TChar>;
};

template <>
struct HashEq<std::wstring> : BasicStringHashEq<wchar_t> {};
template <>
struct HashEq<std::wstring_view> : BasicStringHashEq<wchar_t> {};
template <>
struct HashEq<std::u16string> : BasicStringHashEq<char16_t> {};
template <>
struct HashEq<std::u16string_view> : BasicStringHashEq<char16_t> {};
template <>
struct HashEq<std::u32string> : BasicStringHashEq<char32_t> {};
template <>
struct HashEq<std::u32string_view> : BasicStringHashEq<char32_t> {};

#endif  // Y_ABSL_HAVE_STD_STRING_VIEW

// Supports heterogeneous lookup for pointers and smart pointers.
template <class T>
struct HashEq<T*> {
  struct Hash {
    using is_transparent = void;
    template <class U>
    size_t operator()(const U& ptr) const {
      return y_absl::Hash<const T*>{}(HashEq::ToPtr(ptr));
    }
  };
  struct Eq {
    using is_transparent = void;
    template <class A, class B>
    bool operator()(const A& a, const B& b) const {
      return HashEq::ToPtr(a) == HashEq::ToPtr(b);
    }
  };

 private:
  static const T* ToPtr(const T* ptr) { return ptr; }
  template <class U, class D>
  static const T* ToPtr(const std::unique_ptr<U, D>& ptr) {
    return ptr.get();
  }
  template <class U>
  static const T* ToPtr(const std::shared_ptr<U>& ptr) {
    return ptr.get();
  }
};

template <class T, class D>
struct HashEq<std::unique_ptr<T, D>> : HashEq<T*> {};
template <class T>
struct HashEq<std::shared_ptr<T>> : HashEq<T*> {};

// This header's visibility is restricted.  If you need to access the default
// hasher please use the container's ::hasher alias instead.
//
// Example: typename Hash = typename y_absl::flat_hash_map<K, V>::hasher
template <class T>
using hash_default_hash = typename container_internal::HashEq<T>::Hash;

// This header's visibility is restricted.  If you need to access the default
// key equal please use the container's ::key_equal alias instead.
//
// Example: typename Eq = typename y_absl::flat_hash_map<K, V, Hash>::key_equal
template <class T>
using hash_default_eq = typename container_internal::HashEq<T>::Eq;

}  // namespace container_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_CONTAINER_INTERNAL_HASH_FUNCTION_DEFAULTS_H_
