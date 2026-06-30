//
// Copyright 2017 The Abseil Authors.
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
// -----------------------------------------------------------------------------
// File: str_join.h
// -----------------------------------------------------------------------------
//
// This header file contains functions for joining a range of elements and
// returning the result as a TString. StrJoin operations are specified by
// passing a range, a separator string to use between the elements joined, and
// an optional Formatter responsible for converting each argument in the range
// to a string. If omitted, a default `AlphaNumFormatter()` is called on the
// elements to be joined, using the same formatting that `y_absl::StrCat()` uses.
// This package defines a number of default formatters, and you can define your
// own implementations.
//
// Ranges are specified by passing a container with `std::begin()` and
// `std::end()` iterators, container-specific `begin()` and `end()` iterators, a
// brace-initialized `std::initializer_list`, or a `std::tuple` of heterogeneous
// objects. The separator string is specified as an `y_absl::string_view`.
//
// Because the default formatter uses the `y_absl::AlphaNum` class,
// `y_absl::StrJoin()`, like `y_absl::StrCat()`, will work out-of-the-box on
// collections of strings, ints, floats, doubles, etc.
//
// Example:
//
//   std::vector<TString> v = {"foo", "bar", "baz"};
//   TString s = y_absl::StrJoin(v, "-");
//   EXPECT_EQ("foo-bar-baz", s);
//
// See comments on the `y_absl::StrJoin()` function for more examples.

#ifndef Y_ABSL_STRINGS_STR_JOIN_H_
#define Y_ABSL_STRINGS_STR_JOIN_H_

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <util/generic/string.h>
#include <tuple>
#include <type_traits>
#include <utility>

#include "y_absl/base/macros.h"
#include "y_absl/strings/internal/str_join_internal.h"
#include "y_absl/strings/string_view.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

// -----------------------------------------------------------------------------
// Concept: Formatter
// -----------------------------------------------------------------------------
//
// A Formatter is a function object that is responsible for formatting its
// argument as a string and appending it to a given output TString.
// Formatters may be implemented as function objects, lambdas, or normal
// functions. You may provide your own Formatter to enable `y_absl::StrJoin()` to
// work with arbitrary types.
//
// The following is an example of a custom Formatter that uses
// `y_absl::FormatDuration` to join a list of `y_absl::Duration`s.
//
//   std::vector<y_absl::Duration> v = {y_absl::Seconds(1), y_absl::Milliseconds(10)};
//   TString s =
//       y_absl::StrJoin(v, ", ", [](TString* out, y_absl::Duration dur) {
//         y_absl::StrAppend(out, y_absl::FormatDuration(dur));
//       });
//   EXPECT_EQ(s, "1s, 10ms");
//
// The following standard formatters are provided within this file:
//
// - `AlphaNumFormatter()` (the default)
// - `StreamFormatter()`
// - `PairFormatter()`
// - `DereferenceFormatter()`

// AlphaNumFormatter()
//
// Default formatter used if none is specified. Uses `y_absl::AlphaNum` to convert
// numeric arguments to strings.
inline strings_internal::AlphaNumFormatterImpl AlphaNumFormatter() {
  return strings_internal::AlphaNumFormatterImpl();
}

// StreamFormatter()
//
// Formats its argument using the << operator.
inline strings_internal::StreamFormatterImpl StreamFormatter() {
  return strings_internal::StreamFormatterImpl();
}

// Function Template: PairFormatter(Formatter, y_absl::string_view, Formatter)
//
// Formats a `std::pair` by putting a given separator between the pair's
// `.first` and `.second` members. This formatter allows you to specify
// custom Formatters for both the first and second member of each pair.
template <typename FirstFormatter, typename SecondFormatter>
inline strings_internal::PairFormatterImpl<FirstFormatter, SecondFormatter>
PairFormatter(FirstFormatter f1, y_absl::string_view sep, SecondFormatter f2) {
  return strings_internal::PairFormatterImpl<FirstFormatter, SecondFormatter>(
      std::move(f1), sep, std::move(f2));
}

// Function overload of PairFormatter() for using a default
// `AlphaNumFormatter()` for each Formatter in the pair.
inline strings_internal::PairFormatterImpl<
    strings_internal::AlphaNumFormatterImpl,
    strings_internal::AlphaNumFormatterImpl>
PairFormatter(y_absl::string_view sep) {
  return PairFormatter(AlphaNumFormatter(), sep, AlphaNumFormatter());
}

// Function Template: DereferenceFormatter(Formatter)
//
// Formats its argument by dereferencing it and then applying the given
// formatter. This formatter is useful for formatting a container of
// pointer-to-T. This pattern often shows up when joining repeated fields in
// protocol buffers.
template <typename Formatter>
strings_internal::DereferenceFormatterImpl<Formatter> DereferenceFormatter(
    Formatter&& f) {
  return strings_internal::DereferenceFormatterImpl<Formatter>(
      std::forward<Formatter>(f));
}

// Function overload of `DereferenceFormatter()` for using a default
// `AlphaNumFormatter()`.
inline strings_internal::DereferenceFormatterImpl<
    strings_internal::AlphaNumFormatterImpl>
DereferenceFormatter() {
  return strings_internal::DereferenceFormatterImpl<
      strings_internal::AlphaNumFormatterImpl>(AlphaNumFormatter());
}

// -----------------------------------------------------------------------------
// StrJoin()
// -----------------------------------------------------------------------------
//
// Joins a range of elements and returns the result as a TString.
// `y_absl::StrJoin()` takes a range, a separator string to use between the
// elements joined, and an optional Formatter responsible for converting each
// argument in the range to a string.
//
// If omitted, the default `AlphaNumFormatter()` is called on the elements to be
// joined.
//
// Example 1:
//   // Joins a collection of strings. This pattern also works with a collection
//   // of `y_absl::string_view` or even `const char*`.
//   std::vector<TString> v = {"foo", "bar", "baz"};
//   TString s = y_absl::StrJoin(v, "-");
//   EXPECT_EQ(s, "foo-bar-baz");
//
// Example 2:
//   // Joins the values in the given `std::initializer_list<>` specified using
//   // brace initialization. This pattern also works with an initializer_list
//   // of ints or `y_absl::string_view` -- any `AlphaNum`-compatible type.
//   TString s = y_absl::StrJoin({"foo", "bar", "baz"}, "-");
//   EXPECT_EQs, "foo-bar-baz");
//
// Example 3:
//   // Joins a collection of ints. This pattern also works with floats,
//   // doubles, int64s -- any `StrCat()`-compatible type.
//   std::vector<int> v = {1, 2, 3, -4};
//   TString s = y_absl::StrJoin(v, "-");
//   EXPECT_EQ(s, "1-2-3--4");
//
// Example 4:
//   // Joins a collection of pointer-to-int. By default, pointers are
//   // dereferenced and the pointee is formatted using the default format for
//   // that type; such dereferencing occurs for all levels of indirection, so
//   // this pattern works just as well for `std::vector<int**>` as for
//   // `std::vector<int*>`.
//   int x = 1, y = 2, z = 3;
//   std::vector<int*> v = {&x, &y, &z};
//   TString s = y_absl::StrJoin(v, "-");
//   EXPECT_EQ(s, "1-2-3");
//
// Example 5:
//   // Dereferencing of `std::unique_ptr<>` is also supported:
//   std::vector<std::unique_ptr<int>> v
//   v.emplace_back(new int(1));
//   v.emplace_back(new int(2));
//   v.emplace_back(new int(3));
//   TString s = y_absl::StrJoin(v, "-");
//   EXPECT_EQ(s, "1-2-3");
//
// Example 6:
//   // Joins a `std::map`, with each key-value pair separated by an equals
//   // sign. This pattern would also work with, say, a
//   // `std::vector<std::pair<>>`.
//   std::map<TString, int> m = {
//       {"a", 1},
//       {"b", 2},
//       {"c", 3}};
//   TString s = y_absl::StrJoin(m, ",", y_absl::PairFormatter("="));
//   EXPECT_EQ(s, "a=1,b=2,c=3");
//
// Example 7:
//   // These examples show how `y_absl::StrJoin()` handles a few common edge
//   // cases:
//   std::vector<TString> v_empty;
//   EXPECT_EQ(y_absl::StrJoin(v_empty, "-"), "");
//
//   std::vector<TString> v_one_item = {"foo"};
//   EXPECT_EQ(y_absl::StrJoin(v_one_item, "-"), "foo");
//
//   std::vector<TString> v_empty_string = {""};
//   EXPECT_EQ(y_absl::StrJoin(v_empty_string, "-"), "");
//
//   std::vector<TString> v_one_item_empty_string = {"a", ""};
//   EXPECT_EQ(y_absl::StrJoin(v_one_item_empty_string, "-"), "a-");
//
//   std::vector<TString> v_two_empty_string = {"", ""};
//   EXPECT_EQ(y_absl::StrJoin(v_two_empty_string, "-"), "-");
//
// Example 8:
//   // Joins a `std::tuple<T...>` of heterogeneous types, converting each to
//   // a TString using the `y_absl::AlphaNum` class.
//   TString s = y_absl::StrJoin(std::make_tuple(123, "abc", 0.456), "-");
//   EXPECT_EQ(s, "123-abc-0.456");

template <typename Iterator, typename Formatter>
TString StrJoin(Iterator start, Iterator end, y_absl::string_view sep,
                    Formatter&& fmt) {
  return strings_internal::JoinAlgorithm(start, end, sep, fmt);
}

template <typename Range, typename Formatter>
TString StrJoin(const Range& range, y_absl::string_view separator,
                    Formatter&& fmt) {
  return strings_internal::JoinRange(range, separator, fmt);
}

template <typename T, typename Formatter,
          typename = typename std::enable_if<
              !std::is_convertible<T, y_absl::string_view>::value>::type>
TString StrJoin(std::initializer_list<T> il, y_absl::string_view separator,
                    Formatter&& fmt) {
  return strings_internal::JoinRange(il, separator, fmt);
}

template <typename Formatter>
inline TString StrJoin(std::initializer_list<y_absl::string_view> il,
                           y_absl::string_view separator, Formatter&& fmt) {
  return strings_internal::JoinRange(il, separator, fmt);
}

template <typename... T, typename Formatter>
TString StrJoin(const std::tuple<T...>& value, y_absl::string_view separator,
                    Formatter&& fmt) {
  return strings_internal::JoinAlgorithm(value, separator, fmt);
}

template <typename Iterator>
TString StrJoin(Iterator start, Iterator end, y_absl::string_view separator) {
  return strings_internal::JoinRange(start, end, separator);
}

template <typename Range>
TString StrJoin(const Range& range, y_absl::string_view separator) {
  return strings_internal::JoinRange(range, separator);
}

template <typename T, typename = typename std::enable_if<!std::is_convertible<
                          T, y_absl::string_view>::value>::type>
TString StrJoin(std::initializer_list<T> il, y_absl::string_view separator) {
  return strings_internal::JoinRange(il, separator);
}

inline TString StrJoin(std::initializer_list<y_absl::string_view> il,
                           y_absl::string_view separator) {
  return strings_internal::JoinRange(il, separator);
}

template <typename... T>
TString StrJoin(const std::tuple<T...>& value,
                    y_absl::string_view separator) {
  return strings_internal::JoinTuple(value, separator,
                                     std::index_sequence_for<T...>{});
}

Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_STRINGS_STR_JOIN_H_
