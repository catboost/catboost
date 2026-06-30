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
// File: substitute.h
// -----------------------------------------------------------------------------
//
// This package contains functions for efficiently performing string
// substitutions using a format string with positional notation:
// `Substitute()` and `SubstituteAndAppend()`.
//
// Unlike printf-style format specifiers, `Substitute()` functions do not need
// to specify the type of the substitution arguments. Supported arguments
// following the format string, such as strings, string_views, ints,
// floats, and bools, are automatically converted to strings during the
// substitution process. (See below for a full list of supported types.)
//
// `Substitute()` does not allow you to specify *how* to format a value, beyond
// the default conversion to string. For example, you cannot format an integer
// in hex.
//
// The format string uses positional identifiers indicated by a dollar sign ($)
// and single digit positional ids to indicate which substitution arguments to
// use at that location within the format string.
//
// A '$$' sequence in the format string causes a literal '$' character to be
// output.
//
// Example 1:
//   TString s = Substitute("$1 purchased $0 $2 for $$10. Thanks $1!",
//                              5, "Bob", "Apples");
//   EXPECT_EQ("Bob purchased 5 Apples for $10. Thanks Bob!", s);
//
// Example 2:
//   TString s = "Hi. ";
//   SubstituteAndAppend(&s, "My name is $0 and I am $1 years old.", "Bob", 5);
//   EXPECT_EQ("Hi. My name is Bob and I am 5 years old.", s);
//
// Supported types:
//   * y_absl::string_view, TString, const char* (null is equivalent to "")
//   * int32_t, int64_t, uint32_t, uint64_t
//   * float, double
//   * bool (Printed as "true" or "false")
//   * pointer types other than char* (Printed as "0x<lower case hex string>",
//     except that null is printed as "NULL")
//   * user-defined types via the `AbslStringify()` customization point. See the
//     documentation for `y_absl::StrCat` for an explanation on how to use this.
//
// If an invalid format string is provided, Substitute returns an empty string
// and SubstituteAndAppend does not change the provided output string.
// A format string is invalid if it:
//   * ends in an unescaped $ character,
//     e.g. "Hello $", or
//   * calls for a position argument which is not provided,
//     e.g. Substitute("Hello $2", "world"), or
//   * specifies a non-digit, non-$ character after an unescaped $ character,
//     e.g. "Hello $f".
// In debug mode, i.e. #ifndef NDEBUG, such errors terminate the program.

#ifndef Y_ABSL_STRINGS_SUBSTITUTE_H_
#define Y_ABSL_STRINGS_SUBSTITUTE_H_

#include <cstring>
#include <util/generic/string.h>
#include <type_traits>
#include <vector>

#include "y_absl/base/macros.h"
#include "y_absl/base/nullability.h"
#include "y_absl/base/port.h"
#include "y_absl/strings/ascii.h"
#include "y_absl/strings/escaping.h"
#include "y_absl/strings/internal/stringify_sink.h"
#include "y_absl/strings/numbers.h"
#include "y_absl/strings/str_cat.h"
#include "y_absl/strings/str_split.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/strings/strip.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace substitute_internal {

// Arg
//
// This class provides an argument type for `y_absl::Substitute()` and
// `y_absl::SubstituteAndAppend()`. `Arg` handles implicit conversion of various
// types to a string. (`Arg` is very similar to the `AlphaNum` class in
// `StrCat()`.)
//
// This class has implicit constructors.
class Arg {
 public:
  // Overloads for string-y things
  //
  // Explicitly overload `const char*` so the compiler doesn't cast to `bool`.
  Arg(y_absl::Nullable<const char*> value)  // NOLINT(google-explicit-constructor)
      : piece_(y_absl::NullSafeStringView(value)) {}
  template <typename Allocator>
  Arg(  // NOLINT
      const std::basic_string<char, std::char_traits<char>, Allocator>&
          value) noexcept
      : piece_(value) {}
  Arg(y_absl::string_view value)  // NOLINT(google-explicit-constructor)
      : piece_(value) {}
  Arg(const TString& s)
      : piece_(s.data(), s.size()) {}

  // Overloads for primitives
  //
  // No overloads are available for signed and unsigned char because if people
  // are explicitly declaring their chars as signed or unsigned then they are
  // probably using them as 8-bit integers and would probably prefer an integer
  // representation. However, we can't really know, so we make the caller decide
  // what to do.
  Arg(char value)  // NOLINT(google-explicit-constructor)
      : piece_(scratch_, 1) {
    scratch_[0] = value;
  }
  Arg(short value)  // NOLINT(*)
      : piece_(scratch_,
               static_cast<size_t>(
                   numbers_internal::FastIntToBuffer(value, scratch_) -
                   scratch_)) {}
  Arg(unsigned short value)  // NOLINT(*)
      : piece_(scratch_,
               static_cast<size_t>(
                   numbers_internal::FastIntToBuffer(value, scratch_) -
                   scratch_)) {}
  Arg(int value)  // NOLINT(google-explicit-constructor)
      : piece_(scratch_,
               static_cast<size_t>(
                   numbers_internal::FastIntToBuffer(value, scratch_) -
                   scratch_)) {}
  Arg(unsigned int value)  // NOLINT(google-explicit-constructor)
      : piece_(scratch_,
               static_cast<size_t>(
                   numbers_internal::FastIntToBuffer(value, scratch_) -
                   scratch_)) {}
  Arg(long value)  // NOLINT(*)
      : piece_(scratch_,
               static_cast<size_t>(
                   numbers_internal::FastIntToBuffer(value, scratch_) -
                   scratch_)) {}
  Arg(unsigned long value)  // NOLINT(*)
      : piece_(scratch_,
               static_cast<size_t>(
                   numbers_internal::FastIntToBuffer(value, scratch_) -
                   scratch_)) {}
  Arg(long long value)  // NOLINT(*)
      : piece_(scratch_,
               static_cast<size_t>(
                   numbers_internal::FastIntToBuffer(value, scratch_) -
                   scratch_)) {}
  Arg(unsigned long long value)  // NOLINT(*)
      : piece_(scratch_,
               static_cast<size_t>(
                   numbers_internal::FastIntToBuffer(value, scratch_) -
                   scratch_)) {}
  Arg(float value)  // NOLINT(google-explicit-constructor)
      : piece_(scratch_, numbers_internal::SixDigitsToBuffer(value, scratch_)) {
  }
  Arg(double value)  // NOLINT(google-explicit-constructor)
      : piece_(scratch_, numbers_internal::SixDigitsToBuffer(value, scratch_)) {
  }
  Arg(bool value)  // NOLINT(google-explicit-constructor)
      : piece_(value ? "true" : "false") {}

  template <typename T, typename = typename std::enable_if<
                            HasAbslStringify<T>::value>::type>
  Arg(  // NOLINT(google-explicit-constructor)
      const T& v, strings_internal::StringifySink&& sink = {})
      : piece_(strings_internal::ExtractStringification(sink, v)) {}

  Arg(Hex hex);  // NOLINT(google-explicit-constructor)
  Arg(Dec dec);  // NOLINT(google-explicit-constructor)

  // vector<bool>::reference and const_reference require special help to convert
  // to `Arg` because it requires two user defined conversions.
  template <typename T,
            y_absl::enable_if_t<
                std::is_class<T>::value &&
                (std::is_same<T, std::vector<bool>::reference>::value ||
                 std::is_same<T, std::vector<bool>::const_reference>::value)>* =
                nullptr>
  Arg(T value)  // NOLINT(google-explicit-constructor)
      : Arg(static_cast<bool>(value)) {}

  // `void*` values, with the exception of `char*`, are printed as
  // "0x<hex value>". However, in the case of `nullptr`, "NULL" is printed.
  Arg(  // NOLINT(google-explicit-constructor)
      y_absl::Nullable<const void*> value);

  // Normal enums are already handled by the integer formatters.
  // This overload matches only scoped enums.
  template <typename T,
            typename = typename std::enable_if<
                std::is_enum<T>{} && !std::is_convertible<T, int>{} &&
                !HasAbslStringify<T>::value>::type>
  Arg(T value)  // NOLINT(google-explicit-constructor)
      : Arg(static_cast<typename std::underlying_type<T>::type>(value)) {}

  Arg(const Arg&) = delete;
  Arg& operator=(const Arg&) = delete;

  y_absl::string_view piece() const { return piece_; }

 private:
  y_absl::string_view piece_;
  char scratch_[numbers_internal::kFastToBufferSize];
};

// Internal helper function. Don't call this from outside this implementation.
// This interface may change without notice.
void SubstituteAndAppendArray(
    y_absl::Nonnull<TString*> output, y_absl::string_view format,
    y_absl::Nullable<const y_absl::string_view*> args_array, size_t num_args);

#if defined(Y_ABSL_BAD_CALL_IF)
constexpr int CalculateOneBit(y_absl::Nonnull<const char*> format) {
  // Returns:
  // * 2^N for '$N' when N is in [0-9]
  // * 0 for correct '$' escaping: '$$'.
  // * -1 otherwise.
  return (*format < '0' || *format > '9') ? (*format == '$' ? 0 : -1)
                                          : (1 << (*format - '0'));
}

constexpr const char* SkipNumber(y_absl::Nonnull<const char*> format) {
  return !*format ? format : (format + 1);
}

constexpr int PlaceholderBitmask(y_absl::Nonnull<const char*> format) {
  return !*format
             ? 0
             : *format != '$' ? PlaceholderBitmask(format + 1)
                              : (CalculateOneBit(format + 1) |
                                 PlaceholderBitmask(SkipNumber(format + 1)));
}
#endif  // Y_ABSL_BAD_CALL_IF

}  // namespace substitute_internal

//
// PUBLIC API
//

// SubstituteAndAppend()
//
// Substitutes variables into a given format string and appends to a given
// output string. See file comments above for usage.
//
// The declarations of `SubstituteAndAppend()` below consist of overloads
// for passing 0 to 10 arguments, respectively.
//
// NOTE: A zero-argument `SubstituteAndAppend()` may be used within variadic
// templates to allow a variable number of arguments.
//
// Example:
//  template <typename... Args>
//  void VarMsg(TString* boilerplate, y_absl::string_view format,
//      const Args&... args) {
//    y_absl::SubstituteAndAppend(boilerplate, format, args...);
//  }
//
inline void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                                y_absl::string_view format) {
  substitute_internal::SubstituteAndAppendArray(output, format, nullptr, 0);
}

inline void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                                y_absl::string_view format,
                                const substitute_internal::Arg& a0) {
  const y_absl::string_view args[] = {a0.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

inline void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                                y_absl::string_view format,
                                const substitute_internal::Arg& a0,
                                const substitute_internal::Arg& a1) {
  const y_absl::string_view args[] = {a0.piece(), a1.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

inline void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                                y_absl::string_view format,
                                const substitute_internal::Arg& a0,
                                const substitute_internal::Arg& a1,
                                const substitute_internal::Arg& a2) {
  const y_absl::string_view args[] = {a0.piece(), a1.piece(), a2.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

inline void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                                y_absl::string_view format,
                                const substitute_internal::Arg& a0,
                                const substitute_internal::Arg& a1,
                                const substitute_internal::Arg& a2,
                                const substitute_internal::Arg& a3) {
  const y_absl::string_view args[] = {a0.piece(), a1.piece(), a2.piece(),
                                    a3.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

inline void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                                y_absl::string_view format,
                                const substitute_internal::Arg& a0,
                                const substitute_internal::Arg& a1,
                                const substitute_internal::Arg& a2,
                                const substitute_internal::Arg& a3,
                                const substitute_internal::Arg& a4) {
  const y_absl::string_view args[] = {a0.piece(), a1.piece(), a2.piece(),
                                    a3.piece(), a4.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

inline void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::string_view format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5) {
  const y_absl::string_view args[] = {a0.piece(), a1.piece(), a2.piece(),
                                    a3.piece(), a4.piece(), a5.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

inline void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::string_view format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5,
    const substitute_internal::Arg& a6) {
  const y_absl::string_view args[] = {a0.piece(), a1.piece(), a2.piece(),
                                    a3.piece(), a4.piece(), a5.piece(),
                                    a6.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

inline void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::string_view format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5,
    const substitute_internal::Arg& a6, const substitute_internal::Arg& a7) {
  const y_absl::string_view args[] = {a0.piece(), a1.piece(), a2.piece(),
                                    a3.piece(), a4.piece(), a5.piece(),
                                    a6.piece(), a7.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

inline void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::string_view format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5,
    const substitute_internal::Arg& a6, const substitute_internal::Arg& a7,
    const substitute_internal::Arg& a8) {
  const y_absl::string_view args[] = {a0.piece(), a1.piece(), a2.piece(),
                                    a3.piece(), a4.piece(), a5.piece(),
                                    a6.piece(), a7.piece(), a8.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

inline void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::string_view format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5,
    const substitute_internal::Arg& a6, const substitute_internal::Arg& a7,
    const substitute_internal::Arg& a8, const substitute_internal::Arg& a9) {
  const y_absl::string_view args[] = {
      a0.piece(), a1.piece(), a2.piece(), a3.piece(), a4.piece(),
      a5.piece(), a6.piece(), a7.piece(), a8.piece(), a9.piece()};
  substitute_internal::SubstituteAndAppendArray(output, format, args,
                                                Y_ABSL_ARRAYSIZE(args));
}

#if defined(Y_ABSL_BAD_CALL_IF)
// This body of functions catches cases where the number of placeholders
// doesn't match the number of data arguments.
void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                         y_absl::Nonnull<const char*> format)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 0,
        "There were no substitution arguments "
        "but this format string either has a $[0-9] in it or contains "
        "an unescaped $ character (use $$ instead)");

void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                         y_absl::Nonnull<const char*> format,
                         const substitute_internal::Arg& a0)
    Y_ABSL_BAD_CALL_IF(substitute_internal::PlaceholderBitmask(format) != 1,
                     "There was 1 substitution argument given, but "
                     "this format string is missing its $0, contains "
                     "one of $1-$9, or contains an unescaped $ character (use "
                     "$$ instead)");

void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                         y_absl::Nonnull<const char*> format,
                         const substitute_internal::Arg& a0,
                         const substitute_internal::Arg& a1)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 3,
        "There were 2 substitution arguments given, but this format string is "
        "missing its $0/$1, contains one of $2-$9, or contains an "
        "unescaped $ character (use $$ instead)");

void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                         y_absl::Nonnull<const char*> format,
                         const substitute_internal::Arg& a0,
                         const substitute_internal::Arg& a1,
                         const substitute_internal::Arg& a2)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 7,
        "There were 3 substitution arguments given, but "
        "this format string is missing its $0/$1/$2, contains one of "
        "$3-$9, or contains an unescaped $ character (use $$ instead)");

void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                         y_absl::Nonnull<const char*> format,
                         const substitute_internal::Arg& a0,
                         const substitute_internal::Arg& a1,
                         const substitute_internal::Arg& a2,
                         const substitute_internal::Arg& a3)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 15,
        "There were 4 substitution arguments given, but "
        "this format string is missing its $0-$3, contains one of "
        "$4-$9, or contains an unescaped $ character (use $$ instead)");

void SubstituteAndAppend(y_absl::Nonnull<TString*> output,
                         y_absl::Nonnull<const char*> format,
                         const substitute_internal::Arg& a0,
                         const substitute_internal::Arg& a1,
                         const substitute_internal::Arg& a2,
                         const substitute_internal::Arg& a3,
                         const substitute_internal::Arg& a4)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 31,
        "There were 5 substitution arguments given, but "
        "this format string is missing its $0-$4, contains one of "
        "$5-$9, or contains an unescaped $ character (use $$ instead)");

void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::Nonnull<const char*> format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 63,
        "There were 6 substitution arguments given, but "
        "this format string is missing its $0-$5, contains one of "
        "$6-$9, or contains an unescaped $ character (use $$ instead)");

void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::Nonnull<const char*> format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5,
    const substitute_internal::Arg& a6)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 127,
        "There were 7 substitution arguments given, but "
        "this format string is missing its $0-$6, contains one of "
        "$7-$9, or contains an unescaped $ character (use $$ instead)");

void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::Nonnull<const char*> format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5,
    const substitute_internal::Arg& a6, const substitute_internal::Arg& a7)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 255,
        "There were 8 substitution arguments given, but "
        "this format string is missing its $0-$7, contains one of "
        "$8-$9, or contains an unescaped $ character (use $$ instead)");

void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::Nonnull<const char*> format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5,
    const substitute_internal::Arg& a6, const substitute_internal::Arg& a7,
    const substitute_internal::Arg& a8)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 511,
        "There were 9 substitution arguments given, but "
        "this format string is missing its $0-$8, contains a $9, or "
        "contains an unescaped $ character (use $$ instead)");

void SubstituteAndAppend(
    y_absl::Nonnull<TString*> output, y_absl::Nonnull<const char*> format,
    const substitute_internal::Arg& a0, const substitute_internal::Arg& a1,
    const substitute_internal::Arg& a2, const substitute_internal::Arg& a3,
    const substitute_internal::Arg& a4, const substitute_internal::Arg& a5,
    const substitute_internal::Arg& a6, const substitute_internal::Arg& a7,
    const substitute_internal::Arg& a8, const substitute_internal::Arg& a9)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 1023,
        "There were 10 substitution arguments given, but this "
        "format string either doesn't contain all of $0 through $9 or "
        "contains an unescaped $ character (use $$ instead)");
#endif  // Y_ABSL_BAD_CALL_IF

// Substitute()
//
// Substitutes variables into a given format string. See file comments above
// for usage.
//
// The declarations of `Substitute()` below consist of overloads for passing 0
// to 10 arguments, respectively.
//
// NOTE: A zero-argument `Substitute()` may be used within variadic templates to
// allow a variable number of arguments.
//
// Example:
//  template <typename... Args>
//  void VarMsg(y_absl::string_view format, const Args&... args) {
//    TString s = y_absl::Substitute(format, args...);

Y_ABSL_MUST_USE_RESULT inline TString Substitute(y_absl::string_view format) {
  TString result;
  SubstituteAndAppend(&result, format);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0) {
  TString result;
  SubstituteAndAppend(&result, format, a0);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1) {
  TString result;
  SubstituteAndAppend(&result, format, a0, a1);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2) {
  TString result;
  SubstituteAndAppend(&result, format, a0, a1, a2);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3) {
  TString result;
  SubstituteAndAppend(&result, format, a0, a1, a2, a3);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4) {
  TString result;
  SubstituteAndAppend(&result, format, a0, a1, a2, a3, a4);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4,
    const substitute_internal::Arg& a5) {
  TString result;
  SubstituteAndAppend(&result, format, a0, a1, a2, a3, a4, a5);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4,
    const substitute_internal::Arg& a5, const substitute_internal::Arg& a6) {
  TString result;
  SubstituteAndAppend(&result, format, a0, a1, a2, a3, a4, a5, a6);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4,
    const substitute_internal::Arg& a5, const substitute_internal::Arg& a6,
    const substitute_internal::Arg& a7) {
  TString result;
  SubstituteAndAppend(&result, format, a0, a1, a2, a3, a4, a5, a6, a7);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4,
    const substitute_internal::Arg& a5, const substitute_internal::Arg& a6,
    const substitute_internal::Arg& a7, const substitute_internal::Arg& a8) {
  TString result;
  SubstituteAndAppend(&result, format, a0, a1, a2, a3, a4, a5, a6, a7, a8);
  return result;
}

Y_ABSL_MUST_USE_RESULT inline TString Substitute(
    y_absl::string_view format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4,
    const substitute_internal::Arg& a5, const substitute_internal::Arg& a6,
    const substitute_internal::Arg& a7, const substitute_internal::Arg& a8,
    const substitute_internal::Arg& a9) {
  TString result;
  SubstituteAndAppend(&result, format, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
  return result;
}

#if defined(Y_ABSL_BAD_CALL_IF)
// This body of functions catches cases where the number of placeholders
// doesn't match the number of data arguments.
TString Substitute(y_absl::Nonnull<const char*> format)
    Y_ABSL_BAD_CALL_IF(substitute_internal::PlaceholderBitmask(format) != 0,
                     "There were no substitution arguments "
                     "but this format string either has a $[0-9] in it or "
                     "contains an unescaped $ character (use $$ instead)");

TString Substitute(y_absl::Nonnull<const char*> format,
                       const substitute_internal::Arg& a0)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 1,
        "There was 1 substitution argument given, but "
        "this format string is missing its $0, contains one of $1-$9, "
        "or contains an unescaped $ character (use $$ instead)");

TString Substitute(y_absl::Nonnull<const char*> format,
                       const substitute_internal::Arg& a0,
                       const substitute_internal::Arg& a1)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 3,
        "There were 2 substitution arguments given, but "
        "this format string is missing its $0/$1, contains one of "
        "$2-$9, or contains an unescaped $ character (use $$ instead)");

TString Substitute(y_absl::Nonnull<const char*> format,
                       const substitute_internal::Arg& a0,
                       const substitute_internal::Arg& a1,
                       const substitute_internal::Arg& a2)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 7,
        "There were 3 substitution arguments given, but "
        "this format string is missing its $0/$1/$2, contains one of "
        "$3-$9, or contains an unescaped $ character (use $$ instead)");

TString Substitute(y_absl::Nonnull<const char*> format,
                       const substitute_internal::Arg& a0,
                       const substitute_internal::Arg& a1,
                       const substitute_internal::Arg& a2,
                       const substitute_internal::Arg& a3)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 15,
        "There were 4 substitution arguments given, but "
        "this format string is missing its $0-$3, contains one of "
        "$4-$9, or contains an unescaped $ character (use $$ instead)");

TString Substitute(y_absl::Nonnull<const char*> format,
                       const substitute_internal::Arg& a0,
                       const substitute_internal::Arg& a1,
                       const substitute_internal::Arg& a2,
                       const substitute_internal::Arg& a3,
                       const substitute_internal::Arg& a4)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 31,
        "There were 5 substitution arguments given, but "
        "this format string is missing its $0-$4, contains one of "
        "$5-$9, or contains an unescaped $ character (use $$ instead)");

TString Substitute(y_absl::Nonnull<const char*> format,
                       const substitute_internal::Arg& a0,
                       const substitute_internal::Arg& a1,
                       const substitute_internal::Arg& a2,
                       const substitute_internal::Arg& a3,
                       const substitute_internal::Arg& a4,
                       const substitute_internal::Arg& a5)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 63,
        "There were 6 substitution arguments given, but "
        "this format string is missing its $0-$5, contains one of "
        "$6-$9, or contains an unescaped $ character (use $$ instead)");

TString Substitute(
    y_absl::Nonnull<const char*> format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4,
    const substitute_internal::Arg& a5, const substitute_internal::Arg& a6)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 127,
        "There were 7 substitution arguments given, but "
        "this format string is missing its $0-$6, contains one of "
        "$7-$9, or contains an unescaped $ character (use $$ instead)");

TString Substitute(
    y_absl::Nonnull<const char*> format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4,
    const substitute_internal::Arg& a5, const substitute_internal::Arg& a6,
    const substitute_internal::Arg& a7)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 255,
        "There were 8 substitution arguments given, but "
        "this format string is missing its $0-$7, contains one of "
        "$8-$9, or contains an unescaped $ character (use $$ instead)");

TString Substitute(
    y_absl::Nonnull<const char*> format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4,
    const substitute_internal::Arg& a5, const substitute_internal::Arg& a6,
    const substitute_internal::Arg& a7, const substitute_internal::Arg& a8)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 511,
        "There were 9 substitution arguments given, but "
        "this format string is missing its $0-$8, contains a $9, or "
        "contains an unescaped $ character (use $$ instead)");

TString Substitute(
    y_absl::Nonnull<const char*> format, const substitute_internal::Arg& a0,
    const substitute_internal::Arg& a1, const substitute_internal::Arg& a2,
    const substitute_internal::Arg& a3, const substitute_internal::Arg& a4,
    const substitute_internal::Arg& a5, const substitute_internal::Arg& a6,
    const substitute_internal::Arg& a7, const substitute_internal::Arg& a8,
    const substitute_internal::Arg& a9)
    Y_ABSL_BAD_CALL_IF(
        substitute_internal::PlaceholderBitmask(format) != 1023,
        "There were 10 substitution arguments given, but this "
        "format string either doesn't contain all of $0 through $9 or "
        "contains an unescaped $ character (use $$ instead)");
#endif  // Y_ABSL_BAD_CALL_IF

Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_STRINGS_SUBSTITUTE_H_
