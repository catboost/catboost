//
//  Copyright 2019 The Abseil Authors.
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

#include "y_absl/flags/marshalling.h"

#include <stddef.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <util/generic/string.h>
#include <type_traits>
#include <vector>

#include "y_absl/base/config.h"
#include "y_absl/base/log_severity.h"
#include "y_absl/base/macros.h"
#include "y_absl/numeric/int128.h"
#include "y_absl/strings/ascii.h"
#include "y_absl/strings/match.h"
#include "y_absl/strings/numbers.h"
#include "y_absl/strings/str_cat.h"
#include "y_absl/strings/str_format.h"
#include "y_absl/strings/str_join.h"
#include "y_absl/strings/str_split.h"
#include "y_absl/strings/string_view.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace flags_internal {

// --------------------------------------------------------------------
// AbslParseFlag specializations for boolean type.

bool AbslParseFlag(y_absl::string_view text, bool* dst, TString*) {
  const char* kTrue[] = {"1", "t", "true", "y", "yes"};
  const char* kFalse[] = {"0", "f", "false", "n", "no"};
  static_assert(sizeof(kTrue) == sizeof(kFalse), "true_false_equal");

  text = y_absl::StripAsciiWhitespace(text);

  for (size_t i = 0; i < Y_ABSL_ARRAYSIZE(kTrue); ++i) {
    if (y_absl::EqualsIgnoreCase(text, kTrue[i])) {
      *dst = true;
      return true;
    } else if (y_absl::EqualsIgnoreCase(text, kFalse[i])) {
      *dst = false;
      return true;
    }
  }
  return false;  // didn't match a legal input
}

// --------------------------------------------------------------------
// AbslParseFlag for integral types.

// Return the base to use for parsing text as an integer.  Leading 0x
// puts us in base 16.  But leading 0 does not put us in base 8. It
// caused too many bugs when we had that behavior.
static int NumericBase(y_absl::string_view text) {
  if (text.empty()) return 0;
  size_t num_start = (text[0] == '-' || text[0] == '+') ? 1 : 0;
  const bool hex = (text.size() >= num_start + 2 && text[num_start] == '0' &&
                    (text[num_start + 1] == 'x' || text[num_start + 1] == 'X'));
  return hex ? 16 : 10;
}

template <typename IntType>
inline bool ParseFlagImpl(y_absl::string_view text, IntType& dst) {
  text = y_absl::StripAsciiWhitespace(text);

  return y_absl::numbers_internal::safe_strtoi_base(text, &dst,
                                                  NumericBase(text));
}

bool AbslParseFlag(y_absl::string_view text, short* dst, TString*) {
  int val;
  if (!ParseFlagImpl(text, val)) return false;
  if (static_cast<short>(val) != val)  // worked, but number out of range
    return false;
  *dst = static_cast<short>(val);
  return true;
}

bool AbslParseFlag(y_absl::string_view text, unsigned short* dst, TString*) {
  unsigned int val;
  if (!ParseFlagImpl(text, val)) return false;
  if (static_cast<unsigned short>(val) !=
      val)  // worked, but number out of range
    return false;
  *dst = static_cast<unsigned short>(val);
  return true;
}

bool AbslParseFlag(y_absl::string_view text, int* dst, TString*) {
  return ParseFlagImpl(text, *dst);
}

bool AbslParseFlag(y_absl::string_view text, unsigned int* dst, TString*) {
  return ParseFlagImpl(text, *dst);
}

bool AbslParseFlag(y_absl::string_view text, long* dst, TString*) {
  return ParseFlagImpl(text, *dst);
}

bool AbslParseFlag(y_absl::string_view text, unsigned long* dst, TString*) {
  return ParseFlagImpl(text, *dst);
}

bool AbslParseFlag(y_absl::string_view text, long long* dst, TString*) {
  return ParseFlagImpl(text, *dst);
}

bool AbslParseFlag(y_absl::string_view text, unsigned long long* dst,
                   TString*) {
  return ParseFlagImpl(text, *dst);
}

bool AbslParseFlag(y_absl::string_view text, y_absl::int128* dst, TString*) {
  text = y_absl::StripAsciiWhitespace(text);

  // check hex
  int base = NumericBase(text);
  if (!y_absl::numbers_internal::safe_strto128_base(text, dst, base)) {
    return false;
  }

  return base == 16 ? y_absl::SimpleHexAtoi(text, dst)
                    : y_absl::SimpleAtoi(text, dst);
}

bool AbslParseFlag(y_absl::string_view text, y_absl::uint128* dst, TString*) {
  text = y_absl::StripAsciiWhitespace(text);

  // check hex
  int base = NumericBase(text);
  if (!y_absl::numbers_internal::safe_strtou128_base(text, dst, base)) {
    return false;
  }

  return base == 16 ? y_absl::SimpleHexAtoi(text, dst)
                    : y_absl::SimpleAtoi(text, dst);
}

// --------------------------------------------------------------------
// AbslParseFlag for floating point types.

bool AbslParseFlag(y_absl::string_view text, float* dst, TString*) {
  return y_absl::SimpleAtof(text, dst);
}

bool AbslParseFlag(y_absl::string_view text, double* dst, TString*) {
  return y_absl::SimpleAtod(text, dst);
}

// --------------------------------------------------------------------
// AbslParseFlag for strings.

bool AbslParseFlag(y_absl::string_view text, TString* dst, TString*) {
  dst->assign(text.data(), text.size());
  return true;
}

// --------------------------------------------------------------------
// AbslParseFlag for vector of strings.

bool AbslParseFlag(y_absl::string_view text, std::vector<TString>* dst,
                   TString*) {
  // An empty flag value corresponds to an empty vector, not a vector
  // with a single, empty TString.
  if (text.empty()) {
    dst->clear();
    return true;
  }
  *dst = y_absl::StrSplit(text, ',', y_absl::AllowEmpty());
  return true;
}

// --------------------------------------------------------------------
// AbslUnparseFlag specializations for various builtin flag types.

TString Unparse(bool v) { return v ? "true" : "false"; }
TString Unparse(short v) { return y_absl::StrCat(v); }
TString Unparse(unsigned short v) { return y_absl::StrCat(v); }
TString Unparse(int v) { return y_absl::StrCat(v); }
TString Unparse(unsigned int v) { return y_absl::StrCat(v); }
TString Unparse(long v) { return y_absl::StrCat(v); }
TString Unparse(unsigned long v) { return y_absl::StrCat(v); }
TString Unparse(long long v) { return y_absl::StrCat(v); }
TString Unparse(unsigned long long v) { return y_absl::StrCat(v); }
TString Unparse(y_absl::int128 v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}
TString Unparse(y_absl::uint128 v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

template <typename T>
TString UnparseFloatingPointVal(T v) {
  // digits10 is guaranteed to roundtrip correctly in string -> value -> string
  // conversions, but may not be enough to represent all the values correctly.
  TString digit10_str =
      y_absl::StrFormat("%.*g", std::numeric_limits<T>::digits10, v);
  if (std::isnan(v) || std::isinf(v)) return digit10_str;

  T roundtrip_val = 0;
  TString err;
  if (y_absl::ParseFlag(digit10_str, &roundtrip_val, &err) &&
      roundtrip_val == v) {
    return digit10_str;
  }

  // max_digits10 is the number of base-10 digits that are necessary to uniquely
  // represent all distinct values.
  return y_absl::StrFormat("%.*g", std::numeric_limits<T>::max_digits10, v);
}
TString Unparse(float v) { return UnparseFloatingPointVal(v); }
TString Unparse(double v) { return UnparseFloatingPointVal(v); }
TString AbslUnparseFlag(y_absl::string_view v) { return TString(v); }
TString AbslUnparseFlag(const std::vector<TString>& v) {
  return y_absl::StrJoin(v, ",");
}

}  // namespace flags_internal

bool AbslParseFlag(y_absl::string_view text, y_absl::LogSeverity* dst,
                   TString* err) {
  text = y_absl::StripAsciiWhitespace(text);
  if (text.empty()) {
    *err = "no value provided";
    return false;
  }
  if (y_absl::EqualsIgnoreCase(text, "dfatal")) {
    *dst = y_absl::kLogDebugFatal;
    return true;
  }
  if (y_absl::EqualsIgnoreCase(text, "klogdebugfatal")) {
    *dst = y_absl::kLogDebugFatal;
    return true;
  }
  if (text.front() == 'k' || text.front() == 'K') text.remove_prefix(1);
  if (y_absl::EqualsIgnoreCase(text, "info")) {
    *dst = y_absl::LogSeverity::kInfo;
    return true;
  }
  if (y_absl::EqualsIgnoreCase(text, "warning")) {
    *dst = y_absl::LogSeverity::kWarning;
    return true;
  }
  if (y_absl::EqualsIgnoreCase(text, "error")) {
    *dst = y_absl::LogSeverity::kError;
    return true;
  }
  if (y_absl::EqualsIgnoreCase(text, "fatal")) {
    *dst = y_absl::LogSeverity::kFatal;
    return true;
  }
  std::underlying_type<y_absl::LogSeverity>::type numeric_value;
  if (y_absl::ParseFlag(text, &numeric_value, err)) {
    *dst = static_cast<y_absl::LogSeverity>(numeric_value);
    return true;
  }
  *err =
      "only integers, y_absl::LogSeverity enumerators, and DFATAL are accepted";
  return false;
}

TString AbslUnparseFlag(y_absl::LogSeverity v) {
  if (v == y_absl::NormalizeLogSeverity(v)) return y_absl::LogSeverityName(v);
  return y_absl::UnparseFlag(static_cast<int>(v));
}

Y_ABSL_NAMESPACE_END
}  // namespace y_absl
