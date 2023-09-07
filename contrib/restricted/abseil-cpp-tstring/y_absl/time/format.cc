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

#include <string.h>

#include <cctype>
#include <cstdint>

#include "y_absl/strings/match.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/time/internal/cctz/include/cctz/time_zone.h"
#include "y_absl/time/time.h"

namespace cctz = y_absl::time_internal::cctz;

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

Y_ABSL_DLL extern const char RFC3339_full[] = "%Y-%m-%d%ET%H:%M:%E*S%Ez";
Y_ABSL_DLL extern const char RFC3339_sec[] = "%Y-%m-%d%ET%H:%M:%S%Ez";

Y_ABSL_DLL extern const char RFC1123_full[] = "%a, %d %b %E4Y %H:%M:%S %z";
Y_ABSL_DLL extern const char RFC1123_no_wday[] = "%d %b %E4Y %H:%M:%S %z";

namespace {

const char kInfiniteFutureStr[] = "infinite-future";
const char kInfinitePastStr[] = "infinite-past";

struct cctz_parts {
  cctz::time_point<cctz::seconds> sec;
  cctz::detail::femtoseconds fem;
};

inline cctz::time_point<cctz::seconds> unix_epoch() {
  return std::chrono::time_point_cast<cctz::seconds>(
      std::chrono::system_clock::from_time_t(0));
}

// Splits a Time into seconds and femtoseconds, which can be used with CCTZ.
// Requires that 't' is finite. See duration.cc for details about rep_hi and
// rep_lo.
cctz_parts Split(y_absl::Time t) {
  const auto d = time_internal::ToUnixDuration(t);
  const int64_t rep_hi = time_internal::GetRepHi(d);
  const int64_t rep_lo = time_internal::GetRepLo(d);
  const auto sec = unix_epoch() + cctz::seconds(rep_hi);
  const auto fem = cctz::detail::femtoseconds(rep_lo * (1000 * 1000 / 4));
  return {sec, fem};
}

// Joins the given seconds and femtoseconds into a Time. See duration.cc for
// details about rep_hi and rep_lo.
y_absl::Time Join(const cctz_parts& parts) {
  const int64_t rep_hi = (parts.sec - unix_epoch()).count();
  const uint32_t rep_lo =
      static_cast<uint32_t>(parts.fem.count() / (1000 * 1000 / 4));
  const auto d = time_internal::MakeDuration(rep_hi, rep_lo);
  return time_internal::FromUnixDuration(d);
}

}  // namespace

TString FormatTime(y_absl::string_view format, y_absl::Time t,
                       y_absl::TimeZone tz) {
  if (t == y_absl::InfiniteFuture()) return TString(kInfiniteFutureStr);
  if (t == y_absl::InfinitePast()) return TString(kInfinitePastStr);
  const auto parts = Split(t);
  return cctz::detail::format(TString(format), parts.sec, parts.fem,
                              cctz::time_zone(tz));
}

TString FormatTime(y_absl::Time t, y_absl::TimeZone tz) {
  return FormatTime(RFC3339_full, t, tz);
}

TString FormatTime(y_absl::Time t) {
  return y_absl::FormatTime(RFC3339_full, t, y_absl::LocalTimeZone());
}

bool ParseTime(y_absl::string_view format, y_absl::string_view input,
               y_absl::Time* time, TString* err) {
  return y_absl::ParseTime(format, input, y_absl::UTCTimeZone(), time, err);
}

// If the input string does not contain an explicit UTC offset, interpret
// the fields with respect to the given TimeZone.
bool ParseTime(y_absl::string_view format, y_absl::string_view input,
               y_absl::TimeZone tz, y_absl::Time* time, TString* err) {
  auto strip_leading_space = [](y_absl::string_view* sv) {
    while (!sv->empty()) {
      if (!std::isspace(sv->front())) return;
      sv->remove_prefix(1);
    }
  };

  // Portable toolchains means we don't get nice constexpr here.
  struct Literal {
    const char* name;
    size_t size;
    y_absl::Time value;
  };
  static Literal literals[] = {
      {kInfiniteFutureStr, strlen(kInfiniteFutureStr), InfiniteFuture()},
      {kInfinitePastStr, strlen(kInfinitePastStr), InfinitePast()},
  };
  strip_leading_space(&input);
  for (const auto& lit : literals) {
    if (y_absl::StartsWith(input, y_absl::string_view(lit.name, lit.size))) {
      y_absl::string_view tail = input;
      tail.remove_prefix(lit.size);
      strip_leading_space(&tail);
      if (tail.empty()) {
        *time = lit.value;
        return true;
      }
    }
  }

  TString error;
  cctz_parts parts;
  const bool b =
      cctz::detail::parse(TString(format), TString(input),
                          cctz::time_zone(tz), &parts.sec, &parts.fem, &error);
  if (b) {
    *time = Join(parts);
  } else if (err != nullptr) {
    *err = error;
  }
  return b;
}

// Functions required to support y_absl::Time flags.
bool AbslParseFlag(y_absl::string_view text, y_absl::Time* t, TString* error) {
  return y_absl::ParseTime(RFC3339_full, text, y_absl::UTCTimeZone(), t, error);
}

TString AbslUnparseFlag(y_absl::Time t) {
  return y_absl::FormatTime(RFC3339_full, t, y_absl::UTCTimeZone());
}
bool ParseFlag(const TString& text, y_absl::Time* t, TString* error) {
  return y_absl::ParseTime(RFC3339_full, text, y_absl::UTCTimeZone(), t, error);
}

TString UnparseFlag(y_absl::Time t) {
  return y_absl::FormatTime(RFC3339_full, t, y_absl::UTCTimeZone());
}

Y_ABSL_NAMESPACE_END
}  // namespace y_absl
