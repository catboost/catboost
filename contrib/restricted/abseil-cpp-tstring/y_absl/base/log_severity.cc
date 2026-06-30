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

#include "y_absl/base/log_severity.h"

#include <ostream>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

std::ostream& operator<<(std::ostream& os, y_absl::LogSeverity s) {
  if (s == y_absl::NormalizeLogSeverity(s)) return os << y_absl::LogSeverityName(s);
  return os << "y_absl::LogSeverity(" << static_cast<int>(s) << ")";
}

std::ostream& operator<<(std::ostream& os, y_absl::LogSeverityAtLeast s) {
  switch (s) {
    case y_absl::LogSeverityAtLeast::kInfo:
    case y_absl::LogSeverityAtLeast::kWarning:
    case y_absl::LogSeverityAtLeast::kError:
    case y_absl::LogSeverityAtLeast::kFatal:
      return os << ">=" << static_cast<y_absl::LogSeverity>(s);
    case y_absl::LogSeverityAtLeast::kInfinity:
      return os << "INFINITY";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, y_absl::LogSeverityAtMost s) {
  switch (s) {
    case y_absl::LogSeverityAtMost::kInfo:
    case y_absl::LogSeverityAtMost::kWarning:
    case y_absl::LogSeverityAtMost::kError:
    case y_absl::LogSeverityAtMost::kFatal:
      return os << "<=" << static_cast<y_absl::LogSeverity>(s);
    case y_absl::LogSeverityAtMost::kNegativeInfinity:
      return os << "NEGATIVE_INFINITY";
  }
  return os;
}
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
