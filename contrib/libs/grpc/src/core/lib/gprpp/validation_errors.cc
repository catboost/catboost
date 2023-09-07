// Copyright 2020 gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <grpc/support/port_platform.h>

#include "src/core/lib/gprpp/validation_errors.h"

#include <algorithm>
#include <utility>

#include "y_absl/status/status.h"
#include "y_absl/strings/str_cat.h"
#include "y_absl/strings/str_join.h"
#include "y_absl/strings/strip.h"

namespace grpc_core {

void ValidationErrors::PushField(y_absl::string_view ext) {
  // Skip leading '.' for top-level field names.
  if (fields_.empty()) y_absl::ConsumePrefix(&ext, ".");
  fields_.emplace_back(TString(ext));
}

void ValidationErrors::PopField() { fields_.pop_back(); }

void ValidationErrors::AddError(y_absl::string_view error) {
  field_errors_[y_absl::StrJoin(fields_, "")].emplace_back(error);
}

bool ValidationErrors::FieldHasErrors() const {
  return field_errors_.find(y_absl::StrJoin(fields_, "")) != field_errors_.end();
}

y_absl::Status ValidationErrors::status(y_absl::string_view prefix) const {
  if (field_errors_.empty()) return y_absl::OkStatus();
  std::vector<TString> errors;
  for (const auto& p : field_errors_) {
    if (p.second.size() > 1) {
      errors.emplace_back(y_absl::StrCat("field:", p.first, " errors:[",
                                       y_absl::StrJoin(p.second, "; "), "]"));
    } else {
      errors.emplace_back(
          y_absl::StrCat("field:", p.first, " error:", p.second[0]));
    }
  }
  return y_absl::InvalidArgumentError(
      y_absl::StrCat(prefix, ": [", y_absl::StrJoin(errors, "; "), "]"));
}

}  // namespace grpc_core
