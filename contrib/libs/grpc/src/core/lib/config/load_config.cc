// Copyright 2022 gRPC authors.
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

#include "src/core/lib/config/load_config.h"

#include <stdio.h>

#include "y_absl/flags/marshalling.h"
#include "y_absl/strings/numbers.h"
#include "y_absl/types/optional.h"

#include "src/core/lib/gprpp/env.h"

namespace grpc_core {

namespace {
y_absl::optional<TString> LoadEnv(y_absl::string_view environment_variable) {
  return GetEnv(TString(environment_variable).c_str());
}
}  // namespace

TString LoadConfigFromEnv(y_absl::string_view environment_variable,
                              const char* default_value) {
  return LoadEnv(environment_variable).value_or(default_value);
}

int32_t LoadConfigFromEnv(y_absl::string_view environment_variable,
                          int32_t default_value) {
  auto env = LoadEnv(environment_variable);
  if (env.has_value()) {
    int32_t out;
    if (y_absl::SimpleAtoi(*env, &out)) return out;
    fprintf(stderr, "Error reading int from %s: '%s' is not a number",
            TString(environment_variable).c_str(), env->c_str());
  }
  return default_value;
}

bool LoadConfigFromEnv(y_absl::string_view environment_variable,
                       bool default_value) {
  auto env = LoadEnv(environment_variable);
  if (env.has_value()) {
    bool out;
    TString error;
    if (y_absl::ParseFlag(env->c_str(), &out, &error)) return out;
    fprintf(stderr, "Error reading bool from %s: '%s' is not a bool: %s",
            TString(environment_variable).c_str(), env->c_str(),
            error.c_str());
  }
  return default_value;
}

}  // namespace grpc_core
