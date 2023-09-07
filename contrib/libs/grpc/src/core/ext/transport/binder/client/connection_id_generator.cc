// Copyright 2021 gRPC authors.
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

#include "src/core/ext/transport/binder/client/connection_id_generator.h"

#ifndef GRPC_NO_BINDER

#include "y_absl/strings/str_cat.h"

namespace {
// Make sure `s` does not contain characters other than numbers, alphabets,
// period and underscore
TString Normalize(y_absl::string_view str_view) {
  TString s = TString(str_view);
  for (size_t i = 0; i < s.length(); i++) {
    if (!isalnum(s[i]) && s[i] != '.') {
      s[i] = '_';
    }
  }
  return s;
}

// Remove prefix of the string if the string is longer than len
TString StripToLength(const TString& s, size_t len) {
  if (s.length() > len) {
    return s.substr(s.length() - len, len);
  }
  return s;
}
}  // namespace

namespace grpc_binder {

TString ConnectionIdGenerator::Generate(y_absl::string_view uri) {
  // reserve some room for serial number
  const size_t kReserveForNumbers = 15;
  TString s =
      StripToLength(Normalize(uri), kPathLengthLimit - kReserveForNumbers);
  TString ret;
  {
    grpc_core::MutexLock l(&m_);
    // Insert a hyphen before serial number
    ret = y_absl::StrCat(s, "-", ++count_);
  }
  GPR_ASSERT(ret.length() < kPathLengthLimit);
  return ret;
}

ConnectionIdGenerator* GetConnectionIdGenerator() {
  static ConnectionIdGenerator* cig = new ConnectionIdGenerator();
  return cig;
}

}  // namespace grpc_binder
#endif
