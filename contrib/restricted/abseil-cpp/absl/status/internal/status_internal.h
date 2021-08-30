// Copyright 2019 The Abseil Authors.
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
#ifndef ABSL_STATUS_INTERNAL_STATUS_INTERNAL_H_
#define ABSL_STATUS_INTERNAL_STATUS_INTERNAL_H_

#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/cord.h"

namespace absl {
ABSL_NAMESPACE_BEGIN

enum class StatusCode : int;

namespace status_internal {

// Container for status payloads.
struct Payload {
  std::string type_url;
  absl::Cord payload;
};

using Payloads = absl::InlinedVector<Payload, 1>;

// Reference-counted representation of Status data.
struct StatusRep {
  std::atomic<int32_t> ref;
  absl::StatusCode code;
  std::string message;
  std::unique_ptr<status_internal::Payloads> payloads;
};

absl::StatusCode MapToLocalCode(int value);
}  // namespace status_internal

ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_STATUS_INTERNAL_STATUS_INTERNAL_H_
