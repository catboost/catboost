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
#ifndef Y_ABSL_STATUS_INTERNAL_STATUS_INTERNAL_H_
#define Y_ABSL_STATUS_INTERNAL_STATUS_INTERNAL_H_

#include <memory>
#include <util/generic/string.h>
#include <utility>

#include "y_absl/base/attributes.h"
#include "y_absl/container/inlined_vector.h"
#include "y_absl/strings/cord.h"

#ifndef SWIG
// Disabled for SWIG as it doesn't parse attributes correctly.
namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
// Returned Status objects may not be ignored. Codesearch doesn't handle ifdefs
// as part of a class definitions (b/6995610), so we use a forward declaration.
//
// TODO(b/176172494): Y_ABSL_MUST_USE_RESULT should expand to the more strict
// [[nodiscard]]. For now, just use [[nodiscard]] directly when it is available.
#if Y_ABSL_HAVE_CPP_ATTRIBUTE(nodiscard)
class [[nodiscard]] Status;
#else
class Y_ABSL_MUST_USE_RESULT Status;
#endif
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
#endif  // !SWIG

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

enum class StatusCode : int;

namespace status_internal {

// Container for status payloads.
struct Payload {
  TString type_url;
  y_absl::Cord payload;
};

using Payloads = y_absl::InlinedVector<Payload, 1>;

// Reference-counted representation of Status data.
struct StatusRep {
  StatusRep(y_absl::StatusCode code_arg, y_absl::string_view message_arg,
            std::unique_ptr<status_internal::Payloads> payloads_arg)
      : ref(int32_t{1}),
        code(code_arg),
        message(message_arg),
        payloads(std::move(payloads_arg)) {}

  std::atomic<int32_t> ref;
  y_absl::StatusCode code;

  // As an internal implementation detail, we guarantee that if status.message()
  // is non-empty, then the resulting string_view is null terminated.
  // This is required to implement 'StatusMessageAsCStr(...)'
  TString message;
  std::unique_ptr<status_internal::Payloads> payloads;
};

y_absl::StatusCode MapToLocalCode(int value);

// Returns a pointer to a newly-allocated string with the given `prefix`,
// suitable for output as an error message in assertion/`CHECK()` failures.
//
// This is an internal implementation detail for Abseil logging.
TString* MakeCheckFailString(const y_absl::Status* status,
                                 const char* prefix);

}  // namespace status_internal

Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_STATUS_INTERNAL_STATUS_INTERNAL_H_
