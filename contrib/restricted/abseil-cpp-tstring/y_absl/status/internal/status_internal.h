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

#include <atomic>
#include <cstdint>
#include <memory>
#include <util/generic/string.h>
#include <utility>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"
#include "y_absl/base/nullability.h"
#include "y_absl/container/inlined_vector.h"
#include "y_absl/strings/cord.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/types/optional.h"

#ifndef SWIG
// Disabled for SWIG as it doesn't parse attributes correctly.
namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
class Y_ABSL_ATTRIBUTE_TRIVIAL_ABI Status;
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
#endif  // !SWIG

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

enum class StatusCode : int;
enum class StatusToStringMode : int;

namespace status_internal {

// Container for status payloads.
struct Payload {
  TString type_url;
  y_absl::Cord payload;
};

using Payloads = y_absl::InlinedVector<Payload, 1>;

// Reference-counted representation of Status data.
class StatusRep {
 public:
  StatusRep(y_absl::StatusCode code_arg, y_absl::string_view message_arg,
            std::unique_ptr<status_internal::Payloads> payloads_arg)
      : ref_(int32_t{1}),
        code_(code_arg),
        message_(message_arg),
        payloads_(std::move(payloads_arg)) {}

  y_absl::StatusCode code() const { return code_; }
  const TString& message() const { return message_; }

  // Ref and unref are const to allow access through a const pointer, and are
  // used during copying operations.
  void Ref() const { ref_.fetch_add(1, std::memory_order_relaxed); }
  void Unref() const;

  // Payload methods correspond to the same methods in y_absl::Status.
  y_absl::optional<y_absl::Cord> GetPayload(y_absl::string_view type_url) const;
  void SetPayload(y_absl::string_view type_url, y_absl::Cord payload);
  struct EraseResult {
    bool erased;
    uintptr_t new_rep;
  };
  EraseResult ErasePayload(y_absl::string_view type_url);
  void ForEachPayload(
      y_absl::FunctionRef<void(y_absl::string_view, const y_absl::Cord&)> visitor)
      const;

  TString ToString(StatusToStringMode mode) const;

  bool operator==(const StatusRep& other) const;
  bool operator!=(const StatusRep& other) const { return !(*this == other); }

  // Returns an equivalent heap allocated StatusRep with refcount 1.
  //
  // `this` is not safe to be used after calling as it may have been deleted.
  y_absl::Nonnull<StatusRep*> CloneAndUnref() const;

 private:
  mutable std::atomic<int32_t> ref_;
  y_absl::StatusCode code_;

  // As an internal implementation detail, we guarantee that if status.message()
  // is non-empty, then the resulting string_view is null terminated.
  // This is required to implement 'StatusMessageAsCStr(...)'
  TString message_;
  std::unique_ptr<status_internal::Payloads> payloads_;
};

y_absl::StatusCode MapToLocalCode(int value);

// Returns a pointer to a newly-allocated string with the given `prefix`,
// suitable for output as an error message in assertion/`CHECK()` failures.
//
// This is an internal implementation detail for Abseil logging.
Y_ABSL_ATTRIBUTE_PURE_FUNCTION
y_absl::Nonnull<TString*> MakeCheckFailString(
    y_absl::Nonnull<const y_absl::Status*> status,
    y_absl::Nonnull<const char*> prefix);

}  // namespace status_internal

Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_STATUS_INTERNAL_STATUS_INTERNAL_H_
