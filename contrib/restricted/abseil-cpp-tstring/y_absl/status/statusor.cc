// Copyright 2020 The Abseil Authors.
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
#include "y_absl/status/statusor.h"

#include <cstdlib>
#include <utility>

#include "y_absl/base/call_once.h"
#include "y_absl/base/config.h"
#include "y_absl/base/internal/raw_logging.h"
#include "y_absl/base/nullability.h"
#include "y_absl/status/internal/statusor_internal.h"
#include "y_absl/status/status.h"
#include "y_absl/strings/str_cat.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

BadStatusOrAccess::BadStatusOrAccess(y_absl::Status status)
    : status_(std::move(status)) {}

BadStatusOrAccess::BadStatusOrAccess(const BadStatusOrAccess& other)
    : status_(other.status_) {}

BadStatusOrAccess& BadStatusOrAccess::operator=(
    const BadStatusOrAccess& other) {
  // Ensure assignment is correct regardless of whether this->InitWhat() has
  // already been called.
  other.InitWhat();
  status_ = other.status_;
  what_ = other.what_;
  return *this;
}

BadStatusOrAccess& BadStatusOrAccess::operator=(BadStatusOrAccess&& other) {
  // Ensure assignment is correct regardless of whether this->InitWhat() has
  // already been called.
  other.InitWhat();
  status_ = std::move(other.status_);
  what_ = std::move(other.what_);
  return *this;
}

BadStatusOrAccess::BadStatusOrAccess(BadStatusOrAccess&& other)
    : status_(std::move(other.status_)) {}

y_absl::Nonnull<const char*> BadStatusOrAccess::what() const noexcept {
  InitWhat();
  return what_.c_str();
}

const y_absl::Status& BadStatusOrAccess::status() const { return status_; }

void BadStatusOrAccess::InitWhat() const {
  y_absl::call_once(init_what_, [this] {
    what_ = y_absl::StrCat("Bad StatusOr access: ", status_.ToString());
  });
}

namespace internal_statusor {

void Helper::HandleInvalidStatusCtorArg(y_absl::Nonnull<y_absl::Status*> status) {
  const char* kMessage =
      "An OK status is not a valid constructor argument to StatusOr<T>";
#ifdef NDEBUG
  Y_ABSL_INTERNAL_LOG(ERROR, kMessage);
#else
  Y_ABSL_INTERNAL_LOG(FATAL, kMessage);
#endif
  // In optimized builds, we will fall back to InternalError.
  *status = y_absl::InternalError(kMessage);
}

void Helper::Crash(const y_absl::Status& status) {
  Y_ABSL_INTERNAL_LOG(
      FATAL,
      y_absl::StrCat("Attempting to fetch value instead of handling error ",
                   status.ToString()));
}

void ThrowBadStatusOrAccess(y_absl::Status status) {
#ifdef Y_ABSL_HAVE_EXCEPTIONS
  throw y_absl::BadStatusOrAccess(std::move(status));
#else
  Y_ABSL_INTERNAL_LOG(
      FATAL,
      y_absl::StrCat("Attempting to fetch value instead of handling error ",
                   status.ToString()));
  std::abort();
#endif
}

}  // namespace internal_statusor
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
