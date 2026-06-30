//
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
#include "y_absl/flags/usage.h"

#include <stdlib.h>

#include <util/generic/string.h>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"
#include "y_absl/base/const_init.h"
#include "y_absl/base/internal/raw_logging.h"
#include "y_absl/base/thread_annotations.h"
#include "y_absl/flags/internal/usage.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/synchronization/mutex.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace flags_internal {
namespace {
Y_ABSL_CONST_INIT y_absl::Mutex usage_message_guard(y_absl::kConstInit);
Y_ABSL_CONST_INIT TString* program_usage_message
    Y_ABSL_GUARDED_BY(usage_message_guard) = nullptr;
}  // namespace
}  // namespace flags_internal

// --------------------------------------------------------------------
// Sets the "usage" message to be used by help reporting routines.
void SetProgramUsageMessage(y_absl::string_view new_usage_message) {
  y_absl::MutexLock l(&flags_internal::usage_message_guard);

  if (flags_internal::program_usage_message != nullptr) {
    Y_ABSL_INTERNAL_LOG(FATAL, "SetProgramUsageMessage() called twice.");
    std::exit(1);
  }

  flags_internal::program_usage_message = new TString(new_usage_message);
}

// --------------------------------------------------------------------
// Returns the usage message set by SetProgramUsageMessage().
// Note: We able to return string_view here only because calling
// SetProgramUsageMessage twice is prohibited.
y_absl::string_view ProgramUsageMessage() {
  y_absl::MutexLock l(&flags_internal::usage_message_guard);

  return flags_internal::program_usage_message != nullptr
             ? y_absl::string_view(*flags_internal::program_usage_message)
             : "Warning: SetProgramUsageMessage() never called";
}

Y_ABSL_NAMESPACE_END
}  // namespace y_absl
