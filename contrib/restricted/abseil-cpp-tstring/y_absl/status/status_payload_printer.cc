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
#include "y_absl/status/status_payload_printer.h"

#include "y_absl/base/config.h"
#include "y_absl/base/internal/atomic_hook.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace status_internal {

Y_ABSL_INTERNAL_ATOMIC_HOOK_ATTRIBUTES
static y_absl::base_internal::AtomicHook<StatusPayloadPrinter> storage;

void SetStatusPayloadPrinter(StatusPayloadPrinter printer) {
  storage.Store(printer);
}

StatusPayloadPrinter GetStatusPayloadPrinter() {
  return storage.Load();
}

}  // namespace status_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
