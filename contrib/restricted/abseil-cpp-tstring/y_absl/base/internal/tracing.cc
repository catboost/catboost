// Copyright 2024 The Abseil Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "y_absl/base/internal/tracing.h"

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace base_internal {

extern "C" {

Y_ABSL_ATTRIBUTE_WEAK void Y_ABSL_INTERNAL_C_SYMBOL(AbslInternalTraceWait)(
    const void*, ObjectKind) {}
Y_ABSL_ATTRIBUTE_WEAK void Y_ABSL_INTERNAL_C_SYMBOL(AbslInternalTraceContinue)(
    const void*, ObjectKind) {}
Y_ABSL_ATTRIBUTE_WEAK void Y_ABSL_INTERNAL_C_SYMBOL(AbslInternalTraceSignal)(
    const void*, ObjectKind) {}
Y_ABSL_ATTRIBUTE_WEAK void Y_ABSL_INTERNAL_C_SYMBOL(AbslInternalTraceObserved)(
    const void*, ObjectKind) {}

}  // extern "C"

}  // namespace base_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
