//
//  Copyright 2019 The Abseil Authors.
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

#include "y_absl/flags/internal/program_name.h"

#include <util/generic/string.h>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"
#include "y_absl/base/const_init.h"
#include "y_absl/base/thread_annotations.h"
#include "y_absl/flags/internal/path_util.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/synchronization/mutex.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace flags_internal {

Y_ABSL_CONST_INIT static y_absl::Mutex program_name_guard(y_absl::kConstInit);
Y_ABSL_CONST_INIT static TString* program_name
    Y_ABSL_GUARDED_BY(program_name_guard) = nullptr;

TString ProgramInvocationName() {
  y_absl::MutexLock l(&program_name_guard);

  return program_name ? *program_name : "UNKNOWN";
}

TString ShortProgramInvocationName() {
  y_absl::MutexLock l(&program_name_guard);

  return program_name ? TString(flags_internal::Basename(*program_name))
                      : "UNKNOWN";
}

void SetProgramInvocationName(y_absl::string_view prog_name_str) {
  y_absl::MutexLock l(&program_name_guard);

  if (!program_name)
    program_name = new TString(prog_name_str);
  else
    program_name->assign(prog_name_str.data(), prog_name_str.size());
}

}  // namespace flags_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
