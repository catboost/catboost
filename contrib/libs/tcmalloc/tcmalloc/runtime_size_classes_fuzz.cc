// Copyright 2019 The TCMalloc Authors
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

#include "absl/strings/string_view.h"
#include "tcmalloc/common.h"
#include "tcmalloc/runtime_size_classes.h"
#include "tcmalloc/size_class_info.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* d, size_t size) {
  absl::string_view env =
      absl::string_view(reinterpret_cast<const char*>(d), size);

  tcmalloc::tcmalloc_internal::SizeClassInfo
      parsed[tcmalloc::tcmalloc_internal::kNumClasses];
  tcmalloc::tcmalloc_internal::runtime_size_classes_internal::ParseSizeClasses(
      env, tcmalloc::tcmalloc_internal::kMaxSize,
      tcmalloc::tcmalloc_internal::kNumClasses, parsed);
  return 0;
}
