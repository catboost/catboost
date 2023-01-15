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

#include "absl/base/attributes.h"
#include "tcmalloc/runtime_size_classes.h"
#include "tcmalloc/size_class_info.h"

namespace tcmalloc {

// Default implementation doesn't load runtime size classes.
// To enable runtime size classes, link with :runtime_size_classes.
// This is in a separate library so that it doesn't get inlined inside common.cc
ABSL_ATTRIBUTE_WEAK ABSL_ATTRIBUTE_NOINLINE int MaybeSizeClassesFromEnv(
    int max_size, int max_classes, SizeClassInfo* parsed) {
  return -1;
}

}  // namespace tcmalloc
