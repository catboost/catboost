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
//
// Run-time specification of Size classes
#ifndef TCMALLOC_RUNTIME_SIZE_CLASSES_H_
#define TCMALLOC_RUNTIME_SIZE_CLASSES_H_

#include "absl/strings/string_view.h"
#include "tcmalloc/size_class_info.h"

namespace tcmalloc {
namespace internal {

// Set size classes from a string.
// Format: "size,pages,num_to_move;"
// Example: "8,1,32;16;32;40,1,16;128,2;256;512"
// This function doesn't do validity checking. If a field is missing, its
// value is set to zero.
// The number of size classes parsed is returned.
int ParseSizeClasses(absl::string_view env, int max_size, int max_classes,
                     SizeClassInfo* parsed);

}  // namespace internal

// If the environment variable TCMALLOC_SIZE_CLASSES is defined, its value is
// parsed using ParseSizeClasses and ApplySizeClassDefaults into parsed. The
// number of size classes parsed is returned. On error, a negative value is
// returned.
int MaybeSizeClassesFromEnv(int max_size, int max_classes,
                            SizeClassInfo* parsed);

}  // namespace tcmalloc

#endif  // TCMALLOC_RUNTIME_SIZE_CLASSES_H_
