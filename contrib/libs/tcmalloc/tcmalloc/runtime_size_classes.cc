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

#include "tcmalloc/runtime_size_classes.h"

#include <string.h>

#include "absl/base/attributes.h"
#include "tcmalloc/internal/environment.h"
#include "tcmalloc/internal/logging.h"

using tcmalloc::kLog;

namespace tcmalloc {

namespace internal {

int ParseSizeClasses(absl::string_view env, int max_size, int max_classes,
                     SizeClassInfo* parsed) {
  int c = 1;
  int t = 0;
  memset(parsed, 0, sizeof(parsed[0]) * max_classes);
  for (char e : env) {
    // TODO(b/120885588): replace with absl::from_chars, once it is fully
    // implemented.
    if ('0' <= e && e <= '9') {
      int n = e - '0';
      int v = 10 * parsed[c].Value(t) + n;
      if (v > max_size) {
        Log(kLog, __FILE__, __LINE__, "size class integer overflow", v, n);
        return -3;
      }
      parsed[c].SetValue(t, v);
    } else if (e == ';') {
      // next size class
      t = 0;
      c++;
      if (c >= max_classes) {
        return c;
      }
    } else if (e == ',') {
      t++;
      if (t >= kSizeClassInfoMembers) {
        Log(kLog, __FILE__, __LINE__, "size class too many commas", c);
        return -1;
      }
    } else {
      Log(kLog, __FILE__, __LINE__, "Delimiter not , or ;", c, e);
      return -2;
    }
  }
  // The size class [0, 0, 0] counts as a size class, but is not parsed.
  return c + 1;
}

}  // namespace internal

int ABSL_ATTRIBUTE_NOINLINE MaybeSizeClassesFromEnv(int max_size,
                                                    int max_classes,
                                                    SizeClassInfo* parsed) {
  const char* e =
      tcmalloc::tcmalloc_internal::thread_safe_getenv("TCMALLOC_SIZE_CLASSES");
  if (!e) {
    return 0;
  }
  return internal::ParseSizeClasses(e, max_size, max_classes, parsed);
}

}  // namespace tcmalloc
