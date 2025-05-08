// Copyright 2024 The TCMalloc Authors
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

#include "tcmalloc/internal/memory_tag.h"

#include "absl/strings/string_view.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/optimization.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc::tcmalloc_internal {

absl::string_view MemoryTagToLabel(MemoryTag tag) {
  switch (tag) {
    case MemoryTag::kNormal:
      return "NORMAL";
    case MemoryTag::kNormalP1:
      return "NORMAL_P1";
    case MemoryTag::kSampled:
      return "SAMPLED";
    case MemoryTag::kSelSan:
      return "SELSAN";
    case MemoryTag::kCold:
      return "COLD";
    case MemoryTag::kMetadata:
      return "METADATA";
  }

  ASSUME(false);
}

}  // namespace tcmalloc::tcmalloc_internal
GOOGLE_MALLOC_SECTION_END
