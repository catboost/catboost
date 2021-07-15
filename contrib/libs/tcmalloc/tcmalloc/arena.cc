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

#include "tcmalloc/arena.h"

#include "tcmalloc/internal/logging.h"
#include "tcmalloc/system-alloc.h"

namespace tcmalloc {

void* Arena::Alloc(size_t bytes) {
  char* result;
  bytes = ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
  if (free_avail_ < bytes) {
    size_t ask = bytes > kAllocIncrement ? bytes : kAllocIncrement;
    size_t actual_size;
    // TODO(b/171081864): Arena allocations should be made relatively
    // infrequently.  Consider tagging this memory with sampled objects which
    // are also infrequently allocated.
    free_area_ = reinterpret_cast<char*>(
        SystemAlloc(ask, &actual_size, kPageSize, MemoryTag::kNormal));
    if (ABSL_PREDICT_FALSE(free_area_ == nullptr)) {
      Crash(kCrash, __FILE__, __LINE__,
            "FATAL ERROR: Out of memory trying to allocate internal tcmalloc "
            "data (bytes, object-size)",
            kAllocIncrement, bytes);
    }
    SystemBack(free_area_, actual_size);
    free_avail_ = actual_size;
  }

  ASSERT(reinterpret_cast<uintptr_t>(free_area_) % kAlignment == 0);
  result = free_area_;
  free_area_ += bytes;
  free_avail_ -= bytes;
  bytes_allocated_ += bytes;
  return reinterpret_cast<void*>(result);
}

}  // namespace tcmalloc
