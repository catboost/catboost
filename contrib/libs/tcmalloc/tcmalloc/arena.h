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

#ifndef TCMALLOC_ARENA_H_
#define TCMALLOC_ARENA_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "tcmalloc/common.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// Arena allocation; designed for use by tcmalloc internal data structures like
// spans, profiles, etc.  Always expands.
class Arena {
 public:
  constexpr Arena()
      : free_area_(nullptr), free_avail_(0), bytes_allocated_(0) {}

  // Return a properly aligned byte array of length "bytes".  Crashes if
  // allocation fails.  Requires pageheap_lock is held.
  ABSL_ATTRIBUTE_RETURNS_NONNULL void* Alloc(size_t bytes,
                                             int alignment = kAlignment)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Returns the total number of bytes allocated from this arena.  Requires
  // pageheap_lock is held.
  uint64_t bytes_allocated() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return bytes_allocated_;
  }

 private:
  // How much to allocate from system at a time
  static constexpr int kAllocIncrement = 128 << 10;

  // Free area from which to carve new objects
  char* free_area_ ABSL_GUARDED_BY(pageheap_lock);
  size_t free_avail_ ABSL_GUARDED_BY(pageheap_lock);

  // Total number of bytes allocated from this arena
  uint64_t bytes_allocated_ ABSL_GUARDED_BY(pageheap_lock);

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_ARENA_H_
