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
#include "tcmalloc/static_vars.h"
#include "tcmalloc/system-alloc.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

void* Arena::Alloc(size_t bytes, int alignment) {
  ASSERT(alignment > 0);
  {  // First we need to move up to the correct alignment.
    const int misalignment =
        reinterpret_cast<uintptr_t>(free_area_) % alignment;
    const int alignment_bytes =
        misalignment != 0 ? alignment - misalignment : 0;
    free_area_ += alignment_bytes;
    free_avail_ -= alignment_bytes;
    bytes_allocated_ += alignment_bytes;
  }
  char* result;
  if (free_avail_ < bytes) {
    size_t ask = bytes > kAllocIncrement ? bytes : kAllocIncrement;
    size_t actual_size;
    // TODO(b/171081864): Arena allocations should be made relatively
    // infrequently.  Consider tagging this memory with sampled objects which
    // are also infrequently allocated.
    //
    // In the meantime it is important that we use the current NUMA partition
    // rather than always using a particular one because it's possible that any
    // single partition we choose might only contain nodes that the process is
    // unable to allocate from due to cgroup restrictions.
    MemoryTag tag;
    const auto& numa_topology = Static::numa_topology();
    if (numa_topology.numa_aware()) {
      tag = NumaNormalTag(numa_topology.GetCurrentPartition());
    } else {
      tag = MemoryTag::kNormal;
    }
    free_area_ =
        reinterpret_cast<char*>(SystemAlloc(ask, &actual_size, kPageSize, tag));
    if (ABSL_PREDICT_FALSE(free_area_ == nullptr)) {
      Crash(kCrash, __FILE__, __LINE__,
            "FATAL ERROR: Out of memory trying to allocate internal tcmalloc "
            "data (bytes, object-size); is something preventing mmap from "
            "succeeding (sandbox, VSS limitations)?",
            kAllocIncrement, bytes);
    }
    SystemBack(free_area_, actual_size);
    free_avail_ = actual_size;
  }

  ASSERT(reinterpret_cast<uintptr_t>(free_area_) % alignment == 0);
  result = free_area_;
  free_area_ += bytes;
  free_avail_ -= bytes;
  bytes_allocated_ += bytes;
  return reinterpret_cast<void*>(result);
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
