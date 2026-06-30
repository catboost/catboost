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

#include "tcmalloc/system-alloc.h"

#include <asm/unistd.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/page_size.h"
#include "tcmalloc/malloc_extension.h"

// On systems (like freebsd) that don't define MAP_ANONYMOUS, use the old
// form of the name instead.
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc::tcmalloc_internal::system_allocator_internal {

// Structure for discovering alignment
union MemoryAligner {
  void* p;
  double d;
  size_t s;
} ABSL_CACHELINE_ALIGNED;

static_assert(sizeof(MemoryAligner) < kHugePageSize,
              "hugepage alignment too small");

int MapFixedNoReplaceFlagAvailable() {
  ABSL_CONST_INIT static int noreplace_flag;
  ABSL_CONST_INIT static absl::once_flag flag;

  absl::base_internal::LowLevelCallOnce(&flag, [&]() {
    const size_t page_size = GetPageSize();
    void* ptr =
        mmap(nullptr, page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TC_CHECK_NE(ptr, MAP_FAILED);

    // Try to map over ptr.  If we get a different address, we don't have
    // MAP_FIXED_NOREPLACE.
    //
    // We try to specify a region that overlaps with ptr, but adjust the start
    // downward so it doesn't.  This allows us to detect if the pre-4.19 bug
    // (https://github.com/torvalds/linux/commit/7aa867dd89526e9cfd9714d8b9b587c016eaea34)
    // is present.
    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    TC_CHECK_GT(uptr, page_size);
    void* target = reinterpret_cast<void*>(uptr - page_size);

    void* ptr2 = mmap(target, 2 * page_size, PROT_NONE,
                      MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    const bool rejected = ptr2 == MAP_FAILED;
    if (!rejected) {
      if (ptr2 == target) {
        // [ptr2, 2 * page_size] overlaps with [ptr, page_size], so we only need
        // to unmap [ptr2, page_size].  The second call to munmap below will
        // unmap the rest.
        munmap(ptr2, page_size);
      } else {
        munmap(ptr2, 2 * page_size);
      }
    }
    munmap(ptr, page_size);

    noreplace_flag = rejected ? MAP_FIXED_NOREPLACE : 0;
  });

  return noreplace_flag;
}

}  // namespace tcmalloc::tcmalloc_internal::system_allocator_internal
GOOGLE_MALLOC_SECTION_END
