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

#include "tcmalloc/page_heap.h"

#include <stddef.h>
#include <stdlib.h>

#include <memory>
#include <new>

#include "gtest/gtest.h"
#include "absl/base/internal/spinlock.h"
#include "absl/memory/memory.h"
#include "tcmalloc/common.h"
#include "tcmalloc/pagemap.h"
#include "tcmalloc/static_vars.h"

namespace tcmalloc {
namespace {

// PageHeap expands by kMinSystemAlloc by default, so use this as the minimum
// Span length to not get more memory than expected.
constexpr Length kMinSpanLength = BytesToLengthFloor(kMinSystemAlloc);

void CheckStats(const tcmalloc::PageHeap* ph, Length system_pages,
                Length free_pages, Length unmapped_pages)
    ABSL_LOCKS_EXCLUDED(tcmalloc::pageheap_lock) {
  tcmalloc::BackingStats stats;
  {
    absl::base_internal::SpinLockHolder h(&tcmalloc::pageheap_lock);
    stats = ph->stats();
  }

  ASSERT_EQ(system_pages.in_bytes(), stats.system_bytes);
  ASSERT_EQ(free_pages.in_bytes(), stats.free_bytes);
  ASSERT_EQ(unmapped_pages.in_bytes(), stats.unmapped_bytes);
}

static void Delete(tcmalloc::PageHeap* ph, tcmalloc::Span* s)
    ABSL_LOCKS_EXCLUDED(tcmalloc::pageheap_lock) {
  {
    absl::base_internal::SpinLockHolder h(&tcmalloc::pageheap_lock);
    ph->Delete(s);
  }
}

static Length Release(tcmalloc::PageHeap* ph, Length n) {
  absl::base_internal::SpinLockHolder h(&tcmalloc::pageheap_lock);
  return ph->ReleaseAtLeastNPages(n);
}

class PageHeapTest : public ::testing::Test {
 public:
  PageHeapTest() {
    // If this test is not linked against TCMalloc, the global arena used for
    // metadata will not be initialized.
    Static::InitIfNecessary();
  }
};

// TODO(b/36484267): replace this test wholesale.
TEST_F(PageHeapTest, Stats) {
  auto pagemap = absl::make_unique<tcmalloc::PageMap>();
  void* memory = calloc(1, sizeof(tcmalloc::PageHeap));
  tcmalloc::PageHeap* ph =
      new (memory) tcmalloc::PageHeap(pagemap.get(), MemoryTag::kNormal);

  // Empty page heap
  CheckStats(ph, Length(0), Length(0), Length(0));

  // Allocate a span 's1'
  tcmalloc::Span* s1 = ph->New(kMinSpanLength);
  CheckStats(ph, kMinSpanLength, Length(0), Length(0));

  // Allocate an aligned span 's2'
  static const Length kHalf = kMinSpanLength / 2;
  tcmalloc::Span* s2 = ph->NewAligned(kHalf, kHalf);
  ASSERT_EQ(s2->first_page().index() % kHalf.raw_num(), 0);
  CheckStats(ph, kMinSpanLength * 2, Length(0), kHalf);

  // Delete the old one
  Delete(ph, s1);
  CheckStats(ph, kMinSpanLength * 2, kMinSpanLength, kHalf);

  // Release the space from there:
  Length released = Release(ph, Length(1));
  ASSERT_EQ(released, kMinSpanLength);
  CheckStats(ph, kMinSpanLength * 2, Length(0), kHalf + kMinSpanLength);

  // and delete the new one
  Delete(ph, s2);
  CheckStats(ph, kMinSpanLength * 2, kHalf, kHalf + kMinSpanLength);

  free(memory);
}

}  // namespace
}  // namespace tcmalloc
