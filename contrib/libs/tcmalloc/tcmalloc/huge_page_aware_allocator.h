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

#ifndef TCMALLOC_HUGE_PAGE_AWARE_ALLOCATOR_H_
#define TCMALLOC_HUGE_PAGE_AWARE_ALLOCATOR_H_

#include <stddef.h>

#include "absl/base/thread_annotations.h"
#include "tcmalloc/arena.h"
#include "tcmalloc/common.h"
#include "tcmalloc/huge_allocator.h"
#include "tcmalloc/huge_cache.h"
#include "tcmalloc/huge_page_filler.h"
#include "tcmalloc/huge_pages.h"
#include "tcmalloc/huge_region.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/page_allocator_interface.h"
#include "tcmalloc/page_heap_allocator.h"
#include "tcmalloc/span.h"
#include "tcmalloc/stats.h"
#include "tcmalloc/system-alloc.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

bool decide_subrelease();

// An implementation of the PageAllocator interface that is hugepage-efficent.
// Attempts to pack allocations into full hugepages wherever possible,
// and aggressively returns empty ones to the system.
class HugePageAwareAllocator final : public PageAllocatorInterface {
 public:
  explicit HugePageAwareAllocator(MemoryTag tag);
  ~HugePageAwareAllocator() override = default;

  // Allocate a run of "n" pages.  Returns zero if out of memory.
  // Caller should not pass "n == 0" -- instead, n should have
  // been rounded up already.
  Span* New(Length n) ABSL_LOCKS_EXCLUDED(pageheap_lock) override;

  // As New, but the returned span is aligned to a <align>-page boundary.
  // <align> must be a power of two.
  Span* NewAligned(Length n, Length align)
      ABSL_LOCKS_EXCLUDED(pageheap_lock) override;

  // Delete the span "[p, p+n-1]".
  // REQUIRES: span was returned by earlier call to New() and
  //           has not yet been deleted.
  void Delete(Span* span) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) override;

  BackingStats stats() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) override;

  void GetSmallSpanStats(SmallSpanStats* result)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) override;

  void GetLargeSpanStats(LargeSpanStats* result)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) override;

  // Try to release at least num_pages for reuse by the OS.  Returns
  // the actual number of pages released, which may be less than
  // num_pages if there weren't enough pages to release. The result
  // may also be larger than num_pages since page_heap might decide to
  // release one large range instead of fragmenting it into two
  // smaller released and unreleased ranges.
  Length ReleaseAtLeastNPages(Length num_pages)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) override;

  Length ReleaseAtLeastNPagesBreakingHugepages(Length n)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Prints stats about the page heap to *out.
  void Print(Printer* out) ABSL_LOCKS_EXCLUDED(pageheap_lock) override;

  // Print stats to *out, excluding long/likely uninteresting things
  // unless <everything> is true.
  void Print(Printer* out, bool everything) ABSL_LOCKS_EXCLUDED(pageheap_lock);

  void PrintInPbtxt(PbtxtRegion* region)
      ABSL_LOCKS_EXCLUDED(pageheap_lock) override;

  HugeLength DonatedHugePages() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return donated_huge_pages_;
  }

  const HugeCache* cache() const { return &cache_; }

 private:
  typedef HugePageFiller<PageTracker<SystemRelease>> FillerType;
  FillerType filler_;

  // Calls SystemRelease, but with dropping of pageheap_lock around the call.
  static void UnbackWithoutLock(void* start, size_t length)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  HugeRegionSet<HugeRegion> regions_;

  PageHeapAllocator<FillerType::Tracker> tracker_allocator_;
  PageHeapAllocator<HugeRegion> region_allocator_;

  FillerType::Tracker* GetTracker(HugePage p);

  void SetTracker(HugePage p, FillerType::Tracker* pt);

  template <MemoryTag tag>
  static void* AllocAndReport(size_t bytes, size_t* actual, size_t align)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  static void* MetaDataAlloc(size_t bytes);
  HugeAllocator alloc_;
  HugeCache cache_;

  // donated_huge_pages_ measures the number of huge pages contributed to the
  // filler from left overs of large huge page allocations.  When the large
  // allocation is deallocated, we decrement this count *if* we were able to
  // fully reassemble the address range (that is, the partial hugepage did not
  // get stuck in the filler).
  HugeLength donated_huge_pages_ ABSL_GUARDED_BY(pageheap_lock);

  void GetSpanStats(SmallSpanStats* small, LargeSpanStats* large,
                    PageAgeHistograms* ages);

  PageId RefillFiller(Length n, bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Allocate the first <n> from p, and contribute the rest to the filler.  If
  // "donated" is true, the contribution will be marked as coming from the
  // tail of a multi-hugepage alloc.  Returns the allocated section.
  PageId AllocAndContribute(HugePage p, Length n, bool donated)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  // Helpers for New().

  Span* LockAndAlloc(Length n, bool* from_released);

  Span* AllocSmall(Length n, bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  Span* AllocLarge(Length n, bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  Span* AllocEnormous(Length n, bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  Span* AllocRawHugepages(Length n, bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  bool AddRegion() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  void ReleaseHugepage(FillerType::Tracker* pt)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  // Return an allocation from a single hugepage.
  void DeleteFromHugepage(FillerType::Tracker* pt, PageId p, Length n)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Finish an allocation request - give it a span and mark it in the pagemap.
  Span* Finalize(Length n, PageId page);
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_HUGE_PAGE_AWARE_ALLOCATOR_H_
