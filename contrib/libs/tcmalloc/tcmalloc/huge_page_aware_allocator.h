#pragma clang system_header
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

#include "absl/base/attributes.h"
#include "absl/base/internal/cycleclock.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/time/time.h"
#include "tcmalloc/arena.h"
#include "tcmalloc/central_freelist.h"
#include "tcmalloc/common.h"
#include "tcmalloc/huge_allocator.h"
#include "tcmalloc/huge_cache.h"
#include "tcmalloc/huge_page_filler.h"
#include "tcmalloc/huge_page_subrelease.h"
#include "tcmalloc/huge_pages.h"
#include "tcmalloc/huge_region.h"
#include "tcmalloc/internal/allocation_guard.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/prefetch.h"
#include "tcmalloc/metadata_allocator.h"
#include "tcmalloc/metadata_object_allocator.h"
#include "tcmalloc/page_allocator_interface.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/span.h"
#include "tcmalloc/stats.h"
#include "tcmalloc/system-alloc.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {
namespace huge_page_allocator_internal {

bool decide_subrelease();

HugeRegionUsageOption huge_region_option();
bool use_huge_region_more_often();

class StaticForwarder {
 public:
  // Runtime parameters.  This can change between calls.
  static absl::Duration filler_skip_subrelease_short_interval() {
    return Parameters::filler_skip_subrelease_short_interval();
  }
  static absl::Duration filler_skip_subrelease_long_interval() {
    return Parameters::filler_skip_subrelease_long_interval();
  }
  static absl::Duration cache_demand_release_short_interval() {
    return Parameters::cache_demand_release_short_interval();
  }
  static absl::Duration cache_demand_release_long_interval() {
    return Parameters::cache_demand_release_long_interval();
  }

  static bool release_partial_alloc_pages() {
    return Parameters::release_partial_alloc_pages();
  }

  static bool huge_region_demand_based_release() {
    return Parameters::huge_region_demand_based_release();
  }

  static bool huge_cache_demand_based_release() {
    return Parameters::huge_cache_demand_based_release();
  }

  static bool hpaa_subrelease() { return Parameters::hpaa_subrelease(); }

  // Arena state.
  static Arena& arena();

  // PageAllocator state.

  // Check page heap memory limit.  `n` indicates the size of the allocation
  // currently being made, which will not be included in the sampled memory heap
  // for realized fragmentation estimation.
  static void ShrinkToUsageLimit(Length n)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // PageMap state.
  static void* GetHugepage(HugePage p);
  [[nodiscard]] static bool Ensure(Range r)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  static void Set(PageId page, Span* span);
  static void SetHugepage(HugePage p, void* pt);

  // SpanAllocator state.
  static Span* NewSpan(Range r)
#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock)
#else
      ABSL_LOCKS_EXCLUDED(pageheap_lock)
#endif  // TCMALLOC_INTERNAL_LEGACY_LOCKING
          ABSL_ATTRIBUTE_RETURNS_NONNULL;
  static void DeleteSpan(Span* span)
#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock)
#endif  // TCMALLOC_INTERNAL_LEGACY_LOCKING
          ABSL_ATTRIBUTE_NONNULL();

  // SystemAlloc state.
  [[nodiscard]] static AddressRange AllocatePages(size_t bytes, size_t align,
                                                  MemoryTag tag);
  static void Back(Range r);
  [[nodiscard]] static bool ReleasePages(Range r);
};

struct HugePageAwareAllocatorOptions {
  MemoryTag tag;
  HugeRegionUsageOption use_huge_region_more_often = huge_region_option();
  HugePageFillerDenseTrackerType dense_tracker_type =
      Parameters::dense_trackers_sorted_on_spans_allocated()
          ? HugePageFillerDenseTrackerType::kSpansAllocated
          : HugePageFillerDenseTrackerType::kLongestFreeRangeAndChunks;
  absl::Duration huge_cache_time = Parameters::huge_cache_release_time();
};

// An implementation of the PageAllocator interface that is hugepage-efficient.
// Attempts to pack allocations into full hugepages wherever possible,
// and aggressively returns empty ones to the system.
//
// Some notes: locking discipline here is a bit funny, because
// we want to *not* hold the pageheap lock while backing memory.
//
// We have here a collection of slightly different allocators each
// optimized for slightly different purposes.  This file has two main purposes:
// - pick the right one for a given allocation
// - provide enough data to figure out what we picked last time!

template <typename Forwarder>
class HugePageAwareAllocator final : public PageAllocatorInterface {
 public:
  explicit HugePageAwareAllocator(const HugePageAwareAllocatorOptions& options);
  ~HugePageAwareAllocator() override = default;

  // Allocate a run of "n" pages.  Returns zero if out of memory.
  // Caller should not pass "n == 0" -- instead, n should have
  // been rounded up already.
  Span* New(Length n, SpanAllocInfo span_alloc_info)
      ABSL_LOCKS_EXCLUDED(pageheap_lock) override;

  // As New, but the returned span is aligned to a <align>-page boundary.
  // <align> must be a power of two.
  Span* NewAligned(Length n, Length align, SpanAllocInfo span_alloc_info)
      ABSL_LOCKS_EXCLUDED(pageheap_lock) override;

  // Delete the span "[p, p+n-1]".
  // REQUIRES: span was returned by earlier call to New() and
  //           has not yet been deleted.
#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
  void Delete(Span* span) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) override;
#endif  // TCMALLOC_INTERNAL_LEGACY_LOCKING

  void Delete(AllocationState s)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) override;

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
  Length ReleaseAtLeastNPages(Length num_pages, PageReleaseReason reason)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) override;

  Length ReleaseAtLeastNPagesBreakingHugepages(Length n,
                                               PageReleaseReason reason)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  PageReleaseStats GetReleaseStats() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) override;

  // Prints stats about the page heap to *out.
  void Print(Printer& out) ABSL_LOCKS_EXCLUDED(pageheap_lock) override;

  // Print stats to *out, excluding long/likely uninteresting things
  // unless <everything> is true.
  void Print(Printer& out, bool everything) ABSL_LOCKS_EXCLUDED(pageheap_lock);

  void PrintInPbtxt(PbtxtRegion& region)
      ABSL_LOCKS_EXCLUDED(pageheap_lock) override;

  BackingStats FillerStats() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return filler_.stats();
  }

  BackingStats RegionsStats() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return regions_.stats();
  }

  BackingStats CacheStats() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return cache_.stats();
  }

  HugeLength DonatedHugePages() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return donated_huge_pages_;
  }

  HugeLength RegionsFreeBacked() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return regions_.free_backed();
  }

  // Number of pages that have been retained on huge pages by donations that did
  // not reassemble by the time the larger allocation was deallocated.
  Length AbandonedPages() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return abandoned_pages_;
  }

  const HugeCache* cache() const { return &cache_; }

  const HugeRegionSet<HugeRegion>& region() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return regions_;
  };

  // IsValidSizeClass verifies size class parameters from the HPAA perspective.
  static bool IsValidSizeClass(size_t size, size_t pages);

  Forwarder& forwarder() { return forwarder_; }

 private:
  static constexpr Length kSmallAllocPages = kPagesPerHugePage / 2;

  class Unback final : public MemoryModifyFunction {
   public:
    explicit Unback(HugePageAwareAllocator& hpaa ABSL_ATTRIBUTE_LIFETIME_BOUND)
        : hpaa_(hpaa) {}

    [[nodiscard]] bool operator()(Range r) override
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
#ifndef NDEBUG
      pageheap_lock.AssertHeld();
#endif  // NDEBUG
      return hpaa_.forwarder_.ReleasePages(r);
    }

   public:
    HugePageAwareAllocator& hpaa_;
  };

  class UnbackWithoutLock final : public MemoryModifyFunction {
   public:
    explicit UnbackWithoutLock(
        HugePageAwareAllocator& hpaa ABSL_ATTRIBUTE_LIFETIME_BOUND)
        : hpaa_(hpaa) {}

    [[nodiscard]] bool operator()(Range r) override
        ABSL_NO_THREAD_SAFETY_ANALYSIS {
#ifndef NDEBUG
      pageheap_lock.AssertHeld();
#endif  // NDEBUG
      pageheap_lock.Unlock();
      bool ret = hpaa_.forwarder_.ReleasePages(r);
      pageheap_lock.Lock();
      return ret;
    }

   public:
    HugePageAwareAllocator& hpaa_;
  };

  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Forwarder forwarder_;

  Unback unback_ ABSL_GUARDED_BY(pageheap_lock);
  UnbackWithoutLock unback_without_lock_ ABSL_GUARDED_BY(pageheap_lock);

  typedef HugePageFiller<PageTracker> FillerType;
  FillerType filler_ ABSL_GUARDED_BY(pageheap_lock);

  class VirtualMemoryAllocator final : public VirtualAllocator {
   public:
    explicit VirtualMemoryAllocator(
        HugePageAwareAllocator& hpaa ABSL_ATTRIBUTE_LIFETIME_BOUND)
        : hpaa_(hpaa) {}

    [[nodiscard]] AddressRange operator()(size_t bytes, size_t align) override
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
      return hpaa_.AllocAndReport(bytes, align);
    }

   private:
    HugePageAwareAllocator& hpaa_;
  };

  class ArenaMetadataAllocator final : public MetadataAllocator {
   public:
    explicit ArenaMetadataAllocator(
        HugePageAwareAllocator& hpaa ABSL_ATTRIBUTE_LIFETIME_BOUND)
        : hpaa_(hpaa) {}

    [[nodiscard]] void* operator()(size_t bytes) override {
      return hpaa_.forwarder_.arena().Alloc(bytes);
    }

   public:
    HugePageAwareAllocator& hpaa_;
  };

  HugeRegionSet<HugeRegion> regions_ ABSL_GUARDED_BY(pageheap_lock);

  MetadataObjectAllocator<FillerType::Tracker> tracker_allocator_
      ABSL_GUARDED_BY(pageheap_lock);
  MetadataObjectAllocator<HugeRegion> region_allocator_
      ABSL_GUARDED_BY(pageheap_lock);

  FillerType::Tracker* GetTracker(HugePage p);

  void SetTracker(HugePage p, FillerType::Tracker* pt);

  AddressRange AllocAndReport(size_t bytes, size_t align)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  VirtualMemoryAllocator vm_allocator_ ABSL_GUARDED_BY(pageheap_lock);
  ArenaMetadataAllocator metadata_allocator_ ABSL_GUARDED_BY(pageheap_lock);
  HugeAllocator alloc_ ABSL_GUARDED_BY(pageheap_lock);
  HugeCache cache_ ABSL_GUARDED_BY(pageheap_lock);

  // donated_huge_pages_ measures the number of huge pages contributed to the
  // filler from left overs of large huge page allocations.  When the large
  // allocation is deallocated, we decrement this count *if* we were able to
  // fully reassemble the address range (that is, the partial hugepage did not
  // get stuck in the filler).
  HugeLength donated_huge_pages_ ABSL_GUARDED_BY(pageheap_lock);
  // abandoned_pages_ tracks the number of pages contributed to the filler after
  // a donating allocation is deallocated but the entire huge page has not been
  // reassembled.
  Length abandoned_pages_ ABSL_GUARDED_BY(pageheap_lock);

  void GetSpanStats(SmallSpanStats* small, LargeSpanStats* large)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  PageId RefillFiller(Length n, SpanAllocInfo span_alloc_info,
                      bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Allocate the first <n> from p, and contribute the rest to the filler.  If
  // "donated" is true, the contribution will be marked as coming from the
  // tail of a multi-hugepage alloc.  Returns the allocated section.
  PageId AllocAndContribute(HugePage p, Length n, SpanAllocInfo span_alloc_info,
                            bool donated)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  // Helpers for New().

#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
  using FinalizeType = Span*;
#else   // !TCMALLOC_INTERNAL_LEGACY_LOCKING
  struct FinalizeType {
    Range r;
    bool donated = false;
  };
#endif  // !TCMALLOC_INTERNAL_LEGACY_LOCKING

  FinalizeType LockAndAlloc(Length n, SpanAllocInfo span_alloc_info,
                            bool* from_released);

  FinalizeType AllocSmall(Length n, SpanAllocInfo span_alloc_info,
                          bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  FinalizeType AllocLarge(Length n, SpanAllocInfo span_alloc_info,
                          bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  FinalizeType AllocEnormous(Length n, SpanAllocInfo span_alloc_info,
                             bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  FinalizeType AllocRawHugepages(Length n, SpanAllocInfo span_alloc_info,
                                 bool* from_released)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  bool AddRegion() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  void ReleaseHugepage(FillerType::Tracker* pt)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  // Return an allocation from a single hugepage.
  void DeleteFromHugepage(FillerType::Tracker* pt, Range r, bool might_abandon)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Finish an allocation request - give it a span and mark it in the pagemap.
  FinalizeType Finalize(Range r);

  Span* Spanify(FinalizeType f);

  // Whether this HPAA should use subrelease. This delegates to the appropriate
  // parameter depending whether this is for the cold heap or another heap.
  bool hpaa_subrelease() const;
};

template <class Forwarder>
inline HugePageAwareAllocator<Forwarder>::HugePageAwareAllocator(
    const HugePageAwareAllocatorOptions& options)
    : PageAllocatorInterface("HugePageAware", options.tag),
      unback_(*this),
      unback_without_lock_(*this),
      filler_(options.dense_tracker_type, unback_, unback_without_lock_),
      regions_(options.use_huge_region_more_often),
      tracker_allocator_(forwarder_.arena()),
      region_allocator_(forwarder_.arena()),
      vm_allocator_(*this),
      metadata_allocator_(*this),
      alloc_(vm_allocator_, metadata_allocator_),
      cache_(HugeCache{&alloc_, metadata_allocator_, unback_without_lock_,
                       options.huge_cache_time}) {}

template <class Forwarder>
inline HugePageAwareAllocator<Forwarder>::FillerType::Tracker*
HugePageAwareAllocator<Forwarder>::GetTracker(HugePage p) {
  void* v = forwarder_.GetHugepage(p);
  FillerType::Tracker* pt = reinterpret_cast<FillerType::Tracker*>(v);
  TC_ASSERT(pt == nullptr || pt->location() == p);
  return pt;
}

template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::SetTracker(
    HugePage p, HugePageAwareAllocator<Forwarder>::FillerType::Tracker* pt) {
  forwarder_.SetHugepage(p, pt);
}

template <class Forwarder>
inline PageId HugePageAwareAllocator<Forwarder>::AllocAndContribute(
    HugePage p, Length n, SpanAllocInfo span_alloc_info, bool donated) {
  TC_CHECK_NE(p.start_addr(), nullptr);
  FillerType::Tracker* pt = tracker_allocator_.New(
      p, donated, absl::base_internal::CycleClock::Now());
  TC_ASSERT_GE(pt->longest_free_range(), n);
  TC_ASSERT_EQ(pt->was_donated(), donated);
  // if the page was donated, we track its size so that we can potentially
  // measure it in abandoned_count_ once this large allocation gets deallocated.
  if (pt->was_donated()) {
    pt->set_abandoned_count(n);
  }
  PageId page = pt->Get(n).page;
  TC_ASSERT_EQ(page, p.first_page());
  SetTracker(p, pt);
  filler_.Contribute(pt, donated, span_alloc_info);
  TC_ASSERT_EQ(pt->was_donated(), donated);
  return page;
}

template <class Forwarder>
inline PageId HugePageAwareAllocator<Forwarder>::RefillFiller(
    Length n, SpanAllocInfo span_alloc_info, bool* from_released) {
  HugeRange r = cache_.Get(NHugePages(1), from_released);
  if (!r.valid()) return PageId{0};
  // This is duplicate to Finalize, but if we need to break up
  // hugepages to get to our usage limit it would be very bad to break
  // up what's left of r after we allocate from there--while r is
  // mostly empty, clearly what's left in the filler is too fragmented
  // to be very useful, and we would rather release those
  // pages. Otherwise, we're nearly guaranteed to release r (if n
  // isn't very large), and the next allocation will just repeat this
  // process.
  forwarder_.ShrinkToUsageLimit(n);
  return AllocAndContribute(r.start(), n, span_alloc_info, /*donated=*/false);
}

template <class Forwarder>
inline typename HugePageAwareAllocator<Forwarder>::FinalizeType
HugePageAwareAllocator<Forwarder>::Finalize(Range r)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
  TC_ASSERT_NE(r.p, PageId{0});
  info_.RecordAlloc(r);
  forwarder_.ShrinkToUsageLimit(r.n);
#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
  // TODO(b/175334169): Lift Span creation out of LockAndAlloc.
  Span* ret = forwarder_.NewSpan(r);
  forwarder_.Set(r.p, ret);
  TC_ASSERT(!ret->sampled());
  return ret;
#else
  return {r, false};
#endif
}

// For anything <= half a huge page, we will unconditionally use the filler
// to pack it into a single page.  If we need another page, that's fine.
template <class Forwarder>
inline typename HugePageAwareAllocator<Forwarder>::FinalizeType
HugePageAwareAllocator<Forwarder>::AllocSmall(Length n,
                                              SpanAllocInfo span_alloc_info,
                                              bool* from_released) {
  auto [pt, page, released] = filler_.TryGet(n, span_alloc_info);
  *from_released = released;
  if (ABSL_PREDICT_TRUE(pt != nullptr)) {
    return Finalize(Range(page, n));
  }

  page = RefillFiller(n, span_alloc_info, from_released);
  if (ABSL_PREDICT_FALSE(page == PageId{0})) {
    return {};
  }
  return Finalize(Range(page, n));
}

template <class Forwarder>
inline typename HugePageAwareAllocator<Forwarder>::FinalizeType
HugePageAwareAllocator<Forwarder>::AllocLarge(Length n,
                                              SpanAllocInfo span_alloc_info,
                                              bool* from_released) {
  // If it's an exact page multiple, just pull it from pages directly.
  HugeLength hl = HLFromPages(n);
  if (hl.in_pages() == n) {
    return AllocRawHugepages(n, span_alloc_info, from_released);
  }

  PageId page;
  // If we fit in a single hugepage, try the Filler.p.
  if (n < kPagesPerHugePage) {
    auto [pt, page, released] = filler_.TryGet(n, span_alloc_info);
    *from_released = released;
    if (ABSL_PREDICT_TRUE(pt != nullptr)) {
      return Finalize(Range(page, n));
    }
  }

  // If we're using regions in this binary (see below comment), is
  // there currently available space there?
  if (regions_.MaybeGet(n, &page, from_released)) {
    return Finalize(Range(page, n));
  }

  // We have two choices here: allocate a new region or go to
  // hugepages directly (hoping that slack will be filled by small
  // allocation.) The second strategy is preferable, as it's
  // typically faster and usually more space efficient, but it's sometimes
  // catastrophic.
  //
  // See https://github.com/google/tcmalloc/tree/master/docs/regions-are-not-optional.md
  //
  // So test directly if we're in the bad case--almost no binaries are.
  // If not, just fall back to direct allocation (and hope we do hit that case!)
  const Length slack = info_.slack();
  const Length donated =
      regions_.UseHugeRegionMoreOften() ? abandoned_pages_ + slack : slack;
  // Don't bother at all until the binary is reasonably sized.
  if (donated < HLFromBytes(64 * 1024 * 1024).in_pages()) {
    return AllocRawHugepages(n, span_alloc_info, from_released);
  }

  // In the vast majority of binaries, we have many small allocations which
  // will nicely fill slack.  (Fleetwide, the average ratio is 15:1; only
  // a handful of binaries fall below 1:1.)
  //
  // If we enable an experiment that tries to use huge regions more frequently,
  // we skip the check.
  const Length small = info_.small();
  if (slack < small && !regions_.UseHugeRegionMoreOften()) {
    return AllocRawHugepages(n, span_alloc_info, from_released);
  }

  // We couldn't allocate a new region. They're oversized, so maybe we'd get
  // lucky with a smaller request?
  if (!AddRegion()) {
    return AllocRawHugepages(n, span_alloc_info, from_released);
  }

  TC_CHECK(regions_.MaybeGet(n, &page, from_released));
  return Finalize(Range(page, n));
}

template <class Forwarder>
inline typename HugePageAwareAllocator<Forwarder>::FinalizeType
HugePageAwareAllocator<Forwarder>::AllocEnormous(Length n,
                                                 SpanAllocInfo span_alloc_info,
                                                 bool* from_released) {
  return AllocRawHugepages(n, span_alloc_info, from_released);
}

template <class Forwarder>
inline typename HugePageAwareAllocator<Forwarder>::FinalizeType
HugePageAwareAllocator<Forwarder>::AllocRawHugepages(
    Length n, SpanAllocInfo span_alloc_info, bool* from_released) {
  HugeLength hl = HLFromPages(n);

  HugeRange r = cache_.Get(hl, from_released);
  if (!r.valid()) return {};

  // We now have a huge page range that covers our request.  There
  // might be some slack in it if n isn't a multiple of
  // kPagesPerHugePage. Add the hugepage with slack to the filler,
  // pretending the non-slack portion is a smaller allocation.
  Length total = hl.in_pages();
  Length slack = total - n;
  HugePage first = r.start();
  SetTracker(first, nullptr);
  HugePage last = first + r.len() - NHugePages(1);
  if (slack == Length(0)) {
    SetTracker(last, nullptr);
    return Finalize(Range(r.start().first_page(), total));
  }

  ++donated_huge_pages_;

  Length here = kPagesPerHugePage - slack;
  TC_ASSERT_GT(here, Length(0));
  AllocAndContribute(last, here, span_alloc_info, /*donated=*/true);
  auto span = Finalize(Range(r.start().first_page(), n));
#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
  span->set_donated(/*value=*/true);
  return span;
#else
  span.donated = true;
  return span;
#endif
}

// public
template <class Forwarder>
inline Span* HugePageAwareAllocator<Forwarder>::New(
    Length n, SpanAllocInfo span_alloc_info) {
  TC_CHECK_GT(n, Length(0));
  bool from_released;
  Span* s = Spanify(LockAndAlloc(n, span_alloc_info, &from_released));
  if (s) {
    // Prefetch for writing, as we anticipate using the memory soon.
    PrefetchW(s->start_address());
    if (from_released) {
      forwarder_.Back(Range(s->first_page(), s->num_pages()));
    }
  }
  TC_ASSERT(!s || GetMemoryTag(s->start_address()) == tag_);
  return s;
}

template <class Forwarder>
inline typename HugePageAwareAllocator<Forwarder>::FinalizeType
HugePageAwareAllocator<Forwarder>::LockAndAlloc(Length n,
                                                SpanAllocInfo span_alloc_info,
                                                bool* from_released) {
  PageHeapSpinLockHolder l;
  // Our policy depends on size.  For small things, we will pack them
  // into single hugepages.
  if (n <= kSmallAllocPages) {
    return AllocSmall(n, span_alloc_info, from_released);
  }

  // For anything too big for the filler, we use either a direct hugepage
  // allocation, or possibly the regions if we are worried about slack.
  if (n <= HugeRegion::size().in_pages()) {
    return AllocLarge(n, span_alloc_info, from_released);
  }

  // In the worst case, we just fall back to directly allocating a run
  // of hugepages.
  return AllocEnormous(n, span_alloc_info, from_released);
}

// public
template <class Forwarder>
inline Span* HugePageAwareAllocator<Forwarder>::NewAligned(
    Length n, Length align, SpanAllocInfo span_alloc_info) {
  if (align <= Length(1)) {
    return New(n, span_alloc_info);
  }

  // we can do better than this, but...
  // TODO(b/134690769): support higher align.
  TC_CHECK_LE(align, kPagesPerHugePage);
  bool from_released;
  FinalizeType f;
  {
    PageHeapSpinLockHolder l;
    f = AllocRawHugepages(n, span_alloc_info, &from_released);
  }
  Span* s = Spanify(f);
  if (s && from_released) {
    forwarder_.Back(Range(s->first_page(), s->num_pages()));
  }

  TC_ASSERT(!s || GetMemoryTag(s->start_address()) == tag_);
  return s;
}

template <class Forwarder>
inline Span* HugePageAwareAllocator<Forwarder>::Spanify(FinalizeType f) {
#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
  return f;
#else
  if (ABSL_PREDICT_FALSE(f.r.p == PageId{0})) {
    return nullptr;
  }

  Span* s = forwarder_.NewSpan(f.r);
  forwarder_.Set(f.r.p, s);
  TC_ASSERT(!s->sampled());
  s->set_donated(f.donated);
  return s;
#endif
}

template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::DeleteFromHugepage(
    FillerType::Tracker* pt, Range r, bool might_abandon) {
  if (ABSL_PREDICT_TRUE(filler_.Put(pt, r) == nullptr)) {
    // If this allocation had resulted in a donation to the filler, we record
    // these pages as abandoned.
    if (ABSL_PREDICT_FALSE(might_abandon)) {
      TC_ASSERT(pt->was_donated());
      abandoned_pages_ += pt->abandoned_count();
      pt->set_abandoned(true);
    }
    return;
  }
  if (pt->was_donated()) {
    --donated_huge_pages_;
    if (pt->abandoned()) {
      abandoned_pages_ -= pt->abandoned_count();
      pt->set_abandoned(false);
    }
  } else {
    TC_ASSERT_EQ(pt->abandoned_count(), Length(0));
  }
  ReleaseHugepage(pt);
}

template <class Forwarder>
inline bool HugePageAwareAllocator<Forwarder>::AddRegion() {
  HugeRange r = alloc_.Get(HugeRegion::size());
  if (!r.valid()) return false;
  HugeRegion* region = region_allocator_.New(r, unback_);
  regions_.Contribute(region);
  return true;
}

#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::Delete(Span* span) {
  TC_ASSERT(!span || GetMemoryTag(span->start_address()) == tag_);
  PageId p = span->first_page();
  Length n = span->num_pages();

  bool donated = span->donated();
  forwarder_.DeleteSpan(span);

  Delete(AllocationState{Range{p, n}, donated});
}
#endif  // TCMALLOC_INTERNAL_LEGACY_LOCKING

template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::Delete(AllocationState s) {
  const PageId p = s.r.p;
  const HugePage hp = HugePageContaining(p);
  const Length n = s.r.n;
  info_.RecordFree(Range(p, n));

  // Clear the descriptor of the page so a second pass through the same page
  // could trigger the check on `span != nullptr` in do_free_pages.
  forwarder_.Set(p, nullptr);

  const bool might_abandon = s.donated;

  // The tricky part, as with so many allocators: where did we come from?
  // There are several possibilities.
  FillerType::Tracker* pt = GetTracker(hp);
  // a) We got packed by the filler onto a single hugepage - return our
  //    allocation to that hugepage in the filler.
  if (ABSL_PREDICT_TRUE(pt != nullptr)) {
    TC_ASSERT_EQ(hp, HugePageContaining(p + n - Length(1)));
    DeleteFromHugepage(pt, Range(p, n), might_abandon);
    return;
  }

  // b) We got put into a region, possibly crossing hugepages -
  //    return our allocation to the region.
  if (regions_.MaybePut(Range(p, n))) return;

  // c) we came straight from the HugeCache - return straight there.  (We
  //    might have had slack put into the filler - if so, return that virtual
  //    allocation to the filler too!)
  TC_ASSERT_GE(n, kPagesPerHugePage);
  HugeLength hl = HLFromPages(n);
  HugePage last = hp + hl - NHugePages(1);
  Length slack = hl.in_pages() - n;
  if (slack == Length(0)) {
    TC_ASSERT_EQ(GetTracker(last), nullptr);
  } else {
    pt = GetTracker(last);
    TC_CHECK_NE(pt, nullptr);
    TC_ASSERT(pt->was_donated());
    // We put the slack into the filler (see AllocEnormous.)
    // Handle this page separately as a virtual allocation
    // onto the last hugepage.
    PageId virt = last.first_page();
    Length virt_len = kPagesPerHugePage - slack;
    // We may have used the slack, which would prevent us from returning
    // the entire range now.  If filler returned a Tracker, we are fully empty.
    if (filler_.Put(pt, Range(virt, virt_len)) == nullptr) {
      // Last page isn't empty -- pretend the range was shorter.
      --hl;

      // Note that we abandoned virt_len pages with pt.  These can be reused for
      // other allocations, but this can contribute to excessive slack in the
      // filler.
      abandoned_pages_ += pt->abandoned_count();
      pt->set_abandoned(true);
    } else {
      // Last page was empty - but if we sub-released it, we still
      // have to split it off and release it independently.)
      //
      // We were able to reclaim the donated slack.
      --donated_huge_pages_;
      TC_ASSERT(!pt->abandoned());

      if (pt->released()) {
        --hl;
        ReleaseHugepage(pt);
      } else {
        // Get rid of the tracker *object*, but not the *hugepage* (which is
        // still part of our range.)
        SetTracker(pt->location(), nullptr);
        tracker_allocator_.Delete(pt);
      }
    }
  }
  // We release in the background task instead (i.e., ReleaseAtLeastNPages()) if
  // the demand-based release is enabled.
  cache_.Release(
      {hp, hl},
      /*demand_based_unback=*/forwarder_.huge_cache_demand_based_release());
}

template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::ReleaseHugepage(
    FillerType::Tracker* pt) {
  TC_ASSERT_EQ(pt->used_pages(), Length(0));
  HugeRange r = {pt->location(), NHugePages(1)};
  SetTracker(pt->location(), nullptr);

  if (pt->released()) {
    cache_.ReleaseUnbacked(r);
  } else {
    // We release in the background task instead (i.e., ReleaseAtLeastNPages())
    // if the demand-based release is enabled.
    cache_.Release(
        r,
        /*demand_based_unback=*/forwarder_.huge_cache_demand_based_release());
  }

  tracker_allocator_.Delete(pt);
}

// public
template <class Forwarder>
inline BackingStats HugePageAwareAllocator<Forwarder>::stats() const {
  BackingStats stats = alloc_.stats();
  const auto actual_system = stats.system_bytes;
  stats += cache_.stats();
  stats += filler_.stats();
  stats += regions_.stats();
  // the "system" (total managed) byte count is wildly double counted,
  // since it all comes from HugeAllocator but is then managed by
  // cache/regions/filler. Adjust for that.
  stats.system_bytes = actual_system;
  return stats;
}

// public
template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::GetSmallSpanStats(
    SmallSpanStats* result) {
  GetSpanStats(result, nullptr);
}

// public
template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::GetLargeSpanStats(
    LargeSpanStats* result) {
  GetSpanStats(nullptr, result);
}

template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::GetSpanStats(
    SmallSpanStats* small, LargeSpanStats* large) {
  if (small != nullptr) {
    *small = SmallSpanStats();
  }
  if (large != nullptr) {
    *large = LargeSpanStats();
  }

  alloc_.AddSpanStats(small, large);
  filler_.AddSpanStats(small, large);
  regions_.AddSpanStats(small, large);
  cache_.AddSpanStats(small, large);
}

// public
template <class Forwarder>
inline Length HugePageAwareAllocator<Forwarder>::ReleaseAtLeastNPages(
    Length num_pages, PageReleaseReason reason) {
  // We use demand-based release for the background release but not for the
  // other cases (e.g., limit hit). We achieve this by configuring the intervals
  // and hit_limit accordingly.
  SkipSubreleaseIntervals cache_release_intervals;
  if (reason == PageReleaseReason::kProcessBackgroundActions) {
    cache_release_intervals.short_interval =
        forwarder_.cache_demand_release_short_interval();
    cache_release_intervals.long_interval =
        forwarder_.cache_demand_release_long_interval();
  }
  bool hit_limit = (reason == PageReleaseReason::kSoftLimitExceeded ||
                    reason == PageReleaseReason::kHardLimitExceeded);
  Length released;
  if (forwarder_.huge_cache_demand_based_release()) {
    released +=
        cache_
            .ReleaseCachedPagesByDemand(HLFromPages(num_pages),
                                        cache_release_intervals, hit_limit)
            .in_pages();
  } else {
    released += cache_.ReleaseCachedPages(HLFromPages(num_pages)).in_pages();
  }

  // Release all backed-but-free hugepages from HugeRegion.
  // TODO(b/199203282): We release all the free hugepages from HugeRegions when
  // the experiment is enabled. We can also explore releasing only a desired
  // number of pages.
  if (regions_.UseHugeRegionMoreOften()) {
    if (forwarder_.huge_region_demand_based_release()) {
      Length desired = released > num_pages ? Length(0) : num_pages - released;
      released += regions_.ReleasePagesByPeakDemand(
          desired,
          SkipSubreleaseIntervals{
              .short_interval =
                  forwarder_.filler_skip_subrelease_short_interval(),
              .long_interval =
                  forwarder_.filler_skip_subrelease_long_interval()},
          /*hit_limit*/ false);
    } else {
      released += regions_.ReleasePages(kFractionToReleaseFromRegion);
    }
  }

  // This is our long term plan but in current state will lead to insufficient
  // THP coverage. It is however very useful to have the ability to turn this on
  // for testing.
  // TODO(b/134690769): make this work, remove the flag guard.
  if (hpaa_subrelease()) {
    if (released < num_pages) {
      released += filler_.ReleasePages(
          num_pages - released,
          SkipSubreleaseIntervals{
              .short_interval =
                  forwarder_.filler_skip_subrelease_short_interval(),
              .long_interval =
                  forwarder_.filler_skip_subrelease_long_interval()},
          forwarder_.release_partial_alloc_pages(),
          /*hit_limit*/ false);
    }
  }

  info_.RecordRelease(num_pages, released, reason);
  return released;
}

inline static double BytesToMiB(size_t bytes) {
  const double MiB = 1048576.0;
  return bytes / MiB;
}

inline static void BreakdownStats(Printer& out, const BackingStats& s,
                                  const char* label) {
  out.printf("%s %6.1f MiB used, %6.1f MiB free, %6.1f MiB unmapped\n", label,
             BytesToMiB(s.system_bytes - s.free_bytes - s.unmapped_bytes),
             BytesToMiB(s.free_bytes), BytesToMiB(s.unmapped_bytes));
}

inline static void BreakdownStatsInPbtxt(PbtxtRegion& hpaa,
                                         const BackingStats& s,
                                         const char* key) {
  auto usage = hpaa.CreateSubRegion(key);
  usage.PrintI64("used", s.system_bytes - s.free_bytes - s.unmapped_bytes);
  usage.PrintI64("free", s.free_bytes);
  usage.PrintI64("unmapped", s.unmapped_bytes);
}

// public
template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::Print(Printer& out) {
  Print(out, true);
}

template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::Print(Printer& out,
                                                     bool everything) {
  SmallSpanStats small;
  LargeSpanStats large;
  BackingStats bstats;
  PageHeapSpinLockHolder l;
  bstats = stats();
  GetSpanStats(&small, &large);
  PrintStats("HugePageAware", out, bstats, small, large, everything);
  out.printf(
      "\nHuge page aware allocator components:\n"
      "------------------------------------------------\n");
  out.printf("HugePageAware: breakdown of used / free / unmapped space:\n");

  auto fstats = filler_.stats();
  BreakdownStats(out, fstats, "HugePageAware: filler  ");

  auto rstats = regions_.stats();
  BreakdownStats(out, rstats, "HugePageAware: region  ");

  auto cstats = cache_.stats();
  // Everything in the filler came from the cache -
  // adjust the totals so we see the amount used by the mutator.
  cstats.system_bytes -= fstats.system_bytes;
  BreakdownStats(out, cstats, "HugePageAware: cache   ");

  auto astats = alloc_.stats();
  // Everything in *all* components came from here -
  // so again adjust the totals.
  astats.system_bytes -= (fstats + rstats + cstats).system_bytes;
  BreakdownStats(out, astats, "HugePageAware: alloc   ");
  out.printf("\n");

  out.printf(
      "HugePageAware: filler donations %zu (%zu pages from abandoned "
      "donations)\n",
      donated_huge_pages_.raw_num(), abandoned_pages_.raw_num());

  // Component debug output
  // Filler is by far the most important; print (some) of it
  // unconditionally.
  filler_.Print(out, everything);
  out.printf("\n");
  if (everything) {
    regions_.Print(out);
    out.printf("\n");
    cache_.Print(out);
    alloc_.Print(out);
    out.printf("\n");

    // Use statistics
    info_.Print(out);
  }

  out.printf("PARAMETER use_huge_region_more_often %d\n",
             regions_.UseHugeRegionMoreOften() ? 1 : 0);
  out.printf("PARAMETER hpaa_subrelease %d\n", hpaa_subrelease() ? 1 : 0);
}

template <class Forwarder>
inline void HugePageAwareAllocator<Forwarder>::PrintInPbtxt(
    PbtxtRegion& region) {
  SmallSpanStats small;
  LargeSpanStats large;
  PageHeapSpinLockHolder l;
  GetSpanStats(&small, &large);
  PrintStatsInPbtxt(region, small, large);
  {
    auto hpaa = region.CreateSubRegion("huge_page_allocator");
    hpaa.PrintBool("using_hpaa", true);
    hpaa.PrintBool("using_hpaa_subrelease", hpaa_subrelease());
    hpaa.PrintBool("use_huge_region_more_often",
                   regions_.UseHugeRegionMoreOften());

    // Fill HPAA Usage
    auto fstats = filler_.stats();
    BreakdownStatsInPbtxt(hpaa, fstats, "filler_usage");

    auto rstats = regions_.stats();
    BreakdownStatsInPbtxt(hpaa, rstats, "region_usage");

    auto cstats = cache_.stats();
    // Everything in the filler came from the cache -
    // adjust the totals so we see the amount used by the mutator.
    cstats.system_bytes -= fstats.system_bytes;
    BreakdownStatsInPbtxt(hpaa, cstats, "cache_usage");

    auto astats = alloc_.stats();
    // Everything in *all* components came from here -
    // so again adjust the totals.
    astats.system_bytes -= (fstats + rstats + cstats).system_bytes;

    BreakdownStatsInPbtxt(hpaa, astats, "alloc_usage");

    filler_.PrintInPbtxt(hpaa);
    regions_.PrintInPbtxt(hpaa);
    cache_.PrintInPbtxt(hpaa);
    alloc_.PrintInPbtxt(hpaa);

    // Use statistics
    info_.PrintInPbtxt(hpaa, "hpaa_stat");

    hpaa.PrintI64("filler_donated_huge_pages", donated_huge_pages_.raw_num());
    hpaa.PrintI64("filler_abandoned_pages", abandoned_pages_.raw_num());
  }
}

template <class Forwarder>
inline AddressRange HugePageAwareAllocator<Forwarder>::AllocAndReport(
    size_t bytes, size_t align) {
  auto ret = forwarder_.AllocatePages(bytes, align, tag_);
  if (ret.ptr == nullptr) return ret;
  const PageId page = PageIdContaining(ret.ptr);
  const Length page_len = BytesToLengthFloor(ret.bytes);
  TC_CHECK(forwarder_.Ensure(Range(page, page_len)),
           "Is something limiting virtual address space?");
  return ret;
}

template <class Forwarder>
inline Length
HugePageAwareAllocator<Forwarder>::ReleaseAtLeastNPagesBreakingHugepages(
    Length n, PageReleaseReason reason) {
  // We desperately need to release memory, and are willing to
  // compromise on hugepage usage. That means releasing from the region and
  // filler.

  Length released;

  if (forwarder_.huge_cache_demand_based_release()) {
    released += cache_
                    .ReleaseCachedPagesByDemand(HLFromPages(n),
                                                SkipSubreleaseIntervals{},
                                                /*hit_limit=*/true)
                    .in_pages();
  } else {
    released += cache_.ReleaseCachedPages(HLFromPages(n)).in_pages();
  }

  // We try to release as many free hugepages from HugeRegion as possible.
  if (forwarder_.huge_region_demand_based_release()) {
    released += regions_.ReleasePagesByPeakDemand(
        n - released, SkipSubreleaseIntervals{}, /*hit_limit=*/true);
  } else {
    released += regions_.ReleasePages(/*release_fraction=*/1.0);
  }

  if (released >= n) {
    info_.RecordRelease(n, released, reason);
    return released;
  }

  released += filler_.ReleasePages(n - released, SkipSubreleaseIntervals{},
                                   /*release_partial_alloc_pages=*/false,
                                   /*hit_limit=*/true);

  info_.RecordRelease(n, released, reason);
  return released;
}

template <class Forwarder>
inline PageReleaseStats HugePageAwareAllocator<Forwarder>::GetReleaseStats()
    const {
  return info_.GetRecordedReleases();
}

template <class Forwarder>
bool HugePageAwareAllocator<Forwarder>::IsValidSizeClass(size_t size,
                                                         size_t pages) {
  // We assume that dense spans won't be donated.
  size_t objects = Length(pages).in_bytes() / size;
  if (objects > central_freelist_internal::kFewObjectsAllocMaxLimit &&
      Length(pages) > kSmallAllocPages) {
    return false;
  }
  return true;
}

template <class Forwarder>
inline bool HugePageAwareAllocator<Forwarder>::hpaa_subrelease() const {
  if (tag_ == MemoryTag::kCold) {
    return true;
  } else {
    return forwarder_.hpaa_subrelease();
  }
}

}  // namespace huge_page_allocator_internal

using HugePageAwareAllocator =
    huge_page_allocator_internal::HugePageAwareAllocator<
        huge_page_allocator_internal::StaticForwarder>;

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_HUGE_PAGE_AWARE_ALLOCATOR_H_
