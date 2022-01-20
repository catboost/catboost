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

#include "tcmalloc/huge_page_aware_allocator.h"

#include <stdint.h>
#include <string.h>

#include <new>

#include "absl/base/internal/cycleclock.h"
#include "absl/base/internal/spinlock.h"
#include "absl/time/time.h"
#include "tcmalloc/common.h"
#include "tcmalloc/experiment.h"
#include "tcmalloc/experiment_config.h"
#include "tcmalloc/huge_allocator.h"
#include "tcmalloc/huge_pages.h"
#include "tcmalloc/internal/environment.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/pagemap.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/span.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/stats.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

bool decide_want_hpaa();
ABSL_ATTRIBUTE_WEAK int default_want_hpaa();
ABSL_ATTRIBUTE_WEAK int default_subrelease();

bool decide_subrelease() {
  if (!decide_want_hpaa()) {
    // Subrelease is off if HPAA is off.
    return false;
  }

  const char *e = thread_safe_getenv("TCMALLOC_HPAA_CONTROL");
  if (e) {
    switch (e[0]) {
      case '0':
        if (kPageShift <= 12) {
          return false;
        }

        if (default_want_hpaa != nullptr) {
          int default_hpaa = default_want_hpaa();
          if (default_hpaa < 0) {
            return false;
          }
        }

        Log(kLog, __FILE__, __LINE__,
            "Runtime opt-out from HPAA requires building with "
            "//tcmalloc:want_no_hpaa."
        );
        break;
      case '1':
        return false;
      case '2':
        return true;
      default:
        Crash(kCrash, __FILE__, __LINE__, "bad env var", e);
        return false;
    }
  }

  if (default_subrelease != nullptr) {
    const int decision = default_subrelease();
    if (decision != 0) {
      return decision > 0;
    }
  }

  if (tcmalloc::IsExperimentActive(tcmalloc::Experiment::TCMALLOC_TEMERAIRE)) {
    return false;
  }

  return true;
}

FillerPartialRerelease decide_partial_rerelease() {
  const char *e = thread_safe_getenv("TCMALLOC_PARTIAL_RELEASE_CONTROL");
  if (e) {
    if (e[0] == '0') {
      return FillerPartialRerelease::Return;
    }
    if (e[0] == '1') {
      return FillerPartialRerelease::Retain;
    }
    Crash(kCrash, __FILE__, __LINE__, "bad env var", e);
  }

  return FillerPartialRerelease::Retain;
}

// Some notes: locking discipline here is a bit funny, because
// we want to *not* hold the pageheap lock while backing memory.

// We have here a collection of slightly different allocators each
// optimized for slightly different purposes.  This file has two main purposes:
// - pick the right one for a given allocation
// - provide enough data to figure out what we picked last time!

HugePageAwareAllocator::HugePageAwareAllocator(MemoryTag tag)
    : PageAllocatorInterface("HugePageAware", tag),
      filler_(decide_partial_rerelease()),
      alloc_(
          [](MemoryTag tag) {
            // TODO(ckennelly): Remove the template parameter.
            switch (tag) {
              case MemoryTag::kNormal:
                return AllocAndReport<MemoryTag::kNormal>;
              case MemoryTag::kNormalP1:
                return AllocAndReport<MemoryTag::kNormalP1>;
              case MemoryTag::kSampled:
                return AllocAndReport<MemoryTag::kSampled>;
              default:
                ASSUME(false);
                __builtin_unreachable();
            }
          }(tag),
          MetaDataAlloc),
      cache_(HugeCache{&alloc_, MetaDataAlloc, UnbackWithoutLock}) {
  tracker_allocator_.Init(&Static::arena());
  region_allocator_.Init(&Static::arena());
}

HugePageAwareAllocator::FillerType::Tracker *HugePageAwareAllocator::GetTracker(
    HugePage p) {
  void *v = Static::pagemap().GetHugepage(p.first_page());
  FillerType::Tracker *pt = reinterpret_cast<FillerType::Tracker *>(v);
  ASSERT(pt == nullptr || pt->location() == p);
  return pt;
}

void HugePageAwareAllocator::SetTracker(
    HugePage p, HugePageAwareAllocator::FillerType::Tracker *pt) {
  Static::pagemap().SetHugepage(p.first_page(), pt);
}

PageId HugePageAwareAllocator::AllocAndContribute(HugePage p, Length n,
                                                  bool donated) {
  CHECK_CONDITION(p.start_addr() != nullptr);
  FillerType::Tracker *pt = tracker_allocator_.New();
  new (pt) FillerType::Tracker(p, absl::base_internal::CycleClock::Now());
  ASSERT(pt->longest_free_range() >= n);
  PageId page = pt->Get(n).page;
  ASSERT(page == p.first_page());
  SetTracker(p, pt);
  filler_.Contribute(pt, donated);
  return page;
}

PageId HugePageAwareAllocator::RefillFiller(Length n, bool *from_released) {
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
  Static::page_allocator().ShrinkToUsageLimit();
  return AllocAndContribute(r.start(), n, /*donated=*/false);
}

Span *HugePageAwareAllocator::Finalize(Length n, PageId page)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
  ASSERT(page != PageId{0});
  Span *ret = Span::New(page, n);
  Static::pagemap().Set(page, ret);
  ASSERT(!ret->sampled());
  info_.RecordAlloc(page, n);
  Static::page_allocator().ShrinkToUsageLimit();
  return ret;
}

// For anything <= half a huge page, we will unconditionally use the filler
// to pack it into a single page.  If we need another page, that's fine.
Span *HugePageAwareAllocator::AllocSmall(Length n, bool *from_released) {
  auto [pt, page] = filler_.TryGet(n);
  if (ABSL_PREDICT_TRUE(pt != nullptr)) {
    *from_released = false;
    return Finalize(n, page);
  }

  page = RefillFiller(n, from_released);
  if (ABSL_PREDICT_FALSE(page == PageId{0})) {
    return nullptr;
  }
  return Finalize(n, page);
}

Span *HugePageAwareAllocator::AllocLarge(Length n, bool *from_released) {
  // If it's an exact page multiple, just pull it from pages directly.
  HugeLength hl = HLFromPages(n);
  if (hl.in_pages() == n) {
    return AllocRawHugepages(n, from_released);
  }

  PageId page;
  // If we fit in a single hugepage, try the Filler first.
  if (n < kPagesPerHugePage) {
    auto [pt, page] = filler_.TryGet(n);
    if (ABSL_PREDICT_TRUE(pt != nullptr)) {
      *from_released = false;
      return Finalize(n, page);
    }
  }

  // If we're using regions in this binary (see below comment), is
  // there currently available space there?
  if (regions_.MaybeGet(n, &page, from_released)) {
    return Finalize(n, page);
  }

  // We have two choices here: allocate a new region or go to
  // hugepages directly (hoping that slack will be filled by small
  // allocation.) The second strategy is preferrable, as it's
  // typically faster and usually more space efficient, but it's sometimes
  // catastrophic.
  //
  // See https://github.com/google/tcmalloc/tree/master/docs/regions-are-not-optional.md
  //
  // So test directly if we're in the bad case--almost no binaries are.
  // If not, just fall back to direct allocation (and hope we do hit that case!)
  const Length slack = info_.slack();
  // Don't bother at all until the binary is reasonably sized
  if (slack < HLFromBytes(64 * 1024 * 1024).in_pages()) {
    return AllocRawHugepages(n, from_released);
  }

  // In the vast majority of binaries, we have many small allocations which
  // will nicely fill slack.  (Fleetwide, the average ratio is 15:1; only
  // a handful of binaries fall below 1:1.)
  const Length small = info_.small();
  if (slack < small) {
    return AllocRawHugepages(n, from_released);
  }

  // We couldn't allocate a new region. They're oversized, so maybe we'd get
  // lucky with a smaller request?
  if (!AddRegion()) {
    return AllocRawHugepages(n, from_released);
  }

  CHECK_CONDITION(regions_.MaybeGet(n, &page, from_released));
  return Finalize(n, page);
}

Span *HugePageAwareAllocator::AllocEnormous(Length n, bool *from_released) {
  return AllocRawHugepages(n, from_released);
}

Span *HugePageAwareAllocator::AllocRawHugepages(Length n, bool *from_released) {
  HugeLength hl = HLFromPages(n);

  HugeRange r = cache_.Get(hl, from_released);
  if (!r.valid()) return nullptr;

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
    return Finalize(total, r.start().first_page());
  }

  ++donated_huge_pages_;

  Length here = kPagesPerHugePage - slack;
  ASSERT(here > Length(0));
  AllocAndContribute(last, here, /*donated=*/true);
  return Finalize(n, r.start().first_page());
}

static void BackSpan(Span *span) {
  SystemBack(span->start_address(), span->bytes_in_span());
}

// public
Span *HugePageAwareAllocator::New(Length n) {
  CHECK_CONDITION(n > Length(0));
  bool from_released;
  Span *s = LockAndAlloc(n, &from_released);
  if (s) {
    // Prefetch for writing, as we anticipate using the memory soon.
    __builtin_prefetch(s->start_address(), 1, 3);
    if (from_released) BackSpan(s);
  }
  ASSERT(!s || GetMemoryTag(s->start_address()) == tag_);
  return s;
}

Span *HugePageAwareAllocator::LockAndAlloc(Length n, bool *from_released) {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  // Our policy depends on size.  For small things, we will pack them
  // into single hugepages.
  if (n <= kPagesPerHugePage / 2) {
    return AllocSmall(n, from_released);
  }

  // For anything too big for the filler, we use either a direct hugepage
  // allocation, or possibly the regions if we are worried about slack.
  if (n <= HugeRegion::size().in_pages()) {
    return AllocLarge(n, from_released);
  }

  // In the worst case, we just fall back to directly allocating a run
  // of hugepages.
  return AllocEnormous(n, from_released);
}

// public
Span *HugePageAwareAllocator::NewAligned(Length n, Length align) {
  if (align <= Length(1)) {
    return New(n);
  }

  // we can do better than this, but...
  // TODO(b/134690769): support higher align.
  CHECK_CONDITION(align <= kPagesPerHugePage);
  bool from_released;
  Span *s;
  {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    s = AllocRawHugepages(n, &from_released);
  }
  if (s && from_released) BackSpan(s);
  ASSERT(!s || GetMemoryTag(s->start_address()) == tag_);
  return s;
}

void HugePageAwareAllocator::DeleteFromHugepage(FillerType::Tracker *pt,
                                                PageId p, Length n) {
  if (ABSL_PREDICT_TRUE(filler_.Put(pt, p, n) == nullptr)) return;
  if (pt->donated()) {
    --donated_huge_pages_;
  }
  ReleaseHugepage(pt);
}

bool HugePageAwareAllocator::AddRegion() {
  HugeRange r = alloc_.Get(HugeRegion::size());
  if (!r.valid()) return false;
  HugeRegion *region = region_allocator_.New();
  new (region) HugeRegion(r, SystemRelease);
  regions_.Contribute(region);
  return true;
}

void HugePageAwareAllocator::Delete(Span *span) {
  ASSERT(!span || GetMemoryTag(span->start_address()) == tag_);
  PageId p = span->first_page();
  HugePage hp = HugePageContaining(p);
  Length n = span->num_pages();
  info_.RecordFree(p, n);

  Span::Delete(span);

  // The tricky part, as with so many allocators: where did we come from?
  // There are several possibilities.
  FillerType::Tracker *pt = GetTracker(hp);
  // a) We got packed by the filler onto a single hugepage - return our
  //    allocation to that hugepage in the filler.
  if (ABSL_PREDICT_TRUE(pt != nullptr)) {
    ASSERT(hp == HugePageContaining(p + n - Length(1)));
    DeleteFromHugepage(pt, p, n);
    return;
  }

  // b) We got put into a region, possibly crossing hugepages -
  //    return our allocation to the region.
  if (regions_.MaybePut(p, n)) return;

  // c) we came straight from the HugeCache - return straight there.  (We
  //    might have had slack put into the filler - if so, return that virtual
  //    allocation to the filler too!)
  ASSERT(n >= kPagesPerHugePage);
  HugeLength hl = HLFromPages(n);
  HugePage last = hp + hl - NHugePages(1);
  Length slack = hl.in_pages() - n;
  if (slack == Length(0)) {
    ASSERT(GetTracker(last) == nullptr);
  } else {
    pt = GetTracker(last);
    CHECK_CONDITION(pt != nullptr);
    // We put the slack into the filler (see AllocEnormous.)
    // Handle this page separately as a virtual allocation
    // onto the last hugepage.
    PageId virt = last.first_page();
    Length virt_len = kPagesPerHugePage - slack;
    pt = filler_.Put(pt, virt, virt_len);
    // We may have used the slack, which would prevent us from returning
    // the entire range now.  If filler returned a Tracker, we are fully empty.
    if (pt == nullptr) {
      // Last page isn't empty -- pretend the range was shorter.
      --hl;
    } else {
      // Last page was empty - but if we sub-released it, we still
      // have to split it off and release it independently.)
      if (pt->released()) {
        --hl;
        ReleaseHugepage(pt);
      } else {
        // Get rid of the tracker *object*, but not the *hugepage*
        // (which is still part of our range.)  We were able to reclaim the
        // contributed slack.
        --donated_huge_pages_;
        SetTracker(pt->location(), nullptr);
        tracker_allocator_.Delete(pt);
      }
    }
  }
  cache_.Release({hp, hl});
}

void HugePageAwareAllocator::ReleaseHugepage(FillerType::Tracker *pt) {
  ASSERT(pt->used_pages() == Length(0));
  HugeRange r = {pt->location(), NHugePages(1)};
  SetTracker(pt->location(), nullptr);

  if (pt->released()) {
    cache_.ReleaseUnbacked(r);
  } else {
    cache_.Release(r);
  }

  tracker_allocator_.Delete(pt);
}

// public
BackingStats HugePageAwareAllocator::stats() const {
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
void HugePageAwareAllocator::GetSmallSpanStats(SmallSpanStats *result) {
  GetSpanStats(result, nullptr, nullptr);
}

// public
void HugePageAwareAllocator::GetLargeSpanStats(LargeSpanStats *result) {
  GetSpanStats(nullptr, result, nullptr);
}

void HugePageAwareAllocator::GetSpanStats(SmallSpanStats *small,
                                          LargeSpanStats *large,
                                          PageAgeHistograms *ages) {
  if (small != nullptr) {
    *small = SmallSpanStats();
  }
  if (large != nullptr) {
    *large = LargeSpanStats();
  }

  alloc_.AddSpanStats(small, large, ages);
  filler_.AddSpanStats(small, large, ages);
  regions_.AddSpanStats(small, large, ages);
  cache_.AddSpanStats(small, large, ages);
}

// public
Length HugePageAwareAllocator::ReleaseAtLeastNPages(Length num_pages) {
  Length released;
  released += cache_.ReleaseCachedPages(HLFromPages(num_pages)).in_pages();

  // This is our long term plan but in current state will lead to insufficent
  // THP coverage. It is however very useful to have the ability to turn this on
  // for testing.
  // TODO(b/134690769): make this work, remove the flag guard.
  if (Parameters::hpaa_subrelease()) {
    if (released < num_pages) {
      released += filler_.ReleasePages(
          num_pages - released, Parameters::filler_skip_subrelease_interval(),
          /*hit_limit*/ false);
    }
  }

  // TODO(b/134690769):
  // - perhaps release region?
  // - refuse to release if we're too close to zero?
  info_.RecordRelease(num_pages, released);
  return released;
}

static double BytesToMiB(size_t bytes) {
  const double MiB = 1048576.0;
  return bytes / MiB;
}

static void BreakdownStats(Printer *out, const BackingStats &s,
                           const char *label) {
  out->printf("%s %6.1f MiB used, %6.1f MiB free, %6.1f MiB unmapped\n", label,
              BytesToMiB(s.system_bytes - s.free_bytes - s.unmapped_bytes),
              BytesToMiB(s.free_bytes), BytesToMiB(s.unmapped_bytes));
}

static void BreakdownStatsInPbtxt(PbtxtRegion *hpaa, const BackingStats &s,
                                  const char *key) {
  auto usage = hpaa->CreateSubRegion(key);
  usage.PrintI64("used", s.system_bytes - s.free_bytes - s.unmapped_bytes);
  usage.PrintI64("free", s.free_bytes);
  usage.PrintI64("unmapped", s.unmapped_bytes);
}

// public
void HugePageAwareAllocator::Print(Printer *out) { Print(out, true); }

void HugePageAwareAllocator::Print(Printer *out, bool everything) {
  SmallSpanStats small;
  LargeSpanStats large;
  BackingStats bstats;
  PageAgeHistograms ages(absl::base_internal::CycleClock::Now());
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  bstats = stats();
  GetSpanStats(&small, &large, &ages);
  PrintStats("HugePageAware", out, bstats, small, large, everything);
  out->printf(
      "\nHuge page aware allocator components:\n"
      "------------------------------------------------\n");
  out->printf("HugePageAware: breakdown of used / free / unmapped space:\n");

  auto fstats = filler_.stats();
  BreakdownStats(out, fstats, "HugePageAware: filler");

  auto rstats = regions_.stats();
  BreakdownStats(out, rstats, "HugePageAware: region");

  auto cstats = cache_.stats();
  // Everything in the filler came from the cache -
  // adjust the totals so we see the amount used by the mutator.
  cstats.system_bytes -= fstats.system_bytes;
  BreakdownStats(out, cstats, "HugePageAware: cache ");

  auto astats = alloc_.stats();
  // Everything in *all* components came from here -
  // so again adjust the totals.
  astats.system_bytes -= (fstats + rstats + cstats).system_bytes;
  BreakdownStats(out, astats, "HugePageAware: alloc ");
  out->printf("\n");

  out->printf("HugePageAware: filler donations %zu\n",
              donated_huge_pages_.raw_num());

  // Component debug output
  // Filler is by far the most important; print (some) of it
  // unconditionally.
  filler_.Print(out, everything);
  out->printf("\n");
  if (everything) {
    regions_.Print(out);
    out->printf("\n");
    cache_.Print(out);
    out->printf("\n");
    alloc_.Print(out);
    out->printf("\n");

    // Use statistics
    info_.Print(out);

    // and age tracking.
    ages.Print("HugePageAware", out);
  }

  out->printf("PARAMETER hpaa_subrelease %d\n",
              Parameters::hpaa_subrelease() ? 1 : 0);
}

void HugePageAwareAllocator::PrintInPbtxt(PbtxtRegion *region) {
  SmallSpanStats small;
  LargeSpanStats large;
  PageAgeHistograms ages(absl::base_internal::CycleClock::Now());
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  GetSpanStats(&small, &large, &ages);
  PrintStatsInPbtxt(region, small, large, ages);
  {
    auto hpaa = region->CreateSubRegion("huge_page_allocator");
    hpaa.PrintBool("using_hpaa", true);
    hpaa.PrintBool("using_hpaa_subrelease", Parameters::hpaa_subrelease());

    // Fill HPAA Usage
    auto fstats = filler_.stats();
    BreakdownStatsInPbtxt(&hpaa, fstats, "filler_usage");

    auto rstats = regions_.stats();
    BreakdownStatsInPbtxt(&hpaa, rstats, "region_usage");

    auto cstats = cache_.stats();
    // Everything in the filler came from the cache -
    // adjust the totals so we see the amount used by the mutator.
    cstats.system_bytes -= fstats.system_bytes;
    BreakdownStatsInPbtxt(&hpaa, cstats, "cache_usage");

    auto astats = alloc_.stats();
    // Everything in *all* components came from here -
    // so again adjust the totals.
    astats.system_bytes -= (fstats + rstats + cstats).system_bytes;
    BreakdownStatsInPbtxt(&hpaa, astats, "alloc_usage");

    filler_.PrintInPbtxt(&hpaa);
    regions_.PrintInPbtxt(&hpaa);
    cache_.PrintInPbtxt(&hpaa);
    alloc_.PrintInPbtxt(&hpaa);

    // Use statistics
    info_.PrintInPbtxt(&hpaa, "hpaa_stat");

    hpaa.PrintI64("filler_donated_huge_pages", donated_huge_pages_.raw_num());
  }
}

template <MemoryTag tag>
void *HugePageAwareAllocator::AllocAndReport(size_t bytes, size_t *actual,
                                             size_t align) {
  void *p = SystemAlloc(bytes, actual, align, tag);
  if (p == nullptr) return p;
  const PageId page = PageIdContaining(p);
  const Length page_len = BytesToLengthFloor(*actual);
  Static::pagemap().Ensure(page, page_len);
  return p;
}

void *HugePageAwareAllocator::MetaDataAlloc(size_t bytes)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
  return Static::arena().Alloc(bytes);
}

Length HugePageAwareAllocator::ReleaseAtLeastNPagesBreakingHugepages(Length n) {
  // We desparately need to release memory, and are willing to
  // compromise on hugepage usage. That means releasing from the filler.
  return filler_.ReleasePages(n, absl::ZeroDuration(), /*hit_limit*/ true);
}

void HugePageAwareAllocator::UnbackWithoutLock(void *start, size_t length) {
  pageheap_lock.Unlock();
  SystemRelease(start, length);
  pageheap_lock.Lock();
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
