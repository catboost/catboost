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

#ifndef TCMALLOC_PAGE_ALLOCATOR_H_
#define TCMALLOC_PAGE_ALLOCATOR_H_

#include <inttypes.h>
#include <stddef.h>

#include <utility>

#include "absl/base/thread_annotations.h"
#include "tcmalloc/common.h"
#include "tcmalloc/huge_page_aware_allocator.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/page_allocator_interface.h"
#include "tcmalloc/page_heap.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/span.h"
#include "tcmalloc/stats.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

class PageAllocator {
 public:
  PageAllocator();
  ~PageAllocator() = delete;
  // Allocate a run of "n" pages.  Returns zero if out of memory.
  // Caller should not pass "n == 0" -- instead, n should have
  // been rounded up already.
  //
  // Any address in the returned Span is guaranteed to satisfy
  // GetMemoryTag(addr) == "tag".
  Span* New(Length n, MemoryTag tag) ABSL_LOCKS_EXCLUDED(pageheap_lock);

  // As New, but the returned span is aligned to a <align>-page boundary.
  // <align> must be a power of two.
  Span* NewAligned(Length n, Length align, MemoryTag tag)
      ABSL_LOCKS_EXCLUDED(pageheap_lock);

  // Delete the span "[p, p+n-1]".
  // REQUIRES: span was returned by earlier call to New() with the same value of
  //           "tag" and has not yet been deleted.
  void Delete(Span* span, MemoryTag tag)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  BackingStats stats() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  void GetSmallSpanStats(SmallSpanStats* result)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  void GetLargeSpanStats(LargeSpanStats* result)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Try to release at least num_pages for reuse by the OS.  Returns
  // the actual number of pages released, which may be less than
  // num_pages if there weren't enough pages to release. The result
  // may also be larger than num_pages since page_heap might decide to
  // release one large range instead of fragmenting it into two
  // smaller released and unreleased ranges.
  Length ReleaseAtLeastNPages(Length num_pages)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Prints stats about the page heap to *out.
  void Print(Printer* out, MemoryTag tag) ABSL_LOCKS_EXCLUDED(pageheap_lock);
  void PrintInPbtxt(PbtxtRegion* region, MemoryTag tag)
      ABSL_LOCKS_EXCLUDED(pageheap_lock);

  void set_limit(size_t limit, bool is_hard) ABSL_LOCKS_EXCLUDED(pageheap_lock);
  std::pair<size_t, bool> limit() const ABSL_LOCKS_EXCLUDED(pageheap_lock);
  int64_t limit_hits() const ABSL_LOCKS_EXCLUDED(pageheap_lock);

  // If we have a usage limit set, ensure we're not violating it from our latest
  // allocation.
  void ShrinkToUsageLimit() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  const PageAllocInfo& info(MemoryTag tag) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  enum Algorithm {
    PAGE_HEAP = 0,
    HPAA = 1,
  };

  Algorithm algorithm() const { return alg_; }

 private:
  bool ShrinkHardBy(Length pages) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  ABSL_ATTRIBUTE_RETURNS_NONNULL PageAllocatorInterface* impl(
      MemoryTag tag) const;

  size_t active_numa_partitions() const;

  static constexpr size_t kNumHeaps = kNumaPartitions + 1;

  union Choices {
    Choices() : dummy(0) {}
    ~Choices() {}
    int dummy;
    PageHeap ph;
    HugePageAwareAllocator hpaa;
  } choices_[kNumHeaps];
  std::array<PageAllocatorInterface*, kNumaPartitions> normal_impl_;
  PageAllocatorInterface* sampled_impl_;
  Algorithm alg_;

  bool limit_is_hard_{false};
  // Max size of backed spans we will attempt to maintain.
  size_t limit_{std::numeric_limits<size_t>::max()};
  // The number of times the limit has been hit.
  int64_t limit_hits_{0};
};

inline PageAllocatorInterface* PageAllocator::impl(MemoryTag tag) const {
  switch (tag) {
    case MemoryTag::kNormalP0:
      return normal_impl_[0];
    case MemoryTag::kNormalP1:
      return normal_impl_[1];
    case MemoryTag::kSampled:
      return sampled_impl_;
    default:
      ASSUME(false);
      __builtin_unreachable();
  }
}

inline Span* PageAllocator::New(Length n, MemoryTag tag) {
  return impl(tag)->New(n);
}

inline Span* PageAllocator::NewAligned(Length n, Length align, MemoryTag tag) {
  return impl(tag)->NewAligned(n, align);
}

inline void PageAllocator::Delete(Span* span, MemoryTag tag) {
  impl(tag)->Delete(span);
}

inline BackingStats PageAllocator::stats() const {
  BackingStats ret = normal_impl_[0]->stats();
  for (int partition = 1; partition < active_numa_partitions(); partition++) {
    ret += normal_impl_[partition]->stats();
  }
  ret += sampled_impl_->stats();
  return ret;
}

inline void PageAllocator::GetSmallSpanStats(SmallSpanStats* result) {
  SmallSpanStats normal, sampled;
  for (int partition = 0; partition < active_numa_partitions(); partition++) {
    SmallSpanStats part_stats;
    normal_impl_[partition]->GetSmallSpanStats(&part_stats);
    normal += part_stats;
  }
  sampled_impl_->GetSmallSpanStats(&sampled);
  *result = normal + sampled;
}

inline void PageAllocator::GetLargeSpanStats(LargeSpanStats* result) {
  LargeSpanStats normal, sampled;
  for (int partition = 0; partition < active_numa_partitions(); partition++) {
    LargeSpanStats part_stats;
    normal_impl_[partition]->GetLargeSpanStats(&part_stats);
    normal += part_stats;
  }
  sampled_impl_->GetLargeSpanStats(&sampled);
  *result = normal + sampled;
}

inline Length PageAllocator::ReleaseAtLeastNPages(Length num_pages) {
  Length released;
  for (int partition = 0; partition < active_numa_partitions(); partition++) {
    released +=
        normal_impl_[partition]->ReleaseAtLeastNPages(num_pages - released);
    if (released >= num_pages) {
      return released;
    }
  }

  released += sampled_impl_->ReleaseAtLeastNPages(num_pages - released);
  return released;
}

inline void PageAllocator::Print(Printer* out, MemoryTag tag) {
  const absl::string_view label = MemoryTagToLabel(tag);
  if (tag != MemoryTag::kNormal) {
    out->printf("\n>>>>>>> Begin %s page allocator <<<<<<<\n", label);
  }
  impl(tag)->Print(out);
  if (tag != MemoryTag::kNormal) {
    out->printf(">>>>>>> End %s page allocator <<<<<<<\n", label);
  }
}

inline void PageAllocator::PrintInPbtxt(PbtxtRegion* region, MemoryTag tag) {
  PbtxtRegion pa = region->CreateSubRegion("page_allocator");
  pa.PrintRaw("tag", MemoryTagToLabel(tag));
  impl(tag)->PrintInPbtxt(&pa);
}

inline void PageAllocator::set_limit(size_t limit, bool is_hard) {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  limit_ = limit;
  limit_is_hard_ = is_hard;
}

inline std::pair<size_t, bool> PageAllocator::limit() const {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  return {limit_, limit_is_hard_};
}

inline int64_t PageAllocator::limit_hits() const {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  return limit_hits_;
}

inline const PageAllocInfo& PageAllocator::info(MemoryTag tag) const {
  return impl(tag)->info();
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_PAGE_ALLOCATOR_H_
