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

#include "tcmalloc/page_allocator.h"

#include <cstddef>
#include <limits>

#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "tcmalloc/common.h"
#include "tcmalloc/huge_page_aware_allocator.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/environment.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/selsan/selsan.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/stats.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

using huge_page_allocator_internal::HugePageAwareAllocatorOptions;

PageAllocator::PageAllocator() {
  has_cold_impl_ = ColdFeatureActive();
  size_t part = 0;

  normal_impl_[0] = new (&choices_[part++].hpaa)
      HugePageAwareAllocator(HugePageAwareAllocatorOptions{MemoryTag::kNormal});
  if (tc_globals.numa_topology().numa_aware()) {
    normal_impl_[1] = new (&choices_[part++].hpaa) HugePageAwareAllocator(
        HugePageAwareAllocatorOptions{MemoryTag::kNormalP1});
  }
  sampled_impl_ = new (&choices_[part++].hpaa) HugePageAwareAllocator(
      HugePageAwareAllocatorOptions{MemoryTag::kSampled});
  if (selsan::IsEnabled()) {
    selsan_impl_ = new (&choices_[part++].hpaa) HugePageAwareAllocator(
        HugePageAwareAllocatorOptions{MemoryTag::kSelSan});
  }
  if (has_cold_impl_) {
    cold_impl_ = new (&choices_[part++].hpaa)
        HugePageAwareAllocator(HugePageAwareAllocatorOptions{MemoryTag::kCold});
  } else {
    cold_impl_ = normal_impl_[0];
  }
  alg_ = HPAA;
  TC_CHECK_LE(part, ABSL_ARRAYSIZE(choices_));
}

void PageAllocator::ShrinkToUsageLimit(Length n) {
  BackingStats s = stats();
  const size_t backed =
      s.system_bytes - s.unmapped_bytes + tc_globals.metadata_bytes();
  // New high water marks should be rare.
  if (ABSL_PREDICT_FALSE(backed > peak_backed_bytes_)) {
    peak_backed_bytes_ = backed;
    // This estimate may skew slightly low (and overestimate realized
    // fragmentation), as we allocate successfully from the page heap before
    // updating the sampled object list.
    //
    // TODO(ckennelly): The correction for n overestimates for many-object
    // spans from the CentralFreeList, but those are typically a single page so
    // the error in absolute terms is minimal.
    peak_sampled_application_bytes_ =
        tc_globals.sampled_objects_size_.value() + n.in_bytes();
  }
  // TODO(ckennelly): Consider updating peak_sampled_application_bytes_ if
  // backed == peak_backed_bytes_ but application usage has gone up.  This can
  // occur if we allocate space for many objects preemptively and only later
  // sample them (incrementing sampled_objects_size_).

  if (limits_[kSoft] == std::numeric_limits<size_t>::max()) {
    // Limits are not set.
    return;
  }
  if (backed <= limits_[kSoft]) {
    // We're already fine.
    return;
  }

  ++limit_hits_[kSoft];
  if (limits_[kHard] < backed) ++limit_hits_[kHard];

  const size_t overage = backed - limits_[kSoft];
  const Length pages = LengthFromBytes(overage + kPageSize - 1);
  if (ShrinkHardBy(pages, kSoft)) {
    ++successful_shrinks_after_limit_hit_[kSoft];
    return;
  }

  // We're still not below limit.
  if (limits_[kHard] < std::numeric_limits<size_t>::max()) {
    // Recompute how many pages we still need to release.
    BackingStats s = stats();
    const size_t backed =
        s.system_bytes - s.unmapped_bytes + tc_globals.metadata_bytes();
    if (backed <= limits_[kHard]) {
      // We're already fine in terms of hard limit.
      return;
    }
    const size_t overage = backed - limits_[kHard];
    const Length pages = LengthFromBytes(overage + kPageSize - 1);
    if (ShrinkHardBy(pages, kHard)) {
      ++successful_shrinks_after_limit_hit_[kHard];
      TC_ASSERT_EQ(successful_shrinks_after_limit_hit_[kHard],
                   limit_hits_[kHard]);
      return;
    }
    const size_t hard_limit = limits_[kHard];
    limits_[kHard] = std::numeric_limits<size_t>::max();
    TC_BUG(
        "Hit hard tcmalloc heap limit of %v "
        "(e.g. --tcmalloc_heap_size_hard_limit). "
        "Aborting.\nIt was most likely set to catch "
        "allocations that would crash the process anyway. "
        ,
        hard_limit);
  }

  // Print logs once.
  static bool warned = false;
  if (warned) return;
  warned = true;
  TC_LOG("Couldn't respect usage limit of %v and OOM is likely to follow.",
         limits_[kSoft]);

  if (auto* handler = MallocExtension::GetSoftMemoryLimitHandler()) {
    (*handler)();
  }
}

bool PageAllocator::ShrinkHardBy(Length pages, LimitKind limit_kind) {
  const PageReleaseReason release_reason =
      limit_kind == kHard ? PageReleaseReason::kHardLimitExceeded
                          : PageReleaseReason::kSoftLimitExceeded;
  Length ret = ReleaseAtLeastNPages(pages, release_reason);
  if (alg_ == HPAA) {
    if (pages <= ret) {
      // We released target amount.
      return true;
    }

    // At this point, we have no choice but to break up hugepages.
    // However, if the client has turned off subrelease, and is using hard
    // limits, then respect desire to do no subrelease ever.
    if (limit_kind == kHard && !Parameters::hpaa_subrelease()) return false;

    static bool warned_hugepages = false;
    if (!warned_hugepages) {
      const size_t limit = limits_[limit_kind];
      TC_LOG(
          "Couldn't respect usage limit of %v without breaking hugepages - "
          "performance will drop",
          limit);
      warned_hugepages = true;
    }
    if (has_cold_impl_) {
      ret += static_cast<HugePageAwareAllocator*>(cold_impl_)
                 ->ReleaseAtLeastNPagesBreakingHugepages(pages - ret,
                                                         release_reason);
      if (ret >= pages) {
        return true;
      }
    }
    if (selsan_impl_) {
      ret += static_cast<HugePageAwareAllocator*>(selsan_impl_)
                 ->ReleaseAtLeastNPagesBreakingHugepages(pages - ret,
                                                         release_reason);
      if (ret >= pages) {
        return true;
      }
    }
    for (int partition = 0; partition < active_numa_partitions(); partition++) {
      ret += static_cast<HugePageAwareAllocator*>(normal_impl_[partition])
                 ->ReleaseAtLeastNPagesBreakingHugepages(pages - ret,
                                                         release_reason);
      if (ret >= pages) {
        return true;
      }
    }

    ret += static_cast<HugePageAwareAllocator*>(sampled_impl_)
               ->ReleaseAtLeastNPagesBreakingHugepages(pages - ret,
                                                       release_reason);
  }
  // Return "true", if we got back under the limit.
  return (pages <= ret);
}

size_t PageAllocator::active_numa_partitions() const {
  return tc_globals.numa_topology().active_partitions();
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
