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

#include <new>

#include "tcmalloc/common.h"
#include "tcmalloc/experiment.h"
#include "tcmalloc/experiment_config.h"
#include "tcmalloc/huge_page_aware_allocator.h"
#include "tcmalloc/internal/environment.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/static_vars.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

int ABSL_ATTRIBUTE_WEAK default_want_hpaa();

bool decide_want_hpaa() {
#if defined(__PPC64__) && defined(TCMALLOC_SMALL_BUT_SLOW)
  // In small-but-slow, we choose a kMinSystemAlloc size that smaller than the
  // hugepage size on PPC.  If this situation changes, this static_assert will
  // begin failing.
  static_assert(kHugePageSize > kMinSystemAlloc,
                "HPAA may now support PPC, update tests");
  return false;
#endif

  const char *e =
      tcmalloc::tcmalloc_internal::thread_safe_getenv("TCMALLOC_HPAA_CONTROL");
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
        return true;
      case '2':
        return true;
      default:
        Crash(kCrash, __FILE__, __LINE__, "bad env var", e);
        return false;
    }
  }

  if (default_want_hpaa != nullptr) {
    int default_hpaa = default_want_hpaa();
    if (default_hpaa != 0) {
      return default_hpaa > 0;
    }
  }

#if defined(TCMALLOC_SMALL_BUT_SLOW)
  // HPAA is neither small nor slow :)
  return false;
#else
  return true;
#endif
}

bool want_hpaa() {
  static bool use = decide_want_hpaa();

  return use;
}

PageAllocator::PageAllocator() {
  const bool kUseHPAA = want_hpaa();
  if (kUseHPAA) {
    normal_impl_[0] =
        new (&choices_[0].hpaa) HugePageAwareAllocator(MemoryTag::kNormal);
    if (Static::numa_topology().numa_aware()) {
      normal_impl_[1] =
          new (&choices_[1].hpaa) HugePageAwareAllocator(MemoryTag::kNormalP1);
    }
    sampled_impl_ = new (&choices_[kNumaPartitions + 0].hpaa)
        HugePageAwareAllocator(MemoryTag::kSampled);
    alg_ = HPAA;
  } else {
    normal_impl_[0] = new (&choices_[0].ph) PageHeap(MemoryTag::kNormal);
    if (Static::numa_topology().numa_aware()) {
      normal_impl_[1] = new (&choices_[1].ph) PageHeap(MemoryTag::kNormalP1);
    }
    sampled_impl_ =
        new (&choices_[kNumaPartitions + 0].ph) PageHeap(MemoryTag::kSampled);
    alg_ = PAGE_HEAP;
  }
}

void PageAllocator::ShrinkToUsageLimit() {
  if (limit_ == std::numeric_limits<size_t>::max()) {
    return;
  }
  BackingStats s = stats();
  size_t backed = s.system_bytes - s.unmapped_bytes + Static::metadata_bytes();
  if (backed <= limit_) {
    // We're already fine.
    return;
  }

  limit_hits_++;
  const size_t overage = backed - limit_;
  const Length pages = LengthFromBytes(overage + kPageSize - 1);
  if (ShrinkHardBy(pages)) {
    return;
  }

  // We're still not below limit.
  if (limit_is_hard_) {
    limit_ = std::numeric_limits<decltype(limit_)>::max();
    Crash(
        kCrash, __FILE__, __LINE__,
        "Hit hard tcmalloc heap limit (e.g. --tcmalloc_heap_size_hard_limit). "
        "Aborting.\nIt was most likely set to catch "
        "allocations that would crash the process anyway. "
    );
  }

  // Print logs once.
  static bool warned = false;
  if (warned) return;
  warned = true;
  Log(kLogWithStack, __FILE__, __LINE__, "Couldn't respect usage limit of ",
      limit_, "and OOM is likely to follow.");
}

bool PageAllocator::ShrinkHardBy(Length pages) {
  Length ret = ReleaseAtLeastNPages(pages);
  if (alg_ == HPAA) {
    if (pages <= ret) {
      // We released target amount.
      return true;
    }

    // At this point, we have no choice but to break up hugepages.
    // However, if the client has turned off subrelease, and is using hard
    // limits, then respect desire to do no subrelease ever.
    if (limit_is_hard_ && !Parameters::hpaa_subrelease()) return false;

    static bool warned_hugepages = false;
    if (!warned_hugepages) {
      Log(kLogWithStack, __FILE__, __LINE__, "Couldn't respect usage limit of ",
          limit_, "without breaking hugepages - performance will drop");
      warned_hugepages = true;
    }
    for (int partition = 0; partition < active_numa_partitions(); partition++) {
      ret += static_cast<HugePageAwareAllocator *>(normal_impl_[partition])
                 ->ReleaseAtLeastNPagesBreakingHugepages(pages - ret);
      if (ret >= pages) {
        return true;
      }
    }

    ret += static_cast<HugePageAwareAllocator *>(sampled_impl_)
               ->ReleaseAtLeastNPagesBreakingHugepages(pages - ret);
  }
  // Return "true", if we got back under the limit.
  return (pages <= ret);
}

size_t PageAllocator::active_numa_partitions() const {
  return Static::numa_topology().active_partitions();
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
