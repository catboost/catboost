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

#ifndef TCMALLOC_STATS_H_
#define TCMALLOC_STATS_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/base/internal/cycleclock.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/pages.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

struct BackingStats {
  BackingStats() : system_bytes(0), free_bytes(0), unmapped_bytes(0) {}
  uint64_t system_bytes;    // Total bytes allocated from system
  uint64_t free_bytes;      // Total bytes on normal freelists
  uint64_t unmapped_bytes;  // Total bytes on returned freelists

  BackingStats& operator+=(BackingStats rhs) {
    system_bytes += rhs.system_bytes;
    free_bytes += rhs.free_bytes;
    unmapped_bytes += rhs.unmapped_bytes;
    return *this;
  }
};

inline BackingStats operator+(BackingStats lhs, BackingStats rhs) {
  return lhs += rhs;
}

struct SmallSpanStats {
  // For each free list of small spans, the length (in spans) of the
  // normal and returned free lists for that size.
  int64_t normal_length[kMaxPages.raw_num()] = {0};
  int64_t returned_length[kMaxPages.raw_num()] = {0};

  SmallSpanStats& operator+=(SmallSpanStats rhs) {
    for (size_t i = 0; i < kMaxPages.raw_num(); ++i) {
      normal_length[i] += rhs.normal_length[i];
      returned_length[i] += rhs.returned_length[i];
    }
    return *this;
  }
};

inline SmallSpanStats operator+(SmallSpanStats lhs, SmallSpanStats rhs) {
  return lhs += rhs;
}

// Stats for free large spans (i.e., spans with more than kMaxPages pages).
struct LargeSpanStats {
  size_t spans = 0;       // Number of such spans
  Length normal_pages;    // Combined page length of normal large spans
  Length returned_pages;  // Combined page length of unmapped spans

  LargeSpanStats& operator+=(LargeSpanStats rhs) {
    spans += rhs.spans;
    normal_pages += rhs.normal_pages;
    returned_pages += rhs.returned_pages;
    return *this;
  }
};

inline LargeSpanStats operator+(LargeSpanStats lhs, LargeSpanStats rhs) {
  return lhs += rhs;
}

void PrintStats(const char* label, Printer& out, const BackingStats& backing,
                const SmallSpanStats& small, const LargeSpanStats& large,
                bool everything);

void PrintStatsInPbtxt(PbtxtRegion& region, const SmallSpanStats& small,
                       const LargeSpanStats& large);

enum class PageReleaseReason {
  // The application explicitly requested memory be released from
  // MallocExtension::ReleaseMemoryToSystem().
  kReleaseMemoryToSystem,

  // MallocExtension::ProcessBackgroundActions() released memory because
  // Parameters::background_release_rate() is positive.
  kProcessBackgroundActions,

  // PageAllocator::ShrinkHardBy() released memory because we hit the soft
  // malloc limit.
  kSoftLimitExceeded,

  // PageAllocator::ShrinkHardBy() released memory because we hit the hard
  // malloc limit (and were able to release enough memory to not have to crash
  // the process).
  kHardLimitExceeded,
};

// Counts of how many pages have been released, broken down by
// PageReleaseReason.
struct PageReleaseStats {
  Length total;

  Length release_memory_to_system;
  Length process_background_actions;
  Length soft_limit_exceeded;
  Length hard_limit_exceeded;

  constexpr friend PageReleaseStats operator+(const PageReleaseStats& lhs,
                                              const PageReleaseStats& rhs) {
    return {
        .total = lhs.total + rhs.total,

        .release_memory_to_system =
            lhs.release_memory_to_system + rhs.release_memory_to_system,
        .process_background_actions =
            lhs.process_background_actions + rhs.process_background_actions,
        .soft_limit_exceeded =
            lhs.soft_limit_exceeded + rhs.soft_limit_exceeded,
        .hard_limit_exceeded =
            lhs.hard_limit_exceeded + rhs.hard_limit_exceeded,
    };
  }

  constexpr PageReleaseStats& operator+=(const PageReleaseStats& other) {
    *this = *this + other;

    return *this;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const PageReleaseStats& v) {
    absl::Format(&sink,
                 "{total = %v, release_memory_to_system = %v, "
                 "process_background_actions = %v, soft_limit_exceeded = %v, "
                 "hard_limit_exceeded = %v}",
                 v.total, v.release_memory_to_system,
                 v.process_background_actions, v.soft_limit_exceeded,
                 v.hard_limit_exceeded);
  }

  constexpr friend bool operator==(const PageReleaseStats& lhs,
                                   const PageReleaseStats& rhs) {
    return lhs.total == rhs.total &&
           lhs.release_memory_to_system == rhs.release_memory_to_system &&
           lhs.process_background_actions == rhs.process_background_actions &&
           lhs.soft_limit_exceeded == rhs.soft_limit_exceeded &&
           lhs.hard_limit_exceeded == rhs.hard_limit_exceeded;
  }

  constexpr friend bool operator!=(const PageReleaseStats& lhs,
                                   const PageReleaseStats& rhs) {
    return !(lhs == rhs);
  }
};

class PageAllocInfo {
 private:
  struct Counts;

 public:
  PageAllocInfo(const char* label);

  // Subclasses are responsible for calling these methods when
  // the relevant actions occur
  void RecordAlloc(Range r);
  void RecordFree(Range r);
  void RecordRelease(Length n, Length got, PageReleaseReason reason);

  PageReleaseStats GetRecordedReleases() const;

  // And invoking this in their Print() implementation.
  void Print(Printer& out) const;
  void PrintInPbtxt(PbtxtRegion& region, absl::string_view stat_name) const;

  // Total size of allocations < 1 MiB
  Length small() const { return total_small_; }
  // We define the "slack" of an allocation as the difference
  // between its size and the nearest hugepage multiple (i.e. how
  // much would go unused if we allocated it as an aligned hugepage
  // and didn't use the rest.)
  // Return the total slack of all non-small allocations.
  Length slack() const { return total_slack_; }

  const Counts& counts_for(Length n) const;

  // Returns (approximate) CycleClock ticks since class instantiation.
  int64_t TimeTicks() const;

 private:
  Length total_small_;
  Length total_slack_;

  Length largest_seen_;

  PageReleaseStats released_;

  // How many alloc/frees have we seen (of some size range?)
  struct Counts {
    // raw counts
    size_t nalloc{0}, nfree{0};
    // and total sizes (needed if this struct tracks a nontrivial range
    Length alloc_size;
    Length free_size;

    void Alloc(Length n) {
      nalloc++;
      alloc_size += n;
    }
    void Free(Length n) {
      nfree++;
      free_size += n;
    }
  };

  // Indexed by exact length
  Counts small_[kMaxPages.raw_num()];
  // Indexed by power-of-two-buckets
  Counts large_[kAddressBits - kPageShift];
  const char* label_;

  const int64_t baseline_ticks_{absl::base_internal::CycleClock::Now()};
  const double freq_{absl::base_internal::CycleClock::Frequency()};
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_STATS_H_
