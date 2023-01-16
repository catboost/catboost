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

#ifndef TCMALLOC_HUGE_PAGE_FILLER_H_
#define TCMALLOC_HUGE_PAGE_FILLER_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <limits>

#include "absl/algorithm/container.h"
#include "absl/base/internal/cycleclock.h"
#include "absl/time/time.h"
#include "tcmalloc/common.h"
#include "tcmalloc/huge_allocator.h"
#include "tcmalloc/huge_cache.h"
#include "tcmalloc/huge_pages.h"
#include "tcmalloc/internal/linked_list.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/internal/range_tracker.h"
#include "tcmalloc/internal/timeseries_tracker.h"
#include "tcmalloc/span.h"
#include "tcmalloc/stats.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// This and the following classes implement the adaptive hugepage subrelease
// mechanism and realized fragmentation metric described in "Adaptive Hugepage
// Subrelease for Non-moving Memory Allocators in Warehouse-Scale Computers"
// (ISMM 2021).

// Tracks correctness of skipped subrelease decisions over time.
template <size_t kEpochs = 16>
class SkippedSubreleaseCorrectnessTracker {
 public:
  struct SkippedSubreleaseDecision {
    Length pages;  // Number of pages we skipped subreleasing.
    size_t count;  // Number of times we skipped a subrelease.

    SkippedSubreleaseDecision() : pages(0), count(0) {}
    explicit SkippedSubreleaseDecision(Length pages) : pages(pages), count(1) {}
    explicit SkippedSubreleaseDecision(Length pages, size_t count)
        : pages(pages), count(count) {}

    SkippedSubreleaseDecision& operator+=(SkippedSubreleaseDecision rhs) {
      pages += rhs.pages;
      count += rhs.count;
      return *this;
    }

    static SkippedSubreleaseDecision Zero() {
      return SkippedSubreleaseDecision();
    }
  };

  explicit constexpr SkippedSubreleaseCorrectnessTracker(Clock clock,
                                                         absl::Duration w)
      : window_(w),
        epoch_length_(window_ / kEpochs),
        last_confirmed_peak_(0),
        tracker_(clock, w) {}

  // Not copyable or movable
  SkippedSubreleaseCorrectnessTracker(
      const SkippedSubreleaseCorrectnessTracker&) = delete;
  SkippedSubreleaseCorrectnessTracker& operator=(
      const SkippedSubreleaseCorrectnessTracker&) = delete;

  void ReportSkippedSubreleasePages(
      Length skipped_pages, Length peak_pages,
      absl::Duration expected_time_until_next_peak) {
    total_skipped_ += SkippedSubreleaseDecision(skipped_pages);
    pending_skipped_ += SkippedSubreleaseDecision(skipped_pages);

    SkippedSubreleaseUpdate update;
    update.decision = SkippedSubreleaseDecision(skipped_pages);
    update.num_pages_at_decision = peak_pages;
    update.correctness_interval_epochs =
        expected_time_until_next_peak / epoch_length_;
    tracker_.Report(update);
  }

  void ReportUpdatedPeak(Length current_peak) {
    // Record this peak for the current epoch (so we don't double-count correct
    // predictions later) and advance the tracker.
    SkippedSubreleaseUpdate update;
    update.confirmed_peak = current_peak;
    if (tracker_.Report(update)) {
      // Also keep track of the largest peak we have confirmed this epoch.
      last_confirmed_peak_ = Length(0);
    }

    // Recompute currently pending decisions.
    pending_skipped_ = SkippedSubreleaseDecision::Zero();

    Length largest_peak_already_confirmed = last_confirmed_peak_;

    tracker_.IterBackwards(
        [&](size_t offset, int64_t ts, const SkippedSubreleaseEntry& e) {
          // Do not clear any decisions in the current epoch.
          if (offset == 0) {
            return;
          }

          if (e.decisions.count > 0 &&
              e.max_num_pages_at_decision > largest_peak_already_confirmed &&
              offset <= e.correctness_interval_epochs) {
            if (e.max_num_pages_at_decision <= current_peak) {
              // We can confirm a subrelease decision as correct and it had not
              // been confirmed correct by an earlier peak yet.
              correctly_skipped_ += e.decisions;
            } else {
              pending_skipped_ += e.decisions;
            }
          }

          // Did we clear any earlier decisions based on a peak in this epoch?
          // Keep track of the peak, so we do not clear them again.
          largest_peak_already_confirmed =
              std::max(largest_peak_already_confirmed, e.max_confirmed_peak);
        },
        -1);

    last_confirmed_peak_ = std::max(last_confirmed_peak_, current_peak);
  }

  inline SkippedSubreleaseDecision total_skipped() const {
    return total_skipped_;
  }

  inline SkippedSubreleaseDecision correctly_skipped() const {
    return correctly_skipped_;
  }

  inline SkippedSubreleaseDecision pending_skipped() const {
    return pending_skipped_;
  }

 private:
  struct SkippedSubreleaseUpdate {
    // A subrelease decision that was made at this time step: How much did we
    // decide not to release?
    SkippedSubreleaseDecision decision;

    // What does our future demand have to be for this to be correct? If there
    // were multiple subrelease decisions in the same epoch, use the max.
    Length num_pages_at_decision;

    // How long from the time of the decision do we have before the decision
    // will be determined incorrect?
    int64_t correctness_interval_epochs = 0;

    // At this time step, we confirmed a demand peak at this level, which means
    // all subrelease decisions in earlier time steps that had peak_demand_pages
    // <= this confirmed_peak were confirmed correct and don't need to be
    // considered again in the future.
    Length confirmed_peak;
  };

  struct SkippedSubreleaseEntry {
    SkippedSubreleaseDecision decisions = SkippedSubreleaseDecision::Zero();
    Length max_num_pages_at_decision;
    int64_t correctness_interval_epochs = 0;
    Length max_confirmed_peak;

    static SkippedSubreleaseEntry Nil() { return SkippedSubreleaseEntry(); }

    void Report(SkippedSubreleaseUpdate e) {
      decisions += e.decision;
      correctness_interval_epochs =
          std::max(correctness_interval_epochs, e.correctness_interval_epochs);
      max_num_pages_at_decision =
          std::max(max_num_pages_at_decision, e.num_pages_at_decision);
      max_confirmed_peak = std::max(max_confirmed_peak, e.confirmed_peak);
    }
  };

  const absl::Duration window_;
  const absl::Duration epoch_length_;

  // The largest peak we processed this epoch. This is required to avoid us
  // double-counting correctly predicted decisions.
  Length last_confirmed_peak_;

  SkippedSubreleaseDecision total_skipped_;
  SkippedSubreleaseDecision correctly_skipped_;
  SkippedSubreleaseDecision pending_skipped_;

  TimeSeriesTracker<SkippedSubreleaseEntry, SkippedSubreleaseUpdate, kEpochs>
      tracker_;
};

struct SubreleaseStats {
  Length total_pages_subreleased;  // cumulative since startup
  Length num_pages_subreleased;
  HugeLength total_hugepages_broken{NHugePages(0)};  // cumulative since startup
  HugeLength num_hugepages_broken{NHugePages(0)};

  bool is_limit_hit = false;
  // Keep these limit-related stats cumulative since startup only
  Length total_pages_subreleased_due_to_limit;
  HugeLength total_hugepages_broken_due_to_limit{NHugePages(0)};

  void reset() {
    total_pages_subreleased += num_pages_subreleased;
    total_hugepages_broken += num_hugepages_broken;
    num_pages_subreleased = Length(0);
    num_hugepages_broken = NHugePages(0);
  }

  // Must be called at the beginning of each subrelease request
  void set_limit_hit(bool value) { is_limit_hit = value; }

  // This only has a well-defined meaning within ReleaseCandidates where
  // set_limit_hit() has been called earlier. Do not use anywhere else.
  bool limit_hit() { return is_limit_hit; }
};

// Track filler statistics over a time window.
template <size_t kEpochs = 16>
class FillerStatsTracker {
 public:
  enum Type { kRegular, kDonated, kPartialReleased, kReleased, kNumTypes };

  struct FillerStats {
    Length num_pages;
    Length free_pages;
    Length unmapped_pages;
    Length used_pages_in_subreleased_huge_pages;
    HugeLength huge_pages[kNumTypes];
    Length num_pages_subreleased;
    HugeLength num_hugepages_broken = NHugePages(0);

    HugeLength total_huge_pages() const {
      HugeLength total_huge_pages;
      for (int i = 0; i < kNumTypes; i++) {
        total_huge_pages += huge_pages[i];
      }
      return total_huge_pages;
    }
  };

  struct NumberOfFreePages {
    Length free;
    Length free_backed;
  };

  explicit constexpr FillerStatsTracker(Clock clock, absl::Duration w,
                                        absl::Duration summary_interval)
      : summary_interval_(summary_interval),
        window_(w),
        epoch_length_(window_ / kEpochs),
        tracker_(clock, w),
        skipped_subrelease_correctness_(clock, w) {}

  // Not copyable or movable
  FillerStatsTracker(const FillerStatsTracker&) = delete;
  FillerStatsTracker& operator=(const FillerStatsTracker&) = delete;

  void Report(const FillerStats stats) {
    if (ABSL_PREDICT_FALSE(tracker_.Report(stats))) {
      if (ABSL_PREDICT_FALSE(pending_skipped().count > 0)) {
        // Consider the peak within the just completed epoch to confirm the
        // correctness of any recent subrelease decisions.
        skipped_subrelease_correctness_.ReportUpdatedPeak(std::max(
            stats.num_pages,
            tracker_.GetEpochAtOffset(1).stats[kStatsAtMaxDemand].num_pages));
      }
    }
  }

  void Print(Printer* out) const;
  void PrintInPbtxt(PbtxtRegion* hpaa) const;

  // Calculates recent peaks for skipping subrelease decisions. If our allocated
  // memory is below the demand peak within the last peak_interval, we stop
  // subreleasing. If our demand is going above that peak again within another
  // peak_interval, we report that we made the correct decision.
  FillerStats GetRecentPeak(absl::Duration peak_interval) {
    last_peak_interval_ = peak_interval;
    FillerStats recent_peak;
    Length max_demand_pages;

    int64_t num_epochs = peak_interval / epoch_length_;
    tracker_.IterBackwards(
        [&](size_t offset, int64_t ts, const FillerStatsEntry& e) {
          if (!e.empty()) {
            // Identify the maximum number of demand pages we have seen within
            // the time interval.
            if (e.stats[kStatsAtMaxDemand].num_pages > max_demand_pages) {
              recent_peak = e.stats[kStatsAtMaxDemand];
              max_demand_pages = recent_peak.num_pages;
            }
          }
        },
        num_epochs);

    return recent_peak;
  }

  void ReportSkippedSubreleasePages(
      Length pages, Length peak_pages,
      absl::Duration expected_time_until_next_peak) {
    if (pages == Length(0)) {
      return;
    }

    skipped_subrelease_correctness_.ReportSkippedSubreleasePages(
        pages, peak_pages, expected_time_until_next_peak);
  }

  inline typename SkippedSubreleaseCorrectnessTracker<
      kEpochs>::SkippedSubreleaseDecision
  total_skipped() const {
    return skipped_subrelease_correctness_.total_skipped();
  }

  inline typename SkippedSubreleaseCorrectnessTracker<
      kEpochs>::SkippedSubreleaseDecision
  correctly_skipped() const {
    return skipped_subrelease_correctness_.correctly_skipped();
  }

  inline typename SkippedSubreleaseCorrectnessTracker<
      kEpochs>::SkippedSubreleaseDecision
  pending_skipped() const {
    return skipped_subrelease_correctness_.pending_skipped();
  }

  // Returns the minimum number of free pages throughout the tracker period.
  // The first value of the pair is the number of all free pages, the second
  // value contains only the backed ones.
  NumberOfFreePages min_free_pages(absl::Duration w) const {
    NumberOfFreePages mins;
    mins.free = Length::max();
    mins.free_backed = Length::max();

    int64_t num_epochs = std::clamp(w / epoch_length_, int64_t{0},
                                    static_cast<int64_t>(kEpochs));

    tracker_.IterBackwards(
        [&](size_t offset, int64_t ts, const FillerStatsEntry& e) {
          if (!e.empty()) {
            mins.free = std::min(mins.free, e.min_free_pages);
            mins.free_backed =
                std::min(mins.free_backed, e.min_free_backed_pages);
          }
        },
        num_epochs);
    mins.free = (mins.free == Length::max()) ? Length(0) : mins.free;
    mins.free_backed =
        (mins.free_backed == Length::max()) ? Length(0) : mins.free_backed;
    return mins;
  }

 private:
  // We collect filler statistics at four "interesting points" within each time
  // step: at min/max demand of pages and at min/max use of hugepages. This
  // allows us to approximate the envelope of the different metrics.
  enum StatsType {
    kStatsAtMinDemand,
    kStatsAtMaxDemand,
    kStatsAtMinHugePages,
    kStatsAtMaxHugePages,
    kNumStatsTypes
  };

  struct FillerStatsEntry {
    // Collect filler stats at "interesting points" (minimum/maximum page demand
    // and at minimum/maximum usage of huge pages).
    FillerStats stats[kNumStatsTypes] = {};
    static constexpr Length kDefaultValue = Length::max();
    Length min_free_pages = kDefaultValue;
    Length min_free_backed_pages = kDefaultValue;
    Length num_pages_subreleased;
    HugeLength num_hugepages_broken = NHugePages(0);

    static FillerStatsEntry Nil() { return FillerStatsEntry(); }

    void Report(FillerStats e) {
      if (empty()) {
        for (int i = 0; i < kNumStatsTypes; i++) {
          stats[i] = e;
        }
      }

      if (e.num_pages < stats[kStatsAtMinDemand].num_pages) {
        stats[kStatsAtMinDemand] = e;
      }

      if (e.num_pages > stats[kStatsAtMaxDemand].num_pages) {
        stats[kStatsAtMaxDemand] = e;
      }

      if (e.total_huge_pages() <
          stats[kStatsAtMinHugePages].total_huge_pages()) {
        stats[kStatsAtMinHugePages] = e;
      }

      if (e.total_huge_pages() >
          stats[kStatsAtMaxHugePages].total_huge_pages()) {
        stats[kStatsAtMaxHugePages] = e;
      }

      min_free_pages =
          std::min(min_free_pages, e.free_pages + e.unmapped_pages);
      min_free_backed_pages = std::min(min_free_backed_pages, e.free_pages);

      // Subrelease stats
      num_pages_subreleased += e.num_pages_subreleased;
      num_hugepages_broken += e.num_hugepages_broken;
    }

    bool empty() const { return min_free_pages == kDefaultValue; }
  };

  // The tracker reports pages that have been free for at least this interval,
  // as well as peaks within this interval.
  const absl::Duration summary_interval_;

  const absl::Duration window_;
  const absl::Duration epoch_length_;

  TimeSeriesTracker<FillerStatsEntry, FillerStats, kEpochs> tracker_;
  SkippedSubreleaseCorrectnessTracker<kEpochs> skipped_subrelease_correctness_;

  // Records the last peak_interval value, for reporting and debugging only.
  absl::Duration last_peak_interval_;
};

// Evaluate a/b, avoiding division by zero
inline double safe_div(double a, double b) {
  if (b == 0) {
    return 0.;
  } else {
    return a / b;
  }
}

inline double safe_div(Length a, Length b) {
  return safe_div(a.raw_num(), b.raw_num());
}

template <size_t kEpochs>
void FillerStatsTracker<kEpochs>::Print(Printer* out) const {
  NumberOfFreePages free_pages = min_free_pages(summary_interval_);
  out->printf("HugePageFiller: time series over %d min interval\n\n",
              absl::ToInt64Minutes(summary_interval_));

  // Realized fragmentation is equivalent to backed minimum free pages over a
  // 5-min interval. It is printed for convenience but not included in pbtxt.
  out->printf("HugePageFiller: realized fragmentation: %.1f MiB\n",
              free_pages.free_backed.in_mib());
  out->printf("HugePageFiller: minimum free pages: %zu (%zu backed)\n",
              free_pages.free.raw_num(), free_pages.free_backed.raw_num());

  FillerStatsEntry at_peak_demand;
  FillerStatsEntry at_peak_hps;

  tracker_.IterBackwards(
      [&](size_t offset, int64_t ts, const FillerStatsEntry& e) {
        if (!e.empty()) {
          if (at_peak_demand.empty() ||
              at_peak_demand.stats[kStatsAtMaxDemand].num_pages <
                  e.stats[kStatsAtMaxDemand].num_pages) {
            at_peak_demand = e;
          }

          if (at_peak_hps.empty() ||
              at_peak_hps.stats[kStatsAtMaxHugePages].total_huge_pages() <
                  e.stats[kStatsAtMaxHugePages].total_huge_pages()) {
            at_peak_hps = e;
          }
        }
      },
      summary_interval_ / epoch_length_);

  out->printf(
      "HugePageFiller: at peak demand: %zu pages (and %zu free, %zu unmapped)\n"
      "HugePageFiller: at peak demand: %zu hps (%zu regular, %zu donated, "
      "%zu partial, %zu released)\n",
      at_peak_demand.stats[kStatsAtMaxDemand].num_pages.raw_num(),
      at_peak_demand.stats[kStatsAtMaxDemand].free_pages.raw_num(),
      at_peak_demand.stats[kStatsAtMaxDemand].unmapped_pages.raw_num(),
      at_peak_demand.stats[kStatsAtMaxDemand].total_huge_pages().raw_num(),
      at_peak_demand.stats[kStatsAtMaxDemand].huge_pages[kRegular].raw_num(),
      at_peak_demand.stats[kStatsAtMaxDemand].huge_pages[kDonated].raw_num(),
      at_peak_demand.stats[kStatsAtMaxDemand]
          .huge_pages[kPartialReleased]
          .raw_num(),
      at_peak_demand.stats[kStatsAtMaxDemand].huge_pages[kReleased].raw_num());

  out->printf(
      "HugePageFiller: at peak hps: %zu pages (and %zu free, %zu unmapped)\n"
      "HugePageFiller: at peak hps: %zu hps (%zu regular, %zu donated, "
      "%zu partial, %zu released)\n",
      at_peak_hps.stats[kStatsAtMaxDemand].num_pages.raw_num(),
      at_peak_hps.stats[kStatsAtMaxDemand].free_pages.raw_num(),
      at_peak_hps.stats[kStatsAtMaxDemand].unmapped_pages.raw_num(),
      at_peak_hps.stats[kStatsAtMaxDemand].total_huge_pages().raw_num(),
      at_peak_hps.stats[kStatsAtMaxDemand].huge_pages[kRegular].raw_num(),
      at_peak_hps.stats[kStatsAtMaxDemand].huge_pages[kDonated].raw_num(),
      at_peak_hps.stats[kStatsAtMaxDemand]
          .huge_pages[kPartialReleased]
          .raw_num(),
      at_peak_hps.stats[kStatsAtMaxDemand].huge_pages[kReleased].raw_num());

  out->printf(
      "\nHugePageFiller: Since the start of the execution, %zu subreleases (%zu"
      " pages) were skipped due to recent (%llds) peaks.\n",
      total_skipped().count, total_skipped().pages.raw_num(),
      static_cast<long long>(absl::ToInt64Seconds(last_peak_interval_)));

  Length skipped_pages = total_skipped().pages - pending_skipped().pages;
  double correctly_skipped_pages_percentage =
      safe_div(100.0 * correctly_skipped().pages, skipped_pages);

  size_t skipped_count = total_skipped().count - pending_skipped().count;
  double correctly_skipped_count_percentage =
      safe_div(100.0 * correctly_skipped().count, skipped_count);

  out->printf(
      "HugePageFiller: %.4f%% of decisions confirmed correct, %zu "
      "pending (%.4f%% of pages, %zu pending).\n",
      correctly_skipped_count_percentage, pending_skipped().count,
      correctly_skipped_pages_percentage, pending_skipped().pages.raw_num());

  // Print subrelease stats
  Length total_subreleased;
  HugeLength total_broken = NHugePages(0);
  tracker_.Iter(
      [&](size_t offset, int64_t ts, const FillerStatsEntry& e) {
        total_subreleased += e.num_pages_subreleased;
        total_broken += e.num_hugepages_broken;
      },
      tracker_.kSkipEmptyEntries);
  out->printf(
      "HugePageFiller: Subrelease stats last %d min: total "
      "%zu pages subreleased, %zu hugepages broken\n",
      static_cast<int64_t>(absl::ToInt64Minutes(window_)),
      total_subreleased.raw_num(), total_broken.raw_num());
}

template <size_t kEpochs>
void FillerStatsTracker<kEpochs>::PrintInPbtxt(PbtxtRegion* hpaa) const {
  {
    auto skip_subrelease = hpaa->CreateSubRegion("filler_skipped_subrelease");
    skip_subrelease.PrintI64("skipped_subrelease_interval_ms",
                             absl::ToInt64Milliseconds(last_peak_interval_));
    skip_subrelease.PrintI64("skipped_subrelease_pages",
                             total_skipped().pages.raw_num());
    skip_subrelease.PrintI64("correctly_skipped_subrelease_pages",
                             correctly_skipped().pages.raw_num());
    skip_subrelease.PrintI64("pending_skipped_subrelease_pages",
                             pending_skipped().pages.raw_num());
    skip_subrelease.PrintI64("skipped_subrelease_count", total_skipped().count);
    skip_subrelease.PrintI64("correctly_skipped_subrelease_count",
                             correctly_skipped().count);
    skip_subrelease.PrintI64("pending_skipped_subrelease_count",
                             pending_skipped().count);
  }

  auto filler_stats = hpaa->CreateSubRegion("filler_stats_timeseries");
  filler_stats.PrintI64("window_ms", absl::ToInt64Milliseconds(epoch_length_));
  filler_stats.PrintI64("epochs", kEpochs);

  NumberOfFreePages free_pages = min_free_pages(summary_interval_);
  filler_stats.PrintI64("min_free_pages_interval_ms",
                        absl::ToInt64Milliseconds(summary_interval_));
  filler_stats.PrintI64("min_free_pages", free_pages.free.raw_num());
  filler_stats.PrintI64("min_free_backed_pages",
                        free_pages.free_backed.raw_num());

  static const char* labels[kNumStatsTypes] = {
      "at_minimum_demand", "at_maximum_demand", "at_minimum_huge_pages",
      "at_maximum_huge_pages"};

  tracker_.Iter(
      [&](size_t offset, int64_t ts, const FillerStatsEntry& e) {
        auto region = filler_stats.CreateSubRegion("measurements");
        region.PrintI64("epoch", offset);
        region.PrintI64("timestamp_ms",
                        absl::ToInt64Milliseconds(absl::Nanoseconds(ts)));
        region.PrintI64("min_free_pages", e.min_free_pages.raw_num());
        region.PrintI64("min_free_backed_pages",
                        e.min_free_backed_pages.raw_num());
        region.PrintI64("num_pages_subreleased",
                        e.num_pages_subreleased.raw_num());
        region.PrintI64("num_hugepages_broken",
                        e.num_hugepages_broken.raw_num());
        for (int i = 0; i < kNumStatsTypes; i++) {
          auto m = region.CreateSubRegion(labels[i]);
          FillerStats stats = e.stats[i];
          m.PrintI64("num_pages", stats.num_pages.raw_num());
          m.PrintI64("regular_huge_pages",
                     stats.huge_pages[kRegular].raw_num());
          m.PrintI64("donated_huge_pages",
                     stats.huge_pages[kDonated].raw_num());
          m.PrintI64("partial_released_huge_pages",
                     stats.huge_pages[kPartialReleased].raw_num());
          m.PrintI64("released_huge_pages",
                     stats.huge_pages[kReleased].raw_num());
          m.PrintI64("used_pages_in_subreleased_huge_pages",
                     stats.used_pages_in_subreleased_huge_pages.raw_num());
        }
      },
      tracker_.kSkipEmptyEntries);
}

// PageTracker keeps track of the allocation status of every page in a HugePage.
// It allows allocation and deallocation of a contiguous run of pages.
//
// Its mutating methods are annotated as requiring the pageheap_lock, in order
// to support unlocking the page heap lock in a dynamic annotation-friendly way.
template <MemoryModifyFunction Unback>
class PageTracker : public TList<PageTracker<Unback>>::Elem {
 public:
  static void UnbackImpl(void* p, size_t size) { Unback(p, size); }

  constexpr PageTracker(HugePage p, uint64_t when)
      : location_(p),
        released_count_(0),
        donated_(false),
        unbroken_(true),
        free_{} {
    init_when(when);

#ifndef __ppc64__
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
    // Verify fields are structured so commonly accessed members (as part of
    // Put) are on the first two cache lines.  This allows the CentralFreeList
    // to accelerate deallocations by prefetching PageTracker instances before
    // taking the pageheap_lock.
    //
    // On PPC64, kHugePageSize / kPageSize is typically ~2K (16MB / 8KB),
    // requiring 512 bytes for representing free_.  While its cache line size is
    // larger, the entirety of free_ will not fit on two cache lines.
    static_assert(
        offsetof(PageTracker<Unback>, location_) + sizeof(location_) <=
            2 * ABSL_CACHELINE_SIZE,
        "location_ should fall within the first two cachelines of "
        "PageTracker.");
    static_assert(offsetof(PageTracker<Unback>, when_numerator_) +
                          sizeof(when_numerator_) <=
                      2 * ABSL_CACHELINE_SIZE,
                  "when_numerator_ should fall within the first two cachelines "
                  "of PageTracker.");
    static_assert(offsetof(PageTracker<Unback>, when_denominator_) +
                          sizeof(when_denominator_) <=
                      2 * ABSL_CACHELINE_SIZE,
                  "when_denominator_ should fall within the first two "
                  "cachelines of PageTracker.");
    static_assert(
        offsetof(PageTracker<Unback>, donated_) + sizeof(donated_) <=
            2 * ABSL_CACHELINE_SIZE,
        "donated_ should fall within the first two cachelines of PageTracker.");
    static_assert(
        offsetof(PageTracker<Unback>, free_) + sizeof(free_) <=
            2 * ABSL_CACHELINE_SIZE,
        "free_ should fall within the first two cachelines of PageTracker.");
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif  // __ppc64__
  }

  struct PageAllocation {
    PageId page;
    Length previously_unbacked;
  };

  // REQUIRES: there's a free range of at least n pages
  //
  // Returns a PageId i and a count of previously unbacked pages in the range
  // [i, i+n) in previously_unbacked.
  PageAllocation Get(Length n) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // REQUIRES: p was the result of a previous call to Get(n)
  void Put(PageId p, Length n) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Returns true if any unused pages have been returned-to-system.
  bool released() const { return released_count_ > 0; }

  // Was this tracker donated from the tail of a multi-hugepage allocation?
  // Only up-to-date when the tracker is on a TrackerList in the Filler;
  // otherwise the value is meaningless.
  bool donated() const { return donated_; }
  // Set/reset the donated flag. The donated status is lost, for instance,
  // when further allocations are made on the tracker.
  void set_donated(bool status) { donated_ = status; }

  // These statistics help us measure the fragmentation of a hugepage and
  // the desirability of allocating from this hugepage.
  Length longest_free_range() const { return Length(free_.longest_free()); }
  size_t nallocs() const { return free_.allocs(); }
  Length used_pages() const { return Length(free_.used()); }
  Length released_pages() const { return Length(released_count_); }
  Length free_pages() const;
  bool empty() const;

  bool unbroken() const { return unbroken_; }

  // Returns the hugepage whose availability is being tracked.
  HugePage location() const { return location_; }

  // Return all unused pages to the system, mark future frees to do same.
  // Returns the count of pages unbacked.
  Length ReleaseFree() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Return this allocation to the system, if policy warrants it.
  //
  // As of 3/2020 our policy is to rerelease:  Once we break a hugepage by
  // returning a fraction of it, we return *anything* unused.  This simplifies
  // tracking.
  //
  // TODO(b/141550014):  Make retaining the default/sole policy.
  void MaybeRelease(PageId p, Length n)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    if (released_count_ == 0) {
      return;
    }

    // Mark pages as released.
    Length index = p - location_.first_page();
    ASSERT(released_by_page_.CountBits(index.raw_num(), n.raw_num()) == 0);
    released_by_page_.SetRange(index.raw_num(), n.raw_num());
    released_count_ += n.raw_num();
    ASSERT(released_by_page_.CountBits(0, kPagesPerHugePage.raw_num()) ==
           released_count_);

    // TODO(b/122551676):  If release fails, we should not SetRange above.
    ReleasePagesWithoutLock(p, n);
  }

  void AddSpanStats(SmallSpanStats* small, LargeSpanStats* large,
                    PageAgeHistograms* ages) const;

 private:
  void init_when(uint64_t w) {
    const Length before = Length(free_.total_free());
    when_numerator_ = w * before.raw_num();
    when_denominator_ = before.raw_num();
  }

  HugePage location_;
  // We keep track of an average time weighted by Length::raw_num. In order to
  // avoid doing division on fast path, store the numerator and denominator and
  // only do the division when we need the average.
  uint64_t when_numerator_;
  uint64_t when_denominator_;

  // Cached value of released_by_page_.CountBits(0, kPagesPerHugePages)
  //
  // TODO(b/151663108):  Logically, this is guarded by pageheap_lock.
  uint16_t released_count_;
  bool donated_;
  bool unbroken_;

  RangeTracker<kPagesPerHugePage.raw_num()> free_;
  // Bitmap of pages based on them being released to the OS.
  // * Not yet released pages are unset (considered "free")
  // * Released pages are set.
  //
  // Before releasing any locks to release memory to the OS, we mark the bitmap.
  //
  // Once released, a huge page is considered released *until* free_ is
  // exhausted and no pages released_by_page_ are set.  We may have up to
  // kPagesPerHugePage-1 parallel subreleases in-flight.
  //
  // TODO(b/151663108):  Logically, this is guarded by pageheap_lock.
  Bitmap<kPagesPerHugePage.raw_num()> released_by_page_;

  static_assert(kPagesPerHugePage.raw_num() <
                    std::numeric_limits<uint16_t>::max(),
                "nallocs must be able to support kPagesPerHugePage!");

  void ReleasePages(PageId p, Length n) {
    void* ptr = p.start_addr();
    size_t byte_len = n.in_bytes();
    Unback(ptr, byte_len);
    unbroken_ = false;
  }

  void ReleasePagesWithoutLock(PageId p, Length n)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    pageheap_lock.Unlock();

    void* ptr = p.start_addr();
    size_t byte_len = n.in_bytes();
    Unback(ptr, byte_len);

    pageheap_lock.Lock();
    unbroken_ = false;
  }
};

enum class FillerPartialRerelease : bool {
  // Once we break a hugepage by returning a fraction of it, we return
  // *anything* unused.  This simplifies tracking.
  //
  // As of 2/2020, this is the default behavior.
  Return,
  // When releasing a page onto an already-released huge page, retain the page
  // rather than releasing it back to the OS.  This can reduce minor page
  // faults for hot pages.
  //
  // TODO(b/141550014, b/122551676):  Make this the default behavior.
  Retain,
};

// This tracks a set of unfilled hugepages, and fulfills allocations
// with a goal of filling some hugepages as tightly as possible and emptying
// out the remainder.
template <class TrackerType>
class HugePageFiller {
 public:
  explicit HugePageFiller(FillerPartialRerelease partial_rerelease);
  HugePageFiller(FillerPartialRerelease partial_rerelease, Clock clock);

  typedef TrackerType Tracker;

  struct TryGetResult {
    TrackerType* pt;
    PageId page;
  };

  // Our API is simple, but note that it does not include an unconditional
  // allocation, only a "try"; we expect callers to allocate new hugepages if
  // needed.  This simplifies using it in a few different contexts (and improves
  // the testing story - no dependencies.)
  //
  // On failure, returns nullptr/PageId{0}.
  TryGetResult TryGet(Length n) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Marks [p, p + n) as usable by new allocations into *pt; returns pt
  // if that hugepage is now empty (nullptr otherwise.)
  // REQUIRES: pt is owned by this object (has been Contribute()), and
  // {pt, p, n} was the result of a previous TryGet.
  TrackerType* Put(TrackerType* pt, PageId p, Length n)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Contributes a tracker to the filler. If "donated," then the tracker is
  // marked as having come from the tail of a multi-hugepage allocation, which
  // causes it to be treated slightly differently.
  void Contribute(TrackerType* pt, bool donated);

  HugeLength size() const { return size_; }

  // Useful statistics
  Length pages_allocated() const { return allocated_; }
  Length used_pages() const { return allocated_; }
  Length unmapped_pages() const { return unmapped_; }
  Length free_pages() const;
  Length used_pages_in_released() const { return n_used_released_; }
  Length used_pages_in_partial_released() const {
    return n_used_partial_released_;
  }
  Length used_pages_in_any_subreleased() const {
    return n_used_released_ + n_used_partial_released_;
  }

  // Fraction of used pages that are on non-released hugepages and
  // thus could be backed by kernel hugepages. (Of course, we can't
  // guarantee that the kernel had available 2-mib regions of physical
  // memory--so this being 1 doesn't mean that everything actually
  // *is* hugepage-backed!)
  double hugepage_frac() const;

  // Returns the amount of memory to release if all remaining options of
  // releasing memory involve subreleasing pages.
  Length GetDesiredSubreleasePages(Length desired, Length total_released,
                                   absl::Duration peak_interval)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Tries to release desired pages by iteratively releasing from the emptiest
  // possible hugepage and releasing its free memory to the system.  Return the
  // number of pages actually released.
  Length ReleasePages(Length desired,
                      absl::Duration skip_subrelease_after_peaks_interval,
                      bool hit_limit)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  void AddSpanStats(SmallSpanStats* small, LargeSpanStats* large,
                    PageAgeHistograms* ages) const;

  BackingStats stats() const;
  SubreleaseStats subrelease_stats() const { return subrelease_stats_; }
  void Print(Printer* out, bool everything) const;
  void PrintInPbtxt(PbtxtRegion* hpaa) const;

 private:
  typedef TList<TrackerType> TrackerList;

  // This class wraps an array of N TrackerLists and a Bitmap storing which
  // elements are non-empty.
  template <size_t N>
  class HintedTrackerLists {
   public:
    HintedTrackerLists() : nonempty_{}, size_(NHugePages(0)) {}

    // Removes a TrackerType from the first non-empty freelist with index at
    // least n and returns it. Returns nullptr if there is none.
    TrackerType* GetLeast(const size_t n) {
      ASSERT(n < N);
      size_t i = nonempty_.FindSet(n);
      if (i == N) {
        return nullptr;
      }
      ASSERT(!lists_[i].empty());
      TrackerType* pt = lists_[i].first();
      if (lists_[i].remove(pt)) {
        nonempty_.ClearBit(i);
      }
      --size_;
      return pt;
    }
    void Add(TrackerType* pt, const size_t i) {
      ASSERT(i < N);
      ASSERT(pt != nullptr);
      lists_[i].prepend(pt);
      nonempty_.SetBit(i);
      ++size_;
    }
    void Remove(TrackerType* pt, const size_t i) {
      ASSERT(i < N);
      ASSERT(pt != nullptr);
      if (lists_[i].remove(pt)) {
        nonempty_.ClearBit(i);
      }
      --size_;
    }
    const TrackerList& operator[](const size_t n) const {
      ASSERT(n < N);
      return lists_[n];
    }
    HugeLength size() const { return size_; }
    bool empty() const { return size().raw_num() == 0; }
    // Runs a functor on all HugePages in the TrackerLists.
    // This method is const but the Functor gets passed a non-const pointer.
    // This quirk is inherited from TrackerList.
    template <typename Functor>
    void Iter(const Functor& func, size_t start) const {
      size_t i = nonempty_.FindSet(start);
      while (i < N) {
        auto& list = lists_[i];
        ASSERT(!list.empty());
        for (TrackerType* pt : list) {
          func(pt);
        }
        i++;
        if (i < N) i = nonempty_.FindSet(i);
      }
    }

   private:
    TrackerList lists_[N];
    Bitmap<N> nonempty_;
    HugeLength size_;
  };

  SubreleaseStats subrelease_stats_;

  // We group hugepages first by longest-free (as a measure of fragmentation),
  // then into 8 chunks inside there by desirability of allocation.
  static constexpr size_t kChunks = 8;
  // Which chunk should this hugepage be in?
  // This returns the largest possible value kChunks-1 iff pt has a single
  // allocation.
  size_t IndexFor(TrackerType* pt);
  // Returns index for regular_alloc_.
  static size_t ListFor(Length longest, size_t chunk);
  static constexpr size_t kNumLists = kPagesPerHugePage.raw_num() * kChunks;

  HintedTrackerLists<kNumLists> regular_alloc_;
  HintedTrackerLists<kPagesPerHugePage.raw_num()> donated_alloc_;
  // Partially released ones that we are trying to release.
  //
  // When FillerPartialRerelease == Return:
  //   regular_alloc_partial_released_ is empty and n_used_partial_released_ is
  //   0.
  //
  // When FillerPartialRerelease == Retain:
  //   regular_alloc_partial_released_ contains huge pages that are partially
  //   allocated, partially free, and partially returned to the OS.
  //   n_used_partial_released_ is the number of pages which have been allocated
  //   of the set.
  //
  // regular_alloc_released_:  This list contains huge pages whose pages are
  // either allocated or returned to the OS.  There are no pages that are free,
  // but not returned to the OS.  n_used_released_ contains the number of
  // pages in those huge pages that are not free (i.e., allocated).
  Length n_used_partial_released_;
  Length n_used_released_;
  HintedTrackerLists<kNumLists> regular_alloc_partial_released_;
  HintedTrackerLists<kNumLists> regular_alloc_released_;

  // RemoveFromFillerList pt from the appropriate HintedTrackerList.
  void RemoveFromFillerList(TrackerType* pt);
  // Put pt in the appropriate HintedTrackerList.
  void AddToFillerList(TrackerType* pt);
  // Like AddToFillerList(), but for use when donating from the tail of a
  // multi-hugepage allocation.
  void DonateToFillerList(TrackerType* pt);

  // CompareForSubrelease identifies the worse candidate for subrelease, between
  // the choice of huge pages a and b.
  static bool CompareForSubrelease(TrackerType* a, TrackerType* b) {
    ASSERT(a != nullptr);
    ASSERT(b != nullptr);

    return a->used_pages() < b->used_pages();
  }

  // SelectCandidates identifies the candidates.size() best candidates in the
  // given tracker list.
  //
  // To support gathering candidates from multiple tracker lists,
  // current_candidates is nonzero.
  template <size_t N>
  static int SelectCandidates(absl::Span<TrackerType*> candidates,
                              int current_candidates,
                              const HintedTrackerLists<N>& tracker_list,
                              size_t tracker_start);

  // Release desired pages from the page trackers in candidates.  Returns the
  // number of pages released.
  Length ReleaseCandidates(absl::Span<TrackerType*> candidates, Length desired)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  HugeLength size_;

  Length allocated_;
  Length unmapped_;

  // How much have we eagerly unmapped (in already released hugepages), but
  // not reported to ReleasePages calls?
  Length unmapping_unaccounted_;

  FillerPartialRerelease partial_rerelease_;

  // Functionality related to time series tracking.
  void UpdateFillerStatsTracker();
  using StatsTrackerType = FillerStatsTracker<600>;
  StatsTrackerType fillerstats_tracker_;
};

template <MemoryModifyFunction Unback>
inline typename PageTracker<Unback>::PageAllocation PageTracker<Unback>::Get(
    Length n) {
  size_t index = free_.FindAndMark(n.raw_num());

  ASSERT(released_by_page_.CountBits(0, kPagesPerHugePage.raw_num()) ==
         released_count_);

  size_t unbacked = 0;
  // If release_count_ == 0, CountBits will return 0 and ClearRange will be a
  // no-op (but will touch cachelines) due to the invariants guaranteed by
  // CountBits() == released_count_.
  //
  // This is a performance optimization, not a logical requirement.
  if (ABSL_PREDICT_FALSE(released_count_ > 0)) {
    unbacked = released_by_page_.CountBits(index, n.raw_num());
    released_by_page_.ClearRange(index, n.raw_num());
    ASSERT(released_count_ >= unbacked);
    released_count_ -= unbacked;
  }

  ASSERT(released_by_page_.CountBits(0, kPagesPerHugePage.raw_num()) ==
         released_count_);
  return PageAllocation{location_.first_page() + Length(index),
                        Length(unbacked)};
}

template <MemoryModifyFunction Unback>
inline void PageTracker<Unback>::Put(PageId p, Length n) {
  Length index = p - location_.first_page();
  free_.Unmark(index.raw_num(), n.raw_num());

  when_numerator_ += n.raw_num() * absl::base_internal::CycleClock::Now();
  when_denominator_ += n.raw_num();
}

template <MemoryModifyFunction Unback>
inline Length PageTracker<Unback>::ReleaseFree() {
  size_t count = 0;
  size_t index = 0;
  size_t n;
  // For purposes of tracking, pages which are not yet released are "free" in
  // the released_by_page_ bitmap.  We subrelease these pages in an iterative
  // process:
  //
  // 1.  Identify the next range of still backed pages.
  // 2.  Iterate on the free_ tracker within this range.  For any free range
  //     found, mark these as unbacked.
  // 3.  Release the subrange to the OS.
  while (released_by_page_.NextFreeRange(index, &index, &n)) {
    size_t free_index;
    size_t free_n;

    // Check for freed pages in this unreleased region.
    if (free_.NextFreeRange(index, &free_index, &free_n) &&
        free_index < index + n) {
      // If there is a free range which overlaps with [index, index+n), release
      // it.
      size_t end = std::min(free_index + free_n, index + n);

      // In debug builds, verify [free_index, end) is backed.
      size_t length = end - free_index;
      ASSERT(released_by_page_.CountBits(free_index, length) == 0);
      // Mark pages as released.  Amortize the update to release_count_.
      released_by_page_.SetRange(free_index, length);

      PageId p = location_.first_page() + Length(free_index);
      // TODO(b/122551676):  If release fails, we should not SetRange above.
      ReleasePages(p, Length(length));

      index = end;
      count += length;
    } else {
      // [index, index+n) did not have an overlapping range in free_, move to
      // the next backed range of pages.
      index += n;
    }
  }

  released_count_ += count;
  ASSERT(Length(released_count_) <= kPagesPerHugePage);
  ASSERT(released_by_page_.CountBits(0, kPagesPerHugePage.raw_num()) ==
         released_count_);
  init_when(absl::base_internal::CycleClock::Now());
  return Length(count);
}

template <MemoryModifyFunction Unback>
inline void PageTracker<Unback>::AddSpanStats(SmallSpanStats* small,
                                              LargeSpanStats* large,
                                              PageAgeHistograms* ages) const {
  size_t index = 0, n;

  uint64_t w = when_denominator_ == 0 ? when_numerator_
                                      : when_numerator_ / when_denominator_;
  while (free_.NextFreeRange(index, &index, &n)) {
    bool is_released = released_by_page_.GetBit(index);
    // Find the last bit in the run with the same state (set or cleared) as
    // index.
    size_t end;
    if (index >= kPagesPerHugePage.raw_num() - 1) {
      end = kPagesPerHugePage.raw_num();
    } else {
      end = is_released ? released_by_page_.FindClear(index + 1)
                        : released_by_page_.FindSet(index + 1);
    }
    n = std::min(end - index, n);
    ASSERT(n > 0);

    if (n < kMaxPages.raw_num()) {
      if (small != nullptr) {
        if (is_released) {
          small->returned_length[n]++;
        } else {
          small->normal_length[n]++;
        }
      }
    } else {
      if (large != nullptr) {
        large->spans++;
        if (is_released) {
          large->returned_pages += Length(n);
        } else {
          large->normal_pages += Length(n);
        }
      }
    }

    if (ages) {
      ages->RecordRange(Length(n), is_released, w);
    }
    index += n;
  }
}

template <MemoryModifyFunction Unback>
inline bool PageTracker<Unback>::empty() const {
  return free_.used() == 0;
}

template <MemoryModifyFunction Unback>
inline Length PageTracker<Unback>::free_pages() const {
  return kPagesPerHugePage - used_pages();
}

template <class TrackerType>
inline HugePageFiller<TrackerType>::HugePageFiller(
    FillerPartialRerelease partial_rerelease)
    : HugePageFiller(
          partial_rerelease,
          Clock{.now = absl::base_internal::CycleClock::Now,
                .freq = absl::base_internal::CycleClock::Frequency}) {}

// For testing with mock clock
template <class TrackerType>
inline HugePageFiller<TrackerType>::HugePageFiller(
    FillerPartialRerelease partial_rerelease, Clock clock)
    : size_(NHugePages(0)),
      partial_rerelease_(partial_rerelease),
      fillerstats_tracker_(clock, absl::Minutes(10), absl::Minutes(5)) {}

template <class TrackerType>
inline typename HugePageFiller<TrackerType>::TryGetResult
HugePageFiller<TrackerType>::TryGet(Length n) {
  ASSERT(n > Length(0));

  // How do we choose which hugepage to allocate from (among those with
  // a free range of at least n?) Our goal is to be as space-efficient
  // as possible, which leads to two priorities:
  //
  // (1) avoid fragmentation; keep free ranges in a hugepage as long
  //     as possible. This maintains our ability to satisfy large
  //     requests without allocating new hugepages
  // (2) fill mostly-full hugepages more; let mostly-empty hugepages
  //     empty out.  This lets us recover totally empty hugepages (and
  //     return them to the OS.)
  //
  // In practice, avoiding fragmentation is by far more important:
  // space usage can explode if we don't jealously guard large free ranges.
  //
  // Our primary measure of fragmentation of a hugepage by a proxy measure: the
  // longest free range it contains. If this is short, any free space is
  // probably fairly fragmented.  It also allows us to instantly know if a
  // hugepage can support a given allocation.
  //
  // We quantize the number of allocations in a hugepage (chunked
  // logarithmically.) We favor allocating from hugepages with many allocations
  // already present, which helps with (2) above. Note that using the number of
  // allocations works substantially better than the number of allocated pages;
  // to first order allocations of any size are about as likely to be freed, and
  // so (by simple binomial probability distributions) we're more likely to
  // empty out a hugepage with 2 5-page allocations than one with 5 1-pages.
  //
  // The above suggests using the hugepage with the shortest longest empty
  // range, breaking ties in favor of fewest number of allocations. This works
  // well for most workloads but caused bad page heap fragmentation for some:
  // b/63301358 and b/138618726. The intuition for what went wrong is
  // that although the tail of large allocations is donated to the Filler (see
  // HugePageAwareAllocator::AllocRawHugepages) for use, we don't actually
  // want to use them until the regular Filler hugepages are used up. That
  // way, they can be reassembled as a single large hugepage range if the
  // large allocation is freed.
  // Some workloads can tickle this discrepancy a lot, because they have a lot
  // of large, medium-lifetime allocations. To fix this we treat hugepages
  // that are freshly donated as less preferable than hugepages that have been
  // already used for small allocations, regardless of their longest_free_range.
  //
  // Overall our allocation preference is:
  //  - We prefer allocating from used freelists rather than freshly donated
  //  - We prefer donated pages over previously released hugepages ones.
  //  - Among donated freelists we prefer smaller longest_free_range
  //  - Among used freelists we prefer smaller longest_free_range
  //    with ties broken by (quantized) alloc counts
  //
  // We group hugepages by longest_free_range and quantized alloc count and
  // store each group in a TrackerList. All freshly-donated groups are stored
  // in a "donated" array and the groups with (possibly prior) small allocs are
  // stored in a "regular" array. Each of these arrays is encapsulated in a
  // HintedTrackerLists object, which stores the array together with a bitmap to
  // quickly find non-empty lists. The lists are ordered to satisfy the
  // following two useful properties:
  //
  // - later (nonempty) freelists can always fulfill requests that
  //   earlier ones could.
  // - earlier freelists, by the above criteria, are preferred targets
  //   for allocation.
  //
  // So all we have to do is find the first nonempty freelist in the regular
  // HintedTrackerList that *could* support our allocation, and it will be our
  // best choice. If there is none we repeat with the donated HintedTrackerList.
  ASSUME(n < kPagesPerHugePage);
  TrackerType* pt;

  bool was_released = false;
  do {
    pt = regular_alloc_.GetLeast(ListFor(n, 0));
    if (pt) {
      ASSERT(!pt->donated());
      break;
    }
    pt = donated_alloc_.GetLeast(n.raw_num());
    if (pt) {
      break;
    }
    if (partial_rerelease_ == FillerPartialRerelease::Retain) {
      pt = regular_alloc_partial_released_.GetLeast(ListFor(n, 0));
      if (pt) {
        ASSERT(!pt->donated());
        was_released = true;
        ASSERT(n_used_partial_released_ >= pt->used_pages());
        n_used_partial_released_ -= pt->used_pages();
        break;
      }
    }
    pt = regular_alloc_released_.GetLeast(ListFor(n, 0));
    if (pt) {
      ASSERT(!pt->donated());
      was_released = true;
      ASSERT(n_used_released_ >= pt->used_pages());
      n_used_released_ -= pt->used_pages();
      break;
    }

    return {nullptr, PageId{0}};
  } while (false);
  ASSUME(pt != nullptr);
  ASSERT(pt->longest_free_range() >= n);
  const auto page_allocation = pt->Get(n);
  AddToFillerList(pt);
  allocated_ += n;

  ASSERT(was_released || page_allocation.previously_unbacked == Length(0));
  (void)was_released;
  ASSERT(unmapped_ >= page_allocation.previously_unbacked);
  unmapped_ -= page_allocation.previously_unbacked;
  // We're being used for an allocation, so we are no longer considered
  // donated by this point.
  ASSERT(!pt->donated());
  UpdateFillerStatsTracker();
  return {pt, page_allocation.page};
}

// Marks [p, p + n) as usable by new allocations into *pt; returns pt
// if that hugepage is now empty (nullptr otherwise.)
// REQUIRES: pt is owned by this object (has been Contribute()), and
// {pt, p, n} was the result of a previous TryGet.
template <class TrackerType>
inline TrackerType* HugePageFiller<TrackerType>::Put(TrackerType* pt, PageId p,
                                                     Length n) {
  // Consider releasing [p, p+n).  We do this here:
  // * To unback the memory before we mark it as free.  When partially
  //   unbacking, we release the pageheap_lock.  Another thread could see the
  //   "free" memory and begin using it before we retake the lock.
  // * To maintain maintain the invariant that
  //     pt->released() => regular_alloc_released_.size() > 0 ||
  //                       regular_alloc_partial_released_.size() > 0
  //   We do this before removing pt from our lists, since another thread may
  //   encounter our post-RemoveFromFillerList() update to
  //   regular_alloc_released_.size() and regular_alloc_partial_released_.size()
  //   while encountering pt.
  if (partial_rerelease_ == FillerPartialRerelease::Return) {
    pt->MaybeRelease(p, n);
  }

  RemoveFromFillerList(pt);

  pt->Put(p, n);

  allocated_ -= n;
  if (partial_rerelease_ == FillerPartialRerelease::Return && pt->released()) {
    unmapped_ += n;
    unmapping_unaccounted_ += n;
  }

  if (pt->longest_free_range() == kPagesPerHugePage) {
    --size_;
    if (pt->released()) {
      const Length free_pages = pt->free_pages();
      const Length released_pages = pt->released_pages();
      ASSERT(free_pages >= released_pages);
      ASSERT(unmapped_ >= released_pages);
      unmapped_ -= released_pages;

      if (free_pages > released_pages) {
        // We should only see a difference between free pages and released pages
        // when we retain returned pages.
        ASSERT(partial_rerelease_ == FillerPartialRerelease::Retain);

        // pt is partially released.  As the rest of the hugepage-aware
        // allocator works in terms of whole hugepages, we need to release the
        // rest of the hugepage.  This simplifies subsequent accounting by
        // allowing us to work with hugepage-granularity, rather than needing to
        // retain pt's state indefinitely.
        pageheap_lock.Unlock();
        TrackerType::UnbackImpl(pt->location().start_addr(), kHugePageSize);
        pageheap_lock.Lock();

        unmapping_unaccounted_ += free_pages - released_pages;
      }
    }

    UpdateFillerStatsTracker();
    return pt;
  }
  AddToFillerList(pt);
  UpdateFillerStatsTracker();
  return nullptr;
}

template <class TrackerType>
inline void HugePageFiller<TrackerType>::Contribute(TrackerType* pt,
                                                    bool donated) {
  // A contributed huge page should not yet be subreleased.
  ASSERT(pt->released_pages() == Length(0));

  allocated_ += pt->used_pages();
  if (donated) {
    DonateToFillerList(pt);
  } else {
    AddToFillerList(pt);
  }
  ++size_;
  UpdateFillerStatsTracker();
}

template <class TrackerType>
template <size_t N>
inline int HugePageFiller<TrackerType>::SelectCandidates(
    absl::Span<TrackerType*> candidates, int current_candidates,
    const HintedTrackerLists<N>& tracker_list, size_t tracker_start) {
  auto PushCandidate = [&](TrackerType* pt) {
    // If we have few candidates, we can avoid creating a heap.
    //
    // In ReleaseCandidates(), we unconditionally sort the list and linearly
    // iterate through it--rather than pop_heap repeatedly--so we only need the
    // heap for creating a bounded-size priority queue.
    if (current_candidates < candidates.size()) {
      candidates[current_candidates] = pt;
      current_candidates++;

      if (current_candidates == candidates.size()) {
        std::make_heap(candidates.begin(), candidates.end(),
                       CompareForSubrelease);
      }
      return;
    }

    // Consider popping the worst candidate from our list.
    if (CompareForSubrelease(candidates[0], pt)) {
      // pt is worse than the current worst.
      return;
    }

    std::pop_heap(candidates.begin(), candidates.begin() + current_candidates,
                  CompareForSubrelease);
    candidates[current_candidates - 1] = pt;
    std::push_heap(candidates.begin(), candidates.begin() + current_candidates,
                   CompareForSubrelease);
  };

  tracker_list.Iter(PushCandidate, tracker_start);

  return current_candidates;
}

template <class TrackerType>
inline Length HugePageFiller<TrackerType>::ReleaseCandidates(
    absl::Span<TrackerType*> candidates, Length target) {
  absl::c_sort(candidates, CompareForSubrelease);

  Length total_released;
  HugeLength total_broken = NHugePages(0);
#ifndef NDEBUG
  Length last;
#endif
  for (int i = 0; i < candidates.size() && total_released < target; i++) {
    TrackerType* best = candidates[i];
    ASSERT(best != nullptr);

#ifndef NDEBUG
    // Double check that our sorting criteria were applied correctly.
    ASSERT(last <= best->used_pages());
    last = best->used_pages();
#endif

    if (best->unbroken()) {
      ++total_broken;
    }
    RemoveFromFillerList(best);
    Length ret = best->ReleaseFree();
    unmapped_ += ret;
    ASSERT(unmapped_ >= best->released_pages());
    total_released += ret;
    AddToFillerList(best);
  }

  subrelease_stats_.num_pages_subreleased += total_released;
  subrelease_stats_.num_hugepages_broken += total_broken;

  // Keep separate stats if the on going release is triggered by reaching
  // tcmalloc limit
  if (subrelease_stats_.limit_hit()) {
    subrelease_stats_.total_pages_subreleased_due_to_limit += total_released;
    subrelease_stats_.total_hugepages_broken_due_to_limit += total_broken;
  }
  return total_released;
}

template <class TrackerType>
inline Length HugePageFiller<TrackerType>::GetDesiredSubreleasePages(
    Length desired, Length total_released, absl::Duration peak_interval) {
  // Don't subrelease pages if it wouldn't push you under the latest peak.
  // This is a bit subtle: We want the current *mapped* pages not to be below
  // the recent *demand* peak, i.e., if we have a large amount of free memory
  // right now but demand is below a recent peak, we still want to subrelease.
  ASSERT(total_released < desired);

  if (peak_interval == absl::ZeroDuration()) {
    return desired;
  }

  UpdateFillerStatsTracker();
  Length demand_at_peak =
      fillerstats_tracker_.GetRecentPeak(peak_interval).num_pages;
  Length current_pages = used_pages() + free_pages();

  if (demand_at_peak != Length(0)) {
    Length new_desired;
    if (demand_at_peak >= current_pages) {
      new_desired = total_released;
    } else {
      new_desired = total_released + (current_pages - demand_at_peak);
    }

    if (new_desired >= desired) {
      return desired;
    }

    // Report the amount of memory that we didn't release due to this
    // mechanism, but never more than free_pages, since we would not have
    // been able to release that much memory with or without this mechanism
    // (i.e., reporting more would be confusing).
    Length skipped_pages = std::min(free_pages(), (desired - new_desired));
    fillerstats_tracker_.ReportSkippedSubreleasePages(
        skipped_pages, current_pages, peak_interval);
    return new_desired;
  }

  return desired;
}

// Tries to release desired pages by iteratively releasing from the emptiest
// possible hugepage and releasing its free memory to the system.  Return the
// number of pages actually released.
template <class TrackerType>
inline Length HugePageFiller<TrackerType>::ReleasePages(
    Length desired, absl::Duration skip_subrelease_after_peaks_interval,
    bool hit_limit) {
  Length total_released;

  // We also do eager release, once we've called this at least once:
  // claim credit for anything that gets done.
  if (unmapping_unaccounted_.raw_num() > 0) {
    // TODO(ckennelly):  This may overshoot in releasing more than desired
    // pages.
    Length n = unmapping_unaccounted_;
    unmapping_unaccounted_ = Length(0);
    subrelease_stats_.num_pages_subreleased += n;

    if (n >= desired) {
      return n;
    }

    total_released += n;
  }

  if (skip_subrelease_after_peaks_interval != absl::ZeroDuration()) {
    desired = GetDesiredSubreleasePages(desired, total_released,
                                        skip_subrelease_after_peaks_interval);
    if (desired <= total_released) {
      return total_released;
    }
  }

  subrelease_stats_.set_limit_hit(hit_limit);

  // Optimize for releasing up to a huge page worth of small pages (scattered
  // over many parts of the filler).  Since we hold pageheap_lock, we cannot
  // allocate here.
  constexpr size_t kCandidates = kPagesPerHugePage.raw_num();
  using CandidateArray = std::array<TrackerType*, kCandidates>;

  if (partial_rerelease_ == FillerPartialRerelease::Retain) {
    while (total_released < desired) {
      CandidateArray candidates;
      // We can skip the first kChunks lists as they are known to be 100% full.
      // (Those lists are likely to be long.)
      //
      // We do not examine the regular_alloc_released_ lists, as only contain
      // completely released pages.
      int n_candidates =
          SelectCandidates(absl::MakeSpan(candidates), 0,
                           regular_alloc_partial_released_, kChunks);

      Length released =
          ReleaseCandidates(absl::MakeSpan(candidates.data(), n_candidates),
                            desired - total_released);
      if (released == Length(0)) {
        break;
      }
      total_released += released;
    }
  }

  // Only consider breaking up a hugepage if there are no partially released
  // pages.
  while (total_released < desired) {
    CandidateArray candidates;
    int n_candidates = SelectCandidates(absl::MakeSpan(candidates), 0,
                                        regular_alloc_, kChunks);
    // TODO(b/138864853): Perhaps remove donated_alloc_ from here, it's not a
    // great candidate for partial release.
    n_candidates = SelectCandidates(absl::MakeSpan(candidates), n_candidates,
                                    donated_alloc_, 0);

    Length released =
        ReleaseCandidates(absl::MakeSpan(candidates.data(), n_candidates),
                          desired - total_released);
    if (released == Length(0)) {
      break;
    }
    total_released += released;
  }

  return total_released;
}

template <class TrackerType>
inline void HugePageFiller<TrackerType>::AddSpanStats(
    SmallSpanStats* small, LargeSpanStats* large,
    PageAgeHistograms* ages) const {
  auto loop = [&](const TrackerType* pt) {
    pt->AddSpanStats(small, large, ages);
  };
  // We can skip the first kChunks lists as they are known to be 100% full.
  regular_alloc_.Iter(loop, kChunks);
  donated_alloc_.Iter(loop, 0);

  if (partial_rerelease_ == FillerPartialRerelease::Retain) {
    regular_alloc_partial_released_.Iter(loop, 0);
  } else {
    ASSERT(regular_alloc_partial_released_.empty());
    ASSERT(n_used_partial_released_ == Length(0));
  }
  regular_alloc_released_.Iter(loop, 0);
}

template <class TrackerType>
inline BackingStats HugePageFiller<TrackerType>::stats() const {
  BackingStats s;
  s.system_bytes = size_.in_bytes();
  s.free_bytes = free_pages().in_bytes();
  s.unmapped_bytes = unmapped_pages().in_bytes();
  return s;
}

namespace huge_page_filler_internal {
// Computes some histograms of fullness. Because nearly empty/full huge pages
// are much more interesting, we calculate 4 buckets at each of the beginning
// and end of size one, and then divide the overall space by 16 to have 16
// (mostly) even buckets in the middle.
class UsageInfo {
 public:
  enum Type { kRegular, kDonated, kPartialReleased, kReleased, kNumTypes };

  UsageInfo() {
    size_t i;
    for (i = 0; i <= 4 && i < kPagesPerHugePage.raw_num(); ++i) {
      bucket_bounds_[buckets_size_] = i;
      buckets_size_++;
    }
    if (i < kPagesPerHugePage.raw_num() - 4) {
      // Because kPagesPerHugePage is a power of two, it must be at least 16
      // to get inside this "if" - either i=5 and kPagesPerHugePage=8 and
      // the test fails, or kPagesPerHugePage <= 4 and the test fails.
      ASSERT(kPagesPerHugePage >= Length(16));
      constexpr int step = kPagesPerHugePage.raw_num() / 16;
      // We want to move in "step"-sized increments, aligned every "step".
      // So first we have to round i up to the nearest step boundary. This
      // logic takes advantage of step being a power of two, so step-1 is
      // all ones in the low-order bits.
      i = ((i - 1) | (step - 1)) + 1;
      for (; i < kPagesPerHugePage.raw_num() - 4; i += step) {
        bucket_bounds_[buckets_size_] = i;
        buckets_size_++;
      }
      i = kPagesPerHugePage.raw_num() - 4;
    }
    for (; i < kPagesPerHugePage.raw_num(); ++i) {
      bucket_bounds_[buckets_size_] = i;
      buckets_size_++;
    }
    CHECK_CONDITION(buckets_size_ <= kBucketCapacity);
  }

  template <class TrackerType>
  void Record(const TrackerType* pt, Type which) {
    const Length free = kPagesPerHugePage - pt->used_pages();
    const Length lf = pt->longest_free_range();
    const size_t nalloc = pt->nallocs();
    // This is a little annoying as our buckets *have* to differ;
    // nalloc is in [1,256], free_pages and longest_free are in [0, 255].
    free_page_histo_[which][BucketNum(free.raw_num())]++;
    longest_free_histo_[which][BucketNum(lf.raw_num())]++;
    nalloc_histo_[which][BucketNum(nalloc - 1)]++;
  }

  void Print(Printer* out) {
    PrintHisto(out, free_page_histo_[kRegular],
               "# of regular hps with a<= # of free pages <b", 0);
    PrintHisto(out, free_page_histo_[kDonated],
               "# of donated hps with a<= # of free pages <b", 0);
    PrintHisto(out, free_page_histo_[kPartialReleased],
               "# of partial released hps with a<= # of free pages <b", 0);
    PrintHisto(out, free_page_histo_[kReleased],
               "# of released hps with a<= # of free pages <b", 0);
    // For donated huge pages, number of allocs=1 and longest free range =
    // number of free pages, so it isn't useful to show the next two.
    PrintHisto(out, longest_free_histo_[kRegular],
               "# of regular hps with a<= longest free range <b", 0);
    PrintHisto(out, longest_free_histo_[kPartialReleased],
               "# of partial released hps with a<= longest free range <b", 0);
    PrintHisto(out, longest_free_histo_[kReleased],
               "# of released hps with a<= longest free range <b", 0);
    PrintHisto(out, nalloc_histo_[kRegular],
               "# of regular hps with a<= # of allocations <b", 1);
    PrintHisto(out, nalloc_histo_[kPartialReleased],
               "# of partial released hps with a<= # of allocations <b", 1);
    PrintHisto(out, nalloc_histo_[kReleased],
               "# of released hps with a<= # of allocations <b", 1);
  }

  void Print(PbtxtRegion* hpaa) {
    static constexpr absl::string_view kTrackerTypes[kNumTypes] = {
        "REGULAR", "DONATED", "PARTIAL", "RELEASED"};
    for (int i = 0; i < kNumTypes; ++i) {
      PbtxtRegion scoped = hpaa->CreateSubRegion("filler_tracker");
      scoped.PrintRaw("type", kTrackerTypes[i]);
      PrintHisto(&scoped, free_page_histo_[i], "free_pages_histogram", 0);
      PrintHisto(&scoped, longest_free_histo_[i],
                 "longest_free_range_histogram", 0);
      PrintHisto(&scoped, nalloc_histo_[i], "allocations_histogram", 1);
    }
  }

 private:
  // Maximum of 4 buckets at the start and end, and 16 in the middle.
  static constexpr size_t kBucketCapacity = 4 + 16 + 4;
  using Histo = size_t[kBucketCapacity];

  int BucketNum(size_t page) {
    auto it =
        std::upper_bound(bucket_bounds_, bucket_bounds_ + buckets_size_, page);
    CHECK_CONDITION(it != bucket_bounds_);
    return it - bucket_bounds_ - 1;
  }

  void PrintHisto(Printer* out, Histo h, const char blurb[], size_t offset) {
    out->printf("\nHugePageFiller: %s", blurb);
    for (size_t i = 0; i < buckets_size_; ++i) {
      if (i % 6 == 0) {
        out->printf("\nHugePageFiller:");
      }
      out->printf(" <%3zu<=%6zu", bucket_bounds_[i] + offset, h[i]);
    }
    out->printf("\n");
  }

  void PrintHisto(PbtxtRegion* hpaa, Histo h, const char key[], size_t offset) {
    for (size_t i = 0; i < buckets_size_; ++i) {
      auto hist = hpaa->CreateSubRegion(key);
      hist.PrintI64("lower_bound", bucket_bounds_[i] + offset);
      hist.PrintI64("upper_bound",
                    (i == buckets_size_ - 1 ? bucket_bounds_[i]
                                            : bucket_bounds_[i + 1] - 1) +
                        offset);
      hist.PrintI64("value", h[i]);
    }
  }

  // Arrays, because they are split per alloc type.
  Histo free_page_histo_[kNumTypes]{};
  Histo longest_free_histo_[kNumTypes]{};
  Histo nalloc_histo_[kNumTypes]{};
  size_t bucket_bounds_[kBucketCapacity];
  int buckets_size_ = 0;
};
}  // namespace huge_page_filler_internal

template <class TrackerType>
inline void HugePageFiller<TrackerType>::Print(Printer* out,
                                               bool everything) const {
  out->printf("HugePageFiller: densely pack small requests into hugepages\n");

  HugeLength nrel =
      regular_alloc_released_.size() + regular_alloc_partial_released_.size();
  HugeLength nfull = NHugePages(0);

  // note kChunks, not kNumLists here--we're iterating *full* lists.
  for (size_t chunk = 0; chunk < kChunks; ++chunk) {
    nfull += NHugePages(
        regular_alloc_[ListFor(/*longest=*/Length(0), chunk)].length());
  }
  // A donated alloc full list is impossible because it would have never been
  // donated in the first place. (It's an even hugepage.)
  ASSERT(donated_alloc_[0].empty());
  // Evaluate a/b, avoiding division by zero
  const auto safe_div = [](Length a, Length b) {
    return b == Length(0) ? 0.
                          : static_cast<double>(a.raw_num()) /
                                static_cast<double>(b.raw_num());
  };
  const HugeLength n_partial = size() - nrel - nfull;
  const HugeLength n_nonfull =
      n_partial + regular_alloc_partial_released_.size();
  out->printf(
      "HugePageFiller: %zu total, %zu full, %zu partial, %zu released "
      "(%zu partially), 0 quarantined\n",
      size().raw_num(), nfull.raw_num(), n_partial.raw_num(), nrel.raw_num(),
      regular_alloc_partial_released_.size().raw_num());
  out->printf("HugePageFiller: %zu pages free in %zu hugepages, %.4f free\n",
              free_pages().raw_num(), size().raw_num(),
              safe_div(free_pages(), size().in_pages()));

  ASSERT(free_pages() <= n_nonfull.in_pages());
  out->printf("HugePageFiller: among non-fulls, %.4f free\n",
              safe_div(free_pages(), n_nonfull.in_pages()));

  out->printf(
      "HugePageFiller: %zu used pages in subreleased hugepages (%zu of them in "
      "partially released)\n",
      used_pages_in_any_subreleased().raw_num(),
      used_pages_in_partial_released().raw_num());

  out->printf(
      "HugePageFiller: %zu hugepages partially released, %.4f released\n",
      nrel.raw_num(), safe_div(unmapped_pages(), nrel.in_pages()));
  out->printf("HugePageFiller: %.4f of used pages hugepageable\n",
              hugepage_frac());

  // Subrelease
  out->printf(
      "HugePageFiller: Since startup, %zu pages subreleased, %zu hugepages "
      "broken, (%zu pages, %zu hugepages due to reaching tcmalloc limit)\n",
      subrelease_stats_.total_pages_subreleased.raw_num(),
      subrelease_stats_.total_hugepages_broken.raw_num(),
      subrelease_stats_.total_pages_subreleased_due_to_limit.raw_num(),
      subrelease_stats_.total_hugepages_broken_due_to_limit.raw_num());

  if (!everything) return;

  // Compute some histograms of fullness.
  using huge_page_filler_internal::UsageInfo;
  UsageInfo usage;
  regular_alloc_.Iter(
      [&](const TrackerType* pt) { usage.Record(pt, UsageInfo::kRegular); }, 0);
  donated_alloc_.Iter(
      [&](const TrackerType* pt) { usage.Record(pt, UsageInfo::kDonated); }, 0);
  if (partial_rerelease_ == FillerPartialRerelease::Retain) {
    regular_alloc_partial_released_.Iter(
        [&](const TrackerType* pt) {
          usage.Record(pt, UsageInfo::kPartialReleased);
        },
        0);
  } else {
    ASSERT(regular_alloc_partial_released_.empty());
    ASSERT(n_used_partial_released_.raw_num() == 0);
  }
  regular_alloc_released_.Iter(
      [&](const TrackerType* pt) { usage.Record(pt, UsageInfo::kReleased); },
      0);

  out->printf("\n");
  out->printf("HugePageFiller: fullness histograms\n");
  usage.Print(out);

  out->printf("\n");
  fillerstats_tracker_.Print(out);
}

template <class TrackerType>
inline void HugePageFiller<TrackerType>::PrintInPbtxt(PbtxtRegion* hpaa) const {
  HugeLength nrel =
      regular_alloc_released_.size() + regular_alloc_partial_released_.size();
  HugeLength nfull = NHugePages(0);

  // note kChunks, not kNumLists here--we're iterating *full* lists.
  for (size_t chunk = 0; chunk < kChunks; ++chunk) {
    nfull += NHugePages(
        regular_alloc_[ListFor(/*longest=*/Length(0), chunk)].length());
  }
  // A donated alloc full list is impossible because it would have never been
  // donated in the first place. (It's an even hugepage.)
  ASSERT(donated_alloc_[0].empty());
  // Evaluate a/b, avoiding division by zero
  const auto safe_div = [](Length a, Length b) {
    return b == Length(0) ? 0
                          : static_cast<double>(a.raw_num()) /
                                static_cast<double>(b.raw_num());
  };
  const HugeLength n_partial = size() - nrel - nfull;
  hpaa->PrintI64("filler_full_huge_pages", nfull.raw_num());
  hpaa->PrintI64("filler_partial_huge_pages", n_partial.raw_num());
  hpaa->PrintI64("filler_released_huge_pages", nrel.raw_num());
  hpaa->PrintI64("filler_partially_released_huge_pages",
                 regular_alloc_partial_released_.size().raw_num());
  hpaa->PrintI64("filler_free_pages", free_pages().raw_num());
  hpaa->PrintI64("filler_used_pages_in_subreleased",
                 used_pages_in_any_subreleased().raw_num());
  hpaa->PrintI64("filler_used_pages_in_partial_released",
                 used_pages_in_partial_released().raw_num());
  hpaa->PrintI64(
      "filler_unmapped_bytes",
      static_cast<uint64_t>(nrel.raw_num() *
                            safe_div(unmapped_pages(), nrel.in_pages())));
  hpaa->PrintI64(
      "filler_hugepageable_used_bytes",
      static_cast<uint64_t>(hugepage_frac() *
                            static_cast<double>(allocated_.in_bytes())));
  hpaa->PrintI64("filler_num_pages_subreleased",
                 subrelease_stats_.total_pages_subreleased.raw_num());
  hpaa->PrintI64("filler_num_hugepages_broken",
                 subrelease_stats_.total_hugepages_broken.raw_num());
  hpaa->PrintI64(
      "filler_num_pages_subreleased_due_to_limit",
      subrelease_stats_.total_pages_subreleased_due_to_limit.raw_num());
  hpaa->PrintI64(
      "filler_num_hugepages_broken_due_to_limit",
      subrelease_stats_.total_hugepages_broken_due_to_limit.raw_num());
  // Compute some histograms of fullness.
  using huge_page_filler_internal::UsageInfo;
  UsageInfo usage;
  regular_alloc_.Iter(
      [&](const TrackerType* pt) { usage.Record(pt, UsageInfo::kRegular); }, 0);
  donated_alloc_.Iter(
      [&](const TrackerType* pt) { usage.Record(pt, UsageInfo::kDonated); }, 0);
  if (partial_rerelease_ == FillerPartialRerelease::Retain) {
    regular_alloc_partial_released_.Iter(
        [&](const TrackerType* pt) {
          usage.Record(pt, UsageInfo::kPartialReleased);
        },
        0);
  } else {
    ASSERT(regular_alloc_partial_released_.empty());
    ASSERT(n_used_partial_released_ == Length(0));
  }
  regular_alloc_released_.Iter(
      [&](const TrackerType* pt) { usage.Record(pt, UsageInfo::kReleased); },
      0);

  usage.Print(hpaa);

  fillerstats_tracker_.PrintInPbtxt(hpaa);
}

template <class TrackerType>
inline void HugePageFiller<TrackerType>::UpdateFillerStatsTracker() {
  StatsTrackerType::FillerStats stats;
  stats.num_pages = allocated_;
  stats.free_pages = free_pages();
  stats.unmapped_pages = unmapped_pages();
  stats.used_pages_in_subreleased_huge_pages =
      n_used_partial_released_ + n_used_released_;
  stats.huge_pages[StatsTrackerType::kRegular] = regular_alloc_.size();
  stats.huge_pages[StatsTrackerType::kDonated] = donated_alloc_.size();
  stats.huge_pages[StatsTrackerType::kPartialReleased] =
      regular_alloc_partial_released_.size();
  stats.huge_pages[StatsTrackerType::kReleased] =
      regular_alloc_released_.size();
  stats.num_pages_subreleased = subrelease_stats_.num_pages_subreleased;
  stats.num_hugepages_broken = subrelease_stats_.num_hugepages_broken;
  fillerstats_tracker_.Report(stats);
  subrelease_stats_.reset();
}

template <class TrackerType>
inline size_t HugePageFiller<TrackerType>::IndexFor(TrackerType* pt) {
  ASSERT(!pt->empty());
  // Prefer to allocate from hugepages with many allocations already present;
  // spaced logarithmically.
  const size_t na = pt->nallocs();
  // This equals 63 - ceil(log2(na))
  // (or 31 if size_t is 4 bytes, etc.)
  const size_t neg_ceil_log = __builtin_clzl(2 * na - 1);

  // We want the same spread as neg_ceil_log, but spread over [0,
  // kChunks) (clamped at the left edge) instead of [0, 64). So subtract off
  // the difference (computed by forcing na=1 to kChunks - 1.)
  const size_t kOffset = __builtin_clzl(1) - (kChunks - 1);
  const size_t i = std::max(neg_ceil_log, kOffset) - kOffset;
  ASSERT(i < kChunks);
  return i;
}

template <class TrackerType>
inline size_t HugePageFiller<TrackerType>::ListFor(const Length longest,
                                                   const size_t chunk) {
  ASSERT(chunk < kChunks);
  ASSERT(longest < kPagesPerHugePage);
  return longest.raw_num() * kChunks + chunk;
}

template <class TrackerType>
inline void HugePageFiller<TrackerType>::RemoveFromFillerList(TrackerType* pt) {
  Length longest = pt->longest_free_range();
  ASSERT(longest < kPagesPerHugePage);

  if (pt->donated()) {
    donated_alloc_.Remove(pt, longest.raw_num());
  } else {
    size_t chunk = IndexFor(pt);
    size_t i = ListFor(longest, chunk);
    if (!pt->released()) {
      regular_alloc_.Remove(pt, i);
    } else if (partial_rerelease_ == FillerPartialRerelease::Return ||
               pt->free_pages() <= pt->released_pages()) {
      regular_alloc_released_.Remove(pt, i);
      ASSERT(n_used_released_ >= pt->used_pages());
      n_used_released_ -= pt->used_pages();
    } else {
      regular_alloc_partial_released_.Remove(pt, i);
      ASSERT(n_used_partial_released_ >= pt->used_pages());
      n_used_partial_released_ -= pt->used_pages();
    }
  }
}

template <class TrackerType>
inline void HugePageFiller<TrackerType>::AddToFillerList(TrackerType* pt) {
  size_t chunk = IndexFor(pt);
  Length longest = pt->longest_free_range();
  ASSERT(longest < kPagesPerHugePage);

  // Once a donated alloc is used in any way, it degenerates into being a
  // regular alloc. This allows the algorithm to keep using it (we had to be
  // desperate to use it in the first place), and thus preserves the other
  // donated allocs.
  pt->set_donated(false);

  size_t i = ListFor(longest, chunk);
  if (!pt->released()) {
    regular_alloc_.Add(pt, i);
  } else if (partial_rerelease_ == FillerPartialRerelease::Return ||
             pt->free_pages() == pt->released_pages()) {
    regular_alloc_released_.Add(pt, i);
    n_used_released_ += pt->used_pages();
  } else {
    ASSERT(partial_rerelease_ == FillerPartialRerelease::Retain);
    regular_alloc_partial_released_.Add(pt, i);
    n_used_partial_released_ += pt->used_pages();
  }
}

template <class TrackerType>
inline void HugePageFiller<TrackerType>::DonateToFillerList(TrackerType* pt) {
  Length longest = pt->longest_free_range();
  ASSERT(longest < kPagesPerHugePage);

  // We should never be donating already-released trackers!
  ASSERT(!pt->released());
  pt->set_donated(true);

  donated_alloc_.Add(pt, longest.raw_num());
}

template <class TrackerType>
inline double HugePageFiller<TrackerType>::hugepage_frac() const {
  // How many of our used pages are on non-huge pages? Since
  // everything on a released hugepage is either used or released,
  // just the difference:
  const Length nrel = regular_alloc_released_.size().in_pages();
  const Length used = used_pages();
  const Length unmapped = unmapped_pages();
  ASSERT(n_used_partial_released_ <=
         regular_alloc_partial_released_.size().in_pages());
  const Length used_on_rel = (nrel >= unmapped ? nrel - unmapped : Length(0)) +
                             n_used_partial_released_;
  ASSERT(used >= used_on_rel);
  const Length used_on_huge = used - used_on_rel;

  const Length denom = used > Length(0) ? used : Length(1);
  const double ret =
      static_cast<double>(used_on_huge.raw_num()) / denom.raw_num();
  ASSERT(ret >= 0);
  ASSERT(ret <= 1);
  return std::clamp<double>(ret, 0, 1);
}

// Helper for stat functions.
template <class TrackerType>
inline Length HugePageFiller<TrackerType>::free_pages() const {
  return size().in_pages() - used_pages() - unmapped_pages();
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_HUGE_PAGE_FILLER_H_
