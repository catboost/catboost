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

#include "tcmalloc/huge_cache.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "absl/base/optimization.h"
#include "absl/time/time.h"
#include "tcmalloc/huge_address_map.h"
#include "tcmalloc/huge_page_subrelease.h"
#include "tcmalloc/huge_pages.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/stats.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

template <size_t kEpochs>
void MinMaxTracker<kEpochs>::Report(HugeLength val) {
  timeseries_.Report(val);
}

template <size_t kEpochs>
HugeLength MinMaxTracker<kEpochs>::MaxOverTime(absl::Duration t) const {
  HugeLength m = NHugePages(0);
  size_t num_epochs = ceil(absl::FDivDuration(t, kEpochLength));
  timeseries_.IterBackwards(
      [&](size_t offset, const Extrema& e) { m = std::max(m, e.max); },
      num_epochs);
  return m;
}

template <size_t kEpochs>
HugeLength MinMaxTracker<kEpochs>::MinOverTime(absl::Duration t) const {
  HugeLength m = kMaxVal;
  size_t num_epochs = ceil(absl::FDivDuration(t, kEpochLength));
  timeseries_.IterBackwards(
      [&](size_t offset, const Extrema& e) { m = std::min(m, e.min); },
      num_epochs);
  return m;
}

template <size_t kEpochs>
void MinMaxTracker<kEpochs>::Print(Printer& out) const {
  // Prints timestamp:min_pages:max_pages for each window with records.
  // Timestamp == kEpochs - 1 is the most recent measurement.
  const int64_t millis = absl::ToInt64Milliseconds(kEpochLength);
  out.printf("\nHugeCache: window %lldms * %zu", millis, kEpochs);
  int written = 0;
  timeseries_.Iter(
      [&](size_t offset, const Extrema& e) {
        if ((written++) % 100 == 0) {
          out.printf("\nHugeCache: Usage timeseries ");
        }
        out.printf("%zu:%zu:%zd,", offset, e.min.raw_num(), e.max.raw_num());
      },
      timeseries_.kSkipEmptyEntries);
  out.printf("\n");
}

template <size_t kEpochs>
void MinMaxTracker<kEpochs>::PrintInPbtxt(PbtxtRegion& hpaa) const {
  // Prints content of each non-empty epoch, from oldest to most recent data
  auto huge_cache_history = hpaa.CreateSubRegion("huge_cache_history");
  huge_cache_history.PrintI64("window_ms",
                              absl::ToInt64Milliseconds(kEpochLength));
  huge_cache_history.PrintI64("epochs", kEpochs);

  timeseries_.Iter(
      [&](size_t offset, const Extrema& e) {
        auto m = huge_cache_history.CreateSubRegion("measurements");
        m.PrintI64("epoch", offset);
        m.PrintI64("min_bytes", e.min.in_bytes());
        m.PrintI64("max_bytes", e.max.in_bytes());
      },
      timeseries_.kSkipEmptyEntries);
}

template <size_t kEpochs>
bool MinMaxTracker<kEpochs>::Extrema::operator==(const Extrema& other) const {
  return (other.max == max) && (other.min == min);
}

// Explicit instantiations of template
template class MinMaxTracker<>;
template class MinMaxTracker<600>;

// The logic for actually allocating from the cache or backing, and keeping
// the hit rates specified.
HugeRange HugeCache::DoGet(HugeLength n, bool* from_released) {
  auto* node = Find(n);
  if (!node) {
    misses_++;
    weighted_misses_ += n.raw_num();
    HugeRange res = allocator_->Get(n);
    if (res.valid()) {
      *from_released = true;
    }

    return res;
  }
  hits_++;
  weighted_hits_ += n.raw_num();
  *from_released = false;
  size_ -= n;
  UpdateSize(size());
  HugeRange result, leftover;
  // Put back whatever we have left (or nothing, if it's exact.)
  std::tie(result, leftover) = Split(node->range(), n);
  cache_.Remove(node);
  if (leftover.valid()) {
    cache_.Insert(leftover);
  }
  return result;
}

void HugeCache::MaybeGrowCacheLimit(HugeLength missed) {
  // Our goal is to make the cache size = the largest "brief dip."
  //
  // A "dip" being a case where usage shrinks, then increases back up
  // to previous levels (at least partially).
  //
  // "brief" is "returns to normal usage in < cache_time_." (In
  // other words, we ideally want to be willing to cache memory for
  // cache_time_ before expecting it to be used again--we are loose
  // on the timing..)
  //
  // The interesting part is finding those dips.

  // This is the downward slope: we lost some usage. (This in theory could
  // be as much as 2 * cache_time_ old, which is fine.)
  const HugeLength shrink = off_peak_tracker_.MaxOverTime(cache_time_);

  // This is the upward slope: we are coming back up.
  const HugeLength grow = usage_ - usage_tracker_.MinOverTime(cache_time_);

  // Ideally we now know that we dipped down by some amount, then came
  // up.  Sadly our stats aren't quite good enough to guarantee things
  // happened in the proper order.  Suppose our usage takes the
  // following path (in essentially zero time):
  // 0, 10000, 5000, 5500.
  //
  // Clearly the proven dip here is 500.  But we'll compute shrink = 5000,
  // grow = 5500--we'd prefer to measure from a min *after* that shrink.
  //
  // It's difficult to ensure this, and hopefully this case is rare.
  // TODO(b/134690209): figure out if we can solve that problem.
  const HugeLength dip = std::min(shrink, grow);

  // Fragmentation: we may need to cache a little more than the actual
  // usage jump. 10% seems to be a reasonable addition that doesn't waste
  // much space, but gets good performance on tests.
  const HugeLength slack = dip / 10;

  const HugeLength lim = dip + slack;

  if (lim > limit()) {
    last_limit_change_ = clock_.now();
    limit_ = lim;
  }
}

void HugeCache::IncUsage(HugeLength n) {
  usage_ += n;
  usage_tracker_.Report(usage_);
  detailed_tracker_.Report(usage_);
  off_peak_tracker_.Report(NHugePages(0));
}

void HugeCache::DecUsage(HugeLength n) {
  usage_ -= n;
  usage_tracker_.Report(usage_);
  detailed_tracker_.Report(usage_);
  const HugeLength max = usage_tracker_.MaxOverTime(cache_time_);
  TC_ASSERT_GE(max, usage_);
  const HugeLength off_peak = max - usage_;
  off_peak_tracker_.Report(off_peak);
}

void HugeCache::UpdateSize(HugeLength size) { size_tracker_.Report(size); }

void HugeCache::UpdateStatsTracker() {
  cachestats_tracker_.Report(GetSubreleaseStats());
  hugepage_release_stats_.reset();
}

HugeRange HugeCache::Get(HugeLength n, bool* from_released) {
  HugeRange r = DoGet(n, from_released);
  // failure to get a range should "never" "never" happen (VSS limits
  // or wildly incorrect allocation sizes only...) Don't deal with
  // this case for cache size accounting.
  IncUsage(r.len());

  const bool miss = r.valid() && *from_released;
  if (miss) MaybeGrowCacheLimit(n);
  UpdateStatsTracker();
  return r;
}

void HugeCache::Release(HugeRange r, bool demand_based_unback) {
  DecUsage(r.len());

  cache_.Insert(r);
  size_ += r.len();
  if (size_ <= limit()) {
    fills_++;
  } else {
    overflows_++;
  }
  // Performs a (quick) unback if the demand-based release is disabled.
  if (!demand_based_unback) {
    // Shrink the limit, if we're going to do it, before we shrink to
    // the max size.  (This could reduce the number of regions we break
    // in half to avoid overshrinking.)
    if ((clock_.now() - last_limit_change_) > (cache_time_ticks_ * 2)) {
      total_fast_unbacked_ += MaybeShrinkCacheLimit();
    }
    total_fast_unbacked_ += ShrinkCache(limit());
  }
  UpdateSize(size());
  UpdateStatsTracker();
}

void HugeCache::ReleaseUnbacked(HugeRange r) {
  DecUsage(r.len());
  // No point in trying to cache it, just hand it back.
  allocator_->Release(r);
  UpdateStatsTracker();
}

HugeLength HugeCache::MaybeShrinkCacheLimit() {
  last_limit_change_ = clock_.now();

  const HugeLength min = size_tracker_.MinOverTime(cache_time_ * 2);
  // If cache size has gotten down to at most 20% of max, we assume
  // we're close enough to the optimal size--we don't want to fiddle
  // too much/too often unless we have large gaps in usage.
  if (min < limit() / 5) return NHugePages(0);

  // Take away half of the unused portion.
  HugeLength drop = std::max(min / 2, NHugePages(1));
  limit_ = std::max(limit() <= drop ? NHugePages(0) : limit() - drop,
                    MinCacheLimit());
  return ShrinkCache(limit());
}

HugeLength HugeCache::ShrinkCache(HugeLength target) {
  HugeLength removed = NHugePages(0);
  while (size_ > target) {
    // Remove smallest-ish nodes, to avoid fragmentation where possible.
    auto* node = Find(NHugePages(1));
    TC_CHECK_NE(node, nullptr);
    HugeRange r = node->range();
    cache_.Remove(node);
    // Suppose we're 10 MiB over target but the smallest available node
    // is 100 MiB.  Don't go overboard--split up the range.
    // In particular - this prevents disastrous results if we've decided
    // the cache should be 99 MiB but the actual hot usage is 100 MiB
    // (and it is unfragmented).
    const HugeLength delta = size() - target;
    if (r.len() > delta) {
      HugeRange to_remove, leftover;
      std::tie(to_remove, leftover) = Split(r, delta);
      TC_ASSERT(leftover.valid());
      cache_.Insert(leftover);
      r = to_remove;
    }

    size_ -= r.len();
    // Note, actual unback implementation is temporarily dropping and
    // re-acquiring the page heap lock here.
    if (ABSL_PREDICT_FALSE(!unback_(r))) {
      // We failed to release r.  Retain it in the cache instead of returning it
      // to the HugeAllocator.
      size_ += r.len();
      cache_.Insert(r);
      break;
    }
    allocator_->Release(r);
    removed += r.len();
  }

  return removed;
}

HugeLength HugeCache::ReleaseCachedPages(HugeLength n) {
  // This is a good time to check: is our cache going persistently unused?
  HugeLength released = MaybeShrinkCacheLimit();

  if (released < n) {
    n -= released;
    const HugeLength target = n > size() ? NHugePages(0) : size() - n;
    released += ShrinkCache(target);
  }
  UpdateSize(size());
  UpdateStatsTracker();
  total_periodic_unbacked_ += released;
  return released;
}

HugeLength HugeCache::GetDesiredReleaseablePages(
    HugeLength desired, SkipSubreleaseIntervals intervals) {
  TC_CHECK(intervals.SkipSubreleaseEnabled());
  UpdateStatsTracker();
  HugeLength required_by_demand;
  required_by_demand = HLFromPages(cachestats_tracker_.GetRecentDemand(
      intervals.short_interval, intervals.long_interval, CapDemandInterval()));

  HugeLength current = usage() + size();
  if (required_by_demand != NHugePages(0)) {
    HugeLength new_desired;
    // We can only release if the current capacity is larger than the demand.
    if (required_by_demand < current) {
      new_desired = current - required_by_demand;
    }
    if (new_desired >= desired) {
      return desired;
    }
    // Reports the amount of free hugepages that we didn't release due to this
    // mechanism. As the initial release target is capped by the cache size,
    // here we simply report the reduced amount. Note, only free pages in the
    // smaller of the two (current and required_by_demand) are skipped, so we
    // use that as the reporting peak.
    HugeLength skipped = desired - new_desired;
    cachestats_tracker_.ReportSkippedSubreleasePages(
        skipped.in_pages(),
        std::min(current.in_pages(), required_by_demand.in_pages()));
    return new_desired;
  }
  return desired;
}

HugeLength HugeCache::ReleaseCachedPagesByDemand(
    HugeLength n, SkipSubreleaseIntervals intervals, bool hit_limit) {
  // We get here when one of the three happened: A) hit limit, B) background
  // release, or C) ReleaseMemoryToSystem().
  HugeLength release_target = std::min(n, size());

  // For all those three reasons, we want to release as much as possible to be
  // efficient. However, we do not want to release a large number of hugepages
  // at once because that may impact applications' performance. So we release a
  // fraction of the cache.
  if (size() > MinCacheLimit()) {
    HugeLength increased_release_target =
        std::min(HugeLength(kFractionToReleaseFromCache * size().raw_num()),
                 size() - MinCacheLimit());
    release_target = std::max(release_target, increased_release_target);
  }

  if (release_target == NHugePages(0)) {
    return NHugePages(0);
  }
  if (intervals.SkipSubreleaseEnabled() && !hit_limit) {
    // This will reduce the target if the calculated (future) demand is higher
    // than the current. In other words, we need to reserve some of the free
    // hugepages to meet the future demand. It also makes sure we release the
    // realized fragmentation.
    release_target = GetDesiredReleaseablePages(release_target, intervals);
  }
  HugeLength released = ShrinkCache(size() - release_target);
  hugepage_release_stats_.num_pages_subreleased += released.in_pages();
  hugepage_release_stats_.set_limit_hit(hit_limit);
  if (hugepage_release_stats_.limit_hit()) {
    hugepage_release_stats_.total_pages_subreleased_due_to_limit +=
        released.in_pages();
  }
  UpdateSize(size());
  UpdateStatsTracker();
  total_periodic_unbacked_ += released;
  return released;
}

void HugeCache::AddSpanStats(SmallSpanStats* small,
                             LargeSpanStats* large) const {
  static_assert(kPagesPerHugePage >= kMaxPages);
  for (const HugeAddressMap::Node* node = cache_.first(); node != nullptr;
       node = node->next()) {
    HugeLength n = node->range().len();
    if (large != nullptr) {
      large->spans++;
      large->normal_pages += n.in_pages();
    }
  }
}

HugeAddressMap::Node* HugeCache::Find(HugeLength n) {
  HugeAddressMap::Node* curr = cache_.root();
  // invariant: curr != nullptr && curr->longest >= n
  // we favor smaller gaps and lower nodes and lower addresses, in that
  // order. The net effect is that we are neither a best-fit nor a
  // lowest-address allocator but vaguely close to both.
  HugeAddressMap::Node* best = nullptr;
  while (curr && curr->longest() >= n) {
    if (curr->range().len() >= n) {
      if (!best || best->range().len() > curr->range().len()) {
        best = curr;
      }
    }

    // Either subtree could contain a better fit and we don't want to
    // search the whole tree. Pick a reasonable child to look at.
    auto left = curr->left();
    auto right = curr->right();
    if (!left || left->longest() < n) {
      curr = right;
      continue;
    }

    if (!right || right->longest() < n) {
      curr = left;
      continue;
    }

    // Here, we have a nontrivial choice.
    if (left->range().len() == right->range().len()) {
      if (left->longest() <= right->longest()) {
        curr = left;
      } else {
        curr = right;
      }
    } else if (left->range().len() < right->range().len()) {
      // Here, the longest range in both children is the same...look
      // in the subtree with the smaller root, as that's slightly
      // more likely to be our best.
      curr = left;
    } else {
      curr = right;
    }
  }
  return best;
}

void HugeCache::Print(Printer& out) {
  const int64_t millis = absl::ToInt64Milliseconds(cache_time_);
  out.printf(
      "HugeCache: contains unused, backed hugepage(s) "
      "(cache_time = %lldms)\n",
      millis);
  // a / (a + b), avoiding division by zero
  auto safe_ratio = [](double a, double b) {
    const double total = a + b;
    if (total == 0) return 0.0;
    return a / total;
  };

  const double hit_rate = safe_ratio(hits_, misses_);
  const double overflow_rate = safe_ratio(overflows_, fills_);

  out.printf(
      "HugeCache: %zu / %zu hugepages cached / cache limit "
      "(%.3f hit rate, %.3f overflow rate)\n",
      size_.raw_num(), limit().raw_num(), hit_rate, overflow_rate);
  out.printf("HugeCache: %zu MiB fast unbacked, %zu MiB periodic\n",
             total_fast_unbacked_.in_bytes() / 1024 / 1024,
             total_periodic_unbacked_.in_bytes() / 1024 / 1024);
  UpdateSize(size());

  usage_tracker_.Report(usage_);
  const HugeLength usage_min = usage_tracker_.MinOverTime(cache_time_);
  const HugeLength usage_max = usage_tracker_.MaxOverTime(cache_time_);
  out.printf(
      "HugeCache: recent usage range: %zu min - %zu curr -  %zu max MiB\n",
      usage_min.in_mib(), usage_.in_mib(), usage_max.in_mib());

  const HugeLength off_peak = usage_max - usage_;
  off_peak_tracker_.Report(off_peak);
  const HugeLength off_peak_min = off_peak_tracker_.MinOverTime(cache_time_);
  const HugeLength off_peak_max = off_peak_tracker_.MaxOverTime(cache_time_);
  out.printf(
      "HugeCache: recent offpeak range: %zu min - %zu curr - %zu max MiB\n",
      off_peak_min.in_mib(), off_peak.in_mib(), off_peak_max.in_mib());

  const HugeLength cache_min = size_tracker_.MinOverTime(cache_time_);
  const HugeLength cache_max = size_tracker_.MaxOverTime(cache_time_);
  out.printf(
      "HugeCache: recent cache range: %zu min - %zu curr - %zu max MiB\n",
      cache_min.in_mib(), size_.in_mib(), cache_max.in_mib());

  detailed_tracker_.Print(out);

  // Release stats tracked by the demand-based release mechanism.
  out.printf("\n");
  out.printf(
      "HugeCache: Since startup, %zu hugepages released, "
      "(%zu hugepages due to reaching tcmalloc limit)\n",
      HLFromPages(hugepage_release_stats_.total_pages_subreleased).raw_num(),
      HLFromPages(hugepage_release_stats_.total_pages_subreleased_due_to_limit)
          .raw_num());

  cachestats_tracker_.Print(out, "HugeCache");
}

void HugeCache::PrintInPbtxt(PbtxtRegion& hpaa) {
  hpaa.PrintI64("huge_cache_time_const",
                absl::ToInt64Milliseconds(cache_time_));

  // a / (a + b), avoiding division by zero
  auto safe_ratio = [](double a, double b) {
    const double total = a + b;
    if (total == 0) return 0.0;
    return a / total;
  };

  const double hit_rate = safe_ratio(hits_, misses_);
  const double overflow_rate = safe_ratio(overflows_, fills_);

  // number of bytes in HugeCache
  hpaa.PrintI64("cached_huge_page_bytes", size_.in_bytes());
  // max allowed bytes in HugeCache
  hpaa.PrintI64("max_cached_huge_page_bytes", limit().in_bytes());
  // lifetime cache hit rate
  hpaa.PrintDouble("huge_cache_hit_rate", hit_rate);
  // lifetime cache overflow rate
  hpaa.PrintDouble("huge_cache_overflow_rate", overflow_rate);
  // bytes eagerly unbacked by HugeCache
  hpaa.PrintI64("fast_unbacked_bytes", total_fast_unbacked_.in_bytes());
  // bytes unbacked by periodic releaser thread
  hpaa.PrintI64("periodic_unbacked_bytes", total_periodic_unbacked_.in_bytes());
  UpdateSize(size());

  usage_tracker_.Report(usage_);
  const HugeLength usage_min = usage_tracker_.MinOverTime(cache_time_);
  const HugeLength usage_max = usage_tracker_.MaxOverTime(cache_time_);
  {
    auto usage_stats = hpaa.CreateSubRegion("huge_cache_usage_stats");
    usage_stats.PrintI64("min_bytes", usage_min.in_bytes());
    usage_stats.PrintI64("current_bytes", usage_.in_bytes());
    usage_stats.PrintI64("max_bytes", usage_max.in_bytes());
  }

  const HugeLength off_peak = usage_max - usage_;
  off_peak_tracker_.Report(off_peak);
  const HugeLength off_peak_min = off_peak_tracker_.MinOverTime(cache_time_);
  const HugeLength off_peak_max = off_peak_tracker_.MaxOverTime(cache_time_);
  {
    auto usage_stats = hpaa.CreateSubRegion("huge_cache_offpeak_stats");
    usage_stats.PrintI64("min_bytes", off_peak_min.in_bytes());
    usage_stats.PrintI64("current_bytes", off_peak.in_bytes());
    usage_stats.PrintI64("max_bytes", off_peak_max.in_bytes());
  }

  const HugeLength cache_min = size_tracker_.MinOverTime(cache_time_);
  const HugeLength cache_max = size_tracker_.MaxOverTime(cache_time_);
  {
    auto usage_stats = hpaa.CreateSubRegion("huge_cache_cache_stats");
    usage_stats.PrintI64("min_bytes", cache_min.in_bytes());
    usage_stats.PrintI64("current_bytes", size_.in_bytes());
    usage_stats.PrintI64("max_bytes", cache_max.in_bytes());
  }
  hpaa.PrintI64(
      "cache_num_hugepages_released",
      HLFromPages(hugepage_release_stats_.total_pages_subreleased).raw_num());
  hpaa.PrintI64(
      "cache_num_hugepages_released_due_to_limit",
      HLFromPages(hugepage_release_stats_.total_pages_subreleased_due_to_limit)
          .raw_num());
  detailed_tracker_.PrintInPbtxt(hpaa);
  cachestats_tracker_.PrintTimeseriesStatsInPbtxt(hpaa,
                                                  "cache_stats_timeseries");
  cachestats_tracker_.PrintSubreleaseStatsInPbtxt(hpaa,
                                                  "cache_skipped_subrelease");
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
