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

#include <stdlib.h>
#include <string.h>

#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/internal/cycleclock.h"
#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tcmalloc/huge_pages.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/stats.h"

namespace tcmalloc {
namespace tcmalloc_internal {
namespace {

class HugeCacheTest : public testing::Test {
 private:
  // Allow tests to modify the clock used by the cache.
  static int64_t clock_offset_;
  static double GetClockFrequency() {
    return absl::base_internal::CycleClock::Frequency();
  }
  static int64_t GetClock() {
    return absl::base_internal::CycleClock::Now() +
           clock_offset_ * GetClockFrequency() /
               absl::ToDoubleNanoseconds(absl::Seconds(1));
  }

  // Use a tiny fraction of actual size so we can test aggressively.
  static void* AllocateFake(size_t bytes, size_t* actual, size_t align) {
    if (bytes % kHugePageSize != 0) {
      Crash(kCrash, __FILE__, __LINE__, "not aligned", bytes, kHugePageSize);
    }
    if (align % kHugePageSize != 0) {
      Crash(kCrash, __FILE__, __LINE__, "not aligned", align, kHugePageSize);
    }
    *actual = bytes;
    // we'll actually provide hidden backing, one word per hugepage.
    bytes /= kHugePageSize;
    align /= kHugePageSize;
    size_t index = backing.size();
    if (index % align != 0) {
      index += (align - (index & align));
    }
    backing.resize(index + bytes);
    void* ptr = reinterpret_cast<void*>(index * kHugePageSize);
    return ptr;
  }
  // This isn't super good form but we'll never have more than one HAT
  // extant at once.
  static std::vector<size_t> backing;

  // We use actual malloc for metadata allocations, but we track them so they
  // can be deleted.  (TODO make this an arena if we care, which I doubt)
  static void* MallocMetadata(size_t size) {
    metadata_bytes += size;
    void* ptr = calloc(size, 1);
    metadata_allocs.push_back(ptr);
    return ptr;
  }
  static std::vector<void*> metadata_allocs;
  static size_t metadata_bytes;

  // This is wordy, but necessary for mocking:
  class BackingInterface {
   public:
    virtual void Unback(void* p, size_t len) = 0;
    virtual ~BackingInterface() {}
  };

  class MockBackingInterface : public BackingInterface {
   public:
    MOCK_METHOD2(Unback, void(void* p, size_t len));
  };

  static void MockUnback(void* p, size_t len) { mock_->Unback(p, len); }

 protected:
  static std::unique_ptr<testing::NiceMock<MockBackingInterface>> mock_;

  HugeCacheTest() {
    // We don't use the first few bytes, because things might get weird
    // given zero pointers.
    backing.resize(1024);
    metadata_bytes = 0;
    mock_ = absl::make_unique<testing::NiceMock<MockBackingInterface>>();
  }

  ~HugeCacheTest() override {
    for (void* p : metadata_allocs) {
      free(p);
    }
    metadata_allocs.clear();
    backing.clear();
    mock_.reset(nullptr);

    clock_offset_ = 0;
  }

  void Advance(absl::Duration d) {
    clock_offset_ += absl::ToInt64Nanoseconds(d);
  }

  HugeAllocator alloc_{AllocateFake, MallocMetadata};
  HugeCache cache_{&alloc_, MallocMetadata, MockUnback,
                   Clock{.now = GetClock, .freq = GetClockFrequency}};
};

std::vector<size_t> HugeCacheTest::backing;
std::vector<void*> HugeCacheTest::metadata_allocs;
size_t HugeCacheTest::metadata_bytes;
std::unique_ptr<testing::NiceMock<HugeCacheTest::MockBackingInterface>>
    HugeCacheTest::mock_;

int64_t HugeCacheTest::clock_offset_ = 0;

TEST_F(HugeCacheTest, Basic) {
  bool from;
  for (int i = 0; i < 100 * 1000; ++i) {
    cache_.Release(cache_.Get(NHugePages(1), &from));
  }
}

TEST_F(HugeCacheTest, Backing) {
  bool from;
  cache_.Release(cache_.Get(NHugePages(4), &from));
  EXPECT_TRUE(from);
  // We should be able to split up a large range...
  HugeRange r1 = cache_.Get(NHugePages(3), &from);
  EXPECT_FALSE(from);
  HugeRange r2 = cache_.Get(NHugePages(1), &from);
  EXPECT_FALSE(from);

  // and then merge it back.
  cache_.Release(r1);
  cache_.Release(r2);
  HugeRange r = cache_.Get(NHugePages(4), &from);
  EXPECT_FALSE(from);
  cache_.Release(r);
}

TEST_F(HugeCacheTest, Release) {
  bool from;
  const HugeLength one = NHugePages(1);
  cache_.Release(cache_.Get(NHugePages(5), &from));
  HugeRange r1, r2, r3, r4, r5;
  r1 = cache_.Get(one, &from);
  r2 = cache_.Get(one, &from);
  r3 = cache_.Get(one, &from);
  r4 = cache_.Get(one, &from);
  r5 = cache_.Get(one, &from);
  cache_.Release(r1);
  cache_.Release(r2);
  cache_.Release(r3);
  cache_.Release(r4);
  cache_.Release(r5);

  r1 = cache_.Get(one, &from);
  ASSERT_EQ(false, from);
  r2 = cache_.Get(one, &from);
  ASSERT_EQ(false, from);
  r3 = cache_.Get(one, &from);
  ASSERT_EQ(false, from);
  r4 = cache_.Get(one, &from);
  ASSERT_EQ(false, from);
  r5 = cache_.Get(one, &from);
  ASSERT_EQ(false, from);
  cache_.Release(r1);
  cache_.Release(r2);
  cache_.Release(r5);

  ASSERT_EQ(NHugePages(3), cache_.size());
  EXPECT_CALL(*mock_, Unback(r5.start_addr(), kHugePageSize * 1)).Times(1);
  EXPECT_EQ(NHugePages(1), cache_.ReleaseCachedPages(NHugePages(1)));
  cache_.Release(r3);
  cache_.Release(r4);

  EXPECT_CALL(*mock_, Unback(r1.start_addr(), 4 * kHugePageSize)).Times(1);
  EXPECT_EQ(NHugePages(4), cache_.ReleaseCachedPages(NHugePages(200)));
}

TEST_F(HugeCacheTest, Regret) {
  bool from;
  HugeRange r = cache_.Get(NHugePages(20), &from);
  cache_.Release(r);
  HugeLength cached = cache_.size();
  absl::Duration d = absl::Seconds(20);
  Advance(d);
  char buf[512];
  Printer out(buf, 512);
  cache_.Print(&out);  // To update the regret
  uint64_t expected_regret = absl::ToInt64Nanoseconds(d) * cached.raw_num();
  // Not exactly accurate since the mock clock advances with real time, and
  // when we measure regret will be updated.
  EXPECT_NEAR(cache_.regret(), expected_regret, expected_regret / 1000);
  EXPECT_GE(cache_.regret(), expected_regret);
}

TEST_F(HugeCacheTest, Stats) {
  bool from;
  HugeRange r = cache_.Get(NHugePages(1 + 1 + 2 + 1 + 3), &from);
  HugeRange r1, r2, r3, spacer1, spacer2;
  std::tie(r1, spacer1) = Split(r, NHugePages(1));
  std::tie(spacer1, r2) = Split(spacer1, NHugePages(1));
  std::tie(r2, spacer2) = Split(r2, NHugePages(2));
  std::tie(spacer2, r3) = Split(spacer2, NHugePages(1));
  cache_.Release(r1);
  cache_.Release(r2);
  cache_.Release(r3);

  ASSERT_EQ(NHugePages(6), cache_.size());
  r1 = cache_.Get(NHugePages(1), &from);
  ASSERT_EQ(false, from);
  r2 = cache_.Get(NHugePages(2), &from);
  ASSERT_EQ(false, from);
  r3 = cache_.Get(NHugePages(3), &from);
  ASSERT_EQ(false, from);

  struct Helper {
    static void Stat(const HugeCache& cache, size_t* spans,
                     Length* pages_backed, Length* pages_unbacked,
                     double* avg_age) {
      PageAgeHistograms ages(absl::base_internal::CycleClock::Now());
      LargeSpanStats large;
      cache.AddSpanStats(nullptr, &large, &ages);

      const PageAgeHistograms::Histogram* hist = ages.GetTotalHistogram(false);
      *spans = large.spans;
      *pages_backed = large.normal_pages;
      *pages_unbacked = large.returned_pages;
      *avg_age = hist->avg_age();
    }
  };

  double avg_age;
  size_t spans;
  Length pages_backed;
  Length pages_unbacked;

  cache_.Release(r1);
  absl::SleepFor(absl::Microseconds(5000));
  Helper::Stat(cache_, &spans, &pages_backed, &pages_unbacked, &avg_age);
  EXPECT_EQ(Length(0), pages_unbacked);
  EXPECT_EQ(1, spans);
  EXPECT_EQ(NHugePages(1).in_pages(), pages_backed);
  EXPECT_LE(0.005, avg_age);

  cache_.Release(r2);
  absl::SleepFor(absl::Microseconds(2500));
  Helper::Stat(cache_, &spans, &pages_backed, &pages_unbacked, &avg_age);
  EXPECT_EQ(Length(0), pages_unbacked);
  EXPECT_EQ(2, spans);
  EXPECT_EQ(NHugePages(3).in_pages(), pages_backed);
  EXPECT_LE((0.0075 * 1 + 0.0025 * 2) / (1 + 2), avg_age);

  cache_.Release(r3);
  absl::SleepFor(absl::Microseconds(1250));
  Helper::Stat(cache_, &spans, &pages_backed, &pages_unbacked, &avg_age);
  EXPECT_EQ(Length(0), pages_unbacked);
  EXPECT_EQ(3, spans);
  EXPECT_EQ(NHugePages(6).in_pages(), pages_backed);
  EXPECT_LE((0.00875 * 1 + 0.00375 * 2 + 0.00125 * 3) / (1 + 2 + 3), avg_age);
}

static double Frac(HugeLength num, HugeLength denom) {
  return static_cast<double>(num.raw_num()) / denom.raw_num();
}

TEST_F(HugeCacheTest, Growth) {
  bool released;
  absl::BitGen rng;
  // fragmentation is a bit of a challenge
  std::uniform_int_distribution<size_t> sizes(1, 5);
  // fragment the cache badly.
  std::vector<HugeRange> keep;
  std::vector<HugeRange> drop;
  for (int i = 0; i < 1000; ++i) {
    auto& l = std::bernoulli_distribution()(rng) ? keep : drop;
    l.push_back(cache_.Get(NHugePages(sizes(rng)), &released));
  }

  for (auto r : drop) {
    cache_.Release(r);
  }

  // See the TODO in HugeCache::MaybeGrowCache; without this delay,
  // the above fragmentation plays merry havoc with our instrumentation.
  Advance(absl::Seconds(30));

  // Test that our cache can grow to fit a working set.
  HugeLength hot_set_sizes[] = {NHugePages(5), NHugePages(10), NHugePages(100),
                                NHugePages(10000)};

  for (const HugeLength hot : hot_set_sizes) {
    SCOPED_TRACE(absl::StrCat("cache size = ", hot.in_bytes() / 1024.0 / 1024.0,
                              " MiB"));
    // Exercise the cache allocating about <hot> worth of data. After
    // a brief warmup phase, we should do this without needing to back much.
    auto alloc = [&]() -> std::pair<HugeLength, HugeLength> {
      HugeLength got = NHugePages(0);
      HugeLength needed_backing = NHugePages(0);
      std::vector<HugeRange> items;
      while (got < hot) {
        HugeLength rest = hot - got;
        HugeLength l = std::min(rest, NHugePages(sizes(rng)));
        got += l;
        items.push_back(cache_.Get(l, &released));
        if (released) needed_backing += l;
      }
      for (auto r : items) {
        cache_.Release(r);
      }
      return {needed_backing, got};
    };

    // warmup - we're allowed to incur misses and be too big.
    for (int i = 0; i < 2; ++i) {
      alloc();
    }

    HugeLength needed_backing = NHugePages(0);
    HugeLength total = NHugePages(0);
    for (int i = 0; i < 16; ++i) {
      auto r = alloc();
      needed_backing += r.first;
      total += r.second;
      // Cache shouldn't have just grown arbitrarily
      const HugeLength cached = cache_.size();
      // Allow us 10% slop, but don't get out of bed for tiny caches anyway.
      const double ratio = Frac(cached, hot);
      SCOPED_TRACE(
          absl::StrCat(cached.raw_num(), "hps ", Frac(r.first, r.second)));
      if (ratio > 1 && cached > NHugePages(16)) {
        EXPECT_LE(ratio, 1.1);
      }
    }
    // approximately, given the randomized sizing...

    const double ratio = Frac(needed_backing, total);
    EXPECT_LE(ratio, 0.2);
  }
}

// If we repeatedly grow and shrink, but do so very slowly, we should *not*
// cache the large variation.
TEST_F(HugeCacheTest, SlowGrowthUncached) {
  absl::BitGen rng;
  std::uniform_int_distribution<size_t> sizes(1, 10);
  for (int i = 0; i < 20; ++i) {
    std::vector<HugeRange> rs;
    for (int j = 0; j < 20; ++j) {
      Advance(absl::Milliseconds(600));
      bool released;
      rs.push_back(cache_.Get(NHugePages(sizes(rng)), &released));
    }
    HugeLength max_cached = NHugePages(0);
    for (auto r : rs) {
      Advance(absl::Milliseconds(600));
      cache_.Release(r);
      max_cached = std::max(max_cached, cache_.size());
    }
    EXPECT_GE(NHugePages(10), max_cached);
  }
}

// If very rarely we have a huge increase in usage, it shouldn't be cached.
TEST_F(HugeCacheTest, SpikesUncached) {
  absl::BitGen rng;
  std::uniform_int_distribution<size_t> sizes(1, 10);
  for (int i = 0; i < 20; ++i) {
    std::vector<HugeRange> rs;
    for (int j = 0; j < 2000; ++j) {
      bool released;
      rs.push_back(cache_.Get(NHugePages(sizes(rng)), &released));
    }
    HugeLength max_cached = NHugePages(0);
    for (auto r : rs) {
      cache_.Release(r);
      max_cached = std::max(max_cached, cache_.size());
    }
    EXPECT_GE(NHugePages(10), max_cached);
    Advance(absl::Seconds(30));
  }
}

// If very rarely we have a huge *decrease* in usage, it *should* be cached.
TEST_F(HugeCacheTest, DipsCached) {
  absl::BitGen rng;
  std::uniform_int_distribution<size_t> sizes(1, 10);
  for (int i = 0; i < 20; ++i) {
    std::vector<HugeRange> rs;
    HugeLength got = NHugePages(0);
    HugeLength uncached = NHugePages(0);
    for (int j = 0; j < 2000; ++j) {
      bool released;
      HugeLength n = NHugePages(sizes(rng));
      rs.push_back(cache_.Get(n, &released));
      got += n;
      if (released) uncached += n;
    }
    // Most of our time is at high usage...
    Advance(absl::Seconds(30));
    // Now immediately release and reallocate.
    for (auto r : rs) {
      cache_.Release(r);
    }

    // warmup
    if (i >= 2) {
      EXPECT_GE(0.06, Frac(uncached, got));
    }
  }
}

// Suppose in a previous era of behavior we needed a giant cache,
// but now we don't.  Do we figure this out promptly?
TEST_F(HugeCacheTest, Shrink) {
  absl::BitGen rng;
  std::uniform_int_distribution<size_t> sizes(1, 10);
  for (int i = 0; i < 20; ++i) {
    std::vector<HugeRange> rs;
    for (int j = 0; j < 2000; ++j) {
      HugeLength n = NHugePages(sizes(rng));
      bool released;
      rs.push_back(cache_.Get(n, &released));
    }
    for (auto r : rs) {
      cache_.Release(r);
    }
  }

  ASSERT_LE(NHugePages(10000), cache_.size());

  for (int i = 0; i < 30; ++i) {
    // New working set <= 20 pages.
    Advance(absl::Seconds(1));

    // And do some work.
    for (int j = 0; j < 100; ++j) {
      bool released;
      HugeRange r1 = cache_.Get(NHugePages(sizes(rng)), &released);
      HugeRange r2 = cache_.Get(NHugePages(sizes(rng)), &released);
      cache_.Release(r1);
      cache_.Release(r2);
    }
  }

  ASSERT_GE(NHugePages(25), cache_.limit());
}

TEST_F(HugeCacheTest, Usage) {
  bool released;

  auto r1 = cache_.Get(NHugePages(10), &released);
  EXPECT_EQ(NHugePages(10), cache_.usage());

  auto r2 = cache_.Get(NHugePages(100), &released);
  EXPECT_EQ(NHugePages(110), cache_.usage());

  cache_.Release(r1);
  EXPECT_EQ(NHugePages(100), cache_.usage());

  // Pretend we unbacked this.
  cache_.ReleaseUnbacked(r2);
  EXPECT_EQ(NHugePages(0), cache_.usage());
}

class MinMaxTrackerTest : public testing::Test {
 protected:
  void Advance(absl::Duration d) {
    clock_ += absl::ToDoubleSeconds(d) * GetFakeClockFrequency();
  }

  static int64_t FakeClock() { return clock_; }

  static double GetFakeClockFrequency() {
    return absl::ToDoubleNanoseconds(absl::Seconds(2));
  }

 private:
  static int64_t clock_;
};

int64_t MinMaxTrackerTest::clock_{0};

TEST_F(MinMaxTrackerTest, Works) {
  const absl::Duration kDuration = absl::Seconds(2);
  MinMaxTracker<> tracker{
      Clock{.now = FakeClock, .freq = GetFakeClockFrequency}, kDuration};

  tracker.Report(NHugePages(0));
  EXPECT_EQ(NHugePages(0), tracker.MaxOverTime(kDuration));
  EXPECT_EQ(NHugePages(0), tracker.MinOverTime(kDuration));

  tracker.Report(NHugePages(10));
  EXPECT_EQ(NHugePages(10), tracker.MaxOverTime(kDuration));
  EXPECT_EQ(NHugePages(0), tracker.MinOverTime(kDuration));

  tracker.Report(NHugePages(5));
  EXPECT_EQ(NHugePages(10), tracker.MaxOverTime(kDuration));
  EXPECT_EQ(NHugePages(0), tracker.MinOverTime(kDuration));

  tracker.Report(NHugePages(100));
  EXPECT_EQ(NHugePages(100), tracker.MaxOverTime(kDuration));
  EXPECT_EQ(NHugePages(0), tracker.MinOverTime(kDuration));

  // Some tests for advancing time
  Advance(kDuration / 3);
  tracker.Report(NHugePages(2));
  EXPECT_EQ(NHugePages(2), tracker.MaxOverTime(absl::Nanoseconds(1)));
  EXPECT_EQ(NHugePages(100), tracker.MaxOverTime(kDuration / 2));
  EXPECT_EQ(NHugePages(100), tracker.MaxOverTime(kDuration));
  EXPECT_EQ(NHugePages(2), tracker.MinOverTime(absl::Nanoseconds(1)));
  EXPECT_EQ(NHugePages(0), tracker.MinOverTime(kDuration / 2));
  EXPECT_EQ(NHugePages(0), tracker.MinOverTime(kDuration));

  Advance(kDuration / 3);
  tracker.Report(NHugePages(5));
  EXPECT_EQ(NHugePages(5), tracker.MaxOverTime(absl::Nanoseconds(1)));
  EXPECT_EQ(NHugePages(5), tracker.MaxOverTime(kDuration / 2));
  EXPECT_EQ(NHugePages(100), tracker.MaxOverTime(kDuration));
  EXPECT_EQ(NHugePages(5), tracker.MinOverTime(absl::Nanoseconds(1)));
  EXPECT_EQ(NHugePages(2), tracker.MinOverTime(kDuration / 2));
  EXPECT_EQ(NHugePages(0), tracker.MinOverTime(kDuration));

  // This should annihilate everything.
  Advance(kDuration * 2);
  tracker.Report(NHugePages(1));
  EXPECT_EQ(NHugePages(1), tracker.MaxOverTime(absl::Nanoseconds(1)));
  EXPECT_EQ(NHugePages(1), tracker.MinOverTime(absl::Nanoseconds(1)));
  EXPECT_EQ(NHugePages(1), tracker.MaxOverTime(kDuration));
  EXPECT_EQ(NHugePages(1), tracker.MinOverTime(kDuration));
}

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
