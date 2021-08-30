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

#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <set>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/blocking_counter.h"
#include "tcmalloc/internal/declarations.h"
#include "tcmalloc/internal/linked_list.h"
#include "tcmalloc/malloc_extension.h"
#include "tcmalloc/testing/testutil.h"

namespace tcmalloc {
namespace {

TEST(AllocationSampleTest, TokenAbuse) {
  auto token = MallocExtension::StartAllocationProfiling();
  ::operator delete(::operator new(512 * 1024 * 1024));
  // Repeated Claims should happily return null.
  auto profile = std::move(token).Stop();
  int count = 0;
  profile.Iterate([&](const Profile::Sample &) { count++; });
  EXPECT_EQ(count, 1);

  auto profile2 = std::move(token).Stop();  // NOLINT: use-after-move intended
  int count2 = 0;
  profile2.Iterate([&](const Profile::Sample &) { count2++; });
  EXPECT_EQ(count2, 0);

  // Delete (on the scope ending) without Claim should also be OK.
  { MallocExtension::StartAllocationProfiling(); }
}

// Verify that profiling sessions concurrent with allocations do not crash due
// to mutating pointers accessed by the sampling code (b/143623146).
TEST(AllocationSampleTest, RaceToClaim) {
  MallocExtension::SetProfileSamplingRate(1 << 14);

  absl::BlockingCounter counter(2);
  std::atomic<bool> stop{false};

  std::thread t1([&]() {
    counter.DecrementCount();

    while (!stop) {
      auto token = MallocExtension::StartAllocationProfiling();
      absl::SleepFor(absl::Microseconds(1));
      auto profile = std::move(token).Stop();
    }
  });

  std::thread t2([&]() {
    counter.DecrementCount();

    const int kNum = 1000000;
    std::vector<void *> ptrs;
    while (!stop) {
      for (int i = 0; i < kNum; i++) {
        ptrs.push_back(::operator new(1));
      }
      for (void *p : ptrs) {
        sized_delete(p, 1);
      }
      ptrs.clear();
    }
  });

  // Verify the threads are up and running before we start the clock.
  counter.Wait();

  absl::SleepFor(absl::Seconds(1));

  stop.store(true);

  t1.join();
  t2.join();
}

TEST(AllocationSampleTest, SampleAccuracy) {
  // Disable GWP-ASan, since it allocates different sizes than normal samples.
  MallocExtension::SetGuardedSamplingRate(-1);

  // Allocate about 512 MiB each of various sizes. For _some_ but not all
  // sizes, delete it as we go--it shouldn't matter for the sample count.
  static const size_t kTotalPerSize = 512 * 1024 * 1024;

  // objects we don't delete as we go
  void *list = nullptr;

  // (object size, object alignment, keep objects)
  struct Requests {
    size_t size;
    size_t alignment;
    bool keep;
  };
  std::vector<Requests> sizes = {
      {8, 0, false},          {16, 16, true},        {1024, 0, false},
      {64 * 1024, 64, false}, {512 * 1024, 0, true}, {1024 * 1024, 128, true}};
  std::set<size_t> sizes_expected;
  for (auto s : sizes) {
    sizes_expected.insert(s.size);
  }
  auto token = MallocExtension::StartAllocationProfiling();

  // We use new/delete to allocate memory, as malloc returns objects aligned to
  // std::max_align_t.
  for (auto s : sizes) {
    for (size_t bytes = 0; bytes < kTotalPerSize; bytes += s.size) {
      void *obj;
      if (s.alignment > 0) {
        obj = operator new(s.size, static_cast<std::align_val_t>(s.alignment));
      } else {
        obj = operator new(s.size);
      }
      if (s.keep) {
        tcmalloc::SLL_Push(&list, obj);
      } else {
        operator delete(obj);
      }
    }
  }
  auto profile = std::move(token).Stop();

  // size -> bytes seen
  absl::flat_hash_map<size_t, size_t> m;

  // size -> alignment request
  absl::flat_hash_map<size_t, size_t> alignment;

  for (auto s : sizes) {
    alignment[s.size] = s.alignment;
  }

  profile.Iterate([&](const tcmalloc::Profile::Sample &e) {
    // Don't check stack traces until we have evidence that's broken, it's
    // tedious and done fairly well elsewhere.
    m[e.allocated_size] += e.sum;
    EXPECT_EQ(alignment[e.requested_size], e.requested_alignment);
  });

  size_t max_bytes = 0, min_bytes = std::numeric_limits<size_t>::max();
  EXPECT_EQ(m.size(), sizes_expected.size());
  for (auto seen : m) {
    size_t size = seen.first;
    EXPECT_TRUE(sizes_expected.find(size) != sizes_expected.end()) << size;
    size_t bytes = seen.second;
    min_bytes = std::min(min_bytes, bytes);
    max_bytes = std::max(max_bytes, bytes);
  }
  // Hopefully we're in a fairly small range, that contains our actual
  // allocation.
  // TODO(b/134690164): better statistical tests here.
  EXPECT_GE((min_bytes * 3) / 2, max_bytes);
  EXPECT_LE((min_bytes * 3) / 4, kTotalPerSize);
  EXPECT_LE(kTotalPerSize, (max_bytes * 4) / 3);
  // Remove the objects we left alive
  while (list != nullptr) {
    void *obj = tcmalloc::SLL_Pop(&list);
    operator delete(obj);
  }
}

TEST(FragmentationzTest, Accuracy) {
  // Disable GWP-ASan, since it allocates different sizes than normal samples.
  MallocExtension::SetGuardedSamplingRate(-1);

  // a fairly odd allocation size - will be rounded to 128.  This lets
  // us find our record in the table.
  static const size_t kItemSize = 115;
  // allocate about 3.5 GiB:
  static const size_t kNumItems = 32 * 1024 * 1024;

  std::vector<std::unique_ptr<char[]>> keep;
  std::vector<std::unique_ptr<char[]>> drop;
  // hint expected sizes:
  drop.reserve(kNumItems * 8 / 10);
  keep.reserve(kNumItems * 2 / 10);

  // We allocate many items, then free 80% of them "randomly". (To
  // decrease noise and speed up, we just keep every 5th one exactly.)
  for (int i = 0; i < kNumItems; ++i) {
    // Ideally we should use a malloc() here, for consistency; but unique_ptr
    // doesn't come with a have a "free()" deleter; use ::operator new insted.
    (i % 5 == 0 ? keep : drop)
        .push_back(std::unique_ptr<char[]>(
            static_cast<char *>(::operator new[](kItemSize))));
  }
  drop.resize(0);

  // there are at least 64 items per span here. (8/10)^64 = 6.2e-7 ~= 0
  // probability we actually managed to free a page; every page is fragmented.
  // We still have 20% or so of it allocated, so we should see 80% of it
  // charged to these allocations as fragmentations.
  auto profile = MallocExtension::SnapshotCurrent(ProfileType::kFragmentation);

  // Pull out the fragmentationz entry corresponding to this
  size_t requested_size = 0;
  size_t allocated_size = 0;
  size_t sum = 0;
  size_t count = 0;
  profile.Iterate([&](const Profile::Sample &e) {
    if (e.requested_size != kItemSize) return;

    if (requested_size == 0) {
      allocated_size = e.allocated_size;
      requested_size = e.requested_size;
    } else {
      // we will usually have single entry in
      // profile, but in builds without optimization
      // our fast-path code causes same call-site to
      // have two different stack traces. Thus we
      // expect and deal with second entry for same
      // allocation.
      EXPECT_EQ(requested_size, e.requested_size);
      EXPECT_EQ(allocated_size, e.allocated_size);
    }
    sum += e.sum;
    count += e.count;
  });

  double frag_bytes = sum;
  double real_frag_bytes =
      static_cast<double>(allocated_size * kNumItems) * 0.8;
  // We should be pretty close with this much data:
  // TODO(b/134690164): this is still slightly flaky (<1%) - why?
  EXPECT_NEAR(real_frag_bytes, frag_bytes, real_frag_bytes * 0.15)
      << " sum = " << sum << " allocated = " << allocated_size
      << " requested = " << requested_size << " count = " << count;
}

}  // namespace
}  // namespace tcmalloc
