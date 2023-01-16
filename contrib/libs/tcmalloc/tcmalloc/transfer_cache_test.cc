// Copyright 2020 The TCMalloc Authors
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

#include "tcmalloc/transfer_cache.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/internal/spinlock.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "tcmalloc/central_freelist.h"
#include "tcmalloc/common.h"
#include "tcmalloc/mock_central_freelist.h"
#include "tcmalloc/mock_transfer_cache.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/testing/thread_manager.h"
#include "tcmalloc/transfer_cache_internals.h"

namespace tcmalloc {
namespace tcmalloc_internal {
namespace {

static constexpr int kSizeClass = 0;

template <typename Env>
using TransferCacheTest = ::testing::Test;
TYPED_TEST_SUITE_P(TransferCacheTest);

TYPED_TEST_P(TransferCacheTest, IsolatedSmoke) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;
  EXPECT_CALL(e.central_freelist(), InsertRange)
      .Times(e.transfer_cache().IsFlexible() ? 0 : 1);
  EXPECT_CALL(e.central_freelist(), RemoveRange)
      .Times(e.transfer_cache().IsFlexible() ? 0 : 1);

  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_non_batch_misses, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_misses, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_non_batch_misses, 0);

  e.Insert(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 1);
  e.Insert(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 2);
  e.Insert(batch_size - 1);
  if (e.transfer_cache().IsFlexible()) {
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 3);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 0);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_non_batch_misses, 0);
  } else {
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 2);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 1);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_non_batch_misses, 1);
  }
  e.Remove(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, 1);
  e.Remove(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, 2);
  e.Remove(batch_size - 1);
  if (e.transfer_cache().IsFlexible()) {
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, 3);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_misses, 0);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_non_batch_misses, 0);
  } else {
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, 2);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_misses, 1);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_non_batch_misses, 1);
  }
}

TYPED_TEST_P(TransferCacheTest, ReadStats) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;
  EXPECT_CALL(e.central_freelist(), InsertRange).Times(0);
  EXPECT_CALL(e.central_freelist(), RemoveRange).Times(0);

  // Ensure there is at least one insert hit/remove hit, so we can assert a
  // non-tautology in t2.
  e.Insert(batch_size);
  e.Remove(batch_size);

  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 1);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_non_batch_misses, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, 1);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_misses, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_non_batch_misses, 0);

  std::atomic<bool> stop{false};

  std::thread t1([&]() {
    while (!stop.load(std::memory_order_acquire)) {
      e.Insert(batch_size);
      e.Remove(batch_size);
    }
  });

  std::thread t2([&]() {
    while (!stop.load(std::memory_order_acquire)) {
      auto stats = e.transfer_cache().GetHitRateStats();
      CHECK_CONDITION(stats.insert_hits >= 1);
      CHECK_CONDITION(stats.insert_misses == 0);
      CHECK_CONDITION(stats.insert_non_batch_misses == 0);
      CHECK_CONDITION(stats.remove_hits >= 1);
      CHECK_CONDITION(stats.remove_misses == 0);
      CHECK_CONDITION(stats.remove_non_batch_misses == 0);
    }
  });

  absl::SleepFor(absl::Seconds(1));
  stop.store(true, std::memory_order_release);

  t1.join();
  t2.join();
}

TYPED_TEST_P(TransferCacheTest, SingleItemSmoke) {
  const int batch_size = TypeParam::kBatchSize;
  if (batch_size == 1) {
    GTEST_SKIP() << "skipping trivial batch size";
  }
  TypeParam e;
  const int actions = e.transfer_cache().IsFlexible() ? 2 : 0;
  EXPECT_CALL(e.central_freelist(), InsertRange).Times(2 - actions);
  EXPECT_CALL(e.central_freelist(), RemoveRange).Times(2 - actions);

  e.Insert(1);
  e.Insert(1);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, actions);
  e.Remove(1);
  e.Remove(1);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, actions);
}

TYPED_TEST_P(TransferCacheTest, FetchesFromFreelist) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;
  EXPECT_CALL(e.central_freelist(), InsertRange).Times(0);
  EXPECT_CALL(e.central_freelist(), RemoveRange).Times(1);
  e.Remove(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_misses, 1);
}

TYPED_TEST_P(TransferCacheTest, PartialFetchFromFreelist) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;
  EXPECT_CALL(e.central_freelist(), InsertRange).Times(0);
  EXPECT_CALL(e.central_freelist(), RemoveRange)
      .Times(2)
      .WillOnce([&](void** batch, int n) {
        int returned = static_cast<FakeCentralFreeList&>(e.central_freelist())
                           .RemoveRange(batch, std::min(batch_size / 2, n));
        // Overwrite the elements of batch that were not populated by
        // RemoveRange.
        memset(batch + returned, 0x3f, sizeof(*batch) * (n - returned));
        return returned;
      });
  e.Remove(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_misses, 2);
}

TYPED_TEST_P(TransferCacheTest, EvictsOtherCaches) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;

  EXPECT_CALL(e.transfer_cache_manager(), ShrinkCache).WillOnce([]() {
    return true;
  });
  EXPECT_CALL(e.central_freelist(), InsertRange).Times(0);

  while (e.transfer_cache().HasSpareCapacity(kSizeClass)) {
    e.Insert(batch_size);
  }
  size_t old_hits = e.transfer_cache().GetHitRateStats().insert_hits;
  e.Insert(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, old_hits + 1);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 0);
}

TYPED_TEST_P(TransferCacheTest, EvictsOtherCachesFlex) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;

  EXPECT_CALL(e.transfer_cache_manager(), ShrinkCache).WillRepeatedly([]() {
    return true;
  });
  if (e.transfer_cache().IsFlexible()) {
    EXPECT_CALL(e.central_freelist(), InsertRange).Times(0);
  } else {
    EXPECT_CALL(e.central_freelist(), InsertRange).Times(batch_size - 1);
  }
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 0);

  int total = 0;
  for (int i = 1; i <= batch_size; i++) {
    e.Insert(i);
    total += i;
  }

  if (e.transfer_cache().IsFlexible()) {
    EXPECT_EQ(e.transfer_cache().tc_length(), total);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, batch_size);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 0);
  } else {
    EXPECT_EQ(e.transfer_cache().tc_length(), 1 * batch_size);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 1);
    EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses,
              batch_size - 1);
  }
}

// Similar to EvictsOtherCachesFlex, but with full cache.
TYPED_TEST_P(TransferCacheTest, FullCacheFlex) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;

  EXPECT_CALL(e.transfer_cache_manager(), ShrinkCache).WillRepeatedly([]() {
    return true;
  });
  if (e.transfer_cache().IsFlexible()) {
    EXPECT_CALL(e.central_freelist(), InsertRange).Times(0);
  } else {
    EXPECT_CALL(e.central_freelist(), InsertRange)
        .Times(testing::AtLeast(batch_size));
  }

  while (e.transfer_cache().HasSpareCapacity(kSizeClass)) {
    e.Insert(batch_size);
  }
  for (int i = 1; i < batch_size + 2; i++) {
    e.Insert(i);
  }
}

TYPED_TEST_P(TransferCacheTest, PushesToFreelist) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;

  EXPECT_CALL(e.transfer_cache_manager(), ShrinkCache).WillOnce([]() {
    return false;
  });
  EXPECT_CALL(e.central_freelist(), InsertRange).Times(1);

  while (e.transfer_cache().HasSpareCapacity(kSizeClass)) {
    e.Insert(batch_size);
  }
  size_t old_hits = e.transfer_cache().GetHitRateStats().insert_hits;
  e.Insert(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, old_hits);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 1);
}

TYPED_TEST_P(TransferCacheTest, WrappingWorks) {
  const int batch_size = TypeParam::kBatchSize;

  TypeParam env;
  EXPECT_CALL(env.transfer_cache_manager(), ShrinkCache).Times(0);

  while (env.transfer_cache().HasSpareCapacity(kSizeClass)) {
    env.Insert(batch_size);
  }
  for (int i = 0; i < 100; ++i) {
    env.Remove(batch_size);
    env.Insert(batch_size);
  }
}

TYPED_TEST_P(TransferCacheTest, WrappingFlex) {
  const int batch_size = TypeParam::kBatchSize;

  TypeParam env;
  EXPECT_CALL(env.transfer_cache_manager(), ShrinkCache).Times(0);
  if (env.transfer_cache().IsFlexible()) {
    EXPECT_CALL(env.central_freelist(), InsertRange).Times(0);
    EXPECT_CALL(env.central_freelist(), RemoveRange).Times(0);
  }

  while (env.transfer_cache().HasSpareCapacity(kSizeClass)) {
    env.Insert(batch_size);
  }
  for (int i = 0; i < 100; ++i) {
    for (size_t size = 1; size < batch_size + 2; size++) {
      env.Remove(size);
      env.Insert(size);
    }
  }
}

TYPED_TEST_P(TransferCacheTest, Plunder) {
  TypeParam env;
  //  EXPECT_CALL(env.central_freelist(), RemoveRange).Times(0);
  //  EXPECT_CALL(env.central_freelist(), InsertRange).Times(1);
  // Fill in some elements.
  env.Insert(TypeParam::kBatchSize);
  env.Insert(TypeParam::kBatchSize);
  ASSERT_EQ(env.transfer_cache().tc_length(), 2 * TypeParam::kBatchSize);
  // All these elements will be plundered.
  env.transfer_cache().TryPlunder(kSizeClass);
  ASSERT_EQ(env.transfer_cache().tc_length(), 0);

  env.Insert(TypeParam::kBatchSize);
  env.Insert(TypeParam::kBatchSize);
  ASSERT_EQ(env.transfer_cache().tc_length(), 2 * TypeParam::kBatchSize);

  void* buf[TypeParam::kBatchSize];
  // -1 +1, this sets the low_water_mark (the lowest end-state after a
  // call to RemoveRange to 1 batch.
  (void)env.transfer_cache().RemoveRange(kSizeClass, buf,
                                         TypeParam::kBatchSize);
  env.transfer_cache().InsertRange(kSizeClass, {buf, TypeParam::kBatchSize});
  ASSERT_EQ(env.transfer_cache().tc_length(), 2 * TypeParam::kBatchSize);
  // We have one batch, and this is the same as the low water mark, so nothing
  // gets plundered.
  env.transfer_cache().TryPlunder(kSizeClass);
  ASSERT_EQ(env.transfer_cache().tc_length(), TypeParam::kBatchSize);
  // If we plunder immediately the low_water_mark is at maxint, and eveything
  // gets plundered.
  env.transfer_cache().TryPlunder(kSizeClass);
  ASSERT_EQ(env.transfer_cache().tc_length(), 0);
}

// PickCoprimeBatchSize picks a batch size in [2, max_batch_size) that is
// coprime with 2^32.  We choose the largest possible batch size within that
// constraint to minimize the number of iterations of insert/remove required.
static size_t PickCoprimeBatchSize(size_t max_batch_size) {
  while (max_batch_size > 1) {
    if ((size_t{1} << 32) % max_batch_size != 0) {
      return max_batch_size;
    }
    max_batch_size--;
  }

  return max_batch_size;
}

TEST(RingBufferTest, b172283201) {
  // This test is designed to exercise the wraparound behavior for the
  // RingBufferTransferCache, which manages its indices in uint32_t's.  Because
  // it uses a non-standard batch size (kBatchSize) as part of
  // PickCoprimeBatchSize, it triggers a TransferCache miss to the
  // CentralFreeList, which is uninteresting for exercising b/172283201.

  // For performance reasons, limit to optimized builds.
#if !defined(NDEBUG)
  GTEST_SKIP() << "skipping long running test on debug build";
#elif defined(THREAD_SANITIZER)
  // This test is single threaded, so thread sanitizer will not be useful.
  GTEST_SKIP() << "skipping under thread sanitizer, which slows test execution";
#endif

  using EnvType = FakeTransferCacheEnvironment<
      internal_transfer_cache::RingBufferTransferCache<
          MockCentralFreeList, MockTransferCacheManager>>;
  EnvType env;

  // We pick the largest value <= EnvType::kBatchSize to use as a batch size,
  // such that it is prime relative to 2^32.  This ensures that when we
  // encounter a wraparound, the last operation actually spans both ends of the
  // buffer.
  const size_t batch_size = PickCoprimeBatchSize(EnvType::kBatchSize);
  ASSERT_GT(batch_size, 0);
  ASSERT_NE((size_t{1} << 32) % batch_size, 0) << batch_size;
  // For ease of comparison, allocate a buffer of char's.  We will use these to
  // generate unique addresses.  Since we assert that we will never miss in the
  // TransferCache and go to the CentralFreeList, these do not need to be valid
  // objects for deallocation.
  std::vector<char> buffer(batch_size);
  std::vector<void*> pointers;
  pointers.reserve(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    pointers.push_back(&buffer[i]);
  }

  // To produce wraparound in the RingBufferTransferCache, we fill up the cache
  // completely and then keep inserting new elements. This makes the cache
  // return old elements to the freelist and eventually wrap around.
  EXPECT_CALL(env.central_freelist(), RemoveRange).Times(0);
  // We do return items to the freelist, don't try to actually free them.
  ON_CALL(env.central_freelist(), InsertRange).WillByDefault(testing::Return());
  ON_CALL(env.transfer_cache_manager(), DetermineSizeClassToEvict)
      .WillByDefault(testing::Return(kSizeClass));

  // First fill up the cache to its capacity.

  while (env.transfer_cache().HasSpareCapacity(kSizeClass) ||
         env.transfer_cache().GrowCache(kSizeClass)) {
    env.transfer_cache().InsertRange(kSizeClass, absl::MakeSpan(pointers));
  }

  // The current size of the transfer cache is close to its capacity. Insert
  // enough batches to make sure we wrap around twice (1 batch size should wrap
  // around as we are full currently, then insert the same amount of items
  // again, then one more wrap around).
  const size_t kObjects = env.transfer_cache().tc_length() + 2 * batch_size;

  // From now on, calls to InsertRange() should result in a corresponding call
  // to the freelist whenever the cache is full. This doesn't happen on every
  // call, as we return up to num_to_move (i.e. kBatchSize) items to the free
  // list in one batch.
  EXPECT_CALL(env.central_freelist(),
              InsertRange(testing::SizeIs(EnvType::kBatchSize)))
      .Times(testing::AnyNumber());
  for (size_t i = 0; i < kObjects; i += batch_size) {
    env.transfer_cache().InsertRange(kSizeClass, absl::MakeSpan(pointers));
  }
  // Manually drain the items in the transfercache, otherwise the destructor
  // will try to free them.
  std::vector<void*> to_free(batch_size);
  size_t N = env.transfer_cache().tc_length();
  while (N > 0) {
    const size_t to_remove = std::min(N, batch_size);
    const size_t removed =
        env.transfer_cache().RemoveRange(kSizeClass, to_free.data(), to_remove);
    ASSERT_THAT(removed, testing::Le(to_remove));
    ASSERT_THAT(removed, testing::Gt(0));
    N -= removed;
  }
  ASSERT_EQ(env.transfer_cache().tc_length(), 0);
}

REGISTER_TYPED_TEST_SUITE_P(TransferCacheTest, IsolatedSmoke, ReadStats,
                            FetchesFromFreelist, PartialFetchFromFreelist,
                            EvictsOtherCaches, PushesToFreelist, WrappingWorks,
                            SingleItemSmoke, EvictsOtherCachesFlex,
                            FullCacheFlex, WrappingFlex, Plunder);
template <typename Env>
using FuzzTest = ::testing::Test;
TYPED_TEST_SUITE_P(FuzzTest);

TYPED_TEST_P(FuzzTest, MultiThreadedUnbiased) {
  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(0.3) > absl::Now()) env.RandomlyPoke();
  threads.Stop();
}

TYPED_TEST_P(FuzzTest, MultiThreadedBiasedInsert) {
  const int batch_size = TypeParam::kBatchSize;

  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(5) > absl::Now()) env.Insert(batch_size);
  threads.Stop();
}

TYPED_TEST_P(FuzzTest, MultiThreadedBiasedRemove) {
  const int batch_size = TypeParam::kBatchSize;

  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(5) > absl::Now()) env.Remove(batch_size);
  threads.Stop();
}

TYPED_TEST_P(FuzzTest, MultiThreadedBiasedShrink) {
  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(5) > absl::Now()) env.Shrink();
  threads.Stop();
}

TYPED_TEST_P(FuzzTest, MultiThreadedBiasedGrow) {
  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(5) > absl::Now()) env.Grow();
  threads.Stop();
}

REGISTER_TYPED_TEST_SUITE_P(FuzzTest, MultiThreadedUnbiased,
                            MultiThreadedBiasedInsert,
                            MultiThreadedBiasedRemove, MultiThreadedBiasedGrow,
                            MultiThreadedBiasedShrink);

namespace unit_tests {
using Env = FakeTransferCacheEnvironment<internal_transfer_cache::TransferCache<
    MockCentralFreeList, MockTransferCacheManager>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TransferCache, TransferCacheTest,
                               ::testing::Types<Env>);

using RingBufferEnv = FakeTransferCacheEnvironment<
    internal_transfer_cache::RingBufferTransferCache<MockCentralFreeList,
                                                     MockTransferCacheManager>>;
INSTANTIATE_TYPED_TEST_SUITE_P(RingBuffer, TransferCacheTest,
                               ::testing::Types<RingBufferEnv>);
}  // namespace unit_tests

namespace fuzz_tests {
// Use the FakeCentralFreeList instead of the MockCentralFreeList for fuzz tests
// as it avoids the overheads of mocks and allows more iterations of the fuzzing
// itself.
using Env = FakeTransferCacheEnvironment<internal_transfer_cache::TransferCache<
    MockCentralFreeList, MockTransferCacheManager>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TransferCache, FuzzTest, ::testing::Types<Env>);

using RingBufferEnv = FakeTransferCacheEnvironment<
    internal_transfer_cache::RingBufferTransferCache<MockCentralFreeList,
                                                     MockTransferCacheManager>>;
INSTANTIATE_TYPED_TEST_SUITE_P(RingBuffer, FuzzTest,
                               ::testing::Types<RingBufferEnv>);
}  // namespace fuzz_tests

namespace leak_tests {

template <typename Env>
using TwoSizeClassTest = ::testing::Test;
TYPED_TEST_SUITE_P(TwoSizeClassTest);

TYPED_TEST_P(TwoSizeClassTest, NoLeaks) {
  TypeParam env;

  // The point of this test is to see that adding "random" amounts of
  // allocations to the transfer caches behaves correctly, even in the case that
  // there are multiple size classes interacting by stealing from each other.

  // Fill all caches to their maximum without starting to steal from each other.
  for (int cl = 1; cl < TypeParam::Manager::kSizeClasses; ++cl) {
    const size_t batch_size = TypeParam::Manager::num_objects_to_move(cl);
    while (env.transfer_cache_manager().HasSpareCapacity(cl)) {
      env.Insert(cl, batch_size);
    }
  }

  // Count the number of batches currently in the cache.
  auto count_batches = [&env]() {
    int batch_count = 0;
    for (int cl = 1; cl < TypeParam::Manager::kSizeClasses; ++cl) {
      const size_t batch_size = TypeParam::Manager::num_objects_to_move(cl);
      batch_count += env.transfer_cache_manager().tc_length(cl) / batch_size;
    }
    return batch_count;
  };

  absl::BitGen bitgen;
  const int max_batches = count_batches();
  int expected_batches = max_batches;
  for (int i = 0; i < 100; ++i) {
    {
      // First remove.
      const int cl =
          absl::Uniform<int>(bitgen, 1, TypeParam::Manager::kSizeClasses);
      const size_t batch_size = TypeParam::Manager::num_objects_to_move(cl);
      if (env.transfer_cache_manager().tc_length(cl) >= batch_size) {
        env.Remove(cl, batch_size);
        --expected_batches;
      }
      const int current_batches = count_batches();
      EXPECT_EQ(current_batches, expected_batches) << "iteration " << i;
    }
    {
      // Then add in another size class.
      const int cl =
          absl::Uniform<int>(bitgen, 1, TypeParam::Manager::kSizeClasses);
      // Evict from the "next" size class, skipping 0.
      // This makes sure we are always evicting from somewhere if at all
      // possible.
      env.transfer_cache_manager().evicting_from_ =
          1 + cl % (TypeParam::Manager::kSizeClasses - 1);
      if (expected_batches < max_batches) {
        const size_t batch_size = TypeParam::Manager::num_objects_to_move(cl);
        env.Insert(cl, batch_size);
        ++expected_batches;
      }
      const int current_batches = count_batches();
      EXPECT_EQ(current_batches, expected_batches) << "iteration " << i;
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(TwoSizeClassTest, NoLeaks);

using TwoTransferCacheEnv =
    TwoSizeClassEnv<internal_transfer_cache::TransferCache>;
INSTANTIATE_TYPED_TEST_SUITE_P(TransferCache, TwoSizeClassTest,
                               ::testing::Types<TwoTransferCacheEnv>);

using TwoRingBufferEnv =
    TwoSizeClassEnv<internal_transfer_cache::RingBufferTransferCache>;
INSTANTIATE_TYPED_TEST_SUITE_P(RingBuffer, TwoSizeClassTest,
                               ::testing::Types<TwoRingBufferEnv>);

}  // namespace leak_tests

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
