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
namespace {

template <typename Env>
using TransferCacheTest = ::testing::Test;
TYPED_TEST_SUITE_P(TransferCacheTest);

TYPED_TEST_P(TransferCacheTest, IsolatedSmoke) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;
  EXPECT_CALL(e.central_freelist(), InsertRange).Times(0);
  EXPECT_CALL(e.central_freelist(), RemoveRange).Times(0);

  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, 0);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_misses, 0);

  e.Insert(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 1);
  e.Insert(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, 2);
  e.Remove(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, 1);
  e.Remove(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().remove_hits, 2);
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

  while (e.transfer_cache().HasSpareCapacity()) {
    e.Insert(batch_size);
  }
  size_t old_hits = e.transfer_cache().GetHitRateStats().insert_hits;
  e.Insert(batch_size);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_hits, old_hits + 1);
  EXPECT_EQ(e.transfer_cache().GetHitRateStats().insert_misses, 0);
}

TYPED_TEST_P(TransferCacheTest, PushesToFreelist) {
  const int batch_size = TypeParam::kBatchSize;
  TypeParam e;

  EXPECT_CALL(e.transfer_cache_manager(), ShrinkCache).WillOnce([]() {
    return false;
  });
  EXPECT_CALL(e.central_freelist(), InsertRange).Times(1);

  while (e.transfer_cache().HasSpareCapacity()) {
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

  while (env.transfer_cache().HasSpareCapacity()) {
    env.Insert(batch_size);
  }
  for (int i = 0; i < 100; ++i) {
    env.Remove(batch_size);
    env.Insert(batch_size);
  }
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

TEST(LockTransferCacheTest, b172283201) {
  // This test is designed to exercise the wraparound behavior for the
  // LockFreeTransferCache, which manages its indices in uint32_t's.

  // For performance reasons, limit to optimized builds.
#if !defined(NDEBUG)
  GTEST_SKIP() << "skipping long running test on debug build";
#elif defined(THREAD_SANITIZER)
  // This test is single threaded, so thread sanitizer will not be useful.
  GTEST_SKIP() << "skipping under thread sanitizer, which slows test execution";
#endif

  using EnvType = FakeTransferCacheEnvironment<
      internal_transfer_cache::LockFreeTransferCache<MockCentralFreeList,
                                                     MockTransferCacheManager>>;
  EnvType env;

  // We pick the largest value <= EnvType::kBatchSize to use as a batch size,
  // such that it is prime relative to 2^32.  This ensures that when we
  // encounter a wraparound, the last operation actually spans both ends of the
  // buffer.
  const size_t batch_size = PickCoprimeBatchSize(EnvType::kBatchSize);
  ASSERT_GT(batch_size, 0);
  ASSERT_NE((size_t{1} << 32) % batch_size, 0) << batch_size;
  const size_t kObjects =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) +
      2 * batch_size;

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

  EXPECT_CALL(env.central_freelist(), InsertRange).Times(0);
  EXPECT_CALL(env.central_freelist(), RemoveRange).Times(0);

  for (size_t i = 0; i < kObjects; i += batch_size) {
    env.transfer_cache().InsertRange(absl::MakeSpan(pointers), batch_size);

    ASSERT_EQ(env.transfer_cache().tc_length(), batch_size);

    void* out[kMaxObjectsToMove];
    int out_count = env.transfer_cache().RemoveRange(out, batch_size);
    ASSERT_EQ(out_count, batch_size);

    std::sort(out, out + out_count);
    // Provide an optimized fast path for checking the returned pointers match
    // the inserted pointers.  As discussed in b/172507506, this optimization
    // reduces the runtime of this test by a factor of 10x.
    if (memcmp(out, &pointers[0], sizeof(*out) * out_count) != 0) {
      ASSERT_THAT(pointers, testing::ElementsAreArray(out, out + out_count));
    }
    ASSERT_EQ(env.transfer_cache().tc_length(), 0);
  }
}

REGISTER_TYPED_TEST_SUITE_P(TransferCacheTest, IsolatedSmoke,
                            FetchesFromFreelist, PartialFetchFromFreelist,
                            EvictsOtherCaches, PushesToFreelist, WrappingWorks);
template <typename Env>
using TransferCacheFuzzTest = ::testing::Test;
TYPED_TEST_SUITE_P(TransferCacheFuzzTest);

TYPED_TEST_P(TransferCacheFuzzTest, MultiThreadedUnbiased) {
  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(0.3) > absl::Now()) env.RandomlyPoke();
  threads.Stop();
}

TYPED_TEST_P(TransferCacheFuzzTest, MultiThreadedBiasedInsert) {
  const int batch_size = TypeParam::kBatchSize;

  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(5) > absl::Now()) env.Insert(batch_size);
  threads.Stop();
}

TYPED_TEST_P(TransferCacheFuzzTest, MultiThreadedBiasedRemove) {
  const int batch_size = TypeParam::kBatchSize;

  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(5) > absl::Now()) env.Remove(batch_size);
  threads.Stop();
}

TYPED_TEST_P(TransferCacheFuzzTest, MultiThreadedBiasedShrink) {
  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(5) > absl::Now()) env.Shrink();
  threads.Stop();
}

TYPED_TEST_P(TransferCacheFuzzTest, MultiThreadedBiasedGrow) {
  TypeParam env;
  ThreadManager threads;
  threads.Start(10, [&](int) { env.RandomlyPoke(); });

  auto start = absl::Now();
  while (start + absl::Seconds(5) > absl::Now()) env.Grow();
  threads.Stop();
}

REGISTER_TYPED_TEST_SUITE_P(TransferCacheFuzzTest, MultiThreadedUnbiased,
                            MultiThreadedBiasedInsert,
                            MultiThreadedBiasedRemove, MultiThreadedBiasedGrow,
                            MultiThreadedBiasedShrink);

namespace unit_tests {
using LegacyEnv =
    FakeTransferCacheEnvironment<internal_transfer_cache::TransferCache<
        MockCentralFreeList, MockTransferCacheManager>>;

using LockFreeEnv =
    FakeTransferCacheEnvironment<internal_transfer_cache::LockFreeTransferCache<
        MockCentralFreeList, MockTransferCacheManager>>;

using TransferCacheTypes = ::testing::Types<LegacyEnv, LockFreeEnv>;
INSTANTIATE_TYPED_TEST_SUITE_P(TransferCacheTest, TransferCacheTest,
                               TransferCacheTypes);
}  // namespace unit_tests

namespace fuzz_tests {
// Use the FakeCentralFreeList instead of the MockCentralFreeList for fuzz tests
// as it avoids the overheads of mocks and allows more iterations of the fuzzing
// itself.
using LegacyEnv =
    FakeTransferCacheEnvironment<internal_transfer_cache::TransferCache<
        MockCentralFreeList, MockTransferCacheManager>>;

using LockFreeEnv =
    FakeTransferCacheEnvironment<internal_transfer_cache::LockFreeTransferCache<
        FakeCentralFreeList, MockTransferCacheManager>>;

using TransferCacheFuzzTypes = ::testing::Types<LegacyEnv, LockFreeEnv>;
INSTANTIATE_TYPED_TEST_SUITE_P(TransferCacheFuzzTest, TransferCacheFuzzTest,
                               TransferCacheFuzzTypes);
}  // namespace fuzz_tests

}  // namespace
}  // namespace tcmalloc
