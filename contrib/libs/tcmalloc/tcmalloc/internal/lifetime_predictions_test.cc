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

#include "tcmalloc/internal/lifetime_predictions.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

// This pragma is needed to ensure that the loop does not get unrolled, in which
// case the different Allocate* calls would map to different call sites instead
// of the same ones as expected by this test. Supported pragmas differ between
// GCC and Clang, which is why we need this conditional.
#if (defined(__clang__) || defined(__INTEL_COMPILER))
#define PRAGMA_NO_UNROLL _Pragma("nounroll")
#elif (defined(__GNUC__) || defined(__GCUG__))
// GCC does not always respect "#pragma unroll <N>". The most reliable approach
// is therefore to completely disable optimizations for this source file.
#pragma GCC optimize ("O0")
#define PRAGMA_NO_UNROLL
#else
// If #pragma nounroll is unsupported, the test may still work by compiling with
// equivalent compiler options.
#define PRAGMA_NO_UNROLL
#endif

namespace tcmalloc {
namespace tcmalloc_internal {
namespace {

class LifetimeDatabaseTest : public testing::Test {
 protected:
  LifetimeDatabase lifetime_database_;

  ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NO_TAIL_CALL LifetimeStats*
  AllocateA() {
    LifetimeDatabase::Key key;
    return lifetime_database_.LookupOrAddLifetimeStats(&key);
  }

  ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NO_TAIL_CALL LifetimeStats*
  AllocateB() {
    LifetimeDatabase::Key key;
    return lifetime_database_.LookupOrAddLifetimeStats(&key);
  }

  ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NO_TAIL_CALL LifetimeStats*
  AllocateWithStacktraceId(int id) {
    if (id == 0) {
      LifetimeDatabase::Key key;
      return lifetime_database_.LookupOrAddLifetimeStats(&key);
    } else if (id % 2 == 0) {
      return AllocateWithStacktraceId(id / 2);
    } else {
      return AllocateWithStacktraceId_2(id / 2);
    }
  }

  // Record a sufficiently large number of short-lived allocations to make
  // a prediction short-lived, absent any long-lived allocations.
  void MakeShortLived(LifetimeStats* stats, bool high_certainty) {
    for (int i = 0; i < (high_certainty ? 100 : 2); i++) {
      stats->Update(LifetimeStats::Prediction::kShortLived);
    }
  }

 private:
  ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NO_TAIL_CALL LifetimeStats*
  AllocateWithStacktraceId_2(int id) {
    if (id == 0) {
      LifetimeDatabase::Key key;
      return lifetime_database_.LookupOrAddLifetimeStats(&key);
    } else if (id % 2 == 0) {
      return AllocateWithStacktraceId(id / 2);
    } else {
      return AllocateWithStacktraceId_2(id / 2);
    }
  }
};

TEST_F(LifetimeDatabaseTest, Basic) {
  PRAGMA_NO_UNROLL
  for (int i = 0; i < 2; i++) {
    LifetimeStats* r1 = AllocateA();
    LifetimeStats* r2 = AllocateB();
    LifetimeStats* r3 = AllocateB();
    ASSERT_NE(nullptr, r1);
    ASSERT_NE(nullptr, r2);
    ASSERT_NE(nullptr, r3);

    // First iteration: set short-lived count.
    if (i == 0) {
      MakeShortLived(r1, false);
      MakeShortLived(r2, true);
    } else {
      EXPECT_EQ(LifetimeStats::Prediction::kShortLived,
                r1->Predict(LifetimeStats::Certainty::kLowCertainty));
      EXPECT_EQ(LifetimeStats::Prediction::kLongLived,
                r1->Predict(LifetimeStats::Certainty::kHighCertainty));
      EXPECT_EQ(LifetimeStats::Prediction::kShortLived,
                r2->Predict(LifetimeStats::Certainty::kLowCertainty));
      EXPECT_EQ(LifetimeStats::Prediction::kShortLived,
                r2->Predict(LifetimeStats::Certainty::kHighCertainty));
      EXPECT_EQ(LifetimeStats::Prediction::kLongLived,
                r3->Predict(LifetimeStats::Certainty::kLowCertainty));
      EXPECT_EQ(LifetimeStats::Prediction::kLongLived,
                r3->Predict(LifetimeStats::Certainty::kHighCertainty));
    }

    lifetime_database_.RemoveLifetimeStatsReference(r1);
    lifetime_database_.RemoveLifetimeStatsReference(r2);
    lifetime_database_.RemoveLifetimeStatsReference(r3);
  }
}

TEST_F(LifetimeDatabaseTest, Eviction) {
  const int kEntries = 5 * LifetimeDatabase::kMaxDatabaseSize;

  std::vector<LifetimeStats*> refs;

  PRAGMA_NO_UNROLL
  for (int i = 0; i < kEntries; i++) {
    LifetimeStats* r = AllocateWithStacktraceId(i);
    refs.push_back(r);

    ASSERT_NE(nullptr, r);
    if (i < LifetimeDatabase::kMaxDatabaseSize) {
      MakeShortLived(r, true);
    }
  }

  // Check that even evicted entries are still accessible due to refcounts.
  for (int i = 0; i < kEntries; i++) {
    if (i < LifetimeDatabase::kMaxDatabaseSize) {
      EXPECT_EQ(LifetimeStats::Prediction::kShortLived,
                refs[i]->Predict(LifetimeStats::Certainty::kLowCertainty));
    } else {
      EXPECT_EQ(LifetimeStats::Prediction::kLongLived,
                refs[i]->Predict(LifetimeStats::Certainty::kLowCertainty));
    }
  }

  EXPECT_EQ(LifetimeDatabase::kMaxDatabaseSize, lifetime_database_.size());
  EXPECT_EQ(kEntries - LifetimeDatabase::kMaxDatabaseSize,
            lifetime_database_.evictions());

  uint64_t before_bytes = lifetime_database_.arena_stats()->bytes_allocated;

  // Return all of the references, which should drop the remaining refcounts.
  for (int i = 0; i < kEntries; i++) {
    lifetime_database_.RemoveLifetimeStatsReference(refs[i]);
  }

  uint64_t after_bytes = lifetime_database_.arena_stats()->bytes_allocated;

  // Check that this freed up memory
  EXPECT_LT(after_bytes, before_bytes);
}

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
