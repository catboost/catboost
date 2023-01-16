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

#include "tcmalloc/stack_trace_table.h"

#include <stddef.h>

#include <algorithm>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/attributes.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/macros.h"
#include "absl/debugging/stacktrace.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/static_vars.h"

namespace tcmalloc {
namespace tcmalloc_internal {
namespace {

// Rather than deal with heap allocating stack/tags, AllocationEntry contains
// them inline.
struct AllocationEntry {
  int64_t sum;
  int count;
  size_t requested_size;
  size_t requested_alignment;
  size_t allocated_size;
  int depth;
  void* stack[64];

  friend bool operator==(const AllocationEntry& x, const AllocationEntry& y);
  friend bool operator!=(const AllocationEntry& x, const AllocationEntry& y) {
    return !(x == y);
  }

  friend std::ostream& operator<<(std::ostream& os, const AllocationEntry& e) {
    os << "sum = " << e.sum << "; ";
    os << "count = " << e.count << "; ";

    std::vector<std::string> ptrs;
    for (int i = 0; i < e.depth; i++) {
      ptrs.push_back(absl::StrFormat("%p", e.stack[i]));
    }
    os << "stack = [" << absl::StrJoin(ptrs, ", ") << "]; ";

    os << "requested_size = " << e.requested_size << "; ";
    os << "requested_alignment = " << e.requested_alignment << "; ";
    os << "allocated_size = " << e.allocated_size << "; ";
    return os;
  }
};

inline bool operator==(const AllocationEntry& x, const AllocationEntry& y) {
  if (x.sum != y.sum) {
    return false;
  }

  if (x.count != y.count) {
    return false;
  }

  if (x.depth != y.depth) {
    return false;
  }

  if (x.depth > 0 && !std::equal(x.stack, x.stack + x.depth, y.stack)) {
    return false;
  }

  if (x.requested_size != y.requested_size) {
    return false;
  }

  if (x.requested_alignment != y.requested_alignment) {
    return false;
  }

  if (x.allocated_size != y.allocated_size) {
    return false;
  }

  return true;
}

void CheckTraces(const StackTraceTable& table,
                 std::initializer_list<AllocationEntry> expected) {
  std::vector<AllocationEntry> actual;

  table.Iterate([&](const Profile::Sample& e) {
    AllocationEntry tmp;
    tmp.sum = e.sum;
    tmp.count = e.count;
    tmp.depth = e.depth;
    ASSERT_LE(tmp.depth, ABSL_ARRAYSIZE(tmp.stack));
    std::copy(e.stack, e.stack + e.depth, tmp.stack);

    tmp.requested_size = e.requested_size;
    tmp.requested_alignment = e.requested_alignment;
    tmp.allocated_size = e.allocated_size;

    actual.push_back(tmp);
  });

  EXPECT_THAT(actual, testing::UnorderedElementsAreArray(expected));
}

void AddTrace(StackTraceTable* table, double count, const StackTrace& t) {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  table->AddTrace(count, t);
}

TEST(StackTraceTableTest, StackTraceTable) {
  // If this test is not linked against TCMalloc, the global arena used for
  // StackTraceTable's buckets will not be initialized.
  Static::InitIfNecessary();

  // Empty table
  {
    SCOPED_TRACE("empty");

    StackTraceTable table(ProfileType::kHeap, 1, true, false);
    EXPECT_EQ(0, table.depth_total());
    EXPECT_EQ(0, table.bucket_total());

    CheckTraces(table, {});
  }

  StackTrace t1 = {};
  t1.requested_size = static_cast<uintptr_t>(512);
  t1.requested_alignment = static_cast<uintptr_t>(16);
  t1.allocated_size = static_cast<uintptr_t>(1024);
  t1.depth = static_cast<uintptr_t>(2);
  t1.stack[0] = reinterpret_cast<void*>(1);
  t1.stack[1] = reinterpret_cast<void*>(2);
  t1.weight = 2 << 20;

  const AllocationEntry k1 = {
      1024,
      1,
      512,
      16,
      1024,
      2,
      {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2)},
  };

  StackTrace t2 = {};
  t2.requested_size = static_cast<uintptr_t>(375);
  t2.requested_alignment = static_cast<uintptr_t>(0);
  t2.allocated_size = static_cast<uintptr_t>(512);
  t2.depth = static_cast<uintptr_t>(2);
  t2.stack[0] = reinterpret_cast<void*>(2);
  t2.stack[1] = reinterpret_cast<void*>(1);
  t2.weight = 1;

  const AllocationEntry k2 = {
      512,
      1,
      375,
      0,
      512,
      2,
      {reinterpret_cast<void*>(2), reinterpret_cast<void*>(1)},
  };

  // Table w/ just t1
  {
    SCOPED_TRACE("t1");

    StackTraceTable table(ProfileType::kHeap, 1, true, false);
    AddTrace(&table, 1.0, t1);
    EXPECT_EQ(2, table.depth_total());
    EXPECT_EQ(1, table.bucket_total());

    CheckTraces(table, {k1});
  }

  // We made our last sample at t1.weight (2<<20 bytes).  We sample according to
  // t1.requested_size + 1 (513 bytes).  Therefore we overweight the sample to
  // construct the distribution.
  //
  // We rely on the profiling tests to verify that this correctly reconstructs
  // the distribution (+/- an error tolerance)
  const int t1_sampled_weight =
      static_cast<double>(t1.weight) / (t1.requested_size + 1);
  ASSERT_EQ(t1_sampled_weight, 4088);
  const AllocationEntry k1_unsampled = {
      t1_sampled_weight * 1024,
      t1_sampled_weight,
      512,
      16,
      1024,
      2,
      {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2)},
  };

  // Table w/ just t1 (unsampled)
  {
    SCOPED_TRACE("t1 unsampled");

    StackTraceTable table(ProfileType::kHeap, 1, true, true);
    AddTrace(&table, 1.0, t1);
    EXPECT_EQ(2, table.depth_total());
    EXPECT_EQ(1, table.bucket_total());

    CheckTraces(table, {k1_unsampled});
  }

  const AllocationEntry k1_merged = {
      2048,
      2,
      512,
      16,
      1024,
      2,
      {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2)},
  };

  // Table w/ 2x t1 (merge)
  {
    SCOPED_TRACE("2x t1 merge");

    StackTraceTable table(ProfileType::kHeap, 1, true, false);
    AddTrace(&table, 1.0, t1);
    AddTrace(&table, 1.0, t1);
    EXPECT_EQ(2, table.depth_total());
    EXPECT_EQ(1, table.bucket_total());

    CheckTraces(table, {k1_merged});
  }

  // Table w/ 2x t1 (no merge)
  {
    SCOPED_TRACE("2x t1 no merge");

    StackTraceTable table(ProfileType::kHeap, 1, false, false);
    AddTrace(&table, 1.0, t1);
    AddTrace(&table, 1.0, t1);
    EXPECT_EQ(4, table.depth_total());
    EXPECT_EQ(2, table.bucket_total());

    CheckTraces(table, {k1, k1});
  }

  const AllocationEntry k1_unsampled_merged = {
      2 * t1_sampled_weight * 1024,
      2 * t1_sampled_weight,
      512,
      16,
      1024,
      2,
      {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2)},
  };

  {
    SCOPED_TRACE("2x t1 unsampled");

    StackTraceTable table(ProfileType::kHeap, 1, true, true);
    AddTrace(&table, 1.0, t1);
    AddTrace(&table, 1.0, t1);
    EXPECT_EQ(2, table.depth_total());
    EXPECT_EQ(1, table.bucket_total());

    CheckTraces(table, {k1_unsampled_merged});
  }

  // Table w/ t1, t2
  {
    SCOPED_TRACE("t1, t2");

    StackTraceTable table(ProfileType::kHeap, 1, true, false);
    AddTrace(&table, 1.0, t1);
    AddTrace(&table, 1.0, t2);
    EXPECT_EQ(4, table.depth_total());
    EXPECT_EQ(2, table.bucket_total());
    CheckTraces(table, {k1, k2});
  }

  // Table w/ 1.6 x t1, 1 x t2.
  // Note that t1's 1.6 count will be rounded-up to 2.0.
  {
    SCOPED_TRACE("1.6 t1, t2");

    StackTraceTable table(ProfileType::kHeap, 1, true, false);
    AddTrace(&table, 0.4, t1);
    AddTrace(&table, 1.0, t2);
    AddTrace(&table, 1.2, t1);
    EXPECT_EQ(4, table.depth_total());
    EXPECT_EQ(2, table.bucket_total());

    const AllocationEntry scaled_k1 = {
        2048,
        2,
        512,
        16,
        1024,
        2,
        {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2)},
    };

    CheckTraces(table, {scaled_k1, k2});
  }

  // Same stack as t1, but w/ different size
  StackTrace t3 = {};
  t3.requested_size = static_cast<uintptr_t>(13);
  t3.requested_alignment = static_cast<uintptr_t>(0);
  t3.allocated_size = static_cast<uintptr_t>(17);
  t3.depth = static_cast<uintptr_t>(2);
  t3.stack[0] = reinterpret_cast<void*>(1);
  t3.stack[1] = reinterpret_cast<void*>(2);
  t3.weight = 1;

  const AllocationEntry k3 = {
      17,
      1,
      13,
      0,
      17,
      2,
      {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2)},
  };

  // Table w/ t1, t3
  {
    SCOPED_TRACE("t1, t3");

    StackTraceTable table(ProfileType::kHeap, 1, true, false);
    AddTrace(&table, 1.0, t1);
    AddTrace(&table, 1.0, t3);
    EXPECT_EQ(4, table.depth_total());
    EXPECT_EQ(2, table.bucket_total());

    CheckTraces(table, {k1, k3});
  }

  // Same stack as t1, but w/ different alignment
  StackTrace t4;
  t4.requested_size = static_cast<uintptr_t>(512);
  t4.requested_alignment = static_cast<uintptr_t>(32);
  t4.allocated_size = static_cast<uintptr_t>(1024);
  t4.depth = static_cast<uintptr_t>(2);
  t4.stack[0] = reinterpret_cast<void*>(1);
  t4.stack[1] = reinterpret_cast<void*>(2);
  t4.weight = 1;

  const AllocationEntry k4 = {
      1024,
      1,
      512,
      32,
      1024,
      2,
      {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2)},
  };

  // Table w/ t1, t4
  {
    SCOPED_TRACE("t1, t4");

    StackTraceTable table(ProfileType::kHeap, 1, true, false);
    AddTrace(&table, 1.0, t1);
    AddTrace(&table, 1.0, t4);
    EXPECT_EQ(4, table.depth_total());
    EXPECT_EQ(2, table.bucket_total());

    CheckTraces(table, {k1, k4});
  }
}

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
