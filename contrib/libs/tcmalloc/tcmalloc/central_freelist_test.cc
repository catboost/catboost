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

#include "tcmalloc/central_freelist.h"

#include <algorithm>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "tcmalloc/common.h"
#include "tcmalloc/static_vars.h"

namespace tcmalloc {
namespace {

// TODO(b/162552708) Mock out the page heap to interact with CFL instead
class CFLTest : public testing::TestWithParam<size_t> {
 protected:
  size_t cl_;
  size_t batch_size_;
  size_t objects_per_span_;
  CentralFreeList cfl_;

 private:
  void SetUp() override {
    cl_ = GetParam();
    size_t object_size = Static::sizemap().class_to_size(cl_);
    if (object_size == 0) {
      GTEST_SKIP() << "Skipping empty size class.";
    }

    auto pages_per_span = Length(Static::sizemap().class_to_pages(cl_));
    batch_size_ = Static::sizemap().num_objects_to_move(cl_);
    objects_per_span_ = pages_per_span.in_bytes() / object_size;
    cfl_.Init(cl_);
  }

  void TearDown() override { EXPECT_EQ(cfl_.length(), 0); }
};

TEST_P(CFLTest, SingleBatch) {
  void* batch[kMaxObjectsToMove];
  int got = cfl_.RemoveRange(batch, batch_size_);
  cfl_.InsertRange(batch, got);
  SpanStats stats = cfl_.GetSpanStats();
  EXPECT_EQ(stats.num_spans_requested, 1);
  EXPECT_EQ(stats.num_spans_returned, 1);
  EXPECT_EQ(stats.obj_capacity, 0);
}

TEST_P(CFLTest, MultipleSpans) {
  std::vector<void*> all_objects;

  const size_t num_spans = 10;

  // Request num_spans spans
  void* batch[kMaxObjectsToMove];
  const int num_objects_to_fetch = num_spans * objects_per_span_;
  int total_fetched = 0;
  while (total_fetched < num_objects_to_fetch) {
    int got = cfl_.RemoveRange(batch, batch_size_);
    for (int i = 0; i < got; ++i) {
      all_objects.push_back(batch[i]);
    }
    total_fetched += got;
  }

  SpanStats stats = cfl_.GetSpanStats();
  EXPECT_EQ(stats.num_spans_requested, num_spans);
  EXPECT_EQ(stats.num_spans_returned, 0);

  EXPECT_EQ(all_objects.size(), num_objects_to_fetch);

  // Shuffle
  absl::BitGen rng;
  std::shuffle(all_objects.begin(), all_objects.end(), rng);

  // Return all
  int total_returned = 0;
  bool checked_half = false;
  while (total_returned < num_objects_to_fetch) {
    int size_to_pop =
        std::min(all_objects.size() - total_returned, batch_size_);
    for (int i = 0; i < size_to_pop; ++i) {
      batch[i] = all_objects[i + total_returned];
    }
    total_returned += size_to_pop;
    cfl_.InsertRange(batch, size_to_pop);
    // sanity check
    if (!checked_half && total_returned >= (num_objects_to_fetch / 2)) {
      stats = cfl_.GetSpanStats();
      EXPECT_GT(stats.num_spans_requested, stats.num_spans_returned);
      EXPECT_NE(stats.obj_capacity, 0);
      checked_half = true;
    }
  }

  stats = cfl_.GetSpanStats();
  EXPECT_EQ(stats.num_spans_requested, stats.num_spans_returned);
  EXPECT_EQ(stats.obj_capacity, 0);
}

INSTANTIATE_TEST_SUITE_P(All, CFLTest, testing::Range(size_t(1), kNumClasses));
}  // namespace
}  // namespace tcmalloc
