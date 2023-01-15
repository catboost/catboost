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
#include <stdlib.h>

#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "tcmalloc/common.h"
#include "tcmalloc/size_class_info.h"
#include "tcmalloc/span.h"

namespace tcmalloc {

namespace {

size_t Alignment(size_t size) {
  size_t ret = kAlignment;
  if (size >= 1024) {
    // SizeMap::ClassIndexMaybe requires 128-byte alignment for sizes >=1024.
    ret = 128;
  } else if (size >= 512) {
    // Per //tcmalloc/span.h, we have 64 byte alignment for sizes
    // >=512.
    ret = 64;
  } else if (size >= 8) {
    ret = 8;
  }

  return ret;
}

class SizeClassesTest : public ::testing::Test {
 protected:
  SizeClassesTest() { m_.Init(); }

  SizeMap m_;
};

TEST_F(SizeClassesTest, SmallClassesSinglePage) {
  // Per //tcmalloc/span.h, the compressed index implementation
  // added by cl/126729493 requires small size classes to be placed on a single
  // page span so they can be addressed.
  for (int c = 1; c < kNumClasses; c++) {
    const size_t max_size_in_class = m_.class_to_size(c);
    if (max_size_in_class >= SizeMap::kMultiPageSize) {
      continue;
    }
    if (max_size_in_class == 0) {
      continue;
    }
    EXPECT_EQ(m_.class_to_pages(c), 1) << max_size_in_class;
  }
}

TEST_F(SizeClassesTest, SpanPages) {
  for (int c = 1; c < kNumClasses; c++) {
    const size_t max_size_in_class = m_.class_to_size(c);
    if (max_size_in_class == 0) {
      continue;
    }
    // A span of class_to_pages(c) must be able to hold at least one object.
    EXPECT_GE(Length(m_.class_to_pages(c)).in_bytes(), max_size_in_class);
  }
}

TEST_F(SizeClassesTest, Aligned) {
  // Validate that each size class is properly aligned.
  for (int c = 1; c < kNumClasses; c++) {
    const size_t max_size_in_class = m_.class_to_size(c);
    size_t alignment = Alignment(max_size_in_class);

    EXPECT_EQ(0, max_size_in_class % alignment) << max_size_in_class;
  }
}

TEST_F(SizeClassesTest, Distinguishable) {
  // Validate that the size to class lookup table is able to distinguish each
  // size class from one another.
  //
  // ClassIndexMaybe provides 8 byte granularity below 1024 bytes and 128 byte
  // granularity for larger sizes, so our chosen size classes cannot be any
  // finer (otherwise they would map to the same entry in the lookup table).
  for (int c = 1; c < kNumClasses; c++) {
    const size_t max_size_in_class = m_.class_to_size(c);
    if (max_size_in_class == 0) {
      continue;
    }
    const int class_index = m_.SizeClass(max_size_in_class);

    EXPECT_EQ(c, class_index) << max_size_in_class;
  }
}

// This test is disabled until we use a different span size allocation
// algorithm (such as the one in effect from cl/130150125 until cl/139955211).
TEST_F(SizeClassesTest, DISABLED_WastedSpan) {
  // Validate that each size class does not waste (number of objects) *
  // (alignment) at the end of the span.
  for (int c = 1; c < kNumClasses; c++) {
    const size_t span_size = kPageSize * m_.class_to_pages(c);
    const size_t max_size_in_class = m_.class_to_size(c);
    const size_t alignment = Alignment(max_size_in_class);
    const size_t n_objects = span_size / max_size_in_class;
    const size_t waste = span_size - n_objects * max_size_in_class;

    EXPECT_LT(waste, n_objects * alignment) << max_size_in_class;
  }
}

TEST_F(SizeClassesTest, DoubleCheckedConsistency) {
  // Validate that every size on [0, kMaxSize] maps to a size class that is
  // neither too big nor too small.
  for (size_t size = 0; size <= kMaxSize; size++) {
    const int sc = m_.SizeClass(size);
    EXPECT_GT(sc, 0) << size;
    EXPECT_LT(sc, kNumClasses) << size;

    if (sc > 1) {
      EXPECT_GT(size, m_.class_to_size(sc - 1))
          << "Allocating unnecessarily large class";
    }

    const size_t s = m_.class_to_size(sc);
    EXPECT_LE(size, s);
    EXPECT_NE(s, 0) << size;
  }
}

TEST_F(SizeClassesTest, NumToMove) {
  for (int c = 1; c < kNumClasses; c++) {
    // For non-empty size classes, we should move at least 1 object to/from each
    // layer of the caches.
    const size_t max_size_in_class = m_.class_to_size(c);
    if (max_size_in_class == 0) {
      continue;
    }
    EXPECT_GT(m_.num_objects_to_move(c), 0) << max_size_in_class;
  }
}

class TestingSizeMap : public SizeMap {
 public:
  TestingSizeMap() {}

  bool ValidSizeClasses(int num_classes, const SizeClassInfo* parsed) {
    return SizeMap::ValidSizeClasses(num_classes, parsed);
  }

  const SizeClassInfo* DefaultSizeClasses() const { return kSizeClasses; }
  const int DefaultSizeClassesCount() const { return kSizeClassesCount; }
};

class RunTimeSizeClassesTest : public ::testing::Test {
 protected:
  RunTimeSizeClassesTest() {}

  TestingSizeMap m_;
};

TEST_F(RunTimeSizeClassesTest, ExpandedSizeClasses) {
  // Verify that none of the default size classes are considered expanded size
  // classes.
  for (int i = 0; i < kNumClasses; i++) {
    EXPECT_EQ(i < m_.DefaultSizeClassesCount(), !IsExpandedSizeClass(i)) << i;
  }
}

TEST_F(RunTimeSizeClassesTest, ValidateClassSizeIncreases) {
  SizeClassInfo parsed[] = {
      {0, 0, 0},
      {16, 1, 14},
      {32, 1, 15},
      {kMaxSize, 1, 15},
  };
  EXPECT_TRUE(m_.ValidSizeClasses(4, parsed));

  parsed[2].size = 8;  // Change 32 to 8
  EXPECT_FALSE(m_.ValidSizeClasses(4, parsed));
}

TEST_F(RunTimeSizeClassesTest, ValidateClassSizeMax) {
  SizeClassInfo parsed[] = {
      {0, 0, 0},
      {kMaxSize - 128, 1, 15},
  };
  // Last class must cover kMaxSize
  EXPECT_FALSE(m_.ValidSizeClasses(2, parsed));

  // Check Max Size is allowed 256 KiB = 262144
  parsed[1].size = kMaxSize;
  EXPECT_TRUE(m_.ValidSizeClasses(2, parsed));
  // But kMaxSize + 128 is not allowed
  parsed[1].size = kMaxSize + 128;
  EXPECT_FALSE(m_.ValidSizeClasses(2, parsed));
}

TEST_F(RunTimeSizeClassesTest, ValidateClassSizesAlignment) {
  SizeClassInfo parsed[] = {
      {0, 0, 0},
      {8, 1, 14},
      {kMaxSize, 1, 15},
  };
  EXPECT_TRUE(m_.ValidSizeClasses(3, parsed));
  // Doesn't meet alignment requirements
  parsed[1].size = 7;
  EXPECT_FALSE(m_.ValidSizeClasses(3, parsed));

  // Over 512, expect alignment of 64 bytes.
  // 512 + 64 = 576
  parsed[1].size = 576;
  EXPECT_TRUE(m_.ValidSizeClasses(3, parsed));
  // 512 + 8
  parsed[1].size = 520;
  EXPECT_FALSE(m_.ValidSizeClasses(3, parsed));

  // Over 1024, expect alignment of 128 bytes.
  // 1024 + 128 = 1152
  parsed[1].size = 1024 + 128;
  EXPECT_TRUE(m_.ValidSizeClasses(3, parsed));
  // 1024 + 64 = 1088
  parsed[1].size = 1024 + 64;
  EXPECT_FALSE(m_.ValidSizeClasses(3, parsed));
}

TEST_F(RunTimeSizeClassesTest, ValidateBatchSize) {
  SizeClassInfo parsed[] = {
      {0, 0, 0},
      {8, 1, kMaxObjectsToMove},
      {kMaxSize, 1, 15},
  };
  EXPECT_TRUE(m_.ValidSizeClasses(3, parsed));

  ++parsed[1].num_to_move;
  EXPECT_FALSE(m_.ValidSizeClasses(3, parsed));
}

TEST_F(RunTimeSizeClassesTest, ValidatePageSize) {
  SizeClassInfo parsed[] = {
      {0, 0, 0},
      {1024, 255, kMaxObjectsToMove},
      {kMaxSize, 1, 15},
  };
  EXPECT_TRUE(m_.ValidSizeClasses(3, parsed));

  parsed[1].pages = 256;
  EXPECT_FALSE(m_.ValidSizeClasses(3, parsed));
}

TEST_F(RunTimeSizeClassesTest, ValidateDefaultSizeClasses) {
  // The default size classes also need to be valid.
  EXPECT_TRUE(m_.ValidSizeClasses(m_.DefaultSizeClassesCount(),
                                  m_.DefaultSizeClasses()));
}

TEST_F(RunTimeSizeClassesTest, EnvVariableNotExamined) {
  // Set a valid runtime size class environment variable
  setenv("TCMALLOC_SIZE_CLASSES", "256,1,1", 1);
  m_.Init();
  // Without runtime_size_classes library linked, the environment variable
  // should have no affect.
  EXPECT_NE(m_.class_to_size(1), 256);
}

TEST(SizeMapTest, GetSizeClass) {
  absl::BitGen rng;
  constexpr int kTrials = 1000;

  SizeMap m;
  // Before m.Init(), SizeClass should always return 0.
  for (int i = 0; i < kTrials; ++i) {
    const size_t size = absl::LogUniform(rng, 0, 4 << 20);
    uint32_t cl;
    if (m.GetSizeClass(size, &cl)) {
      EXPECT_EQ(cl, 0) << size;
    } else {
      // We should only fail to lookup the size class when size is outside of
      // the size classes.
      ASSERT_GT(size, kMaxSize);
    }
  }

  // After m.Init(), GetSizeClass should return a size class.
  m.Init();

  for (int i = 0; i < kTrials; ++i) {
    const size_t size = absl::LogUniform(rng, 0, 4 << 20);
    uint32_t cl;
    if (m.GetSizeClass(size, &cl)) {
      const size_t mapped_size = m.class_to_size(cl);
      // The size class needs to hold size.
      ASSERT_GE(mapped_size, size);
    } else {
      // We should only fail to lookup the size class when size is outside of
      // the size classes.
      ASSERT_GT(size, kMaxSize);
    }
  }
}

TEST(SizeMapTest, GetSizeClassWithAlignment) {
  absl::BitGen rng;
  constexpr int kTrials = 1000;

  SizeMap m;
  // Before m.Init(), SizeClass should always return 0.
  for (int i = 0; i < kTrials; ++i) {
    const size_t size = absl::LogUniform(rng, 0, 4 << 20);
    const size_t alignment = 1 << absl::Uniform(rng, 0u, kHugePageShift);
    uint32_t cl;
    if (m.GetSizeClass(size, alignment, &cl)) {
      EXPECT_EQ(cl, 0) << size << " " << alignment;
    } else if (alignment < kPageSize) {
      // When alignment > kPageSize, we do not produce a size class.
      // TODO(b/172060547): alignment == kPageSize could fit into the size
      // classes too.
      //
      // We should only fail to lookup the size class when size is large.
      ASSERT_GT(size, kMaxSize) << alignment;
    }
  }

  // After m.Init(), GetSizeClass should return a size class.
  m.Init();

  for (int i = 0; i < kTrials; ++i) {
    const size_t size = absl::LogUniform(rng, 0, 4 << 20);
    const size_t alignment = 1 << absl::Uniform(rng, 0u, kHugePageShift);
    uint32_t cl;
    if (m.GetSizeClass(size, alignment, &cl)) {
      const size_t mapped_size = m.class_to_size(cl);
      // The size class needs to hold size.
      ASSERT_GE(mapped_size, size);
      // The size needs to be a multiple of alignment.
      ASSERT_EQ(mapped_size % alignment, 0);
    } else if (alignment < kPageSize) {
      // When alignment > kPageSize, we do not produce a size class.
      // TODO(b/172060547): alignment == kPageSize could fit into the size
      // classes too.
      //
      // We should only fail to lookup the size class when size is large.
      ASSERT_GT(size, kMaxSize) << alignment;
    }
  }
}

TEST(SizeMapTest, SizeClass) {
  absl::BitGen rng;
  constexpr int kTrials = 1000;

  SizeMap m;
  // Before m.Init(), SizeClass should always return 0.
  for (int i = 0; i < kTrials; ++i) {
    const size_t size = absl::LogUniform<size_t>(rng, 0u, kMaxSize);
    EXPECT_EQ(m.SizeClass(size), 0) << size;
  }

  // After m.Init(), SizeClass should return a size class.
  m.Init();

  for (int i = 0; i < kTrials; ++i) {
    const size_t size = absl::LogUniform<size_t>(rng, 0u, kMaxSize);
    uint32_t cl = m.SizeClass(size);

    const size_t mapped_size = m.class_to_size(cl);
    // The size class needs to hold size.
    ASSERT_GE(mapped_size, size);
  }
}

TEST(SizeMapTest, Preinit) {
  ABSL_CONST_INIT static SizeMap m;

  for (int cl = 0; cl < kNumClasses; ++cl) {
    EXPECT_EQ(m.class_to_size(cl), 0) << cl;
    EXPECT_EQ(m.class_to_pages(cl), 0) << cl;
    EXPECT_EQ(m.num_objects_to_move(cl), 0) << cl;
  }
}

}  // namespace
}  // namespace tcmalloc
