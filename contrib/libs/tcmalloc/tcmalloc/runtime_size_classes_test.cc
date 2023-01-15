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

#include "tcmalloc/runtime_size_classes.h"

#include <stdlib.h>

#include "gtest/gtest.h"

using tcmalloc::internal::ParseSizeClasses;

namespace tcmalloc {

namespace {

constexpr int kNumClasses = 4;
constexpr int kMaxSize = 1024 * 1024;

TEST(RuntimeSizeClassesTest, EnvSingleFullClass) {
  // Validate simple parsing.
  SizeClassInfo parsed[kNumClasses];
  EXPECT_EQ(ParseSizeClasses("8,1,32", kMaxSize, kNumClasses, parsed), 2);
  EXPECT_EQ(parsed[1].size, 8);
  EXPECT_EQ(parsed[1].pages, 1);
  EXPECT_EQ(parsed[1].num_to_move, 32);

  EXPECT_EQ(parsed[0].size, 0);
  EXPECT_EQ(parsed[0].pages, 0);
  EXPECT_EQ(parsed[0].num_to_move, 0);
}

TEST(RuntimeSizeClassesTest, EnvSingleSizeOnlyClass) {
  // Validate simple parsing.
  SizeClassInfo parsed[kNumClasses];
  EXPECT_EQ(ParseSizeClasses("8,1,2", kMaxSize, kNumClasses, parsed), 2);
  EXPECT_EQ(parsed[1].size, 8);
  EXPECT_EQ(parsed[1].pages, 1);
  EXPECT_EQ(parsed[1].num_to_move, 2);
}

TEST(RuntimeSizeClassesTest, EnvTwoFullClasses) {
  // Validate two classes
  SizeClassInfo parsed[kNumClasses];
  EXPECT_EQ(ParseSizeClasses("8,1,32;1024,2,16", kMaxSize, kNumClasses, parsed),
            3);
  EXPECT_EQ(parsed[1].size, 8);
  EXPECT_EQ(parsed[1].pages, 1);
  EXPECT_EQ(parsed[1].num_to_move, 32);

  EXPECT_EQ(parsed[2].size, 1024);
  EXPECT_EQ(parsed[2].pages, 2);
  EXPECT_EQ(parsed[2].num_to_move, 16);
}

TEST(RuntimeSizeClassesTest, ParseArrayLimit) {
  // Validate that the limit on the number of size classes is enforced.
  SizeClassInfo parsed[kNumClasses] = {
      {0, 0, 0},
      {9, 9, 9},
      {7, 7, 7},
  };
  EXPECT_EQ(ParseSizeClasses("8,1,32;1024,2,16", kMaxSize, 2, parsed), 2);

  EXPECT_EQ(parsed[1].size, 8);
  EXPECT_EQ(parsed[1].pages, 1);
  EXPECT_EQ(parsed[1].num_to_move, 32);

  EXPECT_EQ(parsed[2].size, 7);
  EXPECT_EQ(parsed[2].pages, 7);
  EXPECT_EQ(parsed[2].num_to_move, 7);
}

TEST(RuntimeSizeClassesTest, EnvBadDelimiter) {
  // Invalid class sizes should be caught
  SizeClassInfo parsed[kNumClasses];
  EXPECT_EQ(ParseSizeClasses("8/4,16,3,1", kMaxSize, kNumClasses, parsed), -2);
}

TEST(RuntimeSizeClassesTest, EnvTooManyCommas) {
  // Invalid class sizes should be caught
  SizeClassInfo parsed[kNumClasses];
  EXPECT_EQ(ParseSizeClasses("8,4,16,3", kMaxSize, kNumClasses, parsed), -1);
}

TEST(RuntimeSizeClassesTest, EnvIntOverflow) {
  // Invalid class sizes should be caught
  SizeClassInfo parsed[kNumClasses];
  EXPECT_EQ(ParseSizeClasses("8,4,2147483648", kMaxSize, kNumClasses, parsed),
            -3);
}

TEST(RuntimeSizeClassesTest, EnvVariableExamined) {
  SizeClassInfo parsed[kNumClasses];
  setenv("TCMALLOC_SIZE_CLASSES", "256,13,31", 1);
  EXPECT_EQ(MaybeSizeClassesFromEnv(kMaxSize, kNumClasses, parsed), 2);
  EXPECT_EQ(parsed[1].size, 256);
  EXPECT_EQ(parsed[1].pages, 13);
  EXPECT_EQ(parsed[1].num_to_move, 31);
}

}  // namespace
}  // namespace tcmalloc
