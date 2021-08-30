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

#include <stdlib.h>

#include <string>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tcmalloc/common.h"
#include "tcmalloc/size_class_info.h"
#include "tcmalloc/span.h"

namespace tcmalloc {

namespace {

class TestingSizeMap : public SizeMap {
 public:
  TestingSizeMap() {}

  const SizeClassInfo* DefaultSizeClasses() const { return kSizeClasses; }
  int DefaultSizeClassesCount() const { return kSizeClassesCount; }
};

class RunTimeSizeClassesTest : public ::testing::Test {
 protected:
  RunTimeSizeClassesTest() {}

  TestingSizeMap m_;
};

// Convert size classes into a string that can be passed to ParseSizeClasses().
std::string SizeClassesToString(int num_classes, const SizeClassInfo* parsed) {
  std::string result;
  for (int c = 1; c < num_classes; c++) {
    std::string one_size = absl::StrFormat(
        "%d,%d,%d", parsed[c].size, parsed[c].pages, parsed[c].num_to_move);
    if (c == 1) {
      result = one_size;
    } else {
      absl::StrAppend(&result, ";", one_size);
    }
  }
  return result;
}

std::string ModifiedSizeClassesString(int num_classes,
                                      const SizeClassInfo* source) {
  // Set a valid runtime size class environment variable, which
  // is a modified version of the default class sizes.
  SizeClassInfo parsed[kNumClasses];
  for (int c = 0; c < num_classes; c++) {
    parsed[c] = source[c];
  }
  // Change num_to_move to a different valid value so that
  // loading from the ENV can be detected.
  EXPECT_NE(parsed[1].num_to_move, 3);
  parsed[1].num_to_move = 3;
  return SizeClassesToString(num_classes, parsed);
}

TEST_F(RunTimeSizeClassesTest, EnvVariableExamined) {
  std::string e = ModifiedSizeClassesString(m_.DefaultSizeClassesCount(),
                                            m_.DefaultSizeClasses());
  setenv("TCMALLOC_SIZE_CLASSES", e.c_str(), 1);
  m_.Init();

  // Confirm that the expected change is seen.
  EXPECT_EQ(m_.num_objects_to_move(1), 3);
}

// TODO(b/122839049) - Remove this test after bug is fixed.
TEST_F(RunTimeSizeClassesTest, ReducingSizeClassCountNotAllowed) {
  // Try reducing the mumber of size classes by 1, which is expected to fail.
  std::string e = ModifiedSizeClassesString(m_.DefaultSizeClassesCount() - 1,
                                            m_.DefaultSizeClasses());
  setenv("TCMALLOC_SIZE_CLASSES", e.c_str(), 1);
  m_.Init();

  // Confirm that the expected change is not seen.
  EXPECT_EQ(m_.num_objects_to_move(1), m_.DefaultSizeClasses()[1].num_to_move);
}

// Convert the static classes to a string, parse that string via
// the environement variable and check that we get exactly the same
// results. Note, if the environement variable was not read, this test
// would still pass.
TEST_F(RunTimeSizeClassesTest, EnvRealClasses) {
  const int count = m_.DefaultSizeClassesCount();
  std::string e = SizeClassesToString(count, m_.DefaultSizeClasses());
  setenv("TCMALLOC_SIZE_CLASSES", e.c_str(), 1);
  m_.Init();
  // With the runtime_size_classes library linked, the environment variable
  // will be parsed.

  for (int c = 0; c < count; c++) {
    EXPECT_EQ(m_.class_to_size(c), m_.DefaultSizeClasses()[c].size);
    EXPECT_EQ(m_.class_to_pages(c), m_.DefaultSizeClasses()[c].pages);
    EXPECT_EQ(m_.num_objects_to_move(c),
              m_.DefaultSizeClasses()[c].num_to_move);
  }
  for (int c = count; c < kNumClasses; c++) {
    EXPECT_EQ(m_.class_to_size(c), 0);
    EXPECT_EQ(m_.class_to_pages(c), 0);
    EXPECT_EQ(m_.num_objects_to_move(c), 0);
  }
}

}  // namespace
}  // namespace tcmalloc
