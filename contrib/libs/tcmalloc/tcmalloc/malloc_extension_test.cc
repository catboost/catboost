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
//
// Test for TCMalloc implementation of MallocExtension

#include "tcmalloc/malloc_extension.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tcmalloc {
namespace {

TEST(MallocExtension, BackgroundReleaseRate) {

  // Mutate via MallocExtension.
  MallocExtension::SetBackgroundReleaseRate(
      MallocExtension::BytesPerSecond{100 << 20});

  EXPECT_EQ(static_cast<size_t>(MallocExtension::GetBackgroundReleaseRate()),
            100 << 20);

  // Disable release
  MallocExtension::SetBackgroundReleaseRate(MallocExtension::BytesPerSecond{0});

  EXPECT_EQ(static_cast<size_t>(MallocExtension::GetBackgroundReleaseRate()),
            0);
}

TEST(MallocExtension, Properties) {
  // Verify that every property under GetProperties also works with
  // GetNumericProperty.
  const auto properties = MallocExtension::GetProperties();
  for (const auto& property : properties) {
    absl::optional<size_t> scalar =
        MallocExtension::GetNumericProperty(property.first);
    // The value of the property itself may have changed, so just check that it
    // is present.
    EXPECT_THAT(scalar, testing::Ne(absl::nullopt)) << property.first;
  }
}

}  // namespace
}  // namespace tcmalloc
