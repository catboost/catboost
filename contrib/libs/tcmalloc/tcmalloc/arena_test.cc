// Copyright 2021 The TCMalloc Authors
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

#include "tcmalloc/arena.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tcmalloc {
namespace tcmalloc_internal {
namespace {

TEST(Arena, AlignedAlloc) {
  Arena arena;
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(arena.Alloc(64, 64)) % 64, 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(arena.Alloc(7)) % 8, 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(arena.Alloc(128, 64)) % 64, 0);
  for (int alignment = 1; alignment < 100; ++alignment) {
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(arena.Alloc(7, alignment)) % alignment, 0);
  }
}

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
