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

#include "tcmalloc/guarded_page_allocator.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/internal/sysinfo.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/static_vars.h"

namespace tcmalloc {
namespace tcmalloc_internal {
namespace {

static constexpr size_t kMaxGpaPages = GuardedPageAllocator::kGpaMaxPages;

// Size of pages used by GuardedPageAllocator.
static size_t PageSize() {
  static const size_t page_size =
      std::max(kPageSize, static_cast<size_t>(getpagesize()));
  return page_size;
}

class GuardedPageAllocatorTest : public testing::Test {
 protected:
  GuardedPageAllocatorTest() {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    gpa_.Init(kMaxGpaPages, kMaxGpaPages);
    gpa_.AllowAllocations();
  }

  explicit GuardedPageAllocatorTest(size_t num_pages) {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    gpa_.Init(num_pages, kMaxGpaPages);
    gpa_.AllowAllocations();
  }

  ~GuardedPageAllocatorTest() override { gpa_.Destroy(); }

  GuardedPageAllocator gpa_;
};

class GuardedPageAllocatorParamTest
    : public GuardedPageAllocatorTest,
      public testing::WithParamInterface<size_t> {
 protected:
  GuardedPageAllocatorParamTest() : GuardedPageAllocatorTest(GetParam()) {}
};

TEST_F(GuardedPageAllocatorTest, SingleAllocDealloc) {
  char *buf = reinterpret_cast<char *>(gpa_.Allocate(PageSize(), 0));
  EXPECT_NE(buf, nullptr);
  EXPECT_TRUE(gpa_.PointerIsMine(buf));
  memset(buf, 'A', PageSize());
  EXPECT_DEATH(buf[-1] = 'A', "");
  EXPECT_DEATH(buf[PageSize()] = 'A', "");
  gpa_.Deallocate(buf);
  EXPECT_DEATH(buf[0] = 'B', "");
  EXPECT_DEATH(buf[PageSize() / 2] = 'B', "");
  EXPECT_DEATH(buf[PageSize() - 1] = 'B', "");
}

TEST_F(GuardedPageAllocatorTest, AllocDeallocAligned) {
  for (size_t align = 1; align <= PageSize(); align <<= 1) {
    constexpr size_t alloc_size = 1;
    void *p = gpa_.Allocate(alloc_size, align);
    EXPECT_NE(p, nullptr);
    EXPECT_TRUE(gpa_.PointerIsMine(p));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % align, 0);
  }
}

TEST_P(GuardedPageAllocatorParamTest, AllocDeallocAllPages) {
  size_t num_pages = GetParam();
  char *bufs[kMaxGpaPages];
  for (size_t i = 0; i < num_pages; i++) {
    bufs[i] = reinterpret_cast<char *>(gpa_.Allocate(1, 0));
    EXPECT_NE(bufs[i], nullptr);
    EXPECT_TRUE(gpa_.PointerIsMine(bufs[i]));
  }
  EXPECT_EQ(gpa_.Allocate(1, 0), nullptr);
  gpa_.Deallocate(bufs[0]);
  bufs[0] = reinterpret_cast<char *>(gpa_.Allocate(1, 0));
  EXPECT_NE(bufs[0], nullptr);
  EXPECT_TRUE(gpa_.PointerIsMine(bufs[0]));
  for (size_t i = 0; i < num_pages; i++) {
    bufs[i][0] = 'A';
    gpa_.Deallocate(bufs[i]);
  }
}
INSTANTIATE_TEST_SUITE_P(VaryNumPages, GuardedPageAllocatorParamTest,
                         testing::Values(1, kMaxGpaPages / 2, kMaxGpaPages));

TEST_F(GuardedPageAllocatorTest, PointerIsMine) {
  void *buf = gpa_.Allocate(1, 0);
  int stack_var;
  auto malloc_ptr = absl::make_unique<char>();
  EXPECT_TRUE(gpa_.PointerIsMine(buf));
  EXPECT_FALSE(gpa_.PointerIsMine(&stack_var));
  EXPECT_FALSE(gpa_.PointerIsMine(malloc_ptr.get()));
}

TEST_F(GuardedPageAllocatorTest, Print) {
  char buf[1024] = {};
  Printer out(buf, sizeof(buf));
  gpa_.Print(&out);
  EXPECT_THAT(buf, testing::ContainsRegex("GWP-ASan Status"));
}

// Test that no pages are double-allocated or left unallocated, and that no
// extra pages are allocated when there's concurrent calls to Allocate().
TEST_F(GuardedPageAllocatorTest, ThreadedAllocCount) {
  constexpr size_t kNumThreads = 2;
  void *allocations[kNumThreads][kMaxGpaPages];
  {
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (size_t i = 0; i < kNumThreads; i++) {
      threads.push_back(std::thread([this, &allocations, i]() {
        for (size_t j = 0; j < kMaxGpaPages; j++) {
          allocations[i][j] = gpa_.Allocate(1, 0);
        }
      }));
    }

    for (auto &t : threads) {
      t.join();
    }
  }
  absl::flat_hash_set<void *> allocations_set;
  for (size_t i = 0; i < kNumThreads; i++) {
    for (size_t j = 0; j < kMaxGpaPages; j++) {
      allocations_set.insert(allocations[i][j]);
    }
  }
  allocations_set.erase(nullptr);
  EXPECT_EQ(allocations_set.size(), kMaxGpaPages);
}

// Test that allocator remains in consistent state under high contention and
// doesn't double-allocate pages or fail to deallocate pages.
TEST_F(GuardedPageAllocatorTest, ThreadedHighContention) {
  const size_t kNumThreads = 4 * absl::base_internal::NumCPUs();
  {
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (size_t i = 0; i < kNumThreads; i++) {
      threads.push_back(std::thread([this]() {
        char *buf;
        while ((buf = reinterpret_cast<char *>(gpa_.Allocate(1, 0))) ==
               nullptr) {
          absl::SleepFor(absl::Nanoseconds(5000));
        }

        // Verify that no other thread has access to this page.
        EXPECT_EQ(buf[0], 0);

        // Mark this page and allow some time for another thread to potentially
        // gain access to this page.
        buf[0] = 'A';
        absl::SleepFor(absl::Nanoseconds(5000));

        // Unmark this page and deallocate.
        buf[0] = 0;
        gpa_.Deallocate(buf);
      }));
    }

    for (auto &t : threads) {
      t.join();
    }
  }
  // Verify all pages have been deallocated now that all threads are done.
  for (size_t i = 0; i < kMaxGpaPages; i++) {
    EXPECT_NE(gpa_.Allocate(1, 0), nullptr);
  }
}

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
