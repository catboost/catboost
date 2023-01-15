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
// Tests for infrastructure common to page allocator implementations
// (stats and logging.)
#include "tcmalloc/page_allocator.h"

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <memory>
#include <new>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/internal/spinlock.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/malloc_extension.h"
#include "tcmalloc/page_allocator_test_util.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/stats.h"

namespace tcmalloc {
namespace {

class PageAllocatorTest : public testing::Test {
 protected:
  // Not in constructor so subclasses can mess about with environment
  // variables.
  void SetUp() override {
    // If this test is not linked against TCMalloc, the global arena used for
    // metadata will not be initialized.
    Static::InitIfNecessary();

    before_ = MallocExtension::GetRegionFactory();
    extra_ = new ExtraRegionFactory(before_);
    MallocExtension::SetRegionFactory(extra_);
    void *p = malloc(sizeof(PageAllocator));
    allocator_ = new (p) PageAllocator;
  }
  void TearDown() override {
    MallocExtension::SetRegionFactory(before_);
    delete extra_;
    free(allocator_);
  }

  Span *New(Length n) { return allocator_->New(n, MemoryTag::kNormal); }
  Span *NewAligned(Length n, Length align) {
    return allocator_->NewAligned(n, align, MemoryTag::kNormal);
  }
  void Delete(Span *s) {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    allocator_->Delete(s, MemoryTag::kNormal);
  }

  Length Release(Length n) {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    return allocator_->ReleaseAtLeastNPages(n);
  }

  std::string Print() {
    std::vector<char> buf(1024 * 1024);
    TCMalloc_Printer out(&buf[0], buf.size());
    allocator_->Print(&out, MemoryTag::kNormal);

    return std::string(&buf[0]);
  }

  PageAllocator *allocator_;
  ExtraRegionFactory *extra_;
  AddressRegionFactory *before_;
};

// We've already tested in stats_test that PageAllocInfo keeps good stats;
// here we're just testing that we make the proper Record calls.
TEST_F(PageAllocatorTest, Record) {
  for (int i = 0; i < 15; ++i) {
    Delete(New(Length(1)));
  }

  std::vector<Span *> spans;
  for (int i = 0; i < 20; ++i) {
    spans.push_back(New(Length(2)));
  }

  for (int i = 0; i < 25; ++i) {
    Delete(NewAligned(Length(3), Length(2)));
  }
  {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    auto info = allocator_->info(MemoryTag::kNormal);

    CHECK_CONDITION(15 == info.counts_for(Length(1)).nalloc);
    CHECK_CONDITION(15 == info.counts_for(Length(1)).nfree);

    CHECK_CONDITION(20 == info.counts_for(Length(2)).nalloc);
    CHECK_CONDITION(0 == info.counts_for(Length(2)).nfree);

    CHECK_CONDITION(25 == info.counts_for(Length(3)).nalloc);
    CHECK_CONDITION(25 == info.counts_for(Length(3)).nfree);

    for (auto i = Length(4); i <= kMaxPages; ++i) {
      CHECK_CONDITION(0 == info.counts_for(i).nalloc);
      CHECK_CONDITION(0 == info.counts_for(i).nfree);
    }

    const Length absurd =
        Length(uintptr_t{1} << (kAddressBits - 1 - kPageShift));
    for (Length i = kMaxPages + Length(1); i < absurd; i *= 2) {
      CHECK_CONDITION(0 == info.counts_for(i).nalloc);
      CHECK_CONDITION(0 == info.counts_for(i).nfree);
    }
  }
  for (auto s : spans) Delete(s);
}

// And that we call the print method properly.
TEST_F(PageAllocatorTest, PrintIt) {
  Delete(New(Length(1)));
  std::string output = Print();
  EXPECT_THAT(output, testing::ContainsRegex("stats on allocation sizes"));
}

}  // namespace
}  // namespace tcmalloc
