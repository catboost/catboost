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

#include "tcmalloc/huge_page_aware_allocator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/internal/sysinfo.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tcmalloc/common.h"
#include "tcmalloc/huge_pages.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/malloc_extension.h"
#include "tcmalloc/page_allocator_test_util.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/span.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/stats.h"
#include "tcmalloc/system-alloc.h"
#include "tcmalloc/testing/thread_manager.h"

ABSL_FLAG(std::string, tracefile, "", "file to pull trace from");
ABSL_FLAG(uint64_t, limit, 0, "");
ABSL_FLAG(bool, always_check_usage, false, "enable expensive memory checks");

namespace tcmalloc {
namespace {

using testing::HasSubstr;

class HugePageAwareAllocatorTest : public ::testing::Test {
 protected:
  HugePageAwareAllocatorTest() : rng_() {
    before_ = MallocExtension::GetRegionFactory();
    extra_ = new ExtraRegionFactory(before_);
    MallocExtension::SetRegionFactory(extra_);

    // HugePageAwareAllocator can't be destroyed cleanly, so we store a pointer
    // to one and construct in place.
    void* p = malloc(sizeof(HugePageAwareAllocator));
    allocator_ = new (p) HugePageAwareAllocator(MemoryTag::kNormal);
  }

  ~HugePageAwareAllocatorTest() override {
    CHECK_CONDITION(ids_.empty());
    CHECK_CONDITION(total_ == Length(0));
    // We end up leaking both the backing allocations and the metadata.
    // The backing allocations are unmapped--it's silly, but not
    // costing us muchin a 64-bit address space.
    // The metadata is real memory, but there's barely any of it.
    // It'd be very complicated to rebuild the allocator to support
    // teardown, so we just put up with it.
    {
      absl::base_internal::SpinLockHolder h(&pageheap_lock);
      auto stats = allocator_->stats();
      if (stats.free_bytes + stats.unmapped_bytes != stats.system_bytes) {
        Crash(kCrash, __FILE__, __LINE__, stats.free_bytes,
              stats.unmapped_bytes, "!=", stats.system_bytes);
      }
    }

    free(allocator_);

    MallocExtension::SetRegionFactory(before_);
    delete extra_;
  }

  struct Mark {
    int64_t pad;
    int64_t mark;
    int64_t padding[62];
  };

  void CheckStats() {
    size_t actual_used_bytes = total_.in_bytes();
    BackingStats stats;
    {
      absl::base_internal::SpinLockHolder h2(&pageheap_lock);
      stats = allocator_->stats();
    }
    uint64_t used_bytes =
        stats.system_bytes - stats.free_bytes - stats.unmapped_bytes;
    ASSERT_EQ(used_bytes, actual_used_bytes);
  }

  uint64_t GetFreeBytes() {
    BackingStats stats;
    {
      absl::base_internal::SpinLockHolder h2(&pageheap_lock);
      stats = allocator_->stats();
    }
    return stats.free_bytes;
  }

  Span* AllocatorNew(Length n) { return allocator_->New(n); }

  void AllocatorDelete(Span* s) {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    allocator_->Delete(s);
  }

  Span* New(Length n) {
    absl::base_internal::SpinLockHolder h(&lock_);
    Span* span = AllocatorNew(n);
    CHECK_CONDITION(span != nullptr);
    EXPECT_GE(span->num_pages(), n);
    const size_t id = next_id_++;
    total_ += n;
    CheckStats();
    // and distinct spans...
    CHECK_CONDITION(ids_.insert({span, id}).second);
    return span;
  }

  void Delete(Span* span) {
    Length n = span->num_pages();
    {
      absl::base_internal::SpinLockHolder h(&lock_);
      auto i = ids_.find(span);
      CHECK_CONDITION(i != ids_.end());
      const size_t id = i->second;
      ids_.erase(i);
      AllocatorDelete(span);
      total_ -= n;
      CheckStats();
    }
  }

  // Mostly small things, some large ones.
  Length RandomAllocSize() {
    // TODO(b/128521238): scalable RNG
    absl::base_internal::SpinLockHolder h(&lock_);
    if (absl::Bernoulli(rng_, 1.0 / 1000)) {
      Length n =
          Length(1024) * (1 + absl::LogUniform<int32_t>(rng_, 0, (1 << 8) - 1));
      n += Length(absl::Uniform<int32_t>(rng_, 0, 1024));
      return n;
    }
    return Length(1 + absl::LogUniform<int32_t>(rng_, 0, (1 << 9) - 1));
  }

  Length ReleasePages(Length k) {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    return allocator_->ReleaseAtLeastNPages(k);
  }

  std::string Print() {
    std::string ret;
    const size_t kSize = 1 << 20;
    ret.resize(kSize);
    TCMalloc_Printer p(&ret[0], kSize);
    allocator_->Print(&p);
    ret.erase(p.SpaceRequired());
    return ret;
  }

  std::string PrintInPbTxt() {
    std::string ret;
    const size_t kSize = 1 << 20;
    ret.resize(kSize);
    TCMalloc_Printer p(&ret[0], kSize);
    {
      PbtxtRegion region(&p, kNested, 0);
      allocator_->PrintInPbtxt(&region);
    }
    ret.erase(p.SpaceRequired());
    return ret;
  }

  HugePageAwareAllocator* allocator_;
  ExtraRegionFactory* extra_;
  AddressRegionFactory* before_;
  absl::BitGen rng_;
  absl::base_internal::SpinLock lock_;
  absl::flat_hash_map<Span*, size_t> ids_;
  size_t next_id_{0};
  Length total_;
};

TEST_F(HugePageAwareAllocatorTest, Fuzz) {
  std::vector<Span*> allocs;
  for (int i = 0; i < 5000; ++i) {
    Length n = RandomAllocSize();
    allocs.push_back(New(n));
  }
  static const size_t kReps = 50 * 1000;
  for (int i = 0; i < kReps; ++i) {
    SCOPED_TRACE(absl::StrFormat("%d reps, %d pages", i, total_.raw_num()));
    size_t index = absl::Uniform<int32_t>(rng_, 0, allocs.size());
    Span* old = allocs[index];
    Delete(old);
    Length n = RandomAllocSize();
    allocs[index] = New(n);
  }

  for (auto s : allocs) {
    Delete(s);
  }
}

// Prevent regression of the fragmentation problem that was reported in
// b/63301358, reproduced in CL/161345659 and (partially) fixed in CL/161305971.
TEST_F(HugePageAwareAllocatorTest, JustUnderMultipleOfHugepages) {
  std::vector<Span*> big_allocs, small_allocs;
  // Trigger creation of a hugepage with more than one allocation and plenty of
  // free space.
  small_allocs.push_back(New(Length(1)));
  small_allocs.push_back(New(Length(10)));
  // Limit iterations so that the huge page with the small allocs doesn't fill
  // up.
  size_t n_iter = (kPagesPerHugePage - Length(2)).raw_num();
  // Also limit memory usage to ~1 GB.
  n_iter = std::min((1 << 30) / (2 * kHugePageSize), n_iter);
  for (int i = 0; i < n_iter; ++i) {
    Length n = 2 * kPagesPerHugePage - Length(1);
    big_allocs.push_back(New(n));
    small_allocs.push_back(New(Length(1)));
  }
  for (auto* span : big_allocs) {
    Delete(span);
  }
  // We should have one hugepage that's full of small allocations and a bunch
  // of empty hugepages. The HugeCache will keep some of the empty hugepages
  // backed so free space should drop to a small multiple of the huge page size.
  EXPECT_LE(GetFreeBytes(), 20 * kHugePageSize);
  for (auto* span : small_allocs) {
    Delete(span);
  }
}

TEST_F(HugePageAwareAllocatorTest, Multithreaded) {
  static const size_t kThreads = 16;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  absl::Barrier b1(kThreads);
  absl::Barrier b2(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.push_back(std::thread([this, &b1, &b2]() {
      absl::BitGen rng;
      std::vector<Span*> allocs;
      for (int i = 0; i < 150; ++i) {
        Length n = RandomAllocSize();
        allocs.push_back(New(n));
      }
      b1.Block();
      static const size_t kReps = 4 * 1000;
      for (int i = 0; i < kReps; ++i) {
        size_t index = absl::Uniform<int32_t>(rng, 0, allocs.size());
        Delete(allocs[index]);
        Length n = RandomAllocSize();
        allocs[index] = New(n);
      }
      b2.Block();
      for (auto s : allocs) {
        Delete(s);
      }
    }));
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(HugePageAwareAllocatorTest, ReleasingLarge) {
  // Ensure the HugeCache has some free items:
  Delete(New(kPagesPerHugePage));
  ASSERT_LE(kPagesPerHugePage, ReleasePages(kPagesPerHugePage));
}

TEST_F(HugePageAwareAllocatorTest, ReleasingSmall) {
  const bool old_subrelease = Parameters::hpaa_subrelease();
  Parameters::set_hpaa_subrelease(true);

  const absl::Duration old_skip_subrelease =
      Parameters::filler_skip_subrelease_interval();
  Parameters::set_filler_skip_subrelease_interval(absl::ZeroDuration());

  std::vector<Span*> live, dead;
  static const size_t N = kPagesPerHugePage.raw_num() * 128;
  for (int i = 0; i < N; ++i) {
    Span* span = New(Length(1));
    ((i % 2 == 0) ? live : dead).push_back(span);
  }

  for (auto d : dead) {
    Delete(d);
  }

  EXPECT_EQ(kPagesPerHugePage / 2, ReleasePages(Length(1)));

  for (auto l : live) {
    Delete(l);
  }

  Parameters::set_hpaa_subrelease(old_subrelease);
  Parameters::set_filler_skip_subrelease_interval(old_skip_subrelease);
}

TEST_F(HugePageAwareAllocatorTest, DonatedHugePages) {
  // This test verifies that we accurately measure the amount of RAM that we
  // donate to the huge page filler when making large allocations, including
  // those kept alive after we deallocate.
  static constexpr Length kSlack = Length(2);
  static constexpr Length kLargeSize = 2 * kPagesPerHugePage - kSlack;
  static constexpr Length kSmallSize = Length(1);

  Span* large1 = New(kLargeSize);
  Length slack;
  HugeLength donated_huge_pages;
  {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    slack = allocator_->info().slack();
    donated_huge_pages = allocator_->DonatedHugePages();
  }
  EXPECT_EQ(slack, kSlack);
  EXPECT_EQ(donated_huge_pages, NHugePages(1));

  EXPECT_THAT(Print(), HasSubstr("filler donations 1"));
  EXPECT_THAT(PrintInPbTxt(), HasSubstr("filler_donated_huge_pages: 1"));

  // Make a small allocation and then free the large allocation.  Slack should
  // fall, but we've kept alive our donation to the filler.
  Span* small = New(kSmallSize);
  Delete(large1);
  {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    slack = allocator_->info().slack();
    donated_huge_pages = allocator_->DonatedHugePages();
  }
  EXPECT_EQ(slack, Length(0));
  EXPECT_EQ(donated_huge_pages, NHugePages(1));

  EXPECT_THAT(Print(), HasSubstr("filler donations 1"));
  EXPECT_THAT(PrintInPbTxt(), HasSubstr("filler_donated_huge_pages: 1"));

  // Make another large allocation.  The number of donated huge pages should
  // continue to increase.
  Span* large2 = New(kLargeSize);
  {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    slack = allocator_->info().slack();
    donated_huge_pages = allocator_->DonatedHugePages();
  }
  EXPECT_EQ(slack, kSlack);
  EXPECT_EQ(donated_huge_pages, NHugePages(2));

  EXPECT_THAT(Print(), HasSubstr("filler donations 2"));
  EXPECT_THAT(PrintInPbTxt(), HasSubstr("filler_donated_huge_pages: 2"));

  // Deallocating the small allocation does not reduce the number of donations,
  // as we were unable to reassemble the VSS for large1.
  Delete(small);
  {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    slack = allocator_->info().slack();
    donated_huge_pages = allocator_->DonatedHugePages();
  }
  EXPECT_EQ(slack, kSlack);
  EXPECT_EQ(donated_huge_pages, NHugePages(2));

  EXPECT_THAT(Print(), HasSubstr("filler donations 2"));
  EXPECT_THAT(PrintInPbTxt(), HasSubstr("filler_donated_huge_pages: 2"));

  // Deallocating everything should return slack to 0 and allow large2's
  // contiguous VSS to be reassembled.
  Delete(large2);
  {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    slack = allocator_->info().slack();
    donated_huge_pages = allocator_->DonatedHugePages();
  }
  EXPECT_EQ(slack, Length(0));
  EXPECT_EQ(donated_huge_pages, NHugePages(1));

  EXPECT_THAT(Print(), HasSubstr("filler donations 1"));
  EXPECT_THAT(PrintInPbTxt(), HasSubstr("filler_donated_huge_pages: 1"));
}

TEST_F(HugePageAwareAllocatorTest, PageMapInterference) {
  // This test manipulates the test HugePageAwareAllocator while making
  // allocations/deallocations that interact with the real PageAllocator. The
  // two share a global PageMap.
  //
  // If this test begins failing, the two are likely conflicting by violating
  // invariants in the PageMap.
  std::vector<Span*> allocs;

  for (int i : {10, 20, 30}) {
    auto n = Length(i << 7);
    allocs.push_back(New(n));
  }

  for (auto* a : allocs) {
    Delete(a);
  }

  allocs.clear();

  // Do the same, but allocate something on the real page heap.
  for (int i : {10, 20, 30}) {
    auto n = Length(i << 7);
    allocs.push_back(New(n));

    ::operator delete(::operator new(1 << 20));
  }

  for (auto* a : allocs) {
    Delete(a);
  }
}

static double BytesToMiB(size_t bytes) { return bytes / (1024.0 * 1024.0); }

TEST_F(HugePageAwareAllocatorTest, LargeSmall) {
  const int kIters = 2000;
  const Length kSmallPages = Length(1);
  // Large block must be larger than 1 huge page.
  const Length kLargePages = 2 * kPagesPerHugePage - kSmallPages;
  std::vector<Span*> small_allocs;

  // Repeatedly allocate large and small allocations that fit into a multiple of
  // huge pages.  The large allocations are short lived and the small
  // allocations are long-lived.  We want to refrain from growing the heap size
  // without bound, keeping many huge pages alive because of the small
  // allocations.
  for (int i = 0; i < kIters; i++) {
    Span* large = New(kLargePages);
    ASSERT_NE(large, nullptr);
    Span* small = New(kSmallPages);
    ASSERT_NE(small, nullptr);

    small_allocs.push_back(small);
    Delete(large);
  }

  BackingStats stats;
  {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    stats = allocator_->stats();
  }

  constexpr size_t kBufferSize = 1024 * 1024;
  char buffer[kBufferSize];
  TCMalloc_Printer printer(buffer, kBufferSize);
  allocator_->Print(&printer);
  // Verify that we have less free memory than we allocated in total. We have
  // to account for bytes tied up in the cache.
  EXPECT_LE(stats.free_bytes - allocator_->cache()->size().in_bytes(),
            kSmallPages.in_bytes() * kIters)
      << buffer;

  for (Span* small : small_allocs) {
    Delete(small);
  }
}

// Tests an edge case in hugepage donation behavior.
TEST_F(HugePageAwareAllocatorTest, DonatedPageLists) {
  const Length kSmallPages = Length(1);
  // Large block must be larger than 1 huge page.
  const Length kLargePages = 2 * kPagesPerHugePage - 2 * kSmallPages;

  Span* large = New(kLargePages);
  ASSERT_NE(large, nullptr);

  // Allocating small1 moves the backing huge page off of the donated pages
  // list.
  Span* small1 = New(kSmallPages);
  ASSERT_NE(small1, nullptr);
  // This delete needs to have put the origin PageTracker back onto the right
  // free list.
  Delete(small1);

  // This otherwise fails.
  Span* small2 = New(kSmallPages);
  ASSERT_NE(small2, nullptr);
  Delete(small2);

  // Clean up.
  Delete(large);
}

TEST_F(HugePageAwareAllocatorTest, DonationAccounting) {
  const Length kSmallPages = Length(2);
  const Length kOneHugePageDonation = kPagesPerHugePage - kSmallPages;
  const Length kMultipleHugePagesDonation = 3 * kPagesPerHugePage - kSmallPages;

  // Each of these allocations should count as one donation, but only if they
  // are actually being reused.
  Span* large = New(kOneHugePageDonation);
  ASSERT_NE(large, nullptr);

  // This allocation ensures that the donation is not counted.
  Span* small = New(kSmallPages);
  ASSERT_NE(small, nullptr);

  Span* large2 = New(kMultipleHugePagesDonation);
  ASSERT_NE(large2, nullptr);

  // This allocation ensures that the donation is not counted.
  Span* small2 = New(kSmallPages);
  ASSERT_NE(small2, nullptr);

  Span* large3 = New(kOneHugePageDonation);
  ASSERT_NE(large3, nullptr);

  Span* large4 = New(kMultipleHugePagesDonation);
  ASSERT_NE(large4, nullptr);

  // Clean up.
  Delete(large);
  Delete(large2);
  Delete(large3);
  Delete(large4);
  Delete(small);
  Delete(small2);

  // Check donation count.
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  CHECK_CONDITION(NHugePages(2) == allocator_->DonatedHugePages());
}

// We'd like to test OOM behavior but this, err, OOMs. :)
// (Usable manually in controlled environments.
TEST_F(HugePageAwareAllocatorTest, DISABLED_OOM) {
  std::vector<Span*> objs;
  auto n = Length(1);
  while (true) {
    Span* s = New(n);
    if (!s) break;
    objs.push_back(s);
    n *= 2;
  }
  for (auto s : objs) {
    Delete(s);
  }
}

struct MemoryBytes {
  uint64_t virt;
  uint64_t phys;
};

MemoryBytes operator-(MemoryBytes lhs, MemoryBytes rhs) {
  return {lhs.virt - rhs.virt, lhs.phys - rhs.phys};
}

int64_t pagesize = getpagesize();

static size_t BytesInCore(void* p, size_t len) {
  static const size_t kBufSize = 1024;
  unsigned char buf[kBufSize];
  const size_t kChunk = pagesize * kBufSize;
  size_t resident = 0;
  while (len > 0) {
    // We call mincore in bounded size chunks (though typically one
    // chunk will cover an entire request.)
    const size_t chunk_len = std::min(kChunk, len);
    if (mincore(p, chunk_len, buf) != 0) {
      Crash(kCrash, __FILE__, __LINE__, "mincore failed, errno", errno);
    }
    const size_t lim = chunk_len / pagesize;
    for (size_t i = 0; i < lim; ++i) {
      if (buf[i] & 1) resident += pagesize;
    }
    len -= chunk_len;
    p = static_cast<char*>(p) + chunk_len;
  }

  return resident;
}

// Is any page of this hugepage resident?
bool HugePageResident(HugePage p) {
  return BytesInCore(p.start_addr(), kHugePageSize) > 0;
}

void Touch(PageId p) {
  // a tcmalloc-page may contain more than an actual kernel page
  volatile char* base = reinterpret_cast<char*>(p.start_addr());
  static size_t kActualPages = std::max<size_t>(kPageSize / pagesize, 1);
  for (int i = 0; i < kActualPages; ++i) {
    base[i * pagesize] = 1;
  }
}

// Fault an entire hugepage, as if THP chose to do so on an entirely
// empty hugepage. (In real life, this will usually, but not always,
// happen: we make sure it does so our accounting is accurate.)
void Touch(HugePage hp) {
  PageId p = hp.first_page();
  const PageId lim = p + kPagesPerHugePage;
  while (p < lim) {
    Touch(p);
    ++p;
  }
}

void Touch(Span* s) {
  for (PageId p = s->first_page(); p <= s->last_page(); ++p) {
    Touch(p);
  }
}

// Fault in memory across a span (SystemBack doesn't always do this.)
void TouchTHP(Span* s) {
  PageId p = s->first_page();
  PageId lim = s->last_page();
  HugePage last = HugePageContaining(nullptr);
  while (p <= lim) {
    HugePage hp = HugePageContaining(p);
    // Suppose that we are touching a hugepage for the first time (it
    // is entirely non-resident.) The page fault we take will usually
    // be promoted to a full transparent hugepage, and our accounting
    // assumes this is true.  But we can't actually guarantee that
    // (the kernel won't wait if memory is too fragmented.)  Do it ourselves
    // by hand, to ensure our mincore() calculations return the right answers.
    if (hp != last && !HugePageResident(hp)) {
      last = hp;
      Touch(hp);
    }

    // Regardless of whether we've optimistically faulted in a
    // hugepage, we also touch each page in the span.
    Touch(p);
    ++p;
  }
}

// Similar to above but much more careful about touching memory / mallocing
// and without the validation
class StatTest : public testing::Test {
 protected:
  StatTest() : rng_() {}

  class RegionFactory;

  class Region : public AddressRegion {
   public:
    Region(AddressRegion* underlying, RegionFactory* factory)
        : underlying_(underlying), factory_(factory) {}

    std::pair<void*, size_t> Alloc(size_t size, size_t alignment) override {
      std::pair<void*, size_t> ret = underlying_->Alloc(size, alignment);
      if (!ret.first) return {nullptr, 0};

      // we only support so many allocations here for simplicity
      CHECK_CONDITION(factory_->n_ < factory_->kNumAllocs);
      // Anything coming from the test allocator will request full
      // alignment.  Metadata allocations will not.  Since we can't
      // control the backing of metadata allocations, elide them.
      // TODO(b/128521238): this is not a good way to do this.
      if (alignment >= kHugePageSize) {
        factory_->allocs_[factory_->n_] = ret;
        factory_->n_++;
      }
      return ret;
    }

   private:
    AddressRegion* underlying_;
    RegionFactory* factory_;
  };

  class RegionFactory : public AddressRegionFactory {
   public:
    explicit RegionFactory(AddressRegionFactory* underlying)
        : underlying_(underlying), n_(0) {}

    AddressRegion* Create(void* start, size_t size, UsageHint hint) override {
      AddressRegion* underlying_region = underlying_->Create(start, size, hint);
      CHECK_CONDITION(underlying_region);
      void* region_space = MallocInternal(sizeof(Region));
      CHECK_CONDITION(region_space);
      return new (region_space) Region(underlying_region, this);
    }

    size_t GetStats(absl::Span<char> buffer) override {
      return underlying_->GetStats(buffer);
    }

    MemoryBytes Memory() {
      MemoryBytes b = {0, 0};
      for (int i = 0; i < n_; ++i) {
        void* p = allocs_[i].first;
        size_t len = allocs_[i].second;
        b.virt += len;
        b.phys += BytesInCore(p, len);
      }

      return b;
    }

    AddressRegionFactory* underlying() const { return underlying_; }

   private:
    friend class Region;
    AddressRegionFactory* underlying_;

    static constexpr size_t kNumAllocs = 1000;
    size_t n_;
    std::pair<void*, size_t> allocs_[kNumAllocs];
  };

  // Carefully get memory usage without touching anything.
  MemoryBytes GetSystemBytes() { return replacement_region_factory_.Memory(); }

  // This is essentially a test case set up, but run manually -
  // we can't guarantee gunit won't malloc between.
  void PrepTest() {
    memset(buf, 0, sizeof(buf));
    MallocExtension::ReleaseMemoryToSystem(std::numeric_limits<size_t>::max());
    SetRegionFactory(&replacement_region_factory_);
    alloc = new (buf) HugePageAwareAllocator(MemoryTag::kNormal);
  }

  ~StatTest() override {
    SetRegionFactory(replacement_region_factory_.underlying());
  }

  BackingStats Stats() {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    BackingStats stats = alloc->stats();
    return stats;
  }

  // Use bigger allocs here to ensure growth:
  Length RandomAllocSize() {
    // Since we touch all of the pages, try to avoid OOM'ing by limiting the
    // number of big allocations.
    const Length kMaxBigAllocs = Length(4096);

    if (big_allocs_ < kMaxBigAllocs && absl::Bernoulli(rng_, 1.0 / 50)) {
      auto n =
          Length(1024 * (1 + absl::LogUniform<int32_t>(rng_, 0, (1 << 9) - 1)));
      n += Length(absl::Uniform<int32_t>(rng_, 0, 1024));
      big_allocs_ += n;
      return n;
    }
    return Length(1 + absl::LogUniform<int32_t>(rng_, 0, (1 << 10) - 1));
  }

  Span* Alloc(Length n) {
    Span* span = alloc->New(n);
    TouchTHP(span);
    if (n > span->num_pages()) {
      Crash(kCrash, __FILE__, __LINE__, n.raw_num(),
            "not <=", span->num_pages().raw_num());
    }
    n = span->num_pages();
    if (n > longest_) longest_ = n;
    total_ += n;
    if (total_ > peak_) peak_ = total_;
    return span;
  }

  void Free(Span* s) {
    Length n = s->num_pages();
    total_ -= n;
    {
      absl::base_internal::SpinLockHolder h(&pageheap_lock);
      alloc->Delete(s);
    }
  }

  void CheckStats() {
    MemoryBytes here = GetSystemBytes();
    BackingStats stats = Stats();
    SmallSpanStats small;
    LargeSpanStats large;
    {
      absl::base_internal::SpinLockHolder h(&pageheap_lock);
      alloc->GetSmallSpanStats(&small);
      alloc->GetLargeSpanStats(&large);
    }

    size_t span_stats_free_bytes = 0, span_stats_released_bytes = 0;
    for (auto i = Length(0); i < kMaxPages; ++i) {
      span_stats_free_bytes += i.in_bytes() * small.normal_length[i.raw_num()];
      span_stats_released_bytes +=
          i.in_bytes() * small.returned_length[i.raw_num()];
    }
    span_stats_free_bytes += large.normal_pages.in_bytes();
    span_stats_released_bytes += large.returned_pages.in_bytes();

#ifndef __ppc__
    const size_t alloced_bytes = total_.in_bytes();
#endif
    ASSERT_EQ(here.virt, stats.system_bytes);
#ifndef __ppc__
    const size_t actual_unmapped = here.virt - here.phys;
#endif
    // TODO(b/122551676):  On PPC, our release granularity may be smaller than
    // the system page size, so we may not actually unmap memory that we expect.
    // Pending using the return value of madvise, relax this constraint.
#ifndef __ppc__
    ASSERT_EQ(actual_unmapped, stats.unmapped_bytes);
    ASSERT_EQ(here.phys, stats.free_bytes + alloced_bytes);
    ASSERT_EQ(alloced_bytes,
              stats.system_bytes - stats.free_bytes - stats.unmapped_bytes);
#endif
    ASSERT_EQ(stats.free_bytes, span_stats_free_bytes);
    ASSERT_EQ(stats.unmapped_bytes, span_stats_released_bytes);
  }

  char buf[sizeof(HugePageAwareAllocator)];
  HugePageAwareAllocator* alloc;
  RegionFactory replacement_region_factory_{GetRegionFactory()};
  absl::BitGen rng_;

  Length total_;
  Length longest_;
  Length peak_;
  Length big_allocs_;
};

TEST_F(StatTest, Basic) {
  static const size_t kNumAllocs = 500;
  Span* allocs[kNumAllocs];

  const bool always_check_usage = absl::GetFlag(FLAGS_always_check_usage);

  PrepTest();
  // DO NOT MALLOC ANYTHING BELOW THIS LINE!  WE'RE TRYING TO CAREFULLY COUNT
  // ALLOCATIONS.
  // (note we can't stop background threads, but hopefully they're idle enough.)

  for (int i = 0; i < kNumAllocs; ++i) {
    Length k = RandomAllocSize();
    allocs[i] = Alloc(k);
    // stats are expensive, don't always check
    if (i % 10 != 0 && !always_check_usage) continue;
    CheckStats();
  }

  static const size_t kReps = 1000;
  for (int i = 0; i < kReps; ++i) {
    size_t index = absl::Uniform<int32_t>(rng_, 0, kNumAllocs);

    Free(allocs[index]);
    Length k = RandomAllocSize();
    allocs[index] = Alloc(k);

    if (absl::Bernoulli(rng_, 1.0 / 3)) {
      Length pages(absl::LogUniform<int32_t>(rng_, 0, (1 << 10) - 1) + 1);
      absl::base_internal::SpinLockHolder h(&pageheap_lock);
      alloc->ReleaseAtLeastNPages(pages);
    }

    // stats are expensive, don't always check
    if (i % 10 != 0 && !always_check_usage) continue;
    CheckStats();
  }

  for (int i = 0; i < kNumAllocs; ++i) {
    Free(allocs[i]);
    if (i % 10 != 0 && !always_check_usage) continue;
    CheckStats();
  }

  {
    CheckStats();
    pageheap_lock.Lock();
    auto final_stats = alloc->stats();
    pageheap_lock.Unlock();
    ASSERT_EQ(final_stats.free_bytes + final_stats.unmapped_bytes,
              final_stats.system_bytes);
  }

  // test over, malloc all you like
}

TEST_F(HugePageAwareAllocatorTest, ParallelRelease) {
  ThreadManager threads;
  constexpr int kThreads = 10;

  struct ABSL_CACHELINE_ALIGNED Metadata {
    absl::BitGen rng;
    std::vector<Span*> spans;
  };

  std::vector<Metadata> metadata;
  metadata.resize(kThreads);

  threads.Start(kThreads, [&](int thread_id) {
    Metadata& m = metadata[thread_id];

    if (thread_id == 0) {
      ReleasePages(Length(absl::Uniform(m.rng, 1, 1 << 10)));
      return;
    } else if (thread_id == 1) {
      benchmark::DoNotOptimize(Print());
      return;
    }

    if (absl::Bernoulli(m.rng, 0.6) || m.spans.empty()) {
      Span* s = AllocatorNew(Length(absl::LogUniform(m.rng, 1, 1 << 10)));
      CHECK_CONDITION(s != nullptr);

      // Touch the contents of the buffer.  We later use it to verify we are the
      // only thread manipulating the Span, for example, if another thread
      // madvise DONTNEED'd the contents and zero'd them.
      const uintptr_t key = reinterpret_cast<uintptr_t>(s) ^ thread_id;
      *reinterpret_cast<uintptr_t*>(s->start_address()) = key;

      m.spans.push_back(s);
    } else {
      size_t index = absl::Uniform<size_t>(m.rng, 0, m.spans.size());

      Span* back = m.spans.back();
      Span* s = m.spans[index];
      m.spans[index] = back;
      m.spans.pop_back();

      const uintptr_t key = reinterpret_cast<uintptr_t>(s) ^ thread_id;
      EXPECT_EQ(*reinterpret_cast<uintptr_t*>(s->start_address()), key);

      AllocatorDelete(s);
    }
  });

  absl::SleepFor(absl::Seconds(1));

  threads.Stop();

  for (auto& m : metadata) {
    for (Span* s : m.spans) {
      AllocatorDelete(s);
    }
  }
}

}  // namespace
}  // namespace tcmalloc
