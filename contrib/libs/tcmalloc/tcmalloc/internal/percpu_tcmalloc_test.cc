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

#include "tcmalloc/internal/percpu_tcmalloc.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/mman.h>

#include <atomic>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/internal/sysinfo.h"
#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/debugging/symbolize.h"
#include "absl/random/random.h"
#include "absl/random/seed_sequences.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/util.h"
#include "tcmalloc/malloc_extension.h"

namespace tcmalloc {
namespace subtle {
namespace percpu {
namespace {

using tcmalloc::tcmalloc_internal::AllowedCpus;
using tcmalloc::tcmalloc_internal::ScopedAffinityMask;
using testing::Each;
using testing::UnorderedElementsAreArray;

// Choose an available CPU and executes the passed functor on it. The
// cpu that is chosen, as long as a valid disjoint remote CPU will be passed
// as arguments to it.
//
// If the functor believes that it has failed in a manner attributable to
// external modification, then it should return false and we will attempt to
// retry the operation (up to a constant limit).
void RunOnSingleCpuWithRemoteCpu(std::function<bool(int, int)> test) {
  constexpr int kMaxTries = 1000;

  for (int i = 0; i < kMaxTries; i++) {
    auto allowed = AllowedCpus();

    int target_cpu = allowed[0], remote_cpu;

    // We try to pass something actually within the mask, but, for most tests it
    // only needs to exist.
    if (allowed.size() > 1)
      remote_cpu = allowed[1];
    else
      remote_cpu = target_cpu ? 0 : 1;

    ScopedAffinityMask mask(target_cpu);

    // If the test function failed, assert that the mask was tampered with.
    if (!test(target_cpu, remote_cpu))
      ASSERT_TRUE(mask.Tampered());
    else
      return;
  }

  ASSERT_TRUE(false);
}

// Equivalent to RunOnSingleCpuWithRemoteCpu, except that only the CPU the
// functor is executing on is passed.
void RunOnSingleCpu(std::function<bool(int)> test) {
  auto wrapper = [&test](int this_cpu, int unused) { return test(this_cpu); };
  RunOnSingleCpuWithRemoteCpu(wrapper);
}

// ScopedUnregisterRseq unregisters the current thread from rseq.  On
// destruction, it reregisters it with IsFast().
class ScopedUnregisterRseq {
 public:
  ScopedUnregisterRseq() {
    // Since we expect that we will be able to register the thread for rseq in
    // the destructor, verify that we can do so now.
    CHECK_CONDITION(IsFast());

    syscall(__NR_rseq, &__rseq_abi, sizeof(__rseq_abi), kRseqUnregister,
            TCMALLOC_PERCPU_RSEQ_SIGNATURE);

    // Clear __rseq_refcount.  Otherwise, when we reinitialize the TLS, we
    // will believe that the thread is already registered.
    //
    // IsFast -> InitThreadPerCpu() inspects __rseq_refcount and does not
    // attempt to register __rseq_abi if __rseq_refcount > 0--indiating another
    // library has already registered this thread's data with the kernel.
    __rseq_refcount = 0;

    // Unregistering stores kCpuIdUninitialized to the cpu_id field.
    CHECK_CONDITION(__rseq_abi.cpu_id == kCpuIdUninitialized);
  }

  ~ScopedUnregisterRseq() {
    // Since we have manipulated __rseq_abi.cpu_id, restore the value to
    // uninitialized so that we will successfully re-register.
    __rseq_abi.cpu_id = kCpuIdUninitialized;
    CHECK_CONDITION(IsFast());
  }
};

constexpr size_t kStressSlabs = 4;
constexpr size_t kStressCapacity = 4;

typedef class TcmallocSlab<18ul, kStressSlabs> TcmallocSlab;

enum class SlabInit {
  kEager,
  kLazy,
};

class TcmallocSlabTest : public testing::TestWithParam<SlabInit> {
 protected:
  TcmallocSlabTest() {
    slab_test_ = &slab_;
    metadata_bytes_ = 0;

    slab_.Init(
        &ByteCountingMalloc, [](size_t cl) { return kCapacity; },
        GetParam() == SlabInit::kLazy);

    for (int i = 0; i < kCapacity; ++i) {
      object_ptrs_[i] = &objects_[i];
    }
  }

  ~TcmallocSlabTest() override { slab_.Destroy(free); }

  template <int result>
  static int ExpectOverflow(int cpu, size_t cl, void* item) {
    EXPECT_EQ(cpu, current_cpu_);
    EXPECT_EQ(cl, current_cl_);
    EXPECT_FALSE(overflow_called_);
    overflow_called_ = true;
    return result;
  }

  template <size_t result_object>
  static void* ExpectUnderflow(int cpu, size_t cl) {
    EXPECT_EQ(cpu, current_cpu_);
    EXPECT_EQ(cl, current_cl_);
    EXPECT_LT(result_object, kCapacity);
    EXPECT_FALSE(underflow_called_);
    underflow_called_ = true;
    return &objects_[result_object];
  }

  template <int result>
  bool PushExpectOverflow(TcmallocSlab* slab, size_t cl, void* item) {
    bool res = slab->Push(cl, item, ExpectOverflow<result>);
    EXPECT_TRUE(overflow_called_);
    overflow_called_ = false;
    return res;
  }

  template <size_t result_object>
  void* PopExpectUnderflow(TcmallocSlab* slab, size_t cl) {
    void* res = slab->Pop(cl, ExpectUnderflow<result_object>);
    EXPECT_TRUE(underflow_called_);
    underflow_called_ = false;
    return res;
  }

  static void* ByteCountingMalloc(size_t size) {
    const size_t kPageSize = getpagesize();
    void* ptr;
    CHECK_CONDITION(posix_memalign(&ptr, kPageSize, size) == 0);
    if (ptr) {
      // Emulate obtaining memory as if we got it from mmap (zero'd).
      memset(ptr, 0, size);
      madvise(ptr, size, MADV_DONTNEED);
      metadata_bytes_ += size;
    }
    return ptr;
  }

  TcmallocSlab slab_;

  static constexpr size_t kCapacity = 10;
  static char objects_[kCapacity];
  static void* object_ptrs_[kCapacity];
  static int current_cpu_;
  static size_t current_cl_;
  static bool overflow_called_;
  static bool underflow_called_;
  static TcmallocSlab* slab_test_;
  static size_t metadata_bytes_;
};

static int ExpectNoOverflow(int cpu, size_t cl, void* item) {
  CHECK_CONDITION(false && "overflow is not expected");
  return 0;
}

static void* ExpectNoUnderflow(int cpu, size_t cl) {
  CHECK_CONDITION(false && "underflow is not expected");
  return nullptr;
}

char TcmallocSlabTest::objects_[TcmallocSlabTest::kCapacity];
void* TcmallocSlabTest::object_ptrs_[TcmallocSlabTest::kCapacity];
int TcmallocSlabTest::current_cpu_;
size_t TcmallocSlabTest::current_cl_;
bool TcmallocSlabTest::overflow_called_;
bool TcmallocSlabTest::underflow_called_;
TcmallocSlab* TcmallocSlabTest::slab_test_;
size_t TcmallocSlabTest::metadata_bytes_;

TEST_P(TcmallocSlabTest, Metadata) {
  PerCPUMetadataState r = slab_.MetadataMemoryUsage();

  ASSERT_GT(metadata_bytes_, 0);
  EXPECT_EQ(r.virtual_size, metadata_bytes_);
  if (GetParam() == SlabInit::kLazy) {
    EXPECT_EQ(r.resident_size, 0);

    if (!IsFast()) {
      GTEST_SKIP() << "Need fast percpu. Skipping.";
      return;
    }

    // Initialize a core.  Verify that the increased RSS is proportional to a
    // core.
    slab_.InitCPU(0, [](size_t cl) { return kCapacity; });

    r = slab_.MetadataMemoryUsage();
    // We may fault a whole hugepage, so round up the expected per-core share to
    // a full hugepage.
    size_t expected = r.virtual_size / absl::base_internal::NumCPUs();
    expected = (expected + kHugePageSize - 1) & ~(kHugePageSize - 1);

    // A single core may be less than the full slab for that core, since we do
    // not touch every page within the slab.
    EXPECT_GE(expected, r.resident_size);

    // Read stats from the slab.  This will fault additional memory.
    for (int cpu = 0, n = absl::base_internal::NumCPUs(); cpu < n; ++cpu) {
      // To inhibit optimization, verify the values are sensible.
      for (int cl = 0; cl < kStressSlabs; ++cl) {
        EXPECT_EQ(0, slab_.Length(cpu, cl));
        EXPECT_EQ(0, slab_.Capacity(cpu, cl));
      }
    }

    PerCPUMetadataState post_stats = slab_.MetadataMemoryUsage();
    EXPECT_LE(post_stats.resident_size, metadata_bytes_);
    EXPECT_GT(post_stats.resident_size, r.resident_size);
  } else {
    EXPECT_EQ(r.resident_size, metadata_bytes_);
  }
}

TEST_P(TcmallocSlabTest, Unit) {
  if (MallocExtension::PerCpuCachesActive()) {
    // This test unregisters rseq temporarily, as to decrease flakiness.
    GTEST_SKIP() << "per-CPU TCMalloc is incompatible with unregistering rseq";
  }

  if (!IsFast()) {
    GTEST_SKIP() << "Need fast percpu. Skipping.";
    return;
  }

#ifndef __ppc__
  // On platforms other than PPC, we use __rseq_abi.cpu_id to retrieve the CPU.
  // On PPC, we use a special purpose register, so we cannot fake the CPU.
  ScopedUnregisterRseq rseq;
#endif

  // Decide if we should expect a push or pop to be the first action on the CPU
  // slab to trigger initialization.
  absl::FixedArray<bool, 0> initialized(absl::base_internal::NumCPUs(),
                                        GetParam() != SlabInit::kLazy);

  for (auto cpu : AllowedCpus()) {
    SCOPED_TRACE(cpu);
#ifdef __ppc__
    ScopedAffinityMask aff(cpu);
#else
    __rseq_abi.cpu_id = cpu;

    if (UsingFlatVirtualCpus()) {
      __rseq_abi.vcpu_id = cpu ^ 1;
      cpu = cpu ^ 1;
    }
#endif
    current_cpu_ = cpu;

    for (size_t cl = 0; cl < kStressSlabs; ++cl) {
      SCOPED_TRACE(cl);
      current_cl_ = cl;

#ifdef __ppc__
      // This is imperfect but the window between operations below is small.  We
      // can make this more precise around individual operations if we see
      // measurable flakiness as a result.
      if (aff.Tampered()) break;
#endif

      // Check new slab state.
      ASSERT_EQ(slab_.Length(cpu, cl), 0);
      ASSERT_EQ(slab_.Capacity(cpu, cl), 0);

      if (!initialized[cpu]) {
        void* ptr = slab_.Pop(cl, [](int cpu, size_t cl) {
          slab_test_->InitCPU(cpu, [](size_t cl) { return kCapacity; });

          return static_cast<void*>(slab_test_);
        });

        ASSERT_TRUE(ptr == slab_test_);
        initialized[cpu] = true;
      }

      // Test overflow/underflow handlers.
      ASSERT_EQ(PopExpectUnderflow<5>(&slab_, cl), &objects_[5]);
      ASSERT_FALSE(PushExpectOverflow<-1>(&slab_, cl, &objects_[0]));
      ASSERT_FALSE(PushExpectOverflow<-2>(&slab_, cl, &objects_[0]));
      ASSERT_TRUE(PushExpectOverflow<0>(&slab_, cl, &objects_[0]));

      // Grow capacity to kCapacity / 2.
      ASSERT_EQ(slab_.Grow(cpu, cl, kCapacity / 2, kCapacity), kCapacity / 2);
      ASSERT_EQ(slab_.Length(cpu, cl), 0);
      ASSERT_EQ(slab_.Capacity(cpu, cl), kCapacity / 2);
      ASSERT_EQ(PopExpectUnderflow<5>(&slab_, cl), &objects_[5]);
      ASSERT_TRUE(slab_.Push(cl, &objects_[0], ExpectNoOverflow));
      ASSERT_EQ(slab_.Length(cpu, cl), 1);
      ASSERT_EQ(slab_.Capacity(cpu, cl), kCapacity / 2);
      ASSERT_EQ(slab_.Pop(cl, ExpectNoUnderflow), &objects_[0]);
      ASSERT_EQ(slab_.Length(cpu, cl), 0);
      for (size_t i = 0; i < kCapacity / 2; ++i) {
        ASSERT_TRUE(slab_.Push(cl, &objects_[i], ExpectNoOverflow));
        ASSERT_EQ(slab_.Length(cpu, cl), i + 1);
      }
      ASSERT_FALSE(PushExpectOverflow<-1>(&slab_, cl, &objects_[0]));
      for (size_t i = kCapacity / 2; i > 0; --i) {
        ASSERT_EQ(slab_.Pop(cl, ExpectNoUnderflow), &objects_[i - 1]);
        ASSERT_EQ(slab_.Length(cpu, cl), i - 1);
      }
      // Ensure that Shink don't underflow capacity.
      ASSERT_EQ(slab_.Shrink(cpu, cl, kCapacity), kCapacity / 2);
      ASSERT_EQ(slab_.Capacity(cpu, cl), 0);

      // Grow capacity to kCapacity.
      ASSERT_EQ(slab_.Grow(cpu, cl, kCapacity / 2, kCapacity), kCapacity / 2);
      // Ensure that grow don't overflow max capacity.
      ASSERT_EQ(slab_.Grow(cpu, cl, kCapacity, kCapacity), kCapacity / 2);
      ASSERT_EQ(slab_.Capacity(cpu, cl), kCapacity);
      for (size_t i = 0; i < kCapacity; ++i) {
        ASSERT_TRUE(slab_.Push(cl, &objects_[i], ExpectNoOverflow));
        ASSERT_EQ(slab_.Length(cpu, cl), i + 1);
      }
      ASSERT_FALSE(PushExpectOverflow<-1>(&slab_, cl, &objects_[0]));
      for (size_t i = kCapacity; i > 0; --i) {
        ASSERT_EQ(slab_.Pop(cl, ExpectNoUnderflow), &objects_[i - 1]);
        ASSERT_EQ(slab_.Length(cpu, cl), i - 1);
      }

      // Ensure that we can't shrink below length.
      ASSERT_TRUE(slab_.Push(cl, &objects_[0], ExpectNoOverflow));
      ASSERT_TRUE(slab_.Push(cl, &objects_[1], ExpectNoOverflow));
      ASSERT_EQ(slab_.Shrink(cpu, cl, kCapacity), kCapacity - 2);
      ASSERT_EQ(slab_.Capacity(cpu, cl), 2);

      // Test Drain.
      ASSERT_EQ(slab_.Grow(cpu, cl, 2, kCapacity), 2);
      slab_.Drain(cpu, &cl,
                  [](void* ctx, size_t cl, void** batch, size_t n, size_t cap) {
                    size_t mycl = *static_cast<size_t*>(ctx);
                    if (cl == mycl) {
                      ASSERT_EQ(n, 2);
                      ASSERT_EQ(cap, 4);
                      ASSERT_EQ(batch[0], &objects_[0]);
                      ASSERT_EQ(batch[1], &objects_[1]);
                    } else {
                      ASSERT_EQ(n, 0);
                      ASSERT_EQ(cap, 0);
                    }
                  });
      ASSERT_EQ(slab_.Length(cpu, cl), 0);
      ASSERT_EQ(slab_.Capacity(cpu, cl), 0);

      // Test PushBatch/PopBatch.
      void* batch[kCapacity + 1];
      for (size_t i = 0; i < kCapacity; ++i) {
        batch[i] = &objects_[i];
      }
      ASSERT_EQ(slab_.PopBatch(cl, batch, kCapacity), 0);
      ASSERT_EQ(slab_.PushBatch(cl, batch, kCapacity), 0);
      ASSERT_EQ(slab_.Grow(cpu, cl, kCapacity / 2, kCapacity), kCapacity / 2);
      ASSERT_EQ(slab_.PopBatch(cl, batch, kCapacity), 0);
      // Push a batch of size i into empty slab.
      for (size_t i = 1; i < kCapacity; ++i) {
        const size_t expect = std::min(i, kCapacity / 2);
        ASSERT_EQ(slab_.PushBatch(cl, batch, i), expect);
        ASSERT_EQ(slab_.Length(cpu, cl), expect);
        for (size_t j = 0; j < expect; ++j) {
          ASSERT_EQ(slab_.Pop(cl, ExpectNoUnderflow),
                    &objects_[j + (i - expect)]);
        }
        ASSERT_EQ(PopExpectUnderflow<5>(&slab_, cl), &objects_[5]);
      }
      // Push a batch of size i into non-empty slab.
      for (size_t i = 1; i < kCapacity / 2; ++i) {
        const size_t expect = std::min(i, kCapacity / 2 - i);
        ASSERT_EQ(slab_.PushBatch(cl, batch, i), i);
        ASSERT_EQ(slab_.PushBatch(cl, batch, i), expect);
        ASSERT_EQ(slab_.Length(cpu, cl), i + expect);
        for (size_t j = 0; j < expect; ++j) {
          ASSERT_EQ(slab_.Pop(cl, ExpectNoUnderflow),
                    static_cast<void*>(&objects_[j + (i - expect)]));
        }
        for (size_t j = 0; j < i; ++j) {
          ASSERT_EQ(slab_.Pop(cl, ExpectNoUnderflow),
                    static_cast<void*>(&objects_[j]));
        }
        ASSERT_EQ(PopExpectUnderflow<5>(&slab_, cl), &objects_[5]);
      }
      for (size_t i = 0; i < kCapacity + 1; ++i) {
        batch[i] = nullptr;
      }
      // Pop all elements in a single batch.
      for (size_t i = 1; i < kCapacity / 2; ++i) {
        for (size_t j = 0; j < i; ++j) {
          ASSERT_TRUE(slab_.Push(cl, &objects_[j], ExpectNoOverflow));
        }
        ASSERT_EQ(slab_.PopBatch(cl, batch, i), i);
        ASSERT_EQ(slab_.Length(cpu, cl), 0);
        ASSERT_EQ(PopExpectUnderflow<5>(&slab_, cl), &objects_[5]);

        ASSERT_THAT(absl::MakeSpan(&batch[0], i),
                    UnorderedElementsAreArray(&object_ptrs_[0], i));
        ASSERT_THAT(absl::MakeSpan(&batch[i], kCapacity - i), Each(nullptr));
        for (size_t j = 0; j < kCapacity + 1; ++j) {
          batch[j] = nullptr;
        }
      }
      // Pop half of elements in a single batch.
      for (size_t i = 1; i < kCapacity / 2; ++i) {
        for (size_t j = 0; j < i; ++j) {
          ASSERT_TRUE(slab_.Push(cl, &objects_[j], ExpectNoOverflow));
        }
        size_t want = std::max<size_t>(1, i / 2);
        ASSERT_EQ(slab_.PopBatch(cl, batch, want), want);
        ASSERT_EQ(slab_.Length(cpu, cl), i - want);

        for (size_t j = 0; j < i - want; ++j) {
          ASSERT_EQ(slab_.Pop(cl, ExpectNoUnderflow),
                    static_cast<void*>(&objects_[i - want - j - 1]));
        }

        ASSERT_EQ(PopExpectUnderflow<5>(&slab_, cl), &objects_[5]);

        ASSERT_GE(i, want);
        ASSERT_THAT(absl::MakeSpan(&batch[0], want),
                    UnorderedElementsAreArray(&object_ptrs_[i - want], want));
        ASSERT_THAT(absl::MakeSpan(&batch[want], kCapacity - want),
                    Each(nullptr));
        for (size_t j = 0; j < kCapacity + 1; ++j) {
          batch[j] = nullptr;
        }
      }
      // Pop 2x elements in a single batch.
      for (size_t i = 1; i < kCapacity / 2; ++i) {
        for (size_t j = 0; j < i; ++j) {
          ASSERT_TRUE(slab_.Push(cl, &objects_[j], ExpectNoOverflow));
        }
        ASSERT_EQ(slab_.PopBatch(cl, batch, i * 2), i);
        ASSERT_EQ(slab_.Length(cpu, cl), 0);
        ASSERT_EQ(PopExpectUnderflow<5>(&slab_, cl), &objects_[5]);

        ASSERT_THAT(absl::MakeSpan(&batch[0], i),
                    UnorderedElementsAreArray(&object_ptrs_[0], i));
        ASSERT_THAT(absl::MakeSpan(&batch[i], kCapacity - i), Each(nullptr));
        for (size_t j = 0; j < kCapacity + 1; ++j) {
          batch[j] = nullptr;
        }
      }
      ASSERT_EQ(slab_.Shrink(cpu, cl, kCapacity / 2), kCapacity / 2);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Instant, TcmallocSlabTest,
                         testing::Values(SlabInit::kEager, SlabInit::kLazy));

static void StressThread(size_t thread_id, TcmallocSlab* slab,
                         std::vector<void*>* block,
                         std::vector<absl::Mutex>* mutexes,
                         std::atomic<size_t>* capacity,
                         std::atomic<bool>* stop) {
  EXPECT_TRUE(IsFast());

  struct Handler {
    static int Overflow(int cpu, size_t cl, void* item) {
      EXPECT_GE(cpu, 0);
      EXPECT_LT(cpu, absl::base_internal::NumCPUs());
      EXPECT_LT(cl, kStressSlabs);
      EXPECT_NE(item, nullptr);
      return -1;
    }

    static void* Underflow(int cpu, size_t cl) {
      EXPECT_GE(cpu, 0);
      EXPECT_LT(cpu, absl::base_internal::NumCPUs());
      EXPECT_LT(cl, kStressSlabs);
      return nullptr;
    }
  };

  absl::BitGen rnd(absl::SeedSeq({thread_id}));
  while (!*stop) {
    size_t cl = absl::Uniform<int32_t>(rnd, 0, kStressSlabs);
    const int what = absl::Uniform<int32_t>(rnd, 0, 81);
    if (what < 10) {
      if (!block->empty()) {
        if (slab->Push(cl, block->back(), &Handler::Overflow)) {
          block->pop_back();
        }
      }
    } else if (what < 20) {
      if (void* item = slab->Pop(cl, &Handler::Underflow)) {
        block->push_back(item);
      }
    } else if (what < 30) {
      if (!block->empty()) {
        void* batch[kStressCapacity];
        size_t n = absl::Uniform<int32_t>(
                       rnd, 0, std::min(block->size(), kStressCapacity)) +
                   1;
        for (size_t i = 0; i < n; ++i) {
          batch[i] = block->back();
          block->pop_back();
        }
        size_t pushed = slab->PushBatch(cl, batch, n);
        EXPECT_LE(pushed, n);
        for (size_t i = 0; i < n - pushed; ++i) {
          block->push_back(batch[i]);
        }
      }
    } else if (what < 40) {
      void* batch[kStressCapacity];
      size_t n = absl::Uniform<int32_t>(rnd, 0, kStressCapacity) + 1;
      size_t popped = slab->PopBatch(cl, batch, n);
      EXPECT_LE(popped, n);
      for (size_t i = 0; i < popped; ++i) {
        block->push_back(batch[i]);
      }
    } else if (what < 50) {
      size_t n = absl::Uniform<int32_t>(rnd, 0, kStressCapacity) + 1;
      for (;;) {
        size_t c = capacity->load();
        n = std::min(n, c);
        if (n == 0) {
          break;
        }
        if (capacity->compare_exchange_weak(c, c - n)) {
          break;
        }
      }
      if (n != 0) {
        size_t res =
            slab->Grow(GetCurrentVirtualCpuUnsafe(), cl, n, kStressCapacity);
        EXPECT_LE(res, n);
        capacity->fetch_add(n - res);
      }
    } else if (what < 60) {
      size_t n =
          slab->Shrink(GetCurrentVirtualCpuUnsafe(), cl,
                       absl::Uniform<int32_t>(rnd, 0, kStressCapacity) + 1);
      capacity->fetch_add(n);
    } else if (what < 70) {
      size_t len = slab->Length(
          absl::Uniform<int32_t>(rnd, 0, absl::base_internal::NumCPUs()), cl);
      EXPECT_LE(len, kStressCapacity);
    } else if (what < 80) {
      size_t cap = slab->Capacity(
          absl::Uniform<int32_t>(rnd, 0, absl::base_internal::NumCPUs()), cl);
      EXPECT_LE(cap, kStressCapacity);
    } else {
      struct Context {
        TcmallocSlab* slab;
        std::vector<void*>* block;
        std::atomic<size_t>* capacity;
      };
      Context ctx = {slab, block, capacity};
      int cpu = absl::Uniform<int32_t>(rnd, 0, absl::base_internal::NumCPUs());
      if (mutexes->at(cpu).TryLock()) {
        slab->Drain(
            cpu, &ctx,
            [](void* arg, size_t cl, void** batch, size_t n, size_t cap) {
              Context* ctx = static_cast<Context*>(arg);
              EXPECT_LT(cl, kStressSlabs);
              EXPECT_LE(n, kStressCapacity);
              EXPECT_LE(cap, kStressCapacity);
              for (size_t i = 0; i < n; ++i) {
                EXPECT_NE(batch[i], nullptr);
                ctx->block->push_back(batch[i]);
              }
              ctx->capacity->fetch_add(cap);
            });
        mutexes->at(cpu).Unlock();
      }
    }
  }
}

static void* allocator(size_t bytes) {
  void* ptr = malloc(bytes);
  if (ptr) {
    memset(ptr, 0, bytes);
  }
  return ptr;
}

TEST(TcmallocSlab, Stress) {
  // The test creates 2 * NumCPUs() threads each executing all possible
  // operations on TcmallocSlab. After that we verify that no objects
  // lost/duplicated and that total capacity is preserved.

  if (!IsFast()) {
    GTEST_SKIP() << "Need fast percpu. Skipping.";
    return;
  }

  EXPECT_LE(kStressSlabs, kStressSlabs);
  TcmallocSlab slab;
  slab.Init(
      allocator,
      [](size_t cl) { return cl < kStressSlabs ? kStressCapacity : 0; }, false);
  std::vector<std::thread> threads;
  const int n_threads = 2 * absl::base_internal::NumCPUs();

  // Mutexes protect Drain operation on a CPU.
  std::vector<absl::Mutex> mutexes(absl::base_internal::NumCPUs());
  // Give each thread an initial set of local objects.
  std::vector<std::vector<void*>> blocks(n_threads);
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (size_t j = 0; j < kStressCapacity; ++j) {
      blocks[i].push_back(reinterpret_cast<void*>(i * kStressCapacity + j + 1));
    }
  }
  std::atomic<bool> stop(false);
  // Total capacity shared between all size classes and all CPUs.
  const int kTotalCapacity = blocks.size() * kStressCapacity * 3 / 4;
  std::atomic<size_t> capacity(kTotalCapacity);
  // Create threads and let them work for 5 seconds.
  threads.reserve(n_threads);
  for (size_t t = 0; t < n_threads; ++t) {
    threads.push_back(std::thread(StressThread, t, &slab, &blocks[t], &mutexes,
                                  &capacity, &stop));
  }
  absl::SleepFor(absl::Seconds(5));
  stop = true;
  for (auto& t : threads) {
    t.join();
  }
  // Collect objects and capacity from all slabs.
  std::set<void*> objects;
  struct Context {
    std::set<void*>* objects;
    std::atomic<size_t>* capacity;
  };
  Context ctx = {&objects, &capacity};
  for (int cpu = 0; cpu < absl::base_internal::NumCPUs(); ++cpu) {
    slab.Drain(cpu, &ctx,
               [](void* arg, size_t cl, void** batch, size_t n, size_t cap) {
                 Context* ctx = static_cast<Context*>(arg);
                 for (size_t i = 0; i < n; ++i) {
                   ctx->objects->insert(batch[i]);
                 }
                 ctx->capacity->fetch_add(cap);
               });
    for (size_t cl = 0; cl < kStressSlabs; ++cl) {
      EXPECT_EQ(slab.Length(cpu, cl), 0);
      EXPECT_EQ(slab.Capacity(cpu, cl), 0);
    }
  }
  for (const auto& b : blocks) {
    for (auto o : b) {
      objects.insert(o);
    }
  }
  EXPECT_EQ(objects.size(), blocks.size() * kStressCapacity);
  EXPECT_EQ(capacity.load(), kTotalCapacity);
  slab.Destroy(free);
}

TEST(TcmallocSlab, SMP) {
  // For the other tests here to be meaningful, we need multiple cores.
  ASSERT_GT(absl::base_internal::NumCPUs(), 1);
}

#if ABSL_INTERNAL_HAVE_ELF_SYMBOLIZE
static int FilterElfHeader(struct dl_phdr_info* info, size_t size, void* data) {
  *reinterpret_cast<uintptr_t*>(data) =
      reinterpret_cast<uintptr_t>(info->dlpi_addr);
  // No further iteration wanted.
  return 1;
}
#endif

TEST(TcmallocSlab, CriticalSectionMetadata) {
// We cannot inhibit --gc-sections, except on GCC or Clang 9-or-newer.
#if defined(__clang_major__) && __clang_major__ < 9
  GTEST_SKIP() << "--gc-sections cannot be inhibited on this compiler.";
#endif

  // We expect that restartable sequence critical sections (rseq_cs) are in the
  // __rseq_cs section (by convention, not hard requirement).  Additionally, for
  // each entry in that section, there should be a pointer to it in
  // __rseq_cs_ptr_array.
#if ABSL_INTERNAL_HAVE_ELF_SYMBOLIZE
  uintptr_t relocation = 0;
  dl_iterate_phdr(FilterElfHeader, &relocation);

  int fd = tcmalloc_internal::signal_safe_open("/proc/self/exe", O_RDONLY);
  ASSERT_NE(fd, -1);

  const kernel_rseq_cs* cs_start = nullptr;
  const kernel_rseq_cs* cs_end = nullptr;

  const kernel_rseq_cs** cs_array_start = nullptr;
  const kernel_rseq_cs** cs_array_end = nullptr;

  absl::debugging_internal::ForEachSection(
      fd, [&](const absl::string_view name, const ElfW(Shdr) & hdr) {
        uintptr_t start = relocation + reinterpret_cast<uintptr_t>(hdr.sh_addr);
        uintptr_t end =
            relocation + reinterpret_cast<uintptr_t>(hdr.sh_addr + hdr.sh_size);

        if (name == "__rseq_cs") {
          EXPECT_EQ(cs_start, nullptr);
          EXPECT_EQ(start % alignof(kernel_rseq_cs), 0);
          EXPECT_EQ(end % alignof(kernel_rseq_cs), 0);
          EXPECT_LT(start, end) << "__rseq_cs must not be empty";

          cs_start = reinterpret_cast<const kernel_rseq_cs*>(start);
          cs_end = reinterpret_cast<const kernel_rseq_cs*>(end);
        } else if (name == "__rseq_cs_ptr_array") {
          EXPECT_EQ(cs_array_start, nullptr);
          EXPECT_EQ(start % alignof(kernel_rseq_cs*), 0);
          EXPECT_EQ(end % alignof(kernel_rseq_cs*), 0);
          EXPECT_LT(start, end) << "__rseq_cs_ptr_array must not be empty";

          cs_array_start = reinterpret_cast<const kernel_rseq_cs**>(start);
          cs_array_end = reinterpret_cast<const kernel_rseq_cs**>(end);
        }

        return true;
      });

  close(fd);

  // The length of the array in multiples of rseq_cs should be the same as the
  // length of the array of pointers.
  ASSERT_EQ(cs_end - cs_start, cs_array_end - cs_array_start);

  // The array should not be empty.
  ASSERT_NE(cs_start, nullptr);

  absl::flat_hash_set<const kernel_rseq_cs*> cs_pointers;
  for (auto* ptr = cs_start; ptr != cs_end; ++ptr) {
    cs_pointers.insert(ptr);
  }

  absl::flat_hash_set<const kernel_rseq_cs*> cs_array_pointers;
  for (auto** ptr = cs_array_start; ptr != cs_array_end; ++ptr) {
    // __rseq_cs_ptr_array should have no duplicates.
    EXPECT_TRUE(cs_array_pointers.insert(*ptr).second);
  }

  EXPECT_THAT(cs_pointers, ::testing::ContainerEq(cs_array_pointers));
#endif
}

static void BM_PushPop(benchmark::State& state) {
  CHECK_CONDITION(IsFast());
  RunOnSingleCpu([&](int this_cpu) {
    const int kBatchSize = 32;
    TcmallocSlab slab;
    slab.Init(
        allocator, [](size_t cl) -> size_t { return kBatchSize; }, false);
    CHECK_CONDITION(slab.Grow(this_cpu, 0, kBatchSize, kBatchSize) ==
                    kBatchSize);
    void* batch[kBatchSize];
    for (int i = 0; i < kBatchSize; i++) {
      batch[i] = &batch[i];
    }
    for (auto _ : state) {
      for (size_t x = 0; x < kBatchSize; x++) {
        CHECK_CONDITION(slab.Push(0, batch[x], ExpectNoOverflow));
      }
      for (size_t x = 0; x < kBatchSize; x++) {
        CHECK_CONDITION(slab.Pop(0, ExpectNoUnderflow) ==
                        batch[kBatchSize - x - 1]);
      }
    }
    return true;
  });
}
BENCHMARK(BM_PushPop);

static void BM_PushPopBatch(benchmark::State& state) {
  CHECK_CONDITION(IsFast());
  RunOnSingleCpu([&](int this_cpu) {
    const int kBatchSize = 32;
    TcmallocSlab slab;
    slab.Init(
        allocator, [](size_t cl) -> size_t { return kBatchSize; }, false);
    CHECK_CONDITION(slab.Grow(this_cpu, 0, kBatchSize, kBatchSize) ==
                    kBatchSize);
    void* batch[kBatchSize];
    for (int i = 0; i < kBatchSize; i++) {
      batch[i] = &batch[i];
    }
    for (auto _ : state) {
      CHECK_CONDITION(slab.PushBatch(0, batch, kBatchSize) == kBatchSize);
      CHECK_CONDITION(slab.PopBatch(0, batch, kBatchSize) == kBatchSize);
    }
    return true;
  });
}
BENCHMARK(BM_PushPopBatch);

}  // namespace
}  // namespace percpu
}  // namespace subtle
}  // namespace tcmalloc
