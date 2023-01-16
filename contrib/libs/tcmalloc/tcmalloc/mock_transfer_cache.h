// Copyright 2020 The TCMalloc Authors
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

#ifndef TCMALLOC_MOCK_TRANSFER_CACHE_H_
#define TCMALLOC_MOCK_TRANSFER_CACHE_H_

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <random>

#include "gmock/gmock.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "tcmalloc/common.h"
#include "tcmalloc/mock_central_freelist.h"
#include "tcmalloc/transfer_cache_internals.h"

namespace tcmalloc {
namespace tcmalloc_internal {

inline constexpr size_t kClassSize = 8;
inline constexpr size_t kNumToMove = 32;
inline constexpr int kSizeClass = 0;

class FakeTransferCacheManagerBase {
 public:
  constexpr static size_t class_to_size(int size_class) { return kClassSize; }
  constexpr static size_t num_objects_to_move(int size_class) {
    // TODO(b/170732338): test with multiple different num_objects_to_move
    return kNumToMove;
  }
  void* Alloc(size_t size) {
    memory_.emplace_back(::operator new(size));
    return memory_.back().get();
  }
  struct Free {
    void operator()(void* b) { ::operator delete(b); }
  };

 private:
  std::vector<std::unique_ptr<void, Free>> memory_;
};

// TransferCacheManager with basic stubs for everything.
//
// Useful for benchmarks where you want to unrelated expensive operations.
class FakeTransferCacheManager : public FakeTransferCacheManagerBase {
 public:
  int DetermineSizeClassToEvict();
  bool ShrinkCache(int);
};

// TransferCacheManager which allows intercepting intersting methods.
//
// Useful for intrusive unit tests that want to verify internal behavior.
class RawMockTransferCacheManager : public FakeTransferCacheManagerBase {
 public:
  RawMockTransferCacheManager() : FakeTransferCacheManagerBase() {
    // We want single threaded tests to be deterministic, so we use a
    // deterministic generator.  Because we don't know about the threading for
    // our tests we cannot keep the generator in a local variable.
    ON_CALL(*this, ShrinkCache).WillByDefault([]() {
      thread_local std::mt19937 gen{0};
      return absl::Bernoulli(gen, 0.8);
    });
    ON_CALL(*this, GrowCache).WillByDefault([]() {
      thread_local std::mt19937 gen{0};
      return absl::Bernoulli(gen, 0.8);
    });
    ON_CALL(*this, DetermineSizeClassToEvict).WillByDefault([]() {
      thread_local std::mt19937 gen{0};
      return absl::Uniform<size_t>(gen, 1, kNumClasses);
    });
  }

  MOCK_METHOD(int, DetermineSizeClassToEvict, ());
  MOCK_METHOD(bool, ShrinkCache, (int size_class));
  MOCK_METHOD(bool, GrowCache, (int size_class));
};

using MockTransferCacheManager = testing::NiceMock<RawMockTransferCacheManager>;

// Wires up a largely functional TransferCache + TransferCacheManager +
// MockCentralFreeList.
//
// By default, it fills allocations and responds sensibly.  Because it backs
// onto malloc/free, it will detect leaks and memory misuse when run in asan or
// tsan.
//
// Exposes the underlying mocks to allow for more whitebox tests.
//
// Drains the cache and verifies that no data was lost in the destructor.
template <typename TransferCacheT>
class FakeTransferCacheEnvironment {
 public:
  using TransferCache = TransferCacheT;
  using Manager = typename TransferCache::Manager;
  using FreeList = typename TransferCache::FreeList;

  static constexpr int kMaxObjectsToMove =
      ::tcmalloc::tcmalloc_internal::kMaxObjectsToMove;
  static constexpr int kBatchSize = Manager::num_objects_to_move(1);

  FakeTransferCacheEnvironment() : manager_(), cache_(&manager_, 1) {}

  ~FakeTransferCacheEnvironment() { Drain(); }

  void Shrink() { cache_.ShrinkCache(kSizeClass); }
  void Grow() { cache_.GrowCache(kSizeClass); }

  void Insert(int n) {
    std::vector<void*> bufs;
    while (n > 0) {
      int b = std::min(n, kBatchSize);
      bufs.resize(b);
      central_freelist().AllocateBatch(&bufs[0], b);
      cache_.InsertRange(kSizeClass, absl::MakeSpan(bufs));
      n -= b;
    }
  }

  void Remove(int n) {
    std::vector<void*> bufs;
    while (n > 0) {
      int b = std::min(n, kBatchSize);
      bufs.resize(b);
      int removed = cache_.RemoveRange(kSizeClass, &bufs[0], b);
      // Ensure we make progress.
      ASSERT_GT(removed, 0);
      ASSERT_LE(removed, b);
      central_freelist().FreeBatch({&bufs[0], static_cast<size_t>(removed)});
      n -= removed;
    }
  }

  void Drain() { Remove(cache_.tc_length()); }

  void RandomlyPoke() {
    absl::BitGen gen;
    // We want a probabilistic steady state size:
    // - grow/shrink balance on average
    // - insert/remove balance on average
    double choice = absl::Uniform(gen, 0.0, 1.0);
    if (choice < 0.1) {
      Shrink();
    } else if (choice < 0.2) {
      Grow();
    } else if (choice < 0.3) {
      cache_.HasSpareCapacity(kSizeClass);
    } else if (choice < 0.65) {
      Insert(absl::Uniform(gen, 1, kBatchSize));
    } else {
      Remove(absl::Uniform(gen, 1, kBatchSize));
    }
  }

  TransferCache& transfer_cache() { return cache_; }

  Manager& transfer_cache_manager() { return manager_; }

  FreeList& central_freelist() { return cache_.freelist(); }

 private:
  Manager manager_;
  TransferCache cache_;
};

// A fake transfer cache manager class which supports two size classes instead
// of just the one. To make this work, we have to store the transfer caches
// inside the cache manager, like in production code.
template <typename FreeListT,
          template <typename FreeList, typename Manager> class TransferCacheT>
class TwoSizeClassManager : public FakeTransferCacheManagerBase {
 public:
  using FreeList = FreeListT;
  using TransferCache = TransferCacheT<FreeList, TwoSizeClassManager>;

  // This is 3 instead of 2 because we hard code cl == 0 to be invalid in many
  // places. We only use cl 1 and 2 here.
  static constexpr int kSizeClasses = 3;
  static constexpr size_t kClassSize1 = 8;
  static constexpr size_t kClassSize2 = 16;
  static constexpr size_t kNumToMove1 = 32;
  static constexpr size_t kNumToMove2 = 16;

  TwoSizeClassManager() {
    caches_.push_back(absl::make_unique<TransferCache>(this, 0));
    caches_.push_back(absl::make_unique<TransferCache>(this, 1));
    caches_.push_back(absl::make_unique<TransferCache>(this, 2));
  }

  constexpr static size_t class_to_size(int size_class) {
    switch (size_class) {
      case 1:
        return kClassSize1;
      case 2:
        return kClassSize2;
      default:
        return 0;
    }
  }
  constexpr static size_t num_objects_to_move(int size_class) {
    switch (size_class) {
      case 1:
        return kNumToMove1;
      case 2:
        return kNumToMove2;
      default:
        return 0;
    }
  }

  int DetermineSizeClassToEvict() { return evicting_from_; }

  bool ShrinkCache(int size_class) {
    return caches_[size_class]->ShrinkCache(size_class);
  }

  FreeList& central_freelist(int cl) { return caches_[cl]->freelist(); }

  void InsertRange(int cl, absl::Span<void*> batch) {
    caches_[cl]->InsertRange(cl, batch);
  }

  int RemoveRange(int cl, void** batch, int N) {
    return caches_[cl]->RemoveRange(cl, batch, N);
  }

  bool HasSpareCapacity(int cl) { return caches_[cl]->HasSpareCapacity(cl); }

  size_t tc_length(int cl) { return caches_[cl]->tc_length(); }

  std::vector<std::unique_ptr<TransferCache>> caches_;

  // From which size class to evict.
  int evicting_from_ = 1;
};

template <template <typename FreeList, typename Manager> class TransferCacheT>
class TwoSizeClassEnv {
 public:
  using FreeList = MockCentralFreeList;
  using Manager = TwoSizeClassManager<FreeList, TransferCacheT>;
  using TransferCache = typename Manager::TransferCache;

  static constexpr int kMaxObjectsToMove =
      ::tcmalloc::tcmalloc_internal::kMaxObjectsToMove;

  explicit TwoSizeClassEnv() = default;

  ~TwoSizeClassEnv() { Drain(); }

  void Insert(int cl, int n) {
    const size_t batch_size = Manager::num_objects_to_move(cl);
    std::vector<void*> bufs;
    while (n > 0) {
      int b = std::min<int>(n, batch_size);
      bufs.resize(b);
      central_freelist(cl).AllocateBatch(&bufs[0], b);
      manager_.InsertRange(cl, absl::MakeSpan(bufs));
      n -= b;
    }
  }

  void Remove(int cl, int n) {
    const size_t batch_size = Manager::num_objects_to_move(cl);
    std::vector<void*> bufs;
    while (n > 0) {
      const int b = std::min<int>(n, batch_size);
      bufs.resize(b);
      const int removed = manager_.RemoveRange(cl, &bufs[0], b);
      // Ensure we make progress.
      ASSERT_GT(removed, 0);
      ASSERT_LE(removed, b);
      central_freelist(cl).FreeBatch({&bufs[0], static_cast<size_t>(removed)});
      n -= removed;
    }
  }

  void Drain() {
    for (int i = 0; i < Manager::kSizeClasses; ++i) {
      Remove(i, manager_.tc_length(i));
    }
  }

  Manager& transfer_cache_manager() { return manager_; }

  FreeList& central_freelist(int cl) { return manager_.central_freelist(cl); }

 private:
  Manager manager_;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc

#endif  // TCMALLOC_MOCK_TRANSFER_CACHE_H_
