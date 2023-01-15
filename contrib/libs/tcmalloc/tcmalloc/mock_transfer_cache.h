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

#include <random>

#include "gmock/gmock.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "tcmalloc/common.h"
#include "tcmalloc/mock_central_freelist.h"

namespace tcmalloc {

inline constexpr size_t kClassSize = 8;
inline constexpr size_t kNumToMove = 32;

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
  bool GrowCache(int);
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

  static constexpr int kMaxObjectsToMove = ::kMaxObjectsToMove;
  static constexpr int kMaxCapacityInBatches =
      TransferCache::kMaxCapacityInBatches;
  static constexpr int kInitialCapacityInBatches =
      TransferCache::kInitialCapacityInBatches;
  static constexpr int kBatchSize = Manager::num_objects_to_move(1);

  FakeTransferCacheEnvironment() : manager_(), cache_(&manager_) {
    cache_.Init(1);
  }

  ~FakeTransferCacheEnvironment() { Drain(); }

  void Shrink() { cache_.ShrinkCache(); }
  void Grow() { cache_.GrowCache(); }

  void Insert(int n) {
    std::vector<void*> bufs;
    while (n > 0) {
      int b = std::min(n, kBatchSize);
      bufs.resize(b);
      central_freelist().AllocateBatch(&bufs[0], b);
      cache_.InsertRange(absl::MakeSpan(bufs), b);
      n -= b;
    }
  }

  void Remove(int n) {
    std::vector<void*> bufs;
    while (n > 0) {
      int b = std::min(n, kBatchSize);
      bufs.resize(b);
      int removed = cache_.RemoveRange(&bufs[0], b);
      // Ensure we make progress.
      ASSERT_GT(removed, 0);
      ASSERT_LE(removed, b);
      central_freelist().FreeBatch(&bufs[0], removed);
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
    } else if (choice < 0.6) {
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

}  // namespace tcmalloc

#endif  // TCMALLOC_MOCK_TRANSFER_CACHE_H_
