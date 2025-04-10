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

#ifndef TCMALLOC_TRANSFER_CACHE_H_
#define TCMALLOC_TRANSFER_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <limits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/macros.h"
#include "absl/base/thread_annotations.h"
#include "absl/types/span.h"
#include "tcmalloc/central_freelist.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/transfer_cache_stats.h"

#ifndef TCMALLOC_SMALL_BUT_SLOW
#include "tcmalloc/transfer_cache_internals.h"
#endif

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

enum class TransferCacheImplementation {
  Legacy,
  None,
  Ring,
};

absl::string_view TransferCacheImplementationToLabel(
    TransferCacheImplementation type);

#ifndef TCMALLOC_SMALL_BUT_SLOW

class StaticForwarder {
 public:
  static size_t class_to_size(int size_class);
  static size_t num_objects_to_move(int size_class);
  static void *Alloc(size_t size, int alignment = kAlignment)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
};

// This transfer-cache is set up to be sharded per L3 cache. It is backed by
// the non-sharded "normal" TransferCacheManager.
class ShardedTransferCacheManager {
 public:
  constexpr ShardedTransferCacheManager() {}

  void Init() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  bool should_use(int cl) const { return active_for_class_[cl]; }

  size_t TotalBytes();

  void *Pop(int cl) {
    void *batch[1];
    const int got = cache_[get_index(cl)].tc.RemoveRange(cl, batch, 1);
    return got == 1 ? batch[0] : nullptr;
  }

  void Push(int cl, void *ptr) {
    cache_[get_index(cl)].tc.InsertRange(cl, {&ptr, 1});
  }

  // All caches not touched since last attempt will return all objects
  // to the non-sharded TransferCache.
  void Plunder() {
    if (cache_ == nullptr || num_shards_ == 0) return;
    for (int i = 0; i < num_shards_ * kNumClasses; ++i) {
      cache_[i].tc.TryPlunder(cache_[i].tc.freelist().size_class());
    }
  }

 private:
  // The Manager is set up so that stealing is disabled for this TransferCache.
  class Manager : public StaticForwarder {
   public:
    static constexpr int DetermineSizeClassToEvict() { return -1; }
    static constexpr bool MakeCacheSpace(int) { return false; }
    static constexpr bool ShrinkCache(int) { return false; }
  };

  // Forwards calls to the unsharded TransferCache.
  class BackingTransferCache {
   public:
    void Init(int cl) { size_class_ = cl; }
    void InsertRange(absl::Span<void *> batch) const;
    ABSL_MUST_USE_RESULT int RemoveRange(void **batch, int n) const;
    int size_class() const { return size_class_; }

   private:
    int size_class_ = -1;
  };

  using TransferCache =
      internal_transfer_cache::RingBufferTransferCache<BackingTransferCache,
                                                       Manager>;

  union Cache {
    constexpr Cache() : dummy(false) {}
    ~Cache() {}
    TransferCache tc;
    bool dummy;
  };

  int get_index(int cl) {
    const int cpu = tcmalloc::tcmalloc_internal::subtle::percpu::RseqCpuId();
    ASSERT(cpu < 256);
    ASSERT(cpu >= 0);
    return get_index(cpu, cl);
  }

  int get_index(int cpu, int cl) {
    const int shard = l3_cache_index_[cpu];
    ASSERT(shard < num_shards_);
    const int index = shard * kNumClasses + cl;
    ASSERT(index < num_shards_ * kNumClasses);
    return index;
  }

  // Mapping from cpu to the L3 cache used.
  uint8_t l3_cache_index_[CPU_SETSIZE] = {0};

  Cache *cache_ = nullptr;
  int num_shards_ = 0;
  bool active_for_class_[kNumClasses] = {false};
};

class TransferCacheManager : public StaticForwarder {
  template <typename CentralFreeList, typename Manager>
  friend class internal_transfer_cache::TransferCache;
  using TransferCache =
      internal_transfer_cache::TransferCache<tcmalloc_internal::CentralFreeList,
                                             TransferCacheManager>;

  template <typename CentralFreeList, typename Manager>
  friend class internal_transfer_cache::RingBufferTransferCache;
  using RingBufferTransferCache =
      internal_transfer_cache::RingBufferTransferCache<
          tcmalloc_internal::CentralFreeList, TransferCacheManager>;

 public:
  constexpr TransferCacheManager() : next_to_evict_(1) {}

  TransferCacheManager(const TransferCacheManager &) = delete;
  TransferCacheManager &operator=(const TransferCacheManager &) = delete;

  void Init() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    implementation_ = ChooseImplementation();
    for (int i = 0; i < kNumClasses; ++i) {
      if (implementation_ == TransferCacheImplementation::Ring) {
        new (&cache_[i].rbtc) RingBufferTransferCache(this, i);
      } else {
        new (&cache_[i].tc) TransferCache(this, i);
      }
    }
  }

  void AcquireInternalLocks() {
    for (int i = 0; i < kNumClasses; ++i) {
      if (implementation_ == TransferCacheImplementation::Ring) {
        cache_[i].rbtc.AcquireInternalLocks();
      } else {
        cache_[i].tc.AcquireInternalLocks();
      }
    }
  }

  void ReleaseInternalLocks() {
    for (int i = 0; i < kNumClasses; ++i) {
      if (implementation_ == TransferCacheImplementation::Ring) {
        cache_[i].rbtc.ReleaseInternalLocks();
      } else {
        cache_[i].tc.ReleaseInternalLocks();
      }
    }    
  }

  void InsertRange(int size_class, absl::Span<void *> batch) {
    if (implementation_ == TransferCacheImplementation::Ring) {
      cache_[size_class].rbtc.InsertRange(size_class, batch);
    } else {
      cache_[size_class].tc.InsertRange(size_class, batch);
    }
  }

  ABSL_MUST_USE_RESULT int RemoveRange(int size_class, void **batch, int n) {
    if (implementation_ == TransferCacheImplementation::Ring) {
      return cache_[size_class].rbtc.RemoveRange(size_class, batch, n);
    } else {
      return cache_[size_class].tc.RemoveRange(size_class, batch, n);
    }
  }

  // All caches which have not been modified since the last time this method has
  // been called will return all objects to the freelist.
  void Plunder() {
    for (int i = 0; i < kNumClasses; ++i) {
      if (implementation_ == TransferCacheImplementation::Ring) {
        cache_[i].rbtc.TryPlunder(i);
      } else {
        cache_[i].tc.TryPlunder(i);
      }
    }
  }

  // This is not const because the underlying ring-buffer transfer cache
  // function requires acquiring a lock.
  size_t tc_length(int size_class) {
    if (implementation_ == TransferCacheImplementation::Ring) {
      return cache_[size_class].rbtc.tc_length();
    } else {
      return cache_[size_class].tc.tc_length();
    }
  }

  TransferCacheStats GetHitRateStats(int size_class) const {
    if (implementation_ == TransferCacheImplementation::Ring) {
      return cache_[size_class].rbtc.GetHitRateStats();
    } else {
      return cache_[size_class].tc.GetHitRateStats();
    }
  }

  const CentralFreeList &central_freelist(int size_class) const {
    if (implementation_ == TransferCacheImplementation::Ring) {
      return cache_[size_class].rbtc.freelist();
    } else {
      return cache_[size_class].tc.freelist();
    }
  }

  TransferCacheImplementation implementation() const { return implementation_; }

 private:
  static TransferCacheImplementation ChooseImplementation();

  int DetermineSizeClassToEvict();
  bool ShrinkCache(int size_class) {
    if (implementation_ == TransferCacheImplementation::Ring) {
      return cache_[size_class].rbtc.ShrinkCache(size_class);
    } else {
      return cache_[size_class].tc.ShrinkCache(size_class);
    }
  }

  TransferCacheImplementation implementation_ =
      TransferCacheImplementation::Legacy;
  std::atomic<int32_t> next_to_evict_;
  union Cache {
    constexpr Cache() : dummy(false) {}
    ~Cache() {}

    TransferCache tc;
    RingBufferTransferCache rbtc;
    bool dummy;
  };
  Cache cache_[kNumClasses];
} ABSL_CACHELINE_ALIGNED;

#else

// For the small memory model, the transfer cache is not used.
class TransferCacheManager {
 public:
  constexpr TransferCacheManager() : freelist_() {}
  TransferCacheManager(const TransferCacheManager &) = delete;
  TransferCacheManager &operator=(const TransferCacheManager &) = delete;

  void Init() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    for (int i = 0; i < kNumClasses; ++i) {
      freelist_[i].Init(i);
    }
  }

  void InsertRange(int size_class, absl::Span<void *> batch) {
    freelist_[size_class].InsertRange(batch);
  }

  ABSL_MUST_USE_RESULT int RemoveRange(int size_class, void **batch, int n) {
    return freelist_[size_class].RemoveRange(batch, n);
  }

  static constexpr size_t tc_length(int size_class) { return 0; }

  static constexpr TransferCacheStats GetHitRateStats(int size_class) {
    return {0, 0, 0, 0};
  }

  const CentralFreeList &central_freelist(int size_class) const {
    return freelist_[size_class];
  }

  TransferCacheImplementation implementation() const {
    return TransferCacheImplementation::None;
  }

  void AcquireInternalLocks() {}
  void ReleaseInternalLocks() {}

 private:
  CentralFreeList freelist_[kNumClasses];
} ABSL_CACHELINE_ALIGNED;

// A trivial no-op implementation.
struct ShardedTransferCacheManager {
  static constexpr void Init() {}
  static constexpr bool should_use(int cl) { return false; }
  static constexpr void *Pop(int cl) { return nullptr; }
  static constexpr void Push(int cl, void *ptr) {}
  static constexpr size_t TotalBytes() { return 0; }
  static constexpr void Plunder() {}
};

#endif

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_TRANSFER_CACHE_H_
