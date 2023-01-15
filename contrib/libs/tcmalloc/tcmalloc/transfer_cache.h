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
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/macros.h"
#include "absl/base/thread_annotations.h"
#include "absl/types/span.h"
#include "tcmalloc/central_freelist.h"
#include "tcmalloc/common.h"
#include "tcmalloc/transfer_cache_stats.h"

#ifndef TCMALLOC_SMALL_BUT_SLOW
#include "tcmalloc/transfer_cache_internals.h"
#endif

namespace tcmalloc {

#ifndef TCMALLOC_SMALL_BUT_SLOW

class TransferCacheManager {
  template <typename CentralFreeList, typename Manager>
  friend class internal_transfer_cache::TransferCache;
  using TransferCache =
      internal_transfer_cache::TransferCache<CentralFreeList,
                                             TransferCacheManager>;

  template <typename CentralFreeList, typename Manager>
  friend class internal_transfer_cache::LockFreeTransferCache;
  using LockFreeTransferCache =
      internal_transfer_cache::LockFreeTransferCache<CentralFreeList,
                                                     TransferCacheManager>;

 public:
  constexpr TransferCacheManager()
      : use_lock_free_cache_(false), next_to_evict_(1) {}

  TransferCacheManager(const TransferCacheManager &) = delete;
  TransferCacheManager &operator=(const TransferCacheManager &) = delete;

  void Init() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    use_lock_free_cache_ = false;

    for (int i = 0; i < kNumClasses; ++i) {
      if (use_lock_free_cache_) {
        auto *c = &cache_[i].lock_free;
        new (c) LockFreeTransferCache(this, i);
        c->Init(i);
      } else {
        auto *c = &cache_[i].legacy;
        new (c) TransferCache(this, i);
        c->Init(i);
      }
    }
  }

  void InsertRange(int size_class, absl::Span<void *> batch, int n) {
    if (use_lock_free_cache_)
      cache_[size_class].lock_free.InsertRange(batch, n);
    else
      cache_[size_class].legacy.InsertRange(batch, n);
  }

  ABSL_MUST_USE_RESULT int RemoveRange(int size_class, void **batch, int n) {
    if (use_lock_free_cache_)
      return cache_[size_class].lock_free.RemoveRange(batch, n);
    else
      return cache_[size_class].legacy.RemoveRange(batch, n);
  }

  size_t central_length(int size_class) {
    if (use_lock_free_cache_)
      return cache_[size_class].lock_free.central_length();
    else
      return cache_[size_class].legacy.central_length();
  }

  size_t tc_length(int size_class) {
    if (use_lock_free_cache_)
      return cache_[size_class].lock_free.tc_length();
    else
      return cache_[size_class].legacy.tc_length();
  }

  size_t OverheadBytes(int size_class) {
    if (use_lock_free_cache_)
      return cache_[size_class].lock_free.OverheadBytes();
    else
      return cache_[size_class].legacy.OverheadBytes();
  }

  SpanStats GetSpanStats(int size_class) const {
    if (use_lock_free_cache_)
      return cache_[size_class].lock_free.GetSpanStats();
    else
      return cache_[size_class].legacy.GetSpanStats();
  }

  TransferCacheStats GetHitRateStats(int size_class) {
    if (use_lock_free_cache_)
      return cache_[size_class].lock_free.GetHitRateStats();
    else
      return cache_[size_class].legacy.GetHitRateStats();
  }

 private:
  static size_t class_to_size(int size_class);
  static size_t num_objects_to_move(int size_class);
  void *Alloc(size_t size) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  int DetermineSizeClassToEvict();
  bool ShrinkCache(int size_class) {
    if (use_lock_free_cache_)
      return cache_[size_class].lock_free.ShrinkCache();
    else
      return cache_[size_class].legacy.ShrinkCache();
  }
  bool GrowCache(int size_class) {
    if (use_lock_free_cache_)
      return cache_[size_class].lock_free.GrowCache();
    else
      return cache_[size_class].legacy.GrowCache();
  }

  bool use_lock_free_cache_;
  std::atomic<int32_t> next_to_evict_;
  union Cache {
    constexpr Cache() : dummy(false) {}
    ~Cache() {}

    LockFreeTransferCache lock_free;
    TransferCache legacy;
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

  void InsertRange(int size_class, absl::Span<void *> batch, int n) {
    freelist_[size_class].InsertRange(batch.data(), n);
  }

  ABSL_MUST_USE_RESULT int RemoveRange(int size_class, void **batch, int n) {
    return freelist_[size_class].RemoveRange(batch, n);
  }

  size_t central_length(int size_class) {
    return freelist_[size_class].length();
  }

  size_t tc_length(int size_class) { return 0; }

  size_t OverheadBytes(int size_class) {
    return freelist_[size_class].OverheadBytes();
  }

  SpanStats GetSpanStats(int size_class) const {
    return freelist_[size_class].GetSpanStats();
  }

  TransferCacheStats GetHitRateStats(int size_class) const {
    TransferCacheStats stats;
    stats.insert_hits = 0;
    stats.insert_misses = 0;
    stats.remove_hits = 0;
    stats.remove_misses = 0;
    return stats;
  }

 private:
  CentralFreeList freelist_[kNumClasses];
} ABSL_CACHELINE_ALIGNED;

#endif
}  // namespace tcmalloc

#endif  // TCMALLOC_TRANSFER_CACHE_H_
