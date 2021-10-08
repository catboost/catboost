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

#ifndef TCMALLOC_TRANSFER_CACHE_INTERNAL_H_
#define TCMALLOC_TRANSFER_CACHE_INTERNAL_H_

#include <sched.h>
#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <limits>

#include "absl/numeric/bits.h"
#include "tcmalloc/internal/config.h"

#ifdef __x86_64__
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/casts.h"
#include "absl/base/const_init.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/internal/futex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tcmalloc/central_freelist.h"
#include "tcmalloc/common.h"
#include "tcmalloc/experiment.h"
#include "tcmalloc/internal/atomic_stats_counter.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/tracking.h"
#include "tcmalloc/transfer_cache_stats.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc::tcmalloc_internal::internal_transfer_cache {

struct alignas(8) SizeInfo {
  int32_t used;
  int32_t capacity;
};
static constexpr int kMaxCapacityInBatches = 64;
static constexpr int kInitialCapacityInBatches = 16;

// TransferCache is used to cache transfers of
// sizemap.num_objects_to_move(size_class) back and forth between
// thread caches and the central cache for a given size class.
template <typename CentralFreeList, typename TransferCacheManager>
class TransferCache {
 public:
  using Manager = TransferCacheManager;
  using FreeList = CentralFreeList;

  TransferCache(Manager *owner, int cl)
      : TransferCache(owner, cl, CapacityNeeded(cl)) {}

  struct Capacity {
    int capacity;
    int max_capacity;
  };

  TransferCache(Manager *owner, int cl, Capacity capacity)
      : owner_(owner),
        lock_(absl::kConstInit, absl::base_internal::SCHEDULE_KERNEL_ONLY),
        max_capacity_(capacity.max_capacity),
        slot_info_(SizeInfo({0, capacity.capacity})),
        low_water_mark_(std::numeric_limits<int>::max()),
        slots_(nullptr),
        freelist_do_not_access_directly_() {
    freelist().Init(cl);
    slots_ = max_capacity_ != 0 ? reinterpret_cast<void **>(owner_->Alloc(
                                      max_capacity_ * sizeof(void *)))
                                : nullptr;
  }

  TransferCache(const TransferCache &) = delete;
  TransferCache &operator=(const TransferCache &) = delete;

  // Compute initial and max capacity that we should configure this cache for.
  static Capacity CapacityNeeded(size_t cl) {
    // We need at least 2 slots to store list head and tail.
    static_assert(kMinObjectsToMove >= 2);

    const size_t bytes = Manager::class_to_size(cl);
    if (cl <= 0 || bytes <= 0) return {0, 0};

    // Limit the maximum size of the cache based on the size class.  If this
    // is not done, large size class objects will consume a lot of memory if
    // they just sit in the transfer cache.
    const size_t objs_to_move = Manager::num_objects_to_move(cl);
    ASSERT(objs_to_move > 0);

    // Starting point for the maximum number of entries in the transfer cache.
    // This actual maximum for a given size class may be lower than this
    // maximum value.
    int max_capacity = kMaxCapacityInBatches * objs_to_move;
    // A transfer cache freelist can have anywhere from 0 to
    // max_capacity_ slots to put link list chains into.
    int capacity = kInitialCapacityInBatches * objs_to_move;

    // Limit each size class cache to at most 1MB of objects or one entry,
    // whichever is greater. Total transfer cache memory used across all
    // size classes then can't be greater than approximately
    // 1MB * kMaxNumTransferEntries.
    max_capacity = std::min<int>(
        max_capacity,
        std::max<int>(objs_to_move,
                      (1024 * 1024) / (bytes * objs_to_move) * objs_to_move));
    capacity = std::min(capacity, max_capacity);

    return {capacity, max_capacity};
  }

  // This transfercache implementation does not deal well with non-batch sized
  // inserts and removes.
  static constexpr bool IsFlexible() { return false; }

  // These methods all do internal locking.

  // Insert the specified batch into the transfer cache.  N is the number of
  // elements in the range.  RemoveRange() is the opposite operation.
  void InsertRange(int size_class, absl::Span<void *> batch)
      ABSL_LOCKS_EXCLUDED(lock_) {
    const int N = batch.size();
    const int B = Manager::num_objects_to_move(size_class);
    ASSERT(0 < N && N <= B);
    auto info = slot_info_.load(std::memory_order_relaxed);
    if (N == B) {
      if (info.used + N <= max_capacity_) {
        absl::base_internal::SpinLockHolder h(&lock_);
        if (MakeCacheSpace(size_class, N)) {
          // MakeCacheSpace can drop the lock, so refetch
          info = slot_info_.load(std::memory_order_relaxed);
          info.used += N;
          SetSlotInfo(info);

          void **entry = GetSlot(info.used - N);
          memcpy(entry, batch.data(), sizeof(void *) * N);
          tracking::Report(kTCInsertHit, size_class, 1);
          insert_hits_.LossyAdd(1);
          return;
        }
      }

      insert_misses_.Add(1);
    } else {
      insert_non_batch_misses_.Add(1);
    }

    tracking::Report(kTCInsertMiss, size_class, 1);
    freelist().InsertRange(batch);
  }

  // Returns the actual number of fetched elements and stores elements in the
  // batch.
  ABSL_MUST_USE_RESULT int RemoveRange(int size_class, void **batch, int N)
      ABSL_LOCKS_EXCLUDED(lock_) {
    ASSERT(N > 0);
    const int B = Manager::num_objects_to_move(size_class);
    auto info = slot_info_.load(std::memory_order_relaxed);
    if (N == B) {
      if (info.used >= N) {
        absl::base_internal::SpinLockHolder h(&lock_);
        // Refetch with the lock
        info = slot_info_.load(std::memory_order_relaxed);
        if (info.used >= N) {
          info.used -= N;
          SetSlotInfo(info);
          void **entry = GetSlot(info.used);
          memcpy(batch, entry, sizeof(void *) * N);
          tracking::Report(kTCRemoveHit, size_class, 1);
          remove_hits_.LossyAdd(1);
          low_water_mark_.store(
              std::min(low_water_mark_.load(std::memory_order_acquire),
                       info.used),
              std::memory_order_release);
          return N;
        }
      }

      remove_misses_.Add(1);
    } else {
      remove_non_batch_misses_.Add(1);
    }
    low_water_mark_.store(0, std::memory_order_release);

    tracking::Report(kTCRemoveMiss, size_class, 1);
    return freelist().RemoveRange(batch, N);
  }

  // If this object has not been touched since the last attempt, then
  // return all objects to 'freelist()'.
  void TryPlunder(int size_class) ABSL_LOCKS_EXCLUDED(lock_) {
    if (max_capacity_ == 0) return;
    int low_water_mark = low_water_mark_.load(std::memory_order_acquire);
    low_water_mark_.store(std::numeric_limits<int>::max(),
                          std::memory_order_release);
    while (low_water_mark > 0) {
      if (!lock_.TryLock()) return;
      if (low_water_mark_.load(std::memory_order_acquire) !=
          std::numeric_limits<int>::max()) {
        lock_.Unlock();
        return;
      }
      const int B = Manager::num_objects_to_move(size_class);
      SizeInfo info = GetSlotInfo();
      if (info.used == 0) {
        lock_.Unlock();
        return;
      }
      const size_t num_to_move = std::min(B, info.used);
      void *buf[kMaxObjectsToMove];
      void **const entry = GetSlot(info.used - B);
      memcpy(buf, entry, sizeof(void *) * B);
      info.used -= num_to_move;
      low_water_mark -= num_to_move;
      SetSlotInfo(info);
      lock_.Unlock();
      tracking::Report(kTCElementsPlunder, size_class, num_to_move);
      freelist().InsertRange({buf, num_to_move});
    }
  }
  // Returns the number of free objects in the transfer cache.
  size_t tc_length() const {
    return static_cast<size_t>(slot_info_.load(std::memory_order_relaxed).used);
  }

  // Returns the number of transfer cache insert/remove hits/misses.
  TransferCacheStats GetHitRateStats() const ABSL_LOCKS_EXCLUDED(lock_) {
    TransferCacheStats stats;

    stats.insert_hits = insert_hits_.value();
    stats.remove_hits = remove_hits_.value();
    stats.insert_misses = insert_misses_.value();
    stats.insert_non_batch_misses = insert_non_batch_misses_.value();
    stats.remove_misses = remove_misses_.value();
    stats.remove_non_batch_misses = remove_non_batch_misses_.value();

    // For performance reasons, we only update a single atomic as part of the
    // actual allocation operation.  For reporting, we keep reporting all
    // misses together and separately break-out how many of those misses were
    // non-batch sized.
    stats.insert_misses += stats.insert_non_batch_misses;
    stats.remove_misses += stats.remove_non_batch_misses;

    return stats;
  }

  SizeInfo GetSlotInfo() const {
    return slot_info_.load(std::memory_order_relaxed);
  }

  // REQUIRES: lock is held.
  // Tries to make room for N elements. If the cache is full it will try to
  // expand it at the cost of some other cache size.  Return false if there is
  // no space.
  bool MakeCacheSpace(int size_class, int N)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    auto info = slot_info_.load(std::memory_order_relaxed);
    // Is there room in the cache?
    if (info.used + N <= info.capacity) return true;
    // Check if we can expand this cache?
    if (info.capacity + N > max_capacity_) return false;

    int to_evict = owner_->DetermineSizeClassToEvict();
    if (to_evict == size_class) return false;

    // Release the held lock before the other instance tries to grab its lock.
    lock_.Unlock();
    bool made_space = owner_->ShrinkCache(to_evict);
    lock_.Lock();

    if (!made_space) return false;

    // Succeeded in evicting, we're going to make our cache larger.  However, we
    // may have dropped and re-acquired the lock, so the cache_size may have
    // changed.  Therefore, check and verify that it is still OK to increase the
    // cache_size.
    info = slot_info_.load(std::memory_order_relaxed);
    if (info.capacity + N > max_capacity_) return false;
    info.capacity += N;
    SetSlotInfo(info);
    return true;
  }

  bool HasSpareCapacity(int size_class) const {
    int n = Manager::num_objects_to_move(size_class);
    auto info = GetSlotInfo();
    return info.capacity - info.used >= n;
  }

  // Takes lock_ and invokes MakeCacheSpace() on this cache.  Returns true if it
  // succeeded at growing the cache by a batch size.
  bool GrowCache(int size_class) ABSL_LOCKS_EXCLUDED(lock_) {
    absl::base_internal::SpinLockHolder h(&lock_);
    return MakeCacheSpace(size_class, Manager::num_objects_to_move(size_class));
  }

  // REQUIRES: lock_ is *not* held.
  // Tries to shrink the Cache.  Return false if it failed to shrink the cache.
  // Decreases cache_slots_ on success.
  bool ShrinkCache(int size_class) ABSL_LOCKS_EXCLUDED(lock_) {
    int N = Manager::num_objects_to_move(size_class);

    void *to_free[kMaxObjectsToMove];
    int num_to_free;
    {
      absl::base_internal::SpinLockHolder h(&lock_);
      auto info = slot_info_.load(std::memory_order_relaxed);
      if (info.capacity == 0) return false;
      if (info.capacity < N) return false;

      N = std::min(N, info.capacity);
      int unused = info.capacity - info.used;
      if (N <= unused) {
        info.capacity -= N;
        SetSlotInfo(info);
        return true;
      }

      num_to_free = N - unused;
      info.capacity -= N;
      info.used -= num_to_free;
      SetSlotInfo(info);

      // Our internal slot array may get overwritten as soon as we drop the
      // lock, so copy the items to free to an on stack buffer.
      memcpy(to_free, GetSlot(info.used), sizeof(void *) * num_to_free);
    }

    // Access the freelist without holding the lock.
    freelist().InsertRange({to_free, static_cast<uint64_t>(num_to_free)});
    return true;
  }

  // This is a thin wrapper for the CentralFreeList.  It is intended to ensure
  // that we are not holding lock_ when we access it.
  ABSL_ATTRIBUTE_ALWAYS_INLINE FreeList &freelist() ABSL_LOCKS_EXCLUDED(lock_) {
    return freelist_do_not_access_directly_;
  }

  // The const version of the wrapper, needed to call stats on
  ABSL_ATTRIBUTE_ALWAYS_INLINE const FreeList &freelist() const
      ABSL_LOCKS_EXCLUDED(lock_) {
    return freelist_do_not_access_directly_;
  }

  void AcquireInternalLocks()
  {
    freelist().AcquireInternalLocks();
    lock_.Lock();
  }

  void ReleaseInternalLocks()
  {
    lock_.Unlock();
    freelist().ReleaseInternalLocks();
  }

 private:
  // Returns first object of the i-th slot.
  void **GetSlot(size_t i) ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    return slots_ + i;
  }

  void SetSlotInfo(SizeInfo info) {
    ASSERT(0 <= info.used);
    ASSERT(info.used <= info.capacity);
    ASSERT(info.capacity <= max_capacity_);
    slot_info_.store(info, std::memory_order_relaxed);
  }

  Manager *const owner_;

  // This lock protects all the data members.  used_slots_ and cache_slots_
  // may be looked at without holding the lock.
  absl::base_internal::SpinLock lock_;

  // Maximum size of the cache.
  const int32_t max_capacity_;

  // insert_hits_ and remove_hits_ are logically guarded by lock_ for mutations
  // and use LossyAdd, but the thread annotations cannot indicate that we do not
  // need a lock for reads.
  StatsCounter insert_hits_;
  StatsCounter remove_hits_;
  // Miss counters do not hold lock_, so they use Add.
  StatsCounter insert_misses_;
  StatsCounter insert_non_batch_misses_;
  StatsCounter remove_misses_;
  StatsCounter remove_non_batch_misses_;

  // Number of currently used and available cached entries in slots_. This
  // variable is updated under a lock but can be read without one.
  // INVARIANT: [0 <= slot_info_.used <= slot_info.capacity <= max_cache_slots_]
  std::atomic<SizeInfo> slot_info_;

  // Lowest value of "slot_info_.used" since last call to TryPlunder. All
  // elements not used for a full cycle (2 seconds) are unlikely to get used
  // again.
  std::atomic<int> low_water_mark_;

  // Pointer to array of free objects.  Use GetSlot() to get pointers to
  // entries.
  void **slots_ ABSL_GUARDED_BY(lock_);

  FreeList freelist_do_not_access_directly_;
} ABSL_CACHELINE_ALIGNED;

struct RingBufferSizeInfo {
  // The starting index of data stored in the ring buffer.
  int32_t start;
  // How many elements are stored.
  int32_t used;
  // How many elements are allowed to be stored at most.
  int32_t capacity;
};

// RingBufferTransferCache is a transfer cache which stores cache entries in a
// ring buffer instead of a stack.
template <typename CentralFreeList, typename TransferCacheManager>
class RingBufferTransferCache {
 public:
  using Manager = TransferCacheManager;
  using FreeList = CentralFreeList;

  RingBufferTransferCache(Manager *owner, int cl)
      : RingBufferTransferCache(owner, cl, CapacityNeeded(cl)) {}

  RingBufferTransferCache(
      Manager *owner, int cl,
      typename TransferCache<CentralFreeList, TransferCacheManager>::Capacity
          capacity)
      : lock_(absl::kConstInit, absl::base_internal::SCHEDULE_KERNEL_ONLY),
        slot_info_(RingBufferSizeInfo({0, 0, capacity.capacity})),
        max_capacity_(capacity.max_capacity),
        freelist_do_not_access_directly_(),
        owner_(owner) {
    freelist().Init(cl);
    if (max_capacity_ == 0) {
      // We don't allocate a buffer. Set slots_bitmask_ to 0 to prevent UB.
      slots_bitmask_ = 0;
    } else {
      const size_t slots_size = absl::bit_ceil<size_t>(max_capacity_);
      ASSERT(slots_size >= max_capacity_);
      ASSERT(slots_size < max_capacity_ * 2);
      slots_ =
          reinterpret_cast<void **>(owner_->Alloc(slots_size * sizeof(void *)));
      slots_bitmask_ = slots_size - 1;
    }
  }

  RingBufferTransferCache(const RingBufferTransferCache &) = delete;
  RingBufferTransferCache &operator=(const RingBufferTransferCache &) = delete;

  // This transfercache implementation handles non-batch sized
  // inserts and removes efficiently.
  static constexpr bool IsFlexible() { return true; }

  // These methods all do internal locking.

  void AcquireInternalLocks()
  {
    freelist().AcquireInternalLocks();
    lock_.Lock();
  }

  void ReleaseInternalLocks()
  {
    lock_.Unlock();
    freelist().ReleaseInternalLocks();
  }

  // Insert the specified batch into the transfer cache.  N is the number of
  // elements in the range.  RemoveRange() is the opposite operation.
  void InsertRange(int size_class, absl::Span<void *> batch)
      ABSL_LOCKS_EXCLUDED(lock_) {
    const int N = batch.size();
    const int B = Manager::num_objects_to_move(size_class);
    ASSERT(0 < N && N <= B);
    void *to_free_buf[kMaxObjectsToMove];
    int to_free_num = 0;

    {
      absl::base_internal::SpinLockHolder h(&lock_);
      RingBufferSizeInfo info = GetSlotInfo();
      if (info.used + N <= max_capacity_) {
        const bool cache_grown = MakeCacheSpace(size_class, N);
        // MakeCacheSpace can drop the lock, so refetch
        info = GetSlotInfo();
        if (cache_grown) {
          CopyIntoEnd(batch.data(), N, info);
          SetSlotInfo(info);
          tracking::Report(kTCInsertHit, size_class, 1);
          insert_hits_.LossyAdd(1);
          return;
        }
      }

      // If we arrive here, this means that there is not enough capacity in the
      // current cache to include the new items, and we cannot grow it.

      // We want to return up to `B` items from the transfer cache and currently
      // inserted items.
      const int returned_from_cache = std::min<int>(B, info.used);
      if (returned_from_cache > 0) {
        CopyOutOfStart(to_free_buf, returned_from_cache, info);
      }
      to_free_num = returned_from_cache;
      if (info.used > 0) {
        // We didn't have to return the whole cache. This means we can copy
        // in all of the inserted items.
        ASSERT(info.used + N <= info.capacity);
        CopyIntoEnd(batch.data(), N, info);
      } else {
        // The transfercache is empty. We might still not have enough capacity
        // to store all of the inserted items though.
        const int to_insert_start = std::max(0, N - info.capacity);
        ASSERT(returned_from_cache + to_insert_start <= B);
        if (to_insert_start > 0) {
          // We also want to return some of the inserted items in this case.
          memcpy(to_free_buf + to_free_num, batch.data(),
                 to_insert_start * sizeof(void *));
          to_free_num += to_insert_start;
        }
        // This is only false if info.capacity is 0.
        if (ABSL_PREDICT_TRUE(N > to_insert_start)) {
          CopyIntoEnd(batch.data() + to_insert_start, N - to_insert_start,
                      info);
        }
      }
      SetSlotInfo(info);
    }
    // It can work out that we manage to insert all items into the cache after
    // all.
    if (to_free_num > 0) {
      ASSERT(to_free_num <= kMaxObjectsToMove);
      ASSERT(to_free_num <= B);
      insert_misses_.Add(1);
      tracking::Report(kTCInsertMiss, size_class, 1);
      freelist().InsertRange(absl::Span<void *>(to_free_buf, to_free_num));
    }
  }

  // Returns the actual number of fetched elements and stores elements in the
  // batch. This might return less than N if the transfercache is non-empty but
  // contains fewer elements than N. It is guaranteed to return at least 1 as
  // long as either the transfercache or the free list are not empty.
  ABSL_MUST_USE_RESULT int RemoveRange(int size_class, void **batch, int N)
      ABSL_LOCKS_EXCLUDED(lock_) {
    ASSERT(N > 0);

    {
      absl::base_internal::SpinLockHolder h(&lock_);
      RingBufferSizeInfo info = GetSlotInfo();
      if (info.used > 0) {
        // Return up to however much we have in our local cache.
        const int copied = std::min<int>(N, info.used);
        CopyOutOfEnd(batch, copied, info);
        SetSlotInfo(info);
        tracking::Report(kTCRemoveHit, size_class, 1);
        remove_hits_.LossyAdd(1);
        low_water_mark_ = std::min(low_water_mark_, info.used);
        return copied;
      }
      low_water_mark_ = 0;
    }

    remove_misses_.Add(1);
    tracking::Report(kTCRemoveMiss, size_class, 1);
    return freelist().RemoveRange(batch, N);
  }

  // Return all objects not touched since last call to this function.
  void TryPlunder(int size_class) ABSL_LOCKS_EXCLUDED(lock_) {
    if (max_capacity_ == 0) return;
    // If the lock is being held, someone is modifying the cache.
    if (!lock_.TryLock()) return;
    int low_water_mark = low_water_mark_;
    low_water_mark_ = std::numeric_limits<int>::max();
    const int B = Manager::num_objects_to_move(size_class);
    while (slot_info_.used > 0 && low_water_mark >= B &&
           (low_water_mark_ == std::numeric_limits<int>::max())) {
      const size_t num_to_move(std::min(B, slot_info_.used));
      void *buf[kMaxObjectsToMove];
      CopyOutOfEnd(buf, num_to_move, slot_info_);
      low_water_mark -= num_to_move;
      lock_.Unlock();
      freelist().InsertRange({buf, num_to_move});
      tracking::Report(kTCElementsPlunder, size_class, num_to_move);
      // If someone is starting to use the cache, stop doing this.
      if (!lock_.TryLock()) {
        return;
      }
    }
    lock_.Unlock();
  }

  // Returns the number of free objects in the transfer cache.
  size_t tc_length() ABSL_LOCKS_EXCLUDED(lock_) {
    absl::base_internal::SpinLockHolder h(&lock_);
    return static_cast<size_t>(GetSlotInfo().used);
  }

  // Returns the number of transfer cache insert/remove hits/misses.
  TransferCacheStats GetHitRateStats() const ABSL_LOCKS_EXCLUDED(lock_) {
    TransferCacheStats stats;

    stats.insert_hits = insert_hits_.value();
    stats.remove_hits = remove_hits_.value();
    stats.insert_misses = insert_misses_.value();
    stats.insert_non_batch_misses = 0;
    stats.remove_misses = remove_misses_.value();
    stats.remove_non_batch_misses = 0;

    return stats;
  }

  RingBufferSizeInfo GetSlotInfo() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    return slot_info_;
  }

  // REQUIRES: lock is held.
  // Tries to make room for N elements. If the cache is full it will try to
  // expand it at the cost of some other cache size.  Return false if there is
  // no space.
  bool MakeCacheSpace(int size_class, int N)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    // Increase capacity in number of batches, as we do when reducing capacity.
    const int B = Manager::num_objects_to_move(size_class);
    ASSERT(B >= N);

    auto info = GetSlotInfo();
    // Is there room in the cache?
    if (info.used + N <= info.capacity) return true;
    // Check if we can expand this cache?
    if (info.capacity + B > max_capacity_) return false;

    // Release the held lock before the other instance tries to grab its lock.
    lock_.Unlock();
    int to_evict = owner_->DetermineSizeClassToEvict();
    if (to_evict == size_class) {
      lock_.Lock();
      return false;
    }
    bool made_space = owner_->ShrinkCache(to_evict);
    lock_.Lock();

    if (!made_space) return false;

    // Succeeded in evicting, we're going to make our cache larger.  However, we
    // have dropped and re-acquired the lock, so slot_info_ may have
    // changed.  Therefore, check and verify that it is still OK to increase the
    // cache size.
    info = GetSlotInfo();
    if (info.capacity + B > max_capacity_) return false;
    info.capacity += B;
    SetSlotInfo(info);
    return true;
  }

  bool HasSpareCapacity(int size_class) ABSL_LOCKS_EXCLUDED(lock_) {
    const int n = Manager::num_objects_to_move(size_class);
    absl::base_internal::SpinLockHolder h(&lock_);
    const auto info = GetSlotInfo();
    return info.capacity - info.used >= n;
  }

  // Takes lock_ and invokes MakeCacheSpace() on this cache.  Returns true if it
  // succeeded at growing the cache by a batch size.
  bool GrowCache(int size_class) ABSL_LOCKS_EXCLUDED(lock_) {
    absl::base_internal::SpinLockHolder h(&lock_);
    return MakeCacheSpace(size_class, Manager::num_objects_to_move(size_class));
  }

  // REQUIRES: lock_ is *not* held.
  // Tries to shrink the Cache.  Return false if it failed to shrink the cache.
  // Decreases cache_slots_ on success.
  bool ShrinkCache(int size_class) ABSL_LOCKS_EXCLUDED(lock_) {
    const int N = Manager::num_objects_to_move(size_class);

    void *to_free[kMaxObjectsToMove];
    int num_to_free;
    {
      absl::base_internal::SpinLockHolder h(&lock_);
      auto info = GetSlotInfo();
      if (info.capacity == 0) return false;
      if (info.capacity < N) return false;

      const int unused = info.capacity - info.used;
      if (N <= unused) {
        info.capacity -= N;
        SetSlotInfo(info);
        return true;
      }

      num_to_free = N - unused;

      // Remove from the beginning of the buffer which holds the oldest entries.
      // Our internal slot array may get overwritten as soon as we drop the
      // lock, so copy the items to free to an on stack buffer.
      CopyOutOfStart(to_free, num_to_free, info);
      low_water_mark_ = info.used;
      info.capacity -= N;
      SetSlotInfo(info);
    }

    // Access the freelist without holding the lock.
    freelist().InsertRange({to_free, static_cast<uint64_t>(num_to_free)});
    return true;
  }

  // This is a thin wrapper for the CentralFreeList.  It is intended to ensure
  // that we are not holding lock_ when we access it.
  ABSL_ATTRIBUTE_ALWAYS_INLINE FreeList &freelist() ABSL_LOCKS_EXCLUDED(lock_) {
    return freelist_do_not_access_directly_;
  }

  // The const version of the wrapper, needed to call stats on
  ABSL_ATTRIBUTE_ALWAYS_INLINE const FreeList &freelist() const
      ABSL_LOCKS_EXCLUDED(lock_) {
    return freelist_do_not_access_directly_;
  }

 private:
  // Due to decreased downward pressure, the ring buffer based transfer cache
  // contains on average more bytes than the legacy implementation.
  // To counteract this, decrease the capacity (but not max capacity).
  // TODO(b/161927252):  Revisit TransferCache rebalancing strategy
  static typename TransferCache<CentralFreeList, TransferCacheManager>::Capacity
  CapacityNeeded(int cl) {
    auto capacity =
        TransferCache<CentralFreeList, TransferCacheManager>::CapacityNeeded(
            cl);
    const int N = Manager::num_objects_to_move(cl);
    if (N == 0) return {0, 0};
    ASSERT(capacity.capacity % N == 0);
    // We still want capacity to be in multiples of batches.
    const int capacity_in_batches = capacity.capacity / N;
    // This factor was found by trial and error.
    const int new_batches =
        static_cast<int>(std::ceil(capacity_in_batches / 1.5));
    capacity.capacity = new_batches * N;
    return capacity;
  }

  // Converts a logical index (i.e. i-th element stored in the ring buffer) into
  // a physical index into slots_.
  size_t GetSlotIndex(size_t start, size_t i) const {
    return (start + i) & slots_bitmask_;
  }

  // Copies N elements from source to the end of the ring buffer. It updates
  // `info`, be sure to call SetSlotInfo() to save the modifications.
  // N has to be > 0.
  void CopyIntoEnd(void *const *source, size_t N, RingBufferSizeInfo &info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    ASSERT(N > 0);
    ASSERT(info.used + N <= info.capacity);
    const size_t begin = GetSlotIndex(info.start, info.used);
    const size_t end = GetSlotIndex(info.start, info.used + N);
    if (ABSL_PREDICT_FALSE(end < begin && end != 0)) {
      // We wrap around the buffer.
      memcpy(slots_ + begin, source, sizeof(void *) * (N - end));
      memcpy(slots_, source + (N - end), sizeof(void *) * end);
    } else {
      memcpy(slots_ + begin, source, sizeof(void *) * N);
    }
    info.used += N;
  }

  // Copies N elements stored in slots_ starting at the given logic index into
  // target. Does not do any updates to slot_info_.
  // N has to be > 0.
  // You should use CopyOutOfEnd or CopyOutOfStart instead in most cases.
  void CopyOutOfSlots(void **target, size_t N, size_t start, size_t index) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    ASSERT(N > 0);
    const size_t begin = GetSlotIndex(start, index);
    const size_t end = GetSlotIndex(start, index + N);
    if (ABSL_PREDICT_FALSE(end < begin && end != 0)) {
      // We wrap around the buffer.
      memcpy(target, slots_ + begin, sizeof(void *) * (N - end));
      memcpy(target + (N - end), slots_, sizeof(void *) * end);
    } else {
      memcpy(target, slots_ + begin, sizeof(void *) * N);
    }
  }

  // Copies N elements from the start of the ring buffer into target. Updates
  // `info`, be sure to call SetSlotInfo() to save the modifications.
  // N has to be > 0.
  void CopyOutOfStart(void **target, size_t N, RingBufferSizeInfo &info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    ASSERT(N > 0);
    ASSERT(N <= info.used);
    CopyOutOfSlots(target, N, info.start, 0);
    info.used -= N;
    if (info.used == 0) {
      // This makes it less likely that we will have to do copies that wrap
      // around in the immediate future.
      info.start = 0;
    } else {
      info.start = (info.start + N) & slots_bitmask_;
    }
  }

  // Copies N elements from the end of the ring buffer into target. Updates
  // `info`, be sure to call SetSlotInfo() to save the modifications.
  // N has to be > 0.
  void CopyOutOfEnd(void **target, size_t N, RingBufferSizeInfo &info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    ASSERT(N > 0);
    ASSERT(N <= info.used);
    info.used -= N;
    CopyOutOfSlots(target, N, info.start, info.used);
    if (info.used == 0) {
      // This makes it less likely that we will have to do copies that wrap
      // around in the immediate future.
      info.start = 0;
    }
  }

  void SetSlotInfo(RingBufferSizeInfo info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    ASSERT(0 <= info.start);
    ASSERT((info.start & slots_bitmask_) == info.start);
    ASSERT(0 <= info.used);
    ASSERT(info.used <= info.capacity);
    ASSERT(info.capacity <= max_capacity_);
    slot_info_ = info;
  }

  // Pointer to array of free objects.
  void **slots_ ABSL_GUARDED_BY(lock_);

  // This lock protects all the data members.  used_slots_ and cache_slots_
  // may be looked at without holding the lock.
  absl::base_internal::SpinLock lock_;

  // Number of currently used and available cached entries in slots_. Use
  // GetSlotInfo() to read this.
  // INVARIANT: [0 <= slot_info_.used <= slot_info.capacity <= max_cache_slots_]
  RingBufferSizeInfo slot_info_ ABSL_GUARDED_BY(lock_);

  // Lowest value of "slot_info_.used" since last call to TryPlunder. All
  // elements not used for a full cycle (2 seconds) are unlikely to get used
  // again.
  int low_water_mark_ ABSL_GUARDED_BY(lock_) = std::numeric_limits<int>::max();

  // Maximum size of the cache.
  const int32_t max_capacity_;
  // This is a bitmask used instead of a modulus in the ringbuffer index
  // calculations. This is 1 smaller than the size of slots_ which itself has
  // the size of `absl::bit_ceil(max_capacity_)`, i.e. the smallest power of two
  // >= max_capacity_.
  size_t slots_bitmask_;

  // insert_hits_ and remove_hits_ are logically guarded by lock_ for mutations
  // and use LossyAdd, but the thread annotations cannot indicate that we do not
  // need a lock for reads.
  StatsCounter insert_hits_;
  StatsCounter remove_hits_;
  // Miss counters do not hold lock_, so they use Add.
  StatsCounter insert_misses_;
  StatsCounter remove_misses_;

  FreeList freelist_do_not_access_directly_;
  Manager *const owner_;
} ABSL_CACHELINE_ALIGNED;

}  // namespace tcmalloc::tcmalloc_internal::internal_transfer_cache
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_TRANSFER_CACHE_INTERNAL_H_
