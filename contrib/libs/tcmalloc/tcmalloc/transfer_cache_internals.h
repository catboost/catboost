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

#include <limits>

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
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/tracking.h"
#include "tcmalloc/transfer_cache_stats.h"

namespace tcmalloc::internal_transfer_cache {

struct alignas(8) SizeInfo {
  int32_t used;
  int32_t capacity;
};

// TransferCache is used to cache transfers of
// sizemap.num_objects_to_move(size_class) back and forth between
// thread caches and the central cache for a given size class.
template <typename CentralFreeList, typename TransferCacheManager>
class TransferCache {
 public:
  using Manager = TransferCacheManager;
  using FreeList = CentralFreeList;

  static constexpr int kMaxCapacityInBatches = 64;
  static constexpr int kInitialCapacityInBatches = 16;

  constexpr explicit TransferCache(Manager *owner) : TransferCache(owner, 0) {}

  // C++11 has complex rules for direct initialization of an array of aggregate
  // types that are not copy constructible.  The int parameters allows us to do
  // two distinct things at the same time:
  //  - have an implicit constructor (one arg implicit ctors are dangerous)
  //  - build an array of these in an arg pack expansion without a comma
  //    operator trick
  constexpr TransferCache(Manager *owner, int)
      : owner_(owner),
        lock_(absl::kConstInit, absl::base_internal::SCHEDULE_KERNEL_ONLY),
        max_capacity_(0),
        insert_hits_(0),
        remove_hits_(0),
        insert_misses_(0),
        remove_misses_(0),
        slot_info_{},
        slots_(nullptr),
        freelist_do_not_access_directly_(),
        arbitrary_transfer_(false) {}

  TransferCache(const TransferCache &) = delete;
  TransferCache &operator=(const TransferCache &) = delete;

  // We require the pageheap_lock with some templates, but not in tests, so the
  // thread safety analysis breaks pretty hard here.
  void Init(size_t cl) ABSL_NO_THREAD_SAFETY_ANALYSIS {
    freelist().Init(cl);
    absl::base_internal::SpinLockHolder h(&lock_);

    // We need at least 2 slots to store list head and tail.
    ASSERT(kMinObjectsToMove >= 2);

    // Cache this value, for performance.
    arbitrary_transfer_ =
        IsExperimentActive(Experiment::TCMALLOC_ARBITRARY_TRANSFER_CACHE);

    slots_ = nullptr;
    max_capacity_ = 0;
    SizeInfo info = {0, 0};

    size_t bytes = Manager::class_to_size(cl);
    if (cl > 0 && bytes > 0) {
      // Limit the maximum size of the cache based on the size class.  If this
      // is not done, large size class objects will consume a lot of memory if
      // they just sit in the transfer cache.
      size_t objs_to_move = Manager::num_objects_to_move(cl);
      ASSERT(objs_to_move > 0 && bytes > 0);

      // Starting point for the maximum number of entries in the transfer cache.
      // This actual maximum for a given size class may be lower than this
      // maximum value.
      max_capacity_ = kMaxCapacityInBatches * objs_to_move;
      // A transfer cache freelist can have anywhere from 0 to
      // max_capacity_ slots to put link list chains into.
      info.capacity = kInitialCapacityInBatches * objs_to_move;

      // Limit each size class cache to at most 1MB of objects or one entry,
      // whichever is greater. Total transfer cache memory used across all
      // size classes then can't be greater than approximately
      // 1MB * kMaxNumTransferEntries.
      max_capacity_ = std::min<size_t>(
          max_capacity_,
          std::max<size_t>(
              objs_to_move,
              (1024 * 1024) / (bytes * objs_to_move) * objs_to_move));
      info.capacity = std::min(info.capacity, max_capacity_);
      slots_ = reinterpret_cast<void **>(
          owner_->Alloc(max_capacity_ * sizeof(void *)));
    }
    SetSlotInfo(info);
  }

  // These methods all do internal locking.

  // Insert the specified batch into the transfer cache.  N is the number of
  // elements in the range.  RemoveRange() is the opposite operation.
  void InsertRange(absl::Span<void *> batch, int N) ABSL_LOCKS_EXCLUDED(lock_) {
    const int B = Manager::num_objects_to_move(size_class());
    ASSERT(0 < N && N <= B);
    auto info = slot_info_.load(std::memory_order_relaxed);
    if (N == B && info.used + N <= max_capacity_) {
      absl::base_internal::SpinLockHolder h(&lock_);
      if (MakeCacheSpace()) {
        // MakeCacheSpace can drop the lock, so refetch
        info = slot_info_.load(std::memory_order_relaxed);
        info.used += N;
        SetSlotInfo(info);

        void **entry = GetSlot(info.used - N);
        memcpy(entry, batch.data(), sizeof(void *) * N);
        tracking::Report(kTCInsertHit, size_class(), 1);
        insert_hits_++;
        return;
      }
    } else if (arbitrary_transfer_) {
      absl::base_internal::SpinLockHolder h(&lock_);
      MakeCacheSpace();
      // MakeCacheSpace can drop the lock, so refetch
      info = slot_info_.load(std::memory_order_relaxed);
      int unused = info.capacity - info.used;
      if (N < unused) {
        info.used += N;
        SetSlotInfo(info);
        void **entry = GetSlot(info.used - N);
        memcpy(entry, batch.data(), sizeof(void *) * N);
        tracking::Report(kTCInsertHit, size_class(), 1);
        insert_hits_++;
        return;
      }
      // We could not fit the entire batch into the transfer cache
      // so send the batch to the freelist and also take some elements from
      // the transfer cache so that we amortise the cost of accessing spans
      // in the freelist. Only do this if caller has sufficient space in
      // batch.
      // First of all fill up the rest of the batch with elements from the
      // transfer cache.
      int extra = B - N;
      if (N > 1 && extra > 0 && info.used > 0 && batch.size() >= B) {
        // Take at most all the objects present
        extra = std::min(extra, info.used);
        ASSERT(extra + N <= kMaxObjectsToMove);
        info.used -= extra;
        SetSlotInfo(info);

        void **entry = GetSlot(info.used);
        memcpy(batch.data() + N, entry, sizeof(void *) * extra);
        N += extra;
#ifndef NDEBUG
        int rest = batch.size() - N - 1;
        if (rest > 0) {
          memset(batch.data() + N, 0x3f, rest * sizeof(void *));
        }
#endif
      }
    }
    insert_misses_.fetch_add(1, std::memory_order_relaxed);
    tracking::Report(kTCInsertMiss, size_class(), 1);
    freelist().InsertRange(batch.data(), N);
  }

  // Returns the actual number of fetched elements and stores elements in the
  // batch.
  ABSL_MUST_USE_RESULT int RemoveRange(void **batch, int N)
      ABSL_LOCKS_EXCLUDED(lock_) {
    ASSERT(N > 0);
    const int B = Manager::num_objects_to_move(size_class());
    int fetch = 0;
    auto info = slot_info_.load(std::memory_order_relaxed);
    if (N == B && info.used >= N) {
      absl::base_internal::SpinLockHolder h(&lock_);
      // Refetch with the lock
      info = slot_info_.load(std::memory_order_relaxed);
      if (info.used >= N) {
        info.used -= N;
        SetSlotInfo(info);
        void **entry = GetSlot(info.used);
        memcpy(batch, entry, sizeof(void *) * N);
        tracking::Report(kTCRemoveHit, size_class(), 1);
        remove_hits_++;
        return N;
      }
    } else if (arbitrary_transfer_ && info.used >= 0) {
      absl::base_internal::SpinLockHolder h(&lock_);
      // Refetch with the lock
      info = slot_info_.load(std::memory_order_relaxed);

      fetch = std::min(N, info.used);
      info.used -= fetch;
      SetSlotInfo(info);
      void **entry = GetSlot(info.used);
      memcpy(batch, entry, sizeof(void *) * fetch);
      tracking::Report(kTCRemoveHit, size_class(), 1);
      remove_hits_++;
      if (fetch == N) return N;
    }
    remove_misses_.fetch_add(1, std::memory_order_relaxed);
    tracking::Report(kTCRemoveMiss, size_class(), 1);
    return freelist().RemoveRange(batch + fetch, N - fetch) + fetch;
  }

  // Returns the number of free objects in the central cache.
  size_t central_length() { return freelist().length(); }

  // Returns the number of free objects in the transfer cache.
  size_t tc_length() {
    return static_cast<size_t>(slot_info_.load(std::memory_order_relaxed).used);
  }

  // Returns the number of spans allocated and deallocated from the CFL
  SpanStats GetSpanStats() const { return freelist().GetSpanStats(); }

  // Returns the number of transfer cache insert/remove hits/misses.
  TransferCacheStats GetHitRateStats() ABSL_LOCKS_EXCLUDED(lock_) {
    TransferCacheStats stats;
    {
      absl::base_internal::SpinLockHolder h(&lock_);
      stats.insert_hits = insert_hits_;
      stats.remove_hits = remove_hits_;
    }
    stats.insert_misses = insert_misses_;
    stats.remove_misses = remove_misses_;
    return stats;
  }

  // Returns the memory overhead (internal fragmentation) attributable
  // to the freelist.  This is memory lost when the size of elements
  // in a freelist doesn't exactly divide the page-size (an 8192-byte
  // page full of 5-byte objects would have 2 bytes memory overhead).
  size_t OverheadBytes() { return freelist().OverheadBytes(); }

  SizeInfo GetSlotInfo() const {
    return slot_info_.load(std::memory_order_relaxed);
  }

  // REQUIRES: lock is held.
  // Tries to make room for a batch.  If the cache is full it will try to expand
  // it at the cost of some other cache size.  Return false if there is no
  // space.
  bool MakeCacheSpace() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    const int N = Manager::num_objects_to_move(size_class());

    auto info = slot_info_.load(std::memory_order_relaxed);
    // Is there room in the cache?
    if (info.used + N <= info.capacity) return true;
    // Check if we can expand this cache?
    if (info.capacity + N > max_capacity_) return false;

    int to_evict = owner_->DetermineSizeClassToEvict();
    if (to_evict == size_class()) return false;

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

  bool HasSpareCapacity() {
    int n = Manager::num_objects_to_move(size_class());
    auto info = GetSlotInfo();
    return info.capacity - info.used >= n;
  }

  // Takes lock_ and invokes MakeCacheSpace() on this cache.  Returns true if it
  // succeeded at growing the cache by a btach size.
  bool GrowCache() ABSL_LOCKS_EXCLUDED(lock_) {
    absl::base_internal::SpinLockHolder h(&lock_);
    return MakeCacheSpace();
  }

  // REQUIRES: lock_ is *not* held.
  // Tries to shrink the Cache.  Return false if it failed to shrink the cache.
  // Decreases cache_slots_ on success.
  bool ShrinkCache() ABSL_LOCKS_EXCLUDED(lock_) {
    int N = Manager::num_objects_to_move(size_class());

    void *to_free[kMaxObjectsToMove];
    int num_to_free;
    {
      absl::base_internal::SpinLockHolder h(&lock_);
      auto info = slot_info_.load(std::memory_order_relaxed);
      if (info.capacity == 0) return false;
      if (!arbitrary_transfer_ && info.capacity < N) return false;

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
    freelist().InsertRange(to_free, num_to_free);
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

  // Maximum size of the cache for a given size class. (immutable after Init())
  int32_t max_capacity_;

  size_t insert_hits_ ABSL_GUARDED_BY(lock_);
  size_t remove_hits_ ABSL_GUARDED_BY(lock_);
  std::atomic<size_t> insert_misses_;
  std::atomic<size_t> remove_misses_;

  // Number of currently used and available cached entries in slots_.  This
  // variable is updated under a lock but can be read without one.
  // INVARIANT: [0 <= slot_info_.used <= slot_info.capacity <= max_cache_slots_]
  std::atomic<SizeInfo> slot_info_;

  // Pointer to array of free objects.  Use GetSlot() to get pointers to
  // entries.
  void **slots_ ABSL_GUARDED_BY(lock_);

  size_t size_class() const {
    return freelist_do_not_access_directly_.size_class();
  }

  FreeList freelist_do_not_access_directly_;

  // Cached value of IsExperimentActive(Experiment::TCMALLOC_ARBITRARY_TRANSFER)
  bool arbitrary_transfer_;
} ABSL_CACHELINE_ALIGNED;

// Lock free transfer cache based on LMAX disruptor pattern.
//
// Use `GetSlot()` to get pointers to entries.
// Pointer to array of `max_capacity_ + 1` free objects which forms a circular
// buffer.
//
// Various offsets have a strict ordering invariant:
//   * tail_committed <= tail <= head_committed <= head (viewed circularly).
//   * Empty when tail_committed == head_committed.
//   * Full when tail_committed - 1 == head_committed.
//
// When there are no active threads,
//   *  `tail_committed == tail`
//   *  `head_committed == head`
//
// In terms of atomic sequencing, only committed variables hold dependencies.
// - `RemoveRange` acquires `head_committed` and releases `tail_committed`
// - `InsertRange` acquires `tail_committed` and releases `head_committed`
//
// For example:
//
// The queue is idle with some data in it and a batch size of 3.
//   +--------------------------------------------------------------------+
//   |  |  |  |xx|xx|xx|xx|xx|xx|xx|xx|xx|xx|xx|  |  |  |  |  |  |  |  |  |
//   +--------------------------------------------------------------------+
//             ^                                ^
//             |                                |
//         tail_committed/tail              head_committed/head
//
// Four threads arrive (near simultaneously). Two are concurrently removing
// batches (c1, c2), and two threads are inserting batches (p1, p2).
//   +--------------------------------------------------------------------+
//   |  |  |  |c1|c1|c1|c2|c2|c2|xx|xx|xx|xx|xx|p1|p1|p1|p2|p2|p2|  |  |  |
//   +--------------------------------------------------------------------+
//             ^        ^        ^              ^        ^        ^
//             |        |        |              |        |        |
//             | c1 commit point |              | p1 commit point |
//         tail_committed        tail       head_committed        head
//
// At this point c2 and p2, cannot commit until c1 or p1 commit respectively.
// Let's say c1 commits:
//   +--------------------------------------------------------------------+
//   |  |  |  |  |  |  |c2|c2|c2|xx|xx|xx|xx|xx|p1|p1|p1|p2|p2|p2|  |  |  |
//   +--------------------------------------------------------------------+
//                      ^        ^              ^        ^        ^
//                      |        |              |        |        |
//                      |        |              | p1 commit point |
//         tail_committed        tail       head_committed        head
//
// Now, c2 can commit its batch:
//   +--------------------------------------------------------------------+
//   |  |  |  |  |  |  |  |  |  |xx|xx|xx|xx|xx|p1|p1|p1|p2|p2|p2|  |  |  |
//   +--------------------------------------------------------------------+
//                               ^              ^        ^        ^
//                               |              |        |        |
//                               |              | p1 commit point |
//                tail_committed/tail       head_committed        head
//
// In parallel, p1 could have completed and committed its batch:
//   +--------------------------------------------------------------------+
//   |  |  |  |  |  |  |  |  |  |xx|xx|xx|xx|xx|xx|xx|xx|p2|p2|p2|  |  |  |
//   +--------------------------------------------------------------------+
//                               ^                       ^        ^
//                               |                       |        |
//                tail_committed/tail       head_committed        head
//
// At which point p2 can commit:
//   +--------------------------------------------------------------------+
//   |  |  |  |  |  |  |  |  |  |xx|xx|xx|xx|xx|xx|xx|xx|xx|xx|xx|  |  |  |
//   +--------------------------------------------------------------------+
//                               ^                                ^
//                               |                                |
//                tail_committed/tail              head_committed/head
template <typename CentralFreeList, typename TransferCacheManager>
class LockFreeTransferCache {
 public:
  using Manager = TransferCacheManager;
  using FreeList = CentralFreeList;
  static constexpr int kMaxCapacityInBatches = 64;
  static constexpr int kInitialCapacityInBatches = 16;
  // Initialize the queue sequence at a number close to where it will overflow.
  // There's no cost to doing this and it ensures tests cover the overflow case.
  static constexpr uint32_t kInitSequenceNumber =
      std::numeric_limits<uint32_t>::max() - kMaxObjectsToMove;

  constexpr explicit LockFreeTransferCache(Manager *owner)
      : LockFreeTransferCache(owner, 0) {}

  // C++11 has complex rules for direct initialization of an array of aggregate
  // types that are not copy constructible.  The int parameters allows us to do
  // two distinct things at the same time:
  //  - have an implicit constructor (one arg implicit ctors are dangerous)
  //  - build an array of these in an arg pack expansion without a comma
  //    operator trick
  constexpr LockFreeTransferCache(Manager *owner, int)
      : owner_(owner),
        slots_(nullptr),
        freelist_(),
        max_capacity_(0),
        capacity_(),
        head_(kInitSequenceNumber),
        insert_hits_(0),
        insert_misses_(0),
        head_committed_(kInitSequenceNumber),
        tail_(kInitSequenceNumber),
        remove_hits_(0),
        remove_misses_(0),
        tail_committed_(kInitSequenceNumber) {}

  LockFreeTransferCache(const LockFreeTransferCache &) = delete;
  LockFreeTransferCache &operator=(const LockFreeTransferCache &) = delete;

  // We require the pageheap_lock with some templates, but not in tests, so the
  // thread safety analysis breaks pretty hard here.
  void Init(size_t cl) ABSL_NO_THREAD_SAFETY_ANALYSIS {
    freelist_.Init(cl);

    // We need at least 2 slots to store list head and tail.
    ASSERT(kMinObjectsToMove >= 2);

    slots_ = nullptr;
    max_capacity_ = 0;
    int32_t capacity = 0;

    size_t bytes = Manager::class_to_size(cl);
    if (cl > 0 && bytes > 0) {
      // Limit the maximum size of the cache based on the size class.  If this
      // is not done, large size class objects will consume a lot of memory if
      // they just sit in the transfer cache.
      batch_size_ = Manager::num_objects_to_move(cl);
      ASSERT(batch_size_ > 0 && bytes > 0);

      // Starting point for the maximum number of entries in the transfer cache.
      // This actual maximum for a given size class may be lower than this
      // maximum value.
      max_capacity_ = kMaxCapacityInBatches * batch_size_;
      // A transfer cache freelist can have anywhere from 0 to
      // max_capacity_ slots to put link list chains into.
      capacity = kInitialCapacityInBatches * batch_size_;

      // Limit each size class cache to at most 1MB of objects or one entry,
      // whichever is greater. Total transfer cache memory used across all
      // size classes then can't be greater than approximately
      // 1MB * kMaxNumTransferEntries.
      max_capacity_ = std::min<size_t>(
          max_capacity_,
          std::max<size_t>(batch_size_, (1024 * 1024) / (bytes * batch_size_) *
                                            batch_size_));
      if (tcmalloc_internal::Bits::IsZeroOrPow2(
              static_cast<size_t>(max_capacity_))) {
        --max_capacity_;
        slots_mask_ = max_capacity_;
      } else {
        slots_mask_ = tcmalloc_internal::Bits::RoundUpToPow2(
                          static_cast<size_t>(max_capacity_)) -
                      1;
      }

      capacity = std::min(capacity, max_capacity_);
      capacity_.store(capacity, std::memory_order_relaxed);
      slots_ = reinterpret_cast<void **>(
          owner_->Alloc((slots_mask_ + 1) * sizeof(void *)));
    }
  }

  // Insert the specified batch into the transfer cache.  N is the number of
  // elements in the range.  RemoveRange() is the opposite operation.
  void InsertRange(absl::Span<void *> batch, int n) {
    ASSERT(0 < n && n <= batch_size_);
    absl::optional<Range> r = ClaimInsert(n);
    if (!r.has_value()) {
      tracking::Report(kTCInsertMiss, size_class(), 1);
      insert_misses_.fetch_add(1, std::memory_order_relaxed);
      freelist_.InsertRange(batch.data(), n);
      return;
    }

    tracking::Report(kTCInsertHit, size_class(), 1);
    insert_hits_.fetch_add(1, std::memory_order_relaxed);
    CopyIntoSlots(batch.data(), *r);
    head_committed_.AdvanceCommitLine(*r, batch_size_);
  }

  // Returns the actual number of fetched elements and stores elements in the
  // batch.
  ABSL_MUST_USE_RESULT int RemoveRange(void **batch, int n) {
    ASSERT(n > 0);
    absl::optional<Range> r = ClaimRemove(n);
    if (!r.has_value()) {
      tracking::Report(kTCRemoveMiss, size_class(), 1);
      remove_misses_.fetch_add(1, std::memory_order_relaxed);
      return freelist_.RemoveRange(batch, n);
    }

    tracking::Report(kTCRemoveHit, size_class(), 1);
    remove_hits_.fetch_add(1, std::memory_order_relaxed);
    CopyFromSlots(batch, *r);
    tail_committed_.AdvanceCommitLine(*r, batch_size_);
    return n;
  }

  // Returns the number of free objects in the central cache.
  size_t central_length() { return freelist_.length(); }

  // Returns the number of free objects in the transfer cache.
  size_t tc_length() const {
    return size_from_pos(head_committed_.load(std::memory_order_relaxed),
                         tail_committed_.load(std::memory_order_relaxed));
  }

  uint32_t size_from_pos(uint32_t h, uint32_t t) const { return h - t; }

  // Returns the number of spans allocated and deallocated from the CFL
  SpanStats GetSpanStats() const { return freelist_.GetSpanStats(); }

  // Returns the number of transfer cache insert/remove hits/misses.
  TransferCacheStats GetHitRateStats() const {
    TransferCacheStats stats;
    stats.insert_hits = insert_hits_;
    stats.insert_misses = insert_misses_;
    stats.remove_hits = remove_hits_;
    stats.remove_misses = remove_misses_;
    return stats;
  }

  // Returns the memory overhead (internal fragmentation) attributable
  // to the freelist.  This is memory lost when the size of elements
  // in a freelist doesn't exactly divide the page-size (an 8192-byte
  // page full of 5-byte objects would have 2 bytes memory overhead).
  size_t OverheadBytes() { return freelist_.OverheadBytes(); }

  // Ensures that `size_needed` of total capacity is available.  If the cache is
  // full it will try to expand this cache at the cost of some other cache size.
  // Return false on failure.
  bool HasCacheSpace(uint32_t size_needed) const {
    return size_needed <= capacity_.load(std::memory_order_relaxed);
  }
  bool EnsureCacheSpace(uint32_t size_needed) {
    while (!HasCacheSpace(size_needed)) {
      if (size_needed > max_capacity_) return false;
      int to_evict = owner_->DetermineSizeClassToEvict();
      if (to_evict == size_class()) return false;
      if (!owner_->ShrinkCache(to_evict)) return false;
      if (GrowCache()) continue;

      // At this point, we have successfully taken cache space from someone.  Do
      // not give up until we have given it back to somebody or the entire thing
      // can just start leaking cache capacity.
      while (true) {
        if (++to_evict >= kNumClasses) to_evict = 1;
        if (to_evict == size_class()) {
          // We gave it back to ourselves, which is nice, so we should break
          // from the inner loop and see if we have ensured available space.
          if (GrowCache()) break;
        } else if (owner_->GrowCache(to_evict)) {
          return false;
        }
      }
    }
    return true;
  }

  // Tries to grow the cache. Returns false if it failed.
  bool GrowCache() {
    uint32_t new_c;
    uint32_t old_c = capacity_.load(std::memory_order_relaxed);
    do {
      new_c = old_c + batch_size_;
      if (new_c > max_capacity_) return false;
    } while (!capacity_.compare_exchange_weak(
        old_c, new_c, std::memory_order_relaxed, std::memory_order_relaxed));
    return true;
  }

  // Tries to shrink the Cache.  Return false if it failed.
  bool ShrinkCache() {
    uint32_t new_c;
    uint32_t old_c = capacity_.load(std::memory_order_relaxed);
    do {
      if (old_c < batch_size_) return false;
      new_c = old_c - batch_size_;
    } while (!capacity_.compare_exchange_weak(
        old_c, new_c, std::memory_order_relaxed, std::memory_order_relaxed));

    // TODO(kfm): decide if we want to do this or not
    // if (tc_length() >= capacity_.load(std::memory_order_relaxed)) {
    //   void *buf[kMaxObjectsToMove];
    //   int i = RemoveRange(buf, N);
    //   freelist().InsertRange(buf, i);
    // }
    return true;
  }

  bool HasSpareCapacity() const {
    return HasCacheSpace(tc_length() + batch_size_);
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE FreeList &freelist() { return freelist_; }
  ABSL_ATTRIBUTE_ALWAYS_INLINE const FreeList &freelist() const {
    return freelist_;
  }

 private:
  struct Range {
    uint32_t start;
    uint32_t end;
  };

  // A `uint32_t` like value that exposes extra operations around awaiting
  // specific values.
  //
  // Internally uses a Futex after an initial spinning period.  `quanta`
  // parameters are used for computing a Futex mask and *MUST* be the same as
  // all calls to both `AdvanceCommitLine` and `AwaitEqual`.
  class SpinValue {
   public:
    constexpr SpinValue() : v_(0), sleepers_(0) {}
    explicit constexpr SpinValue(uint32_t v)
        : v_(absl::bit_cast<int32_t>(v)), sleepers_(0) {}

    // Fetches the current value.
    ABSL_ATTRIBUTE_ALWAYS_INLINE
    uint32_t load(std::memory_order order) const {
      return absl::bit_cast<uint32_t>(v_.load(order));
    }

    // Advances the value from `r.start` to `r.end`, waiting for the value to
    // reach `r.start` if it is not there already.
    ABSL_ATTRIBUTE_ALWAYS_INLINE
    void AdvanceCommitLine(Range r, int32_t quanta) {
      const int32_t start = absl::bit_cast<int32_t>(r.start);
      const int32_t end = absl::bit_cast<int32_t>(r.end);
      int32_t temp_pos;

      // To avoid missed wake-ups, it's important that the access to "v_" and
      // "sleepers_" use sequential consistency. Consider a thread T1 here and
      // concurrent thread T2 in Await():
      //
      // T1.1: store new value into v_
      // T1.2: load sleepers_
      // T1.3: if sleepers is non-zero, wake futex
      //
      // T2.1: load v_, see that it's not the expected value
      // T2.2: increment sleepers_
      // T2.3: load v_ again (in futex implementation) and sleep
      //
      // C++11 sequential consistency guarantees that there is a single total
      // order of all atomic operations that are so tagged. Taking as an
      // example the sequence where T1 is transitioning v_=0 -> v_=1,
      // and T2 is waiting for v_ == 1. A missed wakeup would of the order:
      //
      // T1.1: store(v=1)   -> T1.2: load(sleepers) == 0
      // T2.1: load(v) == 0 -> T2.2: sleepers++ -> T2.3: load(v)==0 and sleep
      //
      // Because of the seq_cst, we know the following relationships in addition
      // to the "program order" ones expressed above:
      //
      // T2.1 -> T1.1 (must have come before since T2 didn't see v = 1)
      // T1.2 -> T2.2 (must have come before since T1 didn't see the increment)
      // T2.3 -> T1.1 (must have come before since T2 still didn't see v = 1)
      //
      // Putting all the relationships together:
      //   T2.1 -> T1.1 -> T1.2 -> T2.2 -> T2.3 -> T1.1
      //
      // This consists of a cycle, which violates the "sequential consistency
      // provides a total order" guarantee. Hence, the missed wakeup cannot
      // happen.
      if (ABSL_PREDICT_FALSE(!v_.compare_exchange_strong(
              temp_pos = start, end, std::memory_order_seq_cst,
              std::memory_order_relaxed))) {
        AwaitEqual(r.start, quanta);
        temp_pos = v_.exchange(end, std::memory_order_seq_cst);
        ASSERT(temp_pos == r.start);
      }
      if (ABSL_PREDICT_FALSE(sleepers_.load(std::memory_order_seq_cst))) {
        absl::synchronization_internal::Futex::WakeBitset(
            &v_, std::numeric_limits<int32_t>::max(),
            ComputeFutexBits(r.end, quanta));
      }
    }

    // Waits for the value to be equal to `e`.  `quanta` is used for computing a
    // Futex mask and *MUST* be the same as all calls to quanta passed to
    // AdvanceCommitLine.  Provides an acquire memory order on any release that
    // set the `this->value()` to `e`.
    ABSL_ATTRIBUTE_NOINLINE
    void AwaitEqual(uint32_t e, int32_t quanta) {
      const int32_t expected = absl::bit_cast<int32_t>(e);
      Await([expected](int32_t v) { return v == expected; },
            ComputeFutexBits(expected, quanta));
    }

    // Waits for the value to change from `a` in any way.
    ABSL_ATTRIBUTE_NOINLINE
    void AwaitChange(uint32_t a) {
      const int32_t actual = absl::bit_cast<int32_t>(a);
      Await([actual](int32_t v) { return v != actual; },
            FUTEX_BITSET_MATCH_ANY);
    }

   private:
    // Wait for `condition(current_value)` to return true. If the condition does
    // not become true after briefly spinning, uses "futex_bits" to wait on the
    // futex.
    template <class Condition>
    void Await(const Condition &condition, int32_t futex_bits) {
      int32_t cur;
      do {
        for (int i = 1024; i > 0; --i) {
          cur = v_.load(std::memory_order_seq_cst);
          if (condition(cur)) return;
#ifdef __x86_64__
          _mm_pause();
#endif
        }
        // Increment with seq_cst memory order to ensure this is serialized
        // properly against the read of `sleepers_` in `AdvanceCommitLine`.
        sleepers_.fetch_add(1, std::memory_order_seq_cst);
        // Do one more load of `v_` with seq_cst memory order. The
        // implementation of futex already loads `v_` itself before going to
        // sleep, but futex doesn't participate in the C++11 memory model, so
        // we'd best be on the safe side here and do a C++11-compliant load. On
        // x86 this is a simple unlocked load anyway, so cost isn't high and
        // this path is already expected to be infrequently executed.
        if (ABSL_PREDICT_TRUE(v_.load(std::memory_order_seq_cst) == cur)) {
          absl::synchronization_internal::Futex::WaitBitsetAbsoluteTimeout(
              &v_, cur, futex_bits, nullptr);
        }
        sleepers_.fetch_sub(1, std::memory_order_relaxed);
      } while (true);
    }

    // Futex requires an `int32_t` but treats it like an opaque bag of 4 bytes,
    // so we keep this as an `int32_t` here but have our public interfaces in
    // terms of the preferred type `uint32_t`.
    std::atomic<int32_t> v_;

    // Tracks the number of threads currently sleeping on the futex so that we
    // can avoid the syscall to Wake in the fast path.
    std::atomic<int32_t> sleepers_;
  };

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  void CopyIntoSlots(void **batch, Range r) {
    r.start &= slots_mask_;
    r.end &= slots_mask_;
    if (ABSL_PREDICT_TRUE(r.start < r.end)) {
      void **entry = GetSlot(r.start);
      memcpy(entry, batch, sizeof(void *) * (r.end - r.start));
    } else {
      int32_t overhang = slots_mask_ + 1 - r.start;
      void **entry = GetSlot(r.start);
      memcpy(entry, batch, sizeof(void *) * overhang);
      batch += overhang;
      entry = GetSlot(0);
      memcpy(entry, batch, sizeof(void *) * r.end);
    }
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  void CopyFromSlots(void **batch, Range r) {
    r.start &= slots_mask_;
    r.end &= slots_mask_;
    if (ABSL_PREDICT_TRUE(r.start < r.end)) {
      void **entry = GetSlot(r.start);
      memcpy(batch, entry, sizeof(void *) * (r.end - r.start));
    } else {
      int32_t overhang = slots_mask_ + 1 - r.start;
      void **entry = GetSlot(r.start);
      memcpy(batch, entry, sizeof(void *) * overhang);
      batch += overhang;
      entry = GetSlot(0);
      memcpy(batch, entry, sizeof(void *) * r.end);
    }
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  absl::optional<Range> ClaimInsert(int n) {
    uint32_t old_h = head_.load(std::memory_order_relaxed);
    uint32_t new_h = old_h + n;
    uint32_t tail = tail_committed_.load(std::memory_order_acquire);
    uint32_t size_needed = size_from_pos(old_h, tail) + n;
    if (HasCacheSpace(size_needed) &&
        head_.compare_exchange_strong(old_h, new_h, std::memory_order_relaxed,
                                      std::memory_order_relaxed)) {
      return Range{old_h, new_h};
    }
    return ClaimInsertSlow(n);
  }

  ABSL_ATTRIBUTE_NOINLINE
  absl::optional<Range> ClaimInsertSlow(int n) {
  CLAIM_INSERT_SLOW:
    uint32_t new_h;
    uint32_t old_h = head_.load(std::memory_order_relaxed);
    do {
      uint32_t tail = tail_committed_.load(std::memory_order_acquire);
      uint32_t s = size_from_pos(old_h, tail);
      if (!EnsureCacheSpace(s + n)) {
        if (tail != tail_.load(std::memory_order_relaxed)) {
          // If we have pending removes, wait for some to resolve and try again.
          // - retest capacity in case the pending inserts were too small.
          // - use goto instead of tail call to keep stack bounded in non-opt.
          tail_committed_.AwaitChange(tail);
          goto CLAIM_INSERT_SLOW;
        }
        return absl::nullopt;
      }

      new_h = old_h + n;
    } while (!head_.compare_exchange_weak(
        old_h, new_h, std::memory_order_relaxed, std::memory_order_relaxed));
    return Range{old_h, new_h};
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  absl::optional<Range> ClaimRemove(int n) {
    uint32_t old_t = tail_.load(std::memory_order_relaxed);
    uint32_t head = head_committed_.load(std::memory_order_acquire);
    uint32_t size = size_from_pos(head, old_t);
    uint32_t new_t = old_t + n;
    if (n <= size &&
        tail_.compare_exchange_strong(old_t, new_t, std::memory_order_relaxed,
                                      std::memory_order_relaxed)) {
      return Range{old_t, new_t};
    }
    return ClaimRemoveSlow(n);
  }

  ABSL_ATTRIBUTE_NOINLINE
  absl::optional<Range> ClaimRemoveSlow(int n) {
  CLAIM_REMOVE_SLOW:
    uint32_t new_t;
    uint32_t old_t = tail_.load(std::memory_order_relaxed);
    do {
      uint32_t head = head_committed_.load(std::memory_order_acquire);
      uint32_t s = size_from_pos(head, old_t);
      if (s < n) {
        if (head != head_.load(std::memory_order_relaxed)) {
          // If we have pending inserts, wait for some to resolve and try again.
          // - retest size in case the pending inserts were too small.
          // - use goto instead of tail call to keep stack bounded in non-opt.
          head_committed_.AwaitChange(head);
          goto CLAIM_REMOVE_SLOW;
        }
        return absl::nullopt;
      }
      new_t = old_t + n;
    } while (!tail_.compare_exchange_weak(
        old_t, new_t, std::memory_order_relaxed, std::memory_order_relaxed));
    return Range{old_t, new_t};
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static int32_t ComputeFutexBits(uint32_t pos, int32_t quanta) {
    // We want to bins values according to our quanta so that each of the
    // nearest increment sets a different bit:
    //
    // Value: [0, n), [n, 2*n), ... [k*n, (k+1)*n)
    // Bits:  1 << 0, 1 << 1,       1 << k
    //
    // to avoid overflow, we then do all of this mod 32.
    int32_t shift = ((pos + quanta - 1) / quanta) % 32;
    return absl::bit_cast<int32_t>(uint32_t{1} << shift);
  }

  // Returns first object of the i-th slot.
  void **GetSlot(size_t i) { return slots_ + i; }

  size_t size_class() const { return freelist_.size_class(); }

  Manager *const owner_;

  void **slots_;
  FreeList freelist_;

  // Maximum size of the cache for a given size class. (immutable after
  // Init())
  int32_t max_capacity_;
  int32_t batch_size_;
  uint32_t slots_mask_;

  alignas(ABSL_CACHELINE_SIZE) std::atomic<uint32_t> capacity_;
  alignas(ABSL_CACHELINE_SIZE) std::atomic<uint32_t> head_;
  std::atomic<size_t> insert_hits_;
  std::atomic<size_t> insert_misses_;
  alignas(ABSL_CACHELINE_SIZE) SpinValue head_committed_;
  alignas(ABSL_CACHELINE_SIZE) std::atomic<uint32_t> tail_;
  std::atomic<size_t> remove_hits_;
  std::atomic<size_t> remove_misses_;
  alignas(ABSL_CACHELINE_SIZE) SpinValue tail_committed_;
} ABSL_CACHELINE_ALIGNED;

}  // namespace tcmalloc::internal_transfer_cache

#endif  // TCMALLOC_TRANSFER_CACHE_INTERNAL_H_
