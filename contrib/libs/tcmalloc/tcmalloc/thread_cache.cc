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

#include "tcmalloc/thread_cache.h"

#include <algorithm>

#include "absl/base/internal/spinlock.h"
#include "absl/base/macros.h"
#include "tcmalloc/transfer_cache.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

size_t ThreadCache::per_thread_cache_size_ = kMaxThreadCacheSize;
size_t ThreadCache::overall_thread_cache_size_ = kDefaultOverallThreadCacheSize;
int64_t ThreadCache::unclaimed_cache_space_ = kDefaultOverallThreadCacheSize;
ThreadCache* ThreadCache::thread_heaps_ = nullptr;
int ThreadCache::thread_heap_count_ = 0;
ThreadCache* ThreadCache::next_memory_steal_ = nullptr;
#ifdef ABSL_HAVE_TLS
__thread ThreadCache* ThreadCache::thread_local_data_
    ABSL_ATTRIBUTE_INITIAL_EXEC = nullptr;
#endif
ABSL_CONST_INIT bool ThreadCache::tsd_inited_ = false;
pthread_key_t ThreadCache::heap_key_;

void ThreadCache::Init(pthread_t tid) {
  size_ = 0;

  max_size_ = 0;
  IncreaseCacheLimitLocked();
  if (max_size_ == 0) {
    // There isn't enough memory to go around.  Just give the minimum to
    // this thread.
    max_size_ = kMinThreadCacheSize;

    // Take unclaimed_cache_space_ negative.
    unclaimed_cache_space_ -= kMinThreadCacheSize;
    ASSERT(unclaimed_cache_space_ < 0);
  }

  next_ = nullptr;
  prev_ = nullptr;
  tid_ = tid;
  in_setspecific_ = false;
  for (size_t cl = 0; cl < kNumClasses; ++cl) {
    list_[cl].Init();
  }
}

void ThreadCache::Cleanup() {
  // Put unused memory back into central cache
  for (int cl = 0; cl < kNumClasses; ++cl) {
    if (list_[cl].length() > 0) {
      ReleaseToCentralCache(&list_[cl], cl, list_[cl].length());
    }
  }
}

// Remove some objects of class "cl" from central cache and add to thread heap.
// On success, return the first object for immediate use; otherwise return NULL.
void* ThreadCache::FetchFromCentralCache(size_t cl, size_t byte_size) {
  FreeList* list = &list_[cl];
  ASSERT(list->empty());
  const int batch_size = Static::sizemap().num_objects_to_move(cl);

  const int num_to_move = std::min<int>(list->max_length(), batch_size);
  void* batch[kMaxObjectsToMove];
  int fetch_count =
      Static::transfer_cache().RemoveRange(cl, batch, num_to_move);
  if (fetch_count == 0) {
    return nullptr;
  }

  if (--fetch_count > 0) {
    size_ += byte_size * fetch_count;
    list->PushBatch(fetch_count, batch + 1);
  }

  // Increase max length slowly up to batch_size.  After that,
  // increase by batch_size in one shot so that the length is a
  // multiple of batch_size.
  if (list->max_length() < batch_size) {
    list->set_max_length(list->max_length() + 1);
  } else {
    // Don't let the list get too long.  In 32 bit builds, the length
    // is represented by a 16 bit int, so we need to watch out for
    // integer overflow.
    int new_length = std::min<int>(list->max_length() + batch_size,
                                   kMaxDynamicFreeListLength);
    // The list's max_length must always be a multiple of batch_size,
    // and kMaxDynamicFreeListLength is not necessarily a multiple
    // of batch_size.
    new_length -= new_length % batch_size;
    ASSERT(new_length % batch_size == 0);
    list->set_max_length(new_length);
  }
  return batch[0];
}

void ThreadCache::ListTooLong(FreeList* list, size_t cl) {
  const int batch_size = Static::sizemap().num_objects_to_move(cl);
  ReleaseToCentralCache(list, cl, batch_size);

  // If the list is too long, we need to transfer some number of
  // objects to the central cache.  Ideally, we would transfer
  // num_objects_to_move, so the code below tries to make max_length
  // converge on num_objects_to_move.

  if (list->max_length() < batch_size) {
    // Slow start the max_length so we don't overreserve.
    list->set_max_length(list->max_length() + 1);
  } else if (list->max_length() > batch_size) {
    // If we consistently go over max_length, shrink max_length.  If we don't
    // shrink it, some amount of memory will always stay in this freelist.
    list->set_length_overages(list->length_overages() + 1);
    if (list->length_overages() > kMaxOverages) {
      ASSERT(list->max_length() > batch_size);
      list->set_max_length(list->max_length() - batch_size);
      list->set_length_overages(0);
    }
  }
}

// Remove some objects of class "cl" from thread heap and add to central cache
void ThreadCache::ReleaseToCentralCache(FreeList* src, size_t cl, int N) {
  ASSERT(src == &list_[cl]);
  if (N > src->length()) N = src->length();
  size_t delta_bytes = N * Static::sizemap().class_to_size(cl);

  // We return prepackaged chains of the correct size to the central cache.
  void* batch[kMaxObjectsToMove];
  int batch_size = Static::sizemap().num_objects_to_move(cl);
  while (N > batch_size) {
    src->PopBatch(batch_size, batch);
    static_assert(ABSL_ARRAYSIZE(batch) >= kMaxObjectsToMove,
                  "not enough space in batch");
    Static::transfer_cache().InsertRange(cl,
                                         absl::Span<void*>(batch, batch_size));
    N -= batch_size;
  }
  src->PopBatch(N, batch);
  static_assert(ABSL_ARRAYSIZE(batch) >= kMaxObjectsToMove,
                "not enough space in batch");
  Static::transfer_cache().InsertRange(cl, absl::Span<void*>(batch, N));
  size_ -= delta_bytes;
}

// Release idle memory to the central cache
void ThreadCache::Scavenge() {
  // If the low-water mark for the free list is L, it means we would
  // not have had to allocate anything from the central cache even if
  // we had reduced the free list size by L.  We aim to get closer to
  // that situation by dropping L/2 nodes from the free list.  This
  // may not release much memory, but if so we will call scavenge again
  // pretty soon and the low-water marks will be high on that call.
  for (int cl = 0; cl < kNumClasses; cl++) {
    FreeList* list = &list_[cl];
    const int lowmark = list->lowwatermark();
    if (lowmark > 0) {
      const int drop = (lowmark > 1) ? lowmark / 2 : 1;
      ReleaseToCentralCache(list, cl, drop);

      // Shrink the max length if it isn't used.  Only shrink down to
      // batch_size -- if the thread was active enough to get the max_length
      // above batch_size, it will likely be that active again.  If
      // max_length shinks below batch_size, the thread will have to
      // go through the slow-start behavior again.  The slow-start is useful
      // mainly for threads that stay relatively idle for their entire
      // lifetime.
      const int batch_size = Static::sizemap().num_objects_to_move(cl);
      if (list->max_length() > batch_size) {
        list->set_max_length(
            std::max<int>(list->max_length() - batch_size, batch_size));
      }
    }
    list->clear_lowwatermark();
  }

  IncreaseCacheLimit();
}

void ThreadCache::DeallocateSlow(void* ptr, FreeList* list, size_t cl) {
  tracking::Report(kFreeMiss, cl, 1);
  if (ABSL_PREDICT_FALSE(list->length() > list->max_length())) {
    tracking::Report(kFreeTruncations, cl, 1);
    ListTooLong(list, cl);
  }
  if (size_ >= max_size_) {
    tracking::Report(kFreeScavenges, cl, 1);
    Scavenge();
  }
}

void ThreadCache::IncreaseCacheLimit() {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  IncreaseCacheLimitLocked();
}

void ThreadCache::IncreaseCacheLimitLocked() {
  if (unclaimed_cache_space_ > 0) {
    // Possibly make unclaimed_cache_space_ negative.
    unclaimed_cache_space_ -= kStealAmount;
    max_size_ += kStealAmount;
    return;
  }
  // Don't hold pageheap_lock too long.  Try to steal from 10 other
  // threads before giving up.  The i < 10 condition also prevents an
  // infinite loop in case none of the existing thread heaps are
  // suitable places to steal from.
  for (int i = 0; i < 10; ++i, next_memory_steal_ = next_memory_steal_->next_) {
    // Reached the end of the linked list.  Start at the beginning.
    if (next_memory_steal_ == nullptr) {
      ASSERT(thread_heaps_ != nullptr);
      next_memory_steal_ = thread_heaps_;
    }
    if (next_memory_steal_ == this ||
        next_memory_steal_->max_size_ <= kMinThreadCacheSize) {
      continue;
    }
    next_memory_steal_->max_size_ -= kStealAmount;
    max_size_ += kStealAmount;

    next_memory_steal_ = next_memory_steal_->next_;
    return;
  }
}

void ThreadCache::InitTSD() {
  ASSERT(!tsd_inited_);
  pthread_key_create(&heap_key_, DestroyThreadCache);
  tsd_inited_ = true;
}

ThreadCache* ThreadCache::CreateCacheIfNecessary() {
  // Initialize per-thread data if necessary
  Static::InitIfNecessary();
  ThreadCache* heap = nullptr;

#ifdef ABSL_HAVE_TLS
  const bool maybe_reentrant = !tsd_inited_;
  // If we have set up our TLS, we can avoid a scan of the thread_heaps_ list.
  if (tsd_inited_) {
    if (thread_local_data_) {
      return thread_local_data_;
    }
  }
#else
  const bool maybe_reentrant = true;
#endif

  {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    const pthread_t me = pthread_self();

    // This may be a recursive malloc call from pthread_setspecific()
    // In that case, the heap for this thread has already been created
    // and added to the linked list.  So we search for that first.
    if (maybe_reentrant) {
      for (ThreadCache* h = thread_heaps_; h != nullptr; h = h->next_) {
        if (h->tid_ == me) {
          heap = h;
          break;
        }
      }
    }

    if (heap == nullptr) {
      heap = NewHeap(me);
    }
  }

  // We call pthread_setspecific() outside the lock because it may
  // call malloc() recursively.  We check for the recursive call using
  // the "in_setspecific_" flag so that we can avoid calling
  // pthread_setspecific() if we are already inside pthread_setspecific().
  if (!heap->in_setspecific_ && tsd_inited_) {
    heap->in_setspecific_ = true;
#ifdef ABSL_HAVE_TLS
    // Also keep a copy in __thread for faster retrieval
    thread_local_data_ = heap;
#endif
    pthread_setspecific(heap_key_, heap);
    heap->in_setspecific_ = false;
  }
  return heap;
}

ThreadCache* ThreadCache::NewHeap(pthread_t tid) {
  // Create the heap and add it to the linked list
  ThreadCache* heap = Static::threadcache_allocator().New();
  heap->Init(tid);
  heap->next_ = thread_heaps_;
  heap->prev_ = nullptr;
  if (thread_heaps_ != nullptr) {
    thread_heaps_->prev_ = heap;
  } else {
    // This is the only thread heap at the momment.
    ASSERT(next_memory_steal_ == nullptr);
    next_memory_steal_ = heap;
  }
  thread_heaps_ = heap;
  thread_heap_count_++;
  return heap;
}

void ThreadCache::BecomeIdle() {
  if (!tsd_inited_) return;  // No caches yet
  ThreadCache* heap = GetCacheIfPresent();
  if (heap == nullptr) return;        // No thread cache to remove
  if (heap->in_setspecific_) return;  // Do not disturb the active caller

  heap->in_setspecific_ = true;
  pthread_setspecific(heap_key_, nullptr);
#ifdef ABSL_HAVE_TLS
  // Also update the copy in __thread
  thread_local_data_ = nullptr;
#endif
  heap->in_setspecific_ = false;
  if (GetCacheIfPresent() == heap) {
    // Somehow heap got reinstated by a recursive call to malloc
    // from pthread_setspecific.  We give up in this case.
    return;
  }

  // We can now get rid of the heap
  DeleteCache(heap);
}

void ThreadCache::DestroyThreadCache(void* ptr) {
  // Note that "ptr" cannot be NULL since pthread promises not
  // to invoke the destructor on NULL values, but for safety,
  // we check anyway.
  if (ptr != nullptr) {
#ifdef ABSL_HAVE_TLS
    thread_local_data_ = nullptr;
#endif
    DeleteCache(reinterpret_cast<ThreadCache*>(ptr));
  }
}

void ThreadCache::DeleteCache(ThreadCache* heap) {
  // Remove all memory from heap
  heap->Cleanup();

  // Remove from linked list
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  if (heap->next_ != nullptr) heap->next_->prev_ = heap->prev_;
  if (heap->prev_ != nullptr) heap->prev_->next_ = heap->next_;
  if (thread_heaps_ == heap) thread_heaps_ = heap->next_;
  thread_heap_count_--;

  if (next_memory_steal_ == heap) next_memory_steal_ = heap->next_;
  if (next_memory_steal_ == nullptr) next_memory_steal_ = thread_heaps_;
  unclaimed_cache_space_ += heap->max_size_;

  Static::threadcache_allocator().Delete(heap);
}

void ThreadCache::RecomputePerThreadCacheSize() {
  // Divide available space across threads
  int n = thread_heap_count_ > 0 ? thread_heap_count_ : 1;
  size_t space = overall_thread_cache_size_ / n;

  // Limit to allowed range
  if (space < kMinThreadCacheSize) space = kMinThreadCacheSize;
  if (space > kMaxThreadCacheSize) space = kMaxThreadCacheSize;

  double ratio = space / std::max<double>(1, per_thread_cache_size_);
  size_t claimed = 0;
  for (ThreadCache* h = thread_heaps_; h != nullptr; h = h->next_) {
    // Increasing the total cache size should not circumvent the
    // slow-start growth of max_size_.
    if (ratio < 1.0) {
      h->max_size_ *= ratio;
    }
    claimed += h->max_size_;
  }
  unclaimed_cache_space_ = overall_thread_cache_size_ - claimed;
  per_thread_cache_size_ = space;
}

void ThreadCache::GetThreadStats(uint64_t* total_bytes, uint64_t* class_count) {
  for (ThreadCache* h = thread_heaps_; h != nullptr; h = h->next_) {
    *total_bytes += h->Size();
    if (class_count) {
      for (int cl = 0; cl < kNumClasses; ++cl) {
        class_count[cl] += h->freelist_length(cl);
      }
    }
  }
}

void ThreadCache::set_overall_thread_cache_size(size_t new_size) {
  // Clip the value to a reasonable minimum
  if (new_size < kMinThreadCacheSize) new_size = kMinThreadCacheSize;
  overall_thread_cache_size_ = new_size;

  RecomputePerThreadCacheSize();
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
