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

#ifndef TCMALLOC_CENTRAL_FREELIST_H_
#define TCMALLOC_CENTRAL_FREELIST_H_

#include <stddef.h>

#include <cstddef>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/macros.h"
#include "absl/base/thread_annotations.h"
#include "tcmalloc/internal/atomic_stats_counter.h"
#include "tcmalloc/span.h"
#include "tcmalloc/span_stats.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// Data kept per size-class in central cache.
class CentralFreeList {
 public:
  constexpr CentralFreeList()
      : lock_(absl::kConstInit, absl::base_internal::SCHEDULE_KERNEL_ONLY),
        size_class_(0),
        object_size_(0),
        objects_per_span_(0),
        pages_per_span_(0),
        nonempty_() {}

  CentralFreeList(const CentralFreeList&) = delete;
  CentralFreeList& operator=(const CentralFreeList&) = delete;

  void Init(size_t cl) ABSL_LOCKS_EXCLUDED(lock_);

  // These methods all do internal locking.

  // Insert batch into the central freelist.
  // REQUIRES: batch.size() > 0 && batch.size() <= kMaxObjectsToMove.
  void InsertRange(absl::Span<void*> batch) ABSL_LOCKS_EXCLUDED(lock_);

  // Fill a prefix of batch[0..N-1] with up to N elements removed from central
  // freelist.  Return the number of elements removed.
  ABSL_MUST_USE_RESULT int RemoveRange(void** batch, int N)
      ABSL_LOCKS_EXCLUDED(lock_);

  // Returns the number of free objects in cache.
  size_t length() const { return static_cast<size_t>(counter_.value()); }

  // Returns the memory overhead (internal fragmentation) attributable
  // to the freelist.  This is memory lost when the size of elements
  // in a freelist doesn't exactly divide the page-size (an 8192-byte
  // page full of 5-byte objects would have 2 bytes memory overhead).
  size_t OverheadBytes() const;

  SpanStats GetSpanStats() const;

  void AcquireInternalLocks() {
    lock_.Lock();
  }

  void ReleaseInternalLocks() {
    lock_.Unlock();
  }

 private:
  // Release an object to spans.
  // Returns object's span if it become completely free.
  Span* ReleaseToSpans(void* object, Span* span, size_t object_size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Populate cache by fetching from the page heap.
  // May temporarily release lock_.
  // Fill a prefix of batch[0..N-1] with up to N elements removed from central
  // freelist. Returns the number of elements removed.
  int Populate(void** batch, int N) ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // This lock protects all the mutable data members.
  absl::base_internal::SpinLock lock_;

  size_t size_class_;  // My size class (immutable after Init())
  size_t object_size_;
  size_t objects_per_span_;
  Length pages_per_span_;

  size_t num_spans() const {
    size_t requested = num_spans_requested_.value();
    size_t returned = num_spans_returned_.value();
    if (requested < returned) return 0;
    return (requested - returned);
  }

  void RecordSpanAllocated() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    counter_.LossyAdd(objects_per_span_);
    num_spans_requested_.LossyAdd(1);
  }

  void RecordMultiSpansDeallocated(size_t num_spans_returned)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    counter_.LossyAdd(-num_spans_returned * objects_per_span_);
    num_spans_returned_.LossyAdd(num_spans_returned);
  }

  void UpdateObjectCounts(int num) ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    counter_.LossyAdd(num);
  }

  // The followings are kept as a StatsCounter so that they can read without
  // acquiring a lock. Updates to these variables are guarded by lock_
  // so writes are performed using LossyAdd for speed, the lock still
  // guarantees accuracy.

  // Num free objects in cache entry
  StatsCounter counter_;

  StatsCounter num_spans_requested_;
  StatsCounter num_spans_returned_;

  // Dummy header for non-empty spans
  SpanList nonempty_ ABSL_GUARDED_BY(lock_);
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_CENTRAL_FREELIST_H_
