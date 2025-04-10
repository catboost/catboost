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

#include "tcmalloc/central_freelist.h"

#include <stdint.h>

#include "tcmalloc/internal/linked_list.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/page_heap.h"
#include "tcmalloc/pagemap.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/static_vars.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

static MemoryTag MemoryTagFromSizeClass(size_t cl) {
  if (!Static::numa_topology().numa_aware()) {
    return MemoryTag::kNormal;
  }
  return NumaNormalTag(cl / kNumBaseClasses);
}

// Like a constructor and hence we disable thread safety analysis.
void CentralFreeList::Init(size_t cl) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  size_class_ = cl;
  object_size_ = Static::sizemap().class_to_size(cl);
  pages_per_span_ = Length(Static::sizemap().class_to_pages(cl));
  objects_per_span_ =
      pages_per_span_.in_bytes() / (object_size_ ? object_size_ : 1);
}

static Span* MapObjectToSpan(void* object) {
  const PageId p = PageIdContaining(object);
  Span* span = Static::pagemap().GetExistingDescriptor(p);
  return span;
}

Span* CentralFreeList::ReleaseToSpans(void* object, Span* span,
                                      size_t object_size) {
  if (ABSL_PREDICT_FALSE(span->FreelistEmpty(object_size))) {
    nonempty_.prepend(span);
  }

  if (ABSL_PREDICT_TRUE(span->FreelistPush(object, object_size))) {
    return nullptr;
  }
  span->RemoveFromList();  // from nonempty_
  return span;
}

void CentralFreeList::InsertRange(absl::Span<void*> batch) {
  CHECK_CONDITION(!batch.empty() && batch.size() <= kMaxObjectsToMove);
  Span* spans[kMaxObjectsToMove];
  // Safe to store free spans into freed up space in span array.
  Span** free_spans = spans;
  int free_count = 0;

  // Prefetch Span objects to reduce cache misses.
  for (int i = 0; i < batch.size(); ++i) {
    Span* span = MapObjectToSpan(batch[i]);
    ASSERT(span != nullptr);
    span->Prefetch();
    spans[i] = span;
  }

  // First, release all individual objects into spans under our mutex
  // and collect spans that become completely free.
  {
    // Use local copy of variable to ensure that it is not reloaded.
    size_t object_size = object_size_;
    absl::base_internal::SpinLockHolder h(&lock_);
    for (int i = 0; i < batch.size(); ++i) {
      Span* span = ReleaseToSpans(batch[i], spans[i], object_size);
      if (ABSL_PREDICT_FALSE(span)) {
        free_spans[free_count] = span;
        free_count++;
      }
    }

    RecordMultiSpansDeallocated(free_count);
    UpdateObjectCounts(batch.size());
  }

  // Then, release all free spans into page heap under its mutex.
  if (ABSL_PREDICT_FALSE(free_count)) {
    // Unregister size class doesn't require holding any locks.
    for (int i = 0; i < free_count; ++i) {
      Span* const free_span = free_spans[i];
      ASSERT(IsNormalMemory(free_span->start_address())
      );
      Static::pagemap().UnregisterSizeClass(free_span);

      // Before taking pageheap_lock, prefetch the PageTrackers these spans are
      // on.
      //
      // Small-but-slow does not use the HugePageAwareAllocator (by default), so
      // do not prefetch on this config.
#ifndef TCMALLOC_SMALL_BUT_SLOW
      const PageId p = free_span->first_page();

      // In huge_page_filler.h, we static_assert that PageTracker's key elements
      // for deallocation are within the first two cachelines.
      void* pt = Static::pagemap().GetHugepage(p);
      // Prefetch for writing, as we will issue stores to the PageTracker
      // instance.
      __builtin_prefetch(pt, 1, 3);
      __builtin_prefetch(
          reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(pt) +
                                  ABSL_CACHELINE_SIZE),
          1, 3);
#endif  // TCMALLOC_SMALL_BUT_SLOW
    }

    const MemoryTag tag = MemoryTagFromSizeClass(size_class_);
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    for (int i = 0; i < free_count; ++i) {
      Span* const free_span = free_spans[i];
      ASSERT(tag == GetMemoryTag(free_span->start_address()));
      Static::page_allocator().Delete(free_span, tag);
    }
  }
}

int CentralFreeList::RemoveRange(void** batch, int N) {
  ASSUME(N > 0);
  // Use local copy of variable to ensure that it is not reloaded.
  size_t object_size = object_size_;
  int result = 0;
  absl::base_internal::SpinLockHolder h(&lock_);
  if (ABSL_PREDICT_FALSE(nonempty_.empty())) {
    result = Populate(batch, N);
  } else {
    do {
      Span* span = nonempty_.first();
      int here =
          span->FreelistPopBatch(batch + result, N - result, object_size);
      ASSERT(here > 0);
      if (span->FreelistEmpty(object_size)) {
        span->RemoveFromList();  // from nonempty_
      }
      result += here;
    } while (result < N && !nonempty_.empty());
  }
  UpdateObjectCounts(-result);
  return result;
}

// Fetch memory from the system and add to the central cache freelist.
int CentralFreeList::Populate(void** batch,
                              int N) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  // Release central list lock while operating on pageheap
  // Note, this could result in multiple calls to populate each allocating
  // a new span and the pushing those partially full spans onto nonempty.
  lock_.Unlock();

  const MemoryTag tag = MemoryTagFromSizeClass(size_class_);
  Span* span = Static::page_allocator().New(pages_per_span_, tag);
  if (ABSL_PREDICT_FALSE(span == nullptr)) {
    Log(kLog, __FILE__, __LINE__, "tcmalloc: allocation failed",
        pages_per_span_.in_bytes());
    lock_.Lock();
    return 0;
  }
  ASSERT(tag == GetMemoryTag(span->start_address()));
  ASSERT(span->num_pages() == pages_per_span_);

  Static::pagemap().RegisterSizeClass(span, size_class_);
  size_t objects_per_span = objects_per_span_;
  int result = span->BuildFreelist(object_size_, objects_per_span, batch, N);
  ASSERT(result > 0);
  // This is a cheaper check than using FreelistEmpty().
  bool span_empty = result == objects_per_span;

  lock_.Lock();
  if (!span_empty) {
    nonempty_.prepend(span);
  }
  RecordSpanAllocated();
  return result;
}

size_t CentralFreeList::OverheadBytes() const {
  if (ABSL_PREDICT_FALSE(object_size_ == 0)) {
    return 0;
  }
  const size_t overhead_per_span = pages_per_span_.in_bytes() % object_size_;
  return num_spans() * overhead_per_span;
}

SpanStats CentralFreeList::GetSpanStats() const {
  SpanStats stats;
  if (ABSL_PREDICT_FALSE(objects_per_span_ == 0)) {
    return stats;
  }
  stats.num_spans_requested = static_cast<size_t>(num_spans_requested_.value());
  stats.num_spans_returned = static_cast<size_t>(num_spans_returned_.value());
  stats.obj_capacity = stats.num_live_spans() * objects_per_span_;
  return stats;
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
