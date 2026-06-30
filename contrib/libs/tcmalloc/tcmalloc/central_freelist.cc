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

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/types/span.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/allocation_guard.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/prefetch.h"
#include "tcmalloc/page_allocator_interface.h"
#include "tcmalloc/pagemap.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/selsan/selsan.h"
#include "tcmalloc/span.h"
#include "tcmalloc/static_vars.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {
namespace central_freelist_internal {

static MemoryTag MemoryTagFromSizeClass(size_t size_class) {
  if (IsExpandedSizeClass(size_class)) {
    return MemoryTag::kCold;
  }
  if (selsan::IsEnabled()) {
    return MemoryTag::kSelSan;
  }
  if (!tc_globals.numa_topology().numa_aware()) {
    return MemoryTag::kNormal;
  }
  return NumaNormalTag(size_class / kNumBaseClasses);
}

static AccessDensityPrediction AccessDensity(int objects_per_span) {
  // Use number of objects per span as a proxy for estimating access density of
  // the span. If number of objects per span is higher than
  // kFewObjectsAllocMaxLimit threshold, we assume that the span would be
  // long-lived.
  return objects_per_span > kFewObjectsAllocMaxLimit
             ? AccessDensityPrediction::kDense
             : AccessDensityPrediction::kSparse;
}

size_t StaticForwarder::class_to_size(int size_class) {
  return tc_globals.sizemap().class_to_size(size_class);
}

Length StaticForwarder::class_to_pages(int size_class) {
  return Length(tc_globals.sizemap().class_to_pages(size_class));
}

ABSL_ATTRIBUTE_NOINLINE
static void ReportMismatchedSizeClass(void* object, int page_size_class,
                                      int object_size_class) {
  auto [object_min_size, object_max_size] =
      tc_globals.sizemap().class_to_size_range(object_size_class);
  auto [page_min_size, page_max_size] =
      tc_globals.sizemap().class_to_size_range(page_size_class);

  TC_LOG("*** GWP-ASan (https://google.github.io/tcmalloc/gwp-asan.html) has detected a memory error ***");
  TC_LOG(
      "Mismatched-size-class "
      "(https://github.com/google/tcmalloc/tree/master/docs/mismatched-sized-delete.md) "
      "discovered for pointer %p: this pointer was recently freed "
      "with a size argument in the range [%v, %v], but the "
      "associated span of allocated memory is for allocations with sizes "
      "[%v, %v]. This is not a bug in tcmalloc, but rather is indicative "
      "of an application bug such as buffer overrun/underrun, use-after-free "
      "or double-free.",
      object, object_min_size, object_max_size, page_min_size, page_max_size);
  TC_LOG(
      "NOTE: The blamed stack trace that is about to crash is not likely the "
      "root cause of the issue. We are detecting the invalid deletion at a "
      "later point in time and different code location.");
  RecordCrash("GWP-ASan", "mismatched-size-class");

  tc_globals.mismatched_delete_state().Record(object_min_size, object_max_size,
                                              page_min_size, page_max_size,
                                              std::nullopt, std::nullopt);
  abort();
}

void StaticForwarder::MapObjectsToSpans(absl::Span<void*> batch, Span** spans,
                                        int expected_size_class) {
  // Prefetch Span objects to reduce cache misses.
  for (int i = 0; i < batch.size(); ++i) {
    const PageId p = PageIdContaining(batch[i]);
    auto [span, page_size_class] =
        tc_globals.pagemap().GetExistingDescriptorAndSizeClass(p);
    TC_ASSERT_NE(span, nullptr);
    if (ABSL_PREDICT_FALSE(page_size_class != expected_size_class)) {
      ReportMismatchedSizeClass(span, page_size_class, expected_size_class);
    }
    span->Prefetch();
    spans[i] = span;
  }
}

Span* StaticForwarder::AllocateSpan(int size_class, size_t objects_per_span,
                                    Length pages_per_span) {
  const MemoryTag tag = MemoryTagFromSizeClass(size_class);
  const AccessDensityPrediction density = AccessDensity(objects_per_span);

  SpanAllocInfo span_alloc_info = {.objects_per_span = objects_per_span,
                                   .density = density};
  TC_ASSERT(density == AccessDensityPrediction::kSparse ||
            (density == AccessDensityPrediction::kDense &&
             pages_per_span == Length(1)));
  Span* span =
      tc_globals.page_allocator().New(pages_per_span, span_alloc_info, tag);
  if (ABSL_PREDICT_FALSE(span == nullptr)) {
    return nullptr;
  }
  TC_ASSERT_EQ(tag, GetMemoryTag(span->start_address()));
  TC_ASSERT_EQ(span->num_pages(), pages_per_span);

  tc_globals.pagemap().RegisterSizeClass(span, size_class);
  return span;
}

#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
static void ReturnSpansToPageHeap(MemoryTag tag, absl::Span<Span*> free_spans,
                                  size_t objects_per_span)
    ABSL_LOCKS_EXCLUDED(pageheap_lock) {
  PageHeapSpinLockHolder l;
  for (Span* const free_span : free_spans) {
    TC_ASSERT_EQ(tag, GetMemoryTag(free_span->start_address()));
    tc_globals.page_allocator().Delete(free_span, tag);
  }
}
#endif  // TCMALLOC_INTERNAL_LEGACY_LOCKING

static void ReturnAllocsToPageHeap(
    MemoryTag tag,
    absl::Span<PageAllocatorInterface::AllocationState> free_allocs)
    ABSL_LOCKS_EXCLUDED(pageheap_lock) {
  PageHeapSpinLockHolder l;
  for (const auto& alloc : free_allocs) {
    tc_globals.page_allocator().Delete(alloc, tag);
  }
}

void StaticForwarder::DeallocateSpans(size_t objects_per_span,
                                      absl::Span<Span*> free_spans) {
  TC_ASSERT_NE(free_spans.size(), 0);
  const MemoryTag tag = GetMemoryTag(free_spans[0]->start_address());
  // Unregister size class doesn't require holding any locks.
  for (Span* const free_span : free_spans) {
    TC_ASSERT_EQ(GetMemoryTag(free_span->start_address()), tag);
    TC_ASSERT_NE(GetMemoryTag(free_span->start_address()), MemoryTag::kSampled);
    tc_globals.pagemap().UnregisterSizeClass(free_span);

    // Before taking pageheap_lock, prefetch the PageTrackers these spans are
    // on.
    const PageId p = free_span->first_page();

    // In huge_page_filler.h, we static_assert that PageTracker's key elements
    // for deallocation are within the first two cachelines.
    void* pt = tc_globals.pagemap().GetHugepage(p);
    // Prefetch for writing, as we will issue stores to the PageTracker
    // instance.
    PrefetchW(pt);
    PrefetchW(reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(pt) +
                                      ABSL_CACHELINE_SIZE));
  }

#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
  ReturnSpansToPageHeap(tag, free_spans, objects_per_span);
#else
  PageAllocatorInterface::AllocationState allocs[kMaxObjectsToMove];
  for (int i = 0, n = free_spans.size(); i < n; ++i) {
    Span* s = free_spans[i];
    TC_ASSERT_EQ(tag, GetMemoryTag(s->start_address()));
    allocs[i].r = Range(s->first_page(), s->num_pages());
    allocs[i].donated = s->donated();
    Span::Delete(s);
  }
  ReturnAllocsToPageHeap(tag, absl::MakeSpan(allocs, free_spans.size()));
#endif
}

}  // namespace central_freelist_internal
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
