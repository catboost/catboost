#pragma clang system_header
// Copyright 2022 The TCMalloc Authors
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

#ifndef TCMALLOC_ALLOCATION_SAMPLING_H_
#define TCMALLOC_ALLOCATION_SAMPLING_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "absl/base/attributes.h"
#include "absl/debugging/stacktrace.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/percpu.h"
#include "tcmalloc/malloc_extension.h"
#include "tcmalloc/pagemap.h"
#include "tcmalloc/sampler.h"
#include "tcmalloc/span.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc::tcmalloc_internal {

class Static;

// This function computes a profile that maps a live stack trace to
// the number of bytes of central-cache memory pinned by an allocation
// at that stack trace.
// In the case when span is hosting >= 1 number of small objects (t.proxy !=
// nullptr), we call span::Fragmentation() and read `span->allocated_`. It is
// safe to do so since we hold the per-sample lock while iterating over sampled
// allocations. It prevents the sampled allocation that has the proxy object to
// complete deallocation, thus `proxy` can not be returned to the span yet. It
// thus prevents the central free list to return the span to the page heap.
std::unique_ptr<const ProfileBase> DumpFragmentationProfile(Static& state);

std::unique_ptr<const ProfileBase> DumpHeapProfile(Static& state);

extern "C" ABSL_CONST_INIT thread_local Sampler tcmalloc_sampler
    ABSL_ATTRIBUTE_INITIAL_EXEC;

// Compiler needs to see definition of this variable to generate more
// efficient code for -fPIE/PIC. If the compiler does not see the definition
// it considers it may come from another dynamic library. So even for
// initial-exec model, it need to emit an access via GOT (GOTTPOFF).
// When it sees the definition, it can emit direct %fs:TPOFF access.
// So we provide a weak definition here, but the actual definition is in
// percpu_rseq_asm.S.
ABSL_CONST_INIT ABSL_ATTRIBUTE_WEAK thread_local Sampler tcmalloc_sampler
    ABSL_ATTRIBUTE_INITIAL_EXEC;

inline Sampler* GetThreadSampler() {
  static_assert(sizeof(Sampler) == TCMALLOC_SAMPLER_SIZE,
                "update TCMALLOC_SAMPLER_SIZE");
  static_assert(alignof(Sampler) == TCMALLOC_SAMPLER_ALIGN,
                "update TCMALLOC_SAMPLER_ALIGN");
  static_assert(Sampler::HotDataOffset() == TCMALLOC_SAMPLER_HOT_OFFSET,
                "update TCMALLOC_SAMPLER_HOT_OFFSET");
  return &tcmalloc_sampler;
}

void FreeProxyObject(Static& state, void* ptr, size_t size_class);

// Performs sampling for already occurred allocation of object.
//
// For very small object sizes, object is used as 'proxy' and full
// page with sampled marked is allocated instead.
//
// For medium-sized objects that have single instance per span,
// they're simply freed and fresh page span is allocated to represent
// sampling.
//
// For large objects (i.e. allocated with do_malloc_pages) they are
// also fully reused and their span is marked as sampled.
//
// Note that do_free_with_size assumes sampled objects have
// page-aligned addresses. Please change both functions if need to
// invalidate the assumption.
//
// Note that size_class might not match requested_size in case of
// memalign. I.e. when larger than requested allocation is done to
// satisfy alignment constraint.
//
// In case of out-of-memory condition when allocating span or
// stacktrace struct, this function simply cheats and returns original
// object. As if no sampling was requested.
template <typename Policy>
ABSL_ATTRIBUTE_NOINLINE sized_ptr_t
SampleifyAllocation(Static& state, Policy policy, size_t requested_size,
                    size_t weight, size_t size_class, void* obj, Span* span) {
  TC_CHECK((size_class != 0 && obj != nullptr && span == nullptr) ||
           (size_class == 0 && obj == nullptr && span != nullptr));

  StackTrace stack_trace;
  stack_trace.proxy = nullptr;
  stack_trace.requested_size = requested_size;
  // Grab the stack trace outside the heap lock.
  stack_trace.depth = absl::GetStackTrace(stack_trace.stack, kMaxStackDepth, 0);

  // requested_alignment = 1 means 'small size table alignment was used'
  // Historically this is reported as requested_alignment = 0
  stack_trace.requested_alignment = policy.align();
  if (stack_trace.requested_alignment == 1) {
    stack_trace.requested_alignment = 0;
  }

  stack_trace.requested_size_returning = policy.size_returning();
  stack_trace.access_hint = static_cast<uint8_t>(policy.access());
  stack_trace.weight = weight;

  GuardedAllocWithStatus alloc_with_status{
      nullptr, Profile::Sample::GuardedStatus::NotAttempted};

  size_t capacity = 0;
  if (size_class != 0) {
    TC_ASSERT_EQ(size_class,
                 state.pagemap().sizeclass(PageIdContainingTagged(obj)));

    stack_trace.allocated_size = state.sizemap().class_to_size(size_class);
    stack_trace.cold_allocated = IsExpandedSizeClass(size_class);

    Length num_pages = BytesToLengthCeil(stack_trace.allocated_size);
    alloc_with_status = state.guardedpage_allocator().TrySample(
        requested_size, stack_trace.requested_alignment, num_pages,
        stack_trace);
    if (alloc_with_status.status == Profile::Sample::GuardedStatus::Guarded) {
      TC_ASSERT(!IsNormalMemory(alloc_with_status.alloc));
      const PageId p = PageIdContaining(alloc_with_status.alloc);
#ifdef TCMALLOC_INTERNAL_LEGACY_LOCKING
      PageHeapSpinLockHolder l;
#endif  // TCMALLOC_INTERNAL_LEGACY_LOCKING
      span = Span::New(Range(p, num_pages));
      state.pagemap().Set(p, span);
      // If we report capacity back from a size returning allocation, we can not
      // report the stack_trace.allocated_size, as we guard the size to
      // 'requested_size', and we maintain the invariant that GetAllocatedSize()
      // must match the returned size from size returning allocations. So in
      // that case, we report the requested size for both capacity and
      // GetAllocatedSize().
      if (policy.size_returning()) {
        stack_trace.allocated_size = requested_size;
      }
      capacity = requested_size;
    } else if ((span = state.page_allocator().New(
                    num_pages, {1, AccessDensityPrediction::kSparse},
                    MemoryTag::kSampled)) == nullptr) {
      capacity = stack_trace.allocated_size;
      return {obj, capacity};
    } else {
      capacity = stack_trace.allocated_size;
    }

    size_t span_size =
        Length(state.sizemap().class_to_pages(size_class)).in_bytes();
    size_t objects_per_span = span_size / stack_trace.allocated_size;

    if (objects_per_span != 1) {
      TC_ASSERT_GT(objects_per_span, 1);
      stack_trace.proxy = obj;
      obj = nullptr;
    }
  } else {
    // Set stack_trace.allocated_size to the exact size for a page allocation.
    // NOTE: if we introduce gwp-asan sampling / guarded allocations
    // for page allocations, then we need to revisit do_malloc_pages as
    // the current assumption is that only class sized allocs are sampled
    // for gwp-asan.
    stack_trace.allocated_size = span->bytes_in_span();
    stack_trace.cold_allocated =
        GetMemoryTag(span->start_address()) == MemoryTag::kCold;
    capacity = stack_trace.allocated_size;
  }

  // A span must be provided or created by this point.
  TC_ASSERT_NE(span, nullptr);

  stack_trace.sampled_alloc_handle =
      state.sampled_alloc_handle_generator.fetch_add(
          1, std::memory_order_relaxed) +
      1;
  stack_trace.span_start_address = span->start_address();
  stack_trace.allocation_time = absl::Now();
  stack_trace.guarded_status = alloc_with_status.status;
  stack_trace.allocation_type = policy.allocation_type();
  stack_trace.user_data = SampleUserDataSupport::UserData::Make();

  // How many allocations does this sample represent, given the sampling
  // frequency (weight) and its size.
  const double allocation_estimate =
      static_cast<double>(weight) / (requested_size + 1);

  // Adjust our estimate of internal fragmentation.
  TC_ASSERT_LE(requested_size, stack_trace.allocated_size);
  if (requested_size < stack_trace.allocated_size) {
    state.sampled_internal_fragmentation_.Add(
        allocation_estimate * (stack_trace.allocated_size - requested_size));
  }

  state.allocation_samples.ReportMalloc(stack_trace);

  state.deallocation_samples.ReportMalloc(stack_trace);

  // The SampledAllocation object is visible to readers after this. Readers only
  // care about its various metadata (e.g. stack trace, weight) to generate the
  // heap profile, and won't need any information from Span::Sample() next.
  SampledAllocation* sampled_allocation =
      state.sampled_allocation_recorder().Register(std::move(stack_trace));
  // No pageheap_lock required. The span is freshly allocated and no one else
  // can access it. It is visible after we return from this allocation path.
  span->Sample(sampled_allocation);

  state.peak_heap_tracker().MaybeSaveSample();

  if (obj != nullptr) {
    // We are not maintaining precise statistics on malloc hit/miss rates at our
    // cache tiers.  We can deallocate into our ordinary cache.
    TC_ASSERT_NE(size_class, 0);
    FreeProxyObject(state, obj, size_class);
  }
  TC_ASSERT_EQ(state.pagemap().sizeclass(span->first_page()), 0);
  return {(alloc_with_status.alloc != nullptr) ? alloc_with_status.alloc
                                               : span->start_address(),
          capacity};
}

void MaybeUnsampleAllocation(Static& state, void* ptr,
                             std::optional<size_t> size, Span& span);

template <typename Policy>
static sized_ptr_t SampleLargeAllocation(Static& state, Policy policy,
                                         size_t requested_size, size_t weight,
                                         Span* span) {
  return SampleifyAllocation(state, policy, requested_size, weight, 0, nullptr,
                             span);
}

template <typename Policy>
static sized_ptr_t SampleSmallAllocation(Static& state, Policy policy,
                                         size_t requested_size, size_t weight,
                                         size_t size_class, sized_ptr_t res) {
  return SampleifyAllocation(state, policy, requested_size, weight, size_class,
                             res.p, nullptr);
}
}  // namespace tcmalloc::tcmalloc_internal
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_ALLOCATION_SAMPLING_H_
