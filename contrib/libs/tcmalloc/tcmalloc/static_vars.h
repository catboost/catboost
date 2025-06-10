#pragma clang system_header
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
//
// Static variables shared by multiple classes.

#ifndef TCMALLOC_STATIC_VARS_H_
#define TCMALLOC_STATIC_VARS_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <atomic>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "internal/logging.h"
#include "tcmalloc/allocation_sample.h"
#include "tcmalloc/arena.h"
#include "tcmalloc/central_freelist.h"
#include "tcmalloc/common.h"
#include "tcmalloc/deallocation_profiler.h"
#include "tcmalloc/guarded_page_allocator.h"
#include "tcmalloc/internal/atomic_stats_counter.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/explicitly_constructed.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/mismatched_delete_state.h"
#include "tcmalloc/internal/numa.h"
#include "tcmalloc/internal/sampled_allocation.h"
#include "tcmalloc/internal/sampled_allocation_recorder.h"
#include "tcmalloc/metadata_object_allocator.h"
#include "tcmalloc/page_allocator.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/peak_heap_tracker.h"
#include "tcmalloc/sizemap.h"
#include "tcmalloc/span.h"
#include "tcmalloc/stack_trace_table.h"
#include "tcmalloc/stats.h"
#include "tcmalloc/system-alloc.h"
#include "tcmalloc/transfer_cache.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

class CpuCache;
class PageMap;
class ThreadCache;

using SampledAllocationRecorder = ::tcmalloc::tcmalloc_internal::SampleRecorder<
    SampledAllocation, MetadataObjectAllocator<SampledAllocation>>;

enum class SizeClassConfiguration {
  kPow2Below64 = 1,
  kPow2Only = 2,
  kLegacy = 4,
  kReuse = 6,
};

bool tcmalloc_big_span();

void TCMallocPreFork();
void TCMallocPostFork();

class Static final {
 public:
  constexpr Static() = default;

  // True if InitIfNecessary() has run to completion.
  static bool IsInited();
  // Must be called before calling any of the accessors below.
  // Safe to call multiple times.
  static void InitIfNecessary();

  // Central cache.
  static CentralFreeList& central_freelist(int size_class) {
    return transfer_cache().central_freelist(size_class);
  }
  // Central cache -- an array of free-lists, one per size-class.
  // We have a separate lock per free-list to reduce contention.
  static TransferCacheManager& transfer_cache() { return transfer_cache_; }

  // A per-cache domain TransferCache.
  static ShardedTransferCacheManager& sharded_transfer_cache() {
    return sharded_transfer_cache_;
  }

  static SizeMap& sizemap() { return sizemap_; }

  static CpuCache& cpu_cache() { return cpu_cache_; }

  static PeakHeapTracker& peak_heap_tracker() { return peak_heap_tracker_; }

  static NumaTopology<kNumaPartitions, kNumBaseClasses>& numa_topology() {
    return numa_topology_;
  }

  static SystemAllocator<NumaTopology<kNumaPartitions, kNumBaseClasses>>&
  system_allocator() {
    return system_allocator_;
  }

  static Arena& arena() { return arena_; }

  // Page-level allocator.
  static PageAllocator& page_allocator() {
    return *reinterpret_cast<PageAllocator*>(page_allocator_.memory);
  }

  static PageMap& pagemap() { return pagemap_; }

  static GuardedPageAllocator& guardedpage_allocator() {
    return guardedpage_allocator_;
  }

  static MetadataObjectAllocator<SampledAllocation>&
  sampledallocation_allocator() {
    return sampledallocation_allocator_;
  }

  static MetadataObjectAllocator<Span>& span_allocator() {
    return span_allocator_;
  }

  static MetadataObjectAllocator<ThreadCache>& threadcache_allocator() {
    return threadcache_allocator_;
  }

  static SampledAllocationRecorder& sampled_allocation_recorder() {
    return sampled_allocation_recorder_;
  }

  // State kept for sampled allocations (/heapz support).
  ABSL_CONST_INIT static tcmalloc_internal::StatsCounter sampled_objects_size_;
  // sampled_internal_fragmentation estimates the amount of memory overhead from
  // allocation sizes being rounded up to size class/page boundaries.
  ABSL_CONST_INIT static tcmalloc_internal::StatsCounter
      sampled_internal_fragmentation_;
  // total_sampled_count_ tracks the total number of allocations that are
  // sampled.
  ABSL_CONST_INIT static tcmalloc_internal::StatsCounter total_sampled_count_;

  ABSL_CONST_INIT static AllocationSampleList allocation_samples;

  ABSL_CONST_INIT static deallocationz::DeallocationProfilerList
      deallocation_samples;

  // MallocHook::AllocHandle is a simple 64-bit int, and is not dependent on
  // other data.
  ABSL_CONST_INIT static std::atomic<AllocHandle>
      sampled_alloc_handle_generator;

  static MetadataObjectAllocator<StackTraceTable::LinkedSample>&
  linked_sample_allocator() {
    return linked_sample_allocator_;
  }

  static bool ABSL_ATTRIBUTE_ALWAYS_INLINE CpuCacheActive() {
    return cpu_cache_active_.load(std::memory_order_acquire);
  }
  static void ActivateCpuCache() {
    cpu_cache_active_.store(true, std::memory_order_release);
  }

  static bool ForkSupportEnabled() {
    return fork_support_enabled_;
  }
  static void EnableForkSupport() {
    fork_support_enabled_ = true;
  }


  static void SetSampleUserDataCallbacks(
      SampleUserDataSupport::CreateSampleUserDataCallback create,
      SampleUserDataSupport::CopySampleUserDataCallback copy,
      SampleUserDataSupport::DestroySampleUserDataCallback destroy,
      SampleUserDataSupport::ComputeSampleUserDataHashCallback compute_hash) {
    SampleUserDataSupport::Enable(create, copy, destroy, compute_hash);
  }

  static bool ABSL_ATTRIBUTE_ALWAYS_INLINE HaveHooks() {
    return false;
  }

  static size_t metadata_bytes() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // The root of the pagemap is potentially a large poorly utilized
  // structure, so figure out how much of it is actually resident.
  static size_t pagemap_residence();

  static MismatchedDeleteState& mismatched_delete_state() {
    return mismatched_delete_state_;
  }

  static SizeClassConfiguration size_class_configuration();

 private:
#if defined(__clang__)
  __attribute__((preserve_most))
#endif
  static void
  SlowInitIfNecessary();

  // These static variables require explicit initialization.  We cannot
  // count on their constructors to do any initialization because other
  // static variables may try to allocate memory before these variables
  // can run their constructors.

  ABSL_CONST_INIT static Arena arena_;
  static SizeMap sizemap_;
  TCMALLOC_ATTRIBUTE_NO_DESTROY ABSL_CONST_INIT static TransferCacheManager
      transfer_cache_;
  ABSL_CONST_INIT static ShardedTransferCacheManager sharded_transfer_cache_;
  static CpuCache cpu_cache_;
  ABSL_CONST_INIT static GuardedPageAllocator guardedpage_allocator_;
  static MetadataObjectAllocator<SampledAllocation>
      sampledallocation_allocator_;
  static MetadataObjectAllocator<Span> span_allocator_;
  static MetadataObjectAllocator<ThreadCache> threadcache_allocator_;
  static MetadataObjectAllocator<StackTraceTable::LinkedSample>
      linked_sample_allocator_;
  ABSL_CONST_INIT static std::atomic<bool> inited_;
  ABSL_CONST_INIT static std::atomic<bool> cpu_cache_active_;
  ABSL_CONST_INIT static bool fork_support_enabled_;
  ABSL_CONST_INIT static PeakHeapTracker peak_heap_tracker_;
  ABSL_CONST_INIT static NumaTopology<kNumaPartitions, kNumBaseClasses>
      numa_topology_;
  ABSL_CONST_INIT static MismatchedDeleteState mismatched_delete_state_;

  // PageHeap uses a constructor for initialization.  Like the members above,
  // we can't depend on initialization order, so pageheap is new'd
  // into this buffer.
  union PageAllocatorStorage {
    constexpr PageAllocatorStorage() : extra(0) {}

    char memory[sizeof(PageAllocator)];
    uintptr_t extra;  // To force alignment
  };

  static PageAllocatorStorage page_allocator_;
  static PageMap pagemap_;

  ABSL_CONST_INIT static SystemAllocator<
      NumaTopology<kNumaPartitions, kNumBaseClasses>>
      system_allocator_;

  // Manages sampled allocations and allows iteration over samples free from the
  // global pageheap_lock.
  static SampledAllocationRecorder sampled_allocation_recorder_;
};

ABSL_CONST_INIT extern Static tc_globals;

inline bool Static::IsInited() {
  return inited_.load(std::memory_order_acquire);
}

inline void Static::InitIfNecessary() {
  if (ABSL_PREDICT_FALSE(!IsInited())) {
    SlowInitIfNecessary();
  }
}

// ConstantRatePageAllocatorReleaser() might release more than the requested
// bytes because the page heap releases at the span granularity, and spans are
// of wildly different sizes. This keeps track of the extra bytes bytes released
// so that the app can periodically call Release() to release memory at a
// constant rate.
class ConstantRatePageAllocatorReleaser {
 public:
  size_t Release(size_t num_bytes, PageReleaseReason reason) {
    const PageHeapSpinLockHolder l;

    if (num_bytes <= extra_bytes_released_) {
      // We released too much on a prior call, so don't release any
      // more this time.
      extra_bytes_released_ -= num_bytes;
      num_bytes = 0;
    } else {
      num_bytes -= extra_bytes_released_;
    }

    const Length num_pages = [&] {
      if (num_bytes > 0) {
        // A sub-page size request may round down to zero.  Assume the caller
        // wants some memory released.
        const Length num_pages = BytesToLengthCeil(num_bytes);
        TC_ASSERT_GT(num_pages, Length(0));

        return num_pages;
      } else {
        return Length(0);
      }
    }();

    const size_t bytes_released = tc_globals.page_allocator()
                                      .ReleaseAtLeastNPages(num_pages, reason)
                                      .in_bytes();
    if (bytes_released > num_bytes) {
      extra_bytes_released_ = bytes_released - num_bytes;

      return num_bytes;
    }

    // The PageHeap wasn't able to release num_bytes.  Don't try to compensate
    // with a big release next time.
    extra_bytes_released_ = 0;

    return bytes_released;
  }

 private:
  size_t extra_bytes_released_ = 0;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_STATIC_VARS_H_
