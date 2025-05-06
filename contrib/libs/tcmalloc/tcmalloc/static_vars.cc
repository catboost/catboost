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

#include "tcmalloc/static_vars.h"

#include <stddef.h>

#include <atomic>
#include <cstring>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/optimization.h"
#include "absl/types/span.h"
#include "tcmalloc/allocation_sample.h"
#include "tcmalloc/arena.h"
#include "tcmalloc/common.h"
#include "tcmalloc/cpu_cache.h"
#include "tcmalloc/deallocation_profiler.h"
#include "tcmalloc/experiment.h"
#include "tcmalloc/experiment_config.h"
#include "tcmalloc/guarded_page_allocator.h"
#include "tcmalloc/internal/atomic_stats_counter.h"
#include "tcmalloc/internal/cache_topology.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/environment.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/mincore.h"
#include "tcmalloc/internal/mismatched_delete_state.h"
#include "tcmalloc/internal/numa.h"
#include "tcmalloc/internal/parameter_accessors.h"
#include "tcmalloc/internal/percpu.h"
#include "tcmalloc/internal/sampled_allocation.h"
#include "tcmalloc/internal/sysinfo.h"
#include "tcmalloc/malloc_extension.h"
#include "tcmalloc/metadata_object_allocator.h"
#include "tcmalloc/page_allocator.h"
#include "tcmalloc/pagemap.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/peak_heap_tracker.h"
#include "tcmalloc/size_class_info.h"
#include "tcmalloc/sizemap.h"
#include "tcmalloc/span.h"
#include "tcmalloc/stack_trace_table.h"
#include "tcmalloc/system-alloc.h"
#include "tcmalloc/thread_cache.h"
#include "tcmalloc/transfer_cache.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// TODO(b/304135905): Remove the opt out.
ABSL_ATTRIBUTE_WEAK bool default_want_disable_tcmalloc_big_span();
bool tcmalloc_big_span() {
  // Disable 64B span if built against an opt-out.
  if (default_want_disable_tcmalloc_big_span != nullptr) {
    return false;
  }

  const char* e = thread_safe_getenv("TCMALLOC_DISABLE_BIG_SPAN");
  if (e) {
    switch (e[0]) {
      case '0':
        return true;
      case '1':
        return false;
      default:
        TC_BUG("bad env var '%s'", e);
        return false;
    }
  }

  return true;
}

// Cacheline-align our SizeMap and CpuCache.  They both have very hot arrays as
// their first member variables, and aligning them reduces the number of cache
// lines these arrays use.
//
// IF YOU ADD TO THIS LIST, ADD TO STATIC_VAR_SIZE TOO!
// LINT.IfChange(static_vars)
ABSL_CONST_INIT absl::base_internal::SpinLock pageheap_lock(
    absl::kConstInit, absl::base_internal::SCHEDULE_KERNEL_ONLY);
ABSL_CONST_INIT Arena Static::arena_;
ABSL_CONST_INIT SizeMap ABSL_CACHELINE_ALIGNED Static::sizemap_;
TCMALLOC_ATTRIBUTE_NO_DESTROY ABSL_CONST_INIT TransferCacheManager
    Static::transfer_cache_;
ABSL_CONST_INIT ShardedTransferCacheManager
    Static::sharded_transfer_cache_(nullptr, nullptr);
ABSL_CONST_INIT CpuCache ABSL_CACHELINE_ALIGNED Static::cpu_cache_;
ABSL_CONST_INIT MetadataObjectAllocator<SampledAllocation>
    Static::sampledallocation_allocator_{arena_};
ABSL_CONST_INIT MetadataObjectAllocator<Span> Static::span_allocator_{arena_};
ABSL_CONST_INIT MetadataObjectAllocator<ThreadCache>
    Static::threadcache_allocator_{arena_};
TCMALLOC_ATTRIBUTE_NO_DESTROY ABSL_CONST_INIT SampledAllocationRecorder
    Static::sampled_allocation_recorder_{sampledallocation_allocator_};
ABSL_CONST_INIT tcmalloc_internal::StatsCounter Static::sampled_objects_size_;
ABSL_CONST_INIT tcmalloc_internal::StatsCounter
    Static::sampled_internal_fragmentation_;
ABSL_CONST_INIT tcmalloc_internal::StatsCounter Static::total_sampled_count_;
ABSL_CONST_INIT AllocationSampleList Static::allocation_samples;
ABSL_CONST_INIT deallocationz::DeallocationProfilerList
    Static::deallocation_samples;
ABSL_CONST_INIT std::atomic<AllocHandle> Static::sampled_alloc_handle_generator{
    0};
TCMALLOC_ATTRIBUTE_NO_DESTROY ABSL_CONST_INIT PeakHeapTracker
    Static::peak_heap_tracker_{sampledallocation_allocator_};
ABSL_CONST_INIT MetadataObjectAllocator<StackTraceTable::LinkedSample>
    Static::linked_sample_allocator_{arena_};
ABSL_CONST_INIT std::atomic<bool> Static::inited_{false};
ABSL_CONST_INIT std::atomic<bool> Static::cpu_cache_active_{false};
ABSL_CONST_INIT bool Static::fork_support_enabled_ = false;
ABSL_CONST_INIT Static::PageAllocatorStorage Static::page_allocator_;
ABSL_CONST_INIT PageMap Static::pagemap_;
ABSL_CONST_INIT GuardedPageAllocator Static::guardedpage_allocator_;
ABSL_CONST_INIT NumaTopology<kNumaPartitions, kNumBaseClasses>
    Static::numa_topology_;
ABSL_CONST_INIT MismatchedDeleteState Static::mismatched_delete_state_;
TCMALLOC_ATTRIBUTE_NO_DESTROY ABSL_CONST_INIT
    SystemAllocator<NumaTopology<kNumaPartitions, kNumBaseClasses>>
        Static::system_allocator_{numa_topology_};

// LINT.ThenChange(:static_vars_size)

ABSL_CONST_INIT Static tc_globals;

size_t Static::metadata_bytes() {
  // This is ugly and doesn't nicely account for e.g. alignment losses
  // -- I'd like to put all the above in a struct and take that
  // struct's size.  But we can't due to linking issues.
  //
  // TODO(b/242550501):  Progress on constant initialization guarantees allow
  // state to be consolidated directly into an instance, rather than as a
  // collection of static variables.  Simplify this.
  // LINT.IfChange(static_vars_size)
  const size_t static_var_size =
      sizeof(pageheap_lock) + sizeof(arena_) + sizeof(sizemap_) +
      sizeof(sharded_transfer_cache_) + sizeof(transfer_cache_) +
      sizeof(cpu_cache_) + sizeof(sampledallocation_allocator_) +
      sizeof(span_allocator_) + +sizeof(threadcache_allocator_) +
      sizeof(sampled_allocation_recorder_) + sizeof(linked_sample_allocator_) +
      sizeof(inited_) + sizeof(cpu_cache_active_) + sizeof(page_allocator_) +
      sizeof(pagemap_) + sizeof(sampled_objects_size_) +
      sizeof(sampled_internal_fragmentation_) + sizeof(total_sampled_count_) +
      sizeof(allocation_samples) + sizeof(deallocation_samples) +
      sizeof(sampled_alloc_handle_generator) + sizeof(peak_heap_tracker_) +
      sizeof(guardedpage_allocator_) + sizeof(numa_topology_) +
      sizeof(CacheTopology::Instance()) + sizeof(mismatched_delete_state_) +
      sizeof(system_allocator_);
  // LINT.ThenChange(:static_vars)

  const size_t allocated = arena().stats().bytes_allocated +
                           AddressRegionFactory::InternalBytesAllocated();
  return allocated + static_var_size;
}

size_t Static::pagemap_residence() {
  // Determine residence of the root node of the pagemap.
  size_t total = MInCore::residence(&pagemap_, sizeof(pagemap_));
  return total;
}

int ABSL_ATTRIBUTE_WEAK default_want_legacy_size_classes();

SizeClassConfiguration Static::size_class_configuration() {
  if (IsExperimentActive(Experiment::TEST_ONLY_TCMALLOC_POW2_SIZECLASS)) {
    return SizeClassConfiguration::kPow2Only;
  }

  // TODO(b/242710633): remove this opt out.
  if (default_want_legacy_size_classes != nullptr &&
      default_want_legacy_size_classes() > 0) {
    return SizeClassConfiguration::kLegacy;
  }

  const char* e = thread_safe_getenv("TCMALLOC_LEGACY_SIZE_CLASSES");
  if (e == nullptr) {
    return SizeClassConfiguration::kReuse;
  } else if (!strcmp(e, "pow2below64")) {
    return SizeClassConfiguration::kPow2Below64;
  } else if (!strcmp(e, "0")) {
    return SizeClassConfiguration::kReuse;
  } else {
    TC_BUG("bad TCMALLOC_LEGACY_SIZE_CLASSES env var '%s'", e);
  }
  return SizeClassConfiguration::kReuse;
}

ABSL_ATTRIBUTE_COLD ABSL_ATTRIBUTE_NOINLINE void Static::SlowInitIfNecessary() {
  PageHeapSpinLockHolder l;

  // double-checked locking
  if (!inited_.load(std::memory_order_acquire)) {
    TC_CHECK(sizemap_.Init(SizeMap::CurrentClasses().classes));
    // Verify we can determine the number of CPUs now, since we will need it
    // later for per-CPU caches and initializing the cache topology.
    if (ABSL_PREDICT_FALSE(!NumCPUsMaybe().has_value())) {
      TCMalloc_Internal_SetPerCpuCachesEnabledNoBuildRequirement(false);
    }
    (void)subtle::percpu::IsFast();
    numa_topology_.Init();
    CacheTopology::Instance().Init();

    const bool large_span_experiment = tcmalloc_big_span();
    Parameters::set_max_span_cache_size(
        large_span_experiment ? Span::kLargeCacheSize : Span::kCacheSize);
    Parameters::set_max_span_cache_array_size(
        large_span_experiment ? Span::kLargeCacheArraySize : Span::kCacheSize);

    if (IsExperimentActive(Experiment::TCMALLOC_MIN_HOT_ACCESS_HINT_ABLATION)) {
      TCMalloc_Internal_SetMinHotAccessHint(1);
    }

    // Do a bit of sanitizing: make sure central_cache is aligned properly
    TC_CHECK_EQ((sizeof(transfer_cache_) % ABSL_CACHELINE_SIZE), 0);
    transfer_cache_.Init();
    // The constructor of the sharded transfer cache leaves it in a disabled
    // state.
    sharded_transfer_cache_.Init();
    new (page_allocator_.memory) PageAllocator;
    pagemap_.MapRootWithSmallPages();
    guardedpage_allocator_.Init(/*max_allocated_pages=*/64,
                                /*total_pages=*/128);
    inited_.store(true, std::memory_order_release);

    // TODO: this is called with inited_ = true, so it looks like a race condition
    pageheap_lock.Unlock();
    pthread_atfork(
      TCMallocPreFork,
      TCMallocPostFork,
      TCMallocPostFork);
    pageheap_lock.Lock();
  }
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
