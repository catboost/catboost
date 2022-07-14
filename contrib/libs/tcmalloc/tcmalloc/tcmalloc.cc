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
// tcmalloc is a fast malloc implementation.  See
// https://github.com/google/tcmalloc/tree/master/docs/design.md for a high-level description of
// how this malloc works.
//
// SYNCHRONIZATION
//  1. The thread-/cpu-specific lists are accessed without acquiring any locks.
//     This is safe because each such list is only accessed by one thread/cpu at
//     a time.
//  2. We have a lock per central free-list, and hold it while manipulating
//     the central free list for a particular size.
//  3. The central page allocator is protected by "pageheap_lock".
//  4. The pagemap (which maps from page-number to descriptor),
//     can be read without holding any locks, and written while holding
//     the "pageheap_lock".
//
//     This multi-threaded access to the pagemap is safe for fairly
//     subtle reasons.  We basically assume that when an object X is
//     allocated by thread A and deallocated by thread B, there must
//     have been appropriate synchronization in the handoff of object
//     X from thread A to thread B.
//
// PAGEMAP
// -------
// Page map contains a mapping from page id to Span.
//
// If Span s occupies pages [p..q],
//      pagemap[p] == s
//      pagemap[q] == s
//      pagemap[p+1..q-1] are undefined
//      pagemap[p-1] and pagemap[q+1] are defined:
//         NULL if the corresponding page is not yet in the address space.
//         Otherwise it points to a Span.  This span may be free
//         or allocated.  If free, it is in one of pageheap's freelist.

#include "tcmalloc/tcmalloc.h"

#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/config.h"
#include "absl/base/const_init.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/internal/sysinfo.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/debugging/stacktrace.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/strip.h"
#include "tcmalloc/central_freelist.h"
#include "tcmalloc/common.h"
#include "tcmalloc/cpu_cache.h"
#include "tcmalloc/experiment.h"
#include "tcmalloc/guarded_page_allocator.h"
#include "tcmalloc/internal/linked_list.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/memory_stats.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/internal/percpu.h"
#include "tcmalloc/internal_malloc_extension.h"
#include "tcmalloc/malloc_extension.h"
#include "tcmalloc/page_allocator.h"
#include "tcmalloc/page_heap.h"
#include "tcmalloc/page_heap_allocator.h"
#include "tcmalloc/pagemap.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/sampler.h"
#include "tcmalloc/span.h"
#include "tcmalloc/stack_trace_table.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/stats.h"
#include "tcmalloc/system-alloc.h"
#include "tcmalloc/tcmalloc_policy.h"
#include "tcmalloc/thread_cache.h"
#include "tcmalloc/tracking.h"
#include "tcmalloc/transfer_cache.h"
#include "tcmalloc/transfer_cache_stats.h"

#if defined(TCMALLOC_HAVE_STRUCT_MALLINFO)
#include <malloc.h>
#endif

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// ----------------------- IMPLEMENTATION -------------------------------

// Extract interesting stats
struct TCMallocStats {
  uint64_t thread_bytes;               // Bytes in thread caches
  uint64_t central_bytes;              // Bytes in central cache
  uint64_t transfer_bytes;             // Bytes in central transfer cache
  uint64_t metadata_bytes;             // Bytes alloced for metadata
  uint64_t sharded_transfer_bytes;     // Bytes in per-CCX cache
  uint64_t per_cpu_bytes;              // Bytes in per-CPU cache
  uint64_t pagemap_root_bytes_res;     // Resident bytes of pagemap root node
  uint64_t percpu_metadata_bytes_res;  // Resident bytes of the per-CPU metadata
  AllocatorStats tc_stats;             // ThreadCache objects
  AllocatorStats span_stats;           // Span objects
  AllocatorStats stack_stats;          // StackTrace objects
  AllocatorStats bucket_stats;         // StackTraceTable::Bucket objects
  size_t pagemap_bytes;                // included in metadata bytes
  size_t percpu_metadata_bytes;        // included in metadata bytes
  BackingStats pageheap;               // Stats from page heap

  // Explicitly declare the ctor to put it in the google_malloc section.
  TCMallocStats() = default;
};

// Get stats into "r".  Also, if class_count != NULL, class_count[k]
// will be set to the total number of objects of size class k in the
// central cache, transfer cache, and per-thread and per-CPU caches.
// If small_spans is non-NULL, it is filled.  Same for large_spans.
// The boolean report_residence determines whether residence information
// should be captured or not. Residence info requires a potentially
// costly OS call, and is not necessary in all situations.
static void ExtractStats(TCMallocStats* r, uint64_t* class_count,
                         SpanStats* span_stats, SmallSpanStats* small_spans,
                         LargeSpanStats* large_spans,
                         TransferCacheStats* tc_stats, bool report_residence) {
  r->central_bytes = 0;
  r->transfer_bytes = 0;
  for (int cl = 0; cl < kNumClasses; ++cl) {
    const size_t length = Static::central_freelist(cl).length();
    const size_t tc_length = Static::transfer_cache().tc_length(cl);
    const size_t cache_overhead = Static::central_freelist(cl).OverheadBytes();
    const size_t size = Static::sizemap().class_to_size(cl);
    r->central_bytes += (size * length) + cache_overhead;
    r->transfer_bytes += (size * tc_length);
    if (class_count) {
      // Sum the lengths of all per-class freelists, except the per-thread
      // freelists, which get counted when we call GetThreadStats(), below.
      class_count[cl] = length + tc_length;
      if (UsePerCpuCache()) {
        class_count[cl] += Static::cpu_cache().TotalObjectsOfClass(cl);
      }
    }
    if (span_stats) {
      span_stats[cl] = Static::central_freelist(cl).GetSpanStats();
    }
    if (tc_stats) {
      tc_stats[cl] = Static::transfer_cache().GetHitRateStats(cl);
    }
  }

  // Add stats from per-thread heaps
  r->thread_bytes = 0;
  {  // scope
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    ThreadCache::GetThreadStats(&r->thread_bytes, class_count);
    r->tc_stats = ThreadCache::HeapStats();
    r->span_stats = Static::span_allocator().stats();
    r->stack_stats = Static::stacktrace_allocator().stats();
    r->bucket_stats = Static::bucket_allocator().stats();
    r->metadata_bytes = Static::metadata_bytes();
    r->pagemap_bytes = Static::pagemap().bytes();
    r->pageheap = Static::page_allocator().stats();
    if (small_spans != nullptr) {
      Static::page_allocator().GetSmallSpanStats(small_spans);
    }
    if (large_spans != nullptr) {
      Static::page_allocator().GetLargeSpanStats(large_spans);
    }
  }
  // We can access the pagemap without holding the pageheap_lock since it
  // is static data, and we are only taking address and size which are
  // constants.
  if (report_residence) {
    auto resident_bytes = Static::pagemap_residence();
    r->pagemap_root_bytes_res = resident_bytes;
    ASSERT(r->metadata_bytes >= r->pagemap_bytes);
    r->metadata_bytes = r->metadata_bytes - r->pagemap_bytes + resident_bytes;
  } else {
    r->pagemap_root_bytes_res = 0;
  }

  r->per_cpu_bytes = 0;
  r->sharded_transfer_bytes = 0;
  r->percpu_metadata_bytes_res = 0;
  r->percpu_metadata_bytes = 0;
  if (UsePerCpuCache()) {
    r->per_cpu_bytes = Static::cpu_cache().TotalUsedBytes();
    r->sharded_transfer_bytes = Static::sharded_transfer_cache().TotalBytes();

    if (report_residence) {
      auto percpu_metadata = Static::cpu_cache().MetadataMemoryUsage();
      r->percpu_metadata_bytes_res = percpu_metadata.resident_size;
      r->percpu_metadata_bytes = percpu_metadata.virtual_size;

      ASSERT(r->metadata_bytes >= r->percpu_metadata_bytes);
      r->metadata_bytes = r->metadata_bytes - r->percpu_metadata_bytes +
                          r->percpu_metadata_bytes_res;
    }
  }
}

static void ExtractTCMallocStats(TCMallocStats* r, bool report_residence) {
  ExtractStats(r, nullptr, nullptr, nullptr, nullptr, nullptr,
               report_residence);
}

// Because different fields of stats are computed from state protected
// by different locks, they may be inconsistent.  Prevent underflow
// when subtracting to avoid gigantic results.
static uint64_t StatSub(uint64_t a, uint64_t b) {
  return (a >= b) ? (a - b) : 0;
}

// Return approximate number of bytes in use by app.
static uint64_t InUseByApp(const TCMallocStats& stats) {
  return StatSub(stats.pageheap.system_bytes,
                 stats.thread_bytes + stats.central_bytes +
                     stats.transfer_bytes + stats.per_cpu_bytes +
                     stats.sharded_transfer_bytes + stats.pageheap.free_bytes +
                     stats.pageheap.unmapped_bytes);
}

static uint64_t VirtualMemoryUsed(const TCMallocStats& stats) {
  return stats.pageheap.system_bytes + stats.metadata_bytes;
}

static uint64_t PhysicalMemoryUsed(const TCMallocStats& stats) {
  return StatSub(VirtualMemoryUsed(stats), stats.pageheap.unmapped_bytes);
}

// The number of bytes either in use by the app or fragmented so that
// it cannot be (arbitrarily) reused.
static uint64_t RequiredBytes(const TCMallocStats& stats) {
  return StatSub(PhysicalMemoryUsed(stats), stats.pageheap.free_bytes);
}

static int CountAllowedCpus() {
  cpu_set_t allowed_cpus;
  if (sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus) != 0) {
    return 0;
  }

  return CPU_COUNT(&allowed_cpus);
}

// WRITE stats to "out"
static void DumpStats(Printer* out, int level) {
  TCMallocStats stats;
  uint64_t class_count[kNumClasses];
  SpanStats span_stats[kNumClasses];
  TransferCacheStats tc_stats[kNumClasses];
  if (level >= 2) {
    ExtractStats(&stats, class_count, span_stats, nullptr, nullptr, tc_stats,
                 true);
  } else {
    ExtractTCMallocStats(&stats, true);
  }

  static const double MiB = 1048576.0;

  out->printf(
      "See https://github.com/google/tcmalloc/tree/master/docs/stats.md for an explanation of "
      "this page\n");

  const uint64_t virtual_memory_used = VirtualMemoryUsed(stats);
  const uint64_t physical_memory_used = PhysicalMemoryUsed(stats);
  const uint64_t bytes_in_use_by_app = InUseByApp(stats);

#ifdef TCMALLOC_SMALL_BUT_SLOW
  out->printf("NOTE:  SMALL MEMORY MODEL IS IN USE, PERFORMANCE MAY SUFFER.\n");
#endif
  // clang-format off
  // Avoid clang-format complaining about the way that this text is laid out.
  out->printf(
      "------------------------------------------------\n"
      "MALLOC:   %12" PRIu64 " (%7.1f MiB) Bytes in use by application\n"
      "MALLOC: + %12" PRIu64 " (%7.1f MiB) Bytes in page heap freelist\n"
      "MALLOC: + %12" PRIu64 " (%7.1f MiB) Bytes in central cache freelist\n"
      "MALLOC: + %12" PRIu64 " (%7.1f MiB) Bytes in per-CPU cache freelist\n"
      "MALLOC: + %12" PRIu64 " (%7.1f MiB) Bytes in Sharded cache freelist\n"
      "MALLOC: + %12" PRIu64 " (%7.1f MiB) Bytes in transfer cache freelist\n"
      "MALLOC: + %12" PRIu64 " (%7.1f MiB) Bytes in thread cache freelists\n"
      "MALLOC: + %12" PRIu64 " (%7.1f MiB) Bytes in malloc metadata\n"
      "MALLOC:   ------------\n"
      "MALLOC: = %12" PRIu64 " (%7.1f MiB) Actual memory used (physical + swap)\n"
      "MALLOC: + %12" PRIu64 " (%7.1f MiB) Bytes released to OS (aka unmapped)\n"
      "MALLOC:   ------------\n"
      "MALLOC: = %12" PRIu64 " (%7.1f MiB) Virtual address space used\n"
      "MALLOC:\n"
      "MALLOC:   %12" PRIu64 "               Spans in use\n"
      "MALLOC:   %12" PRIu64 " (%7.1f MiB) Spans created\n"
      "MALLOC:   %12" PRIu64 "               Thread heaps in use\n"
      "MALLOC:   %12" PRIu64 " (%7.1f MiB) Thread heaps created\n"
      "MALLOC:   %12" PRIu64 "               Stack traces in use\n"
      "MALLOC:   %12" PRIu64 " (%7.1f MiB) Stack traces created\n"
      "MALLOC:   %12" PRIu64 "               Table buckets in use\n"
      "MALLOC:   %12" PRIu64 " (%7.1f MiB) Table buckets created\n"
      "MALLOC:   %12" PRIu64 " (%7.1f MiB) Pagemap bytes used\n"
      "MALLOC:   %12" PRIu64 " (%7.1f MiB) Pagemap root resident bytes\n"
      "MALLOC:   %12" PRIu64 " (%7.1f MiB) per-CPU slab bytes used\n"
      "MALLOC:   %12" PRIu64 " (%7.1f MiB) per-CPU slab resident bytes\n"
      "MALLOC:   %12" PRIu64 "               Tcmalloc page size\n"
      "MALLOC:   %12" PRIu64 "               Tcmalloc hugepage size\n"
      "MALLOC:   %12" PRIu64 "               CPUs Allowed in Mask\n",
      bytes_in_use_by_app, bytes_in_use_by_app / MiB,
      stats.pageheap.free_bytes, stats.pageheap.free_bytes / MiB,
      stats.central_bytes, stats.central_bytes / MiB,
      stats.per_cpu_bytes, stats.per_cpu_bytes / MiB,
      stats.sharded_transfer_bytes, stats.sharded_transfer_bytes / MiB,
      stats.transfer_bytes, stats.transfer_bytes / MiB,
      stats.thread_bytes, stats.thread_bytes / MiB,
      stats.metadata_bytes, stats.metadata_bytes / MiB,
      physical_memory_used, physical_memory_used / MiB,
      stats.pageheap.unmapped_bytes, stats.pageheap.unmapped_bytes / MiB,
      virtual_memory_used, virtual_memory_used / MiB,
      uint64_t(stats.span_stats.in_use),
      uint64_t(stats.span_stats.total),
      (stats.span_stats.total * sizeof(Span)) / MiB,
      uint64_t(stats.tc_stats.in_use),
      uint64_t(stats.tc_stats.total),
      (stats.tc_stats.total * sizeof(ThreadCache)) / MiB,
      uint64_t(stats.stack_stats.in_use),
      uint64_t(stats.stack_stats.total),
      (stats.stack_stats.total * sizeof(StackTrace)) / MiB,
      uint64_t(stats.bucket_stats.in_use),
      uint64_t(stats.bucket_stats.total),
      (stats.bucket_stats.total * sizeof(StackTraceTable::Bucket)) / MiB,
      uint64_t(stats.pagemap_bytes),
      stats.pagemap_bytes / MiB,
      stats.pagemap_root_bytes_res, stats.pagemap_root_bytes_res / MiB,
      uint64_t(stats.percpu_metadata_bytes),
      stats.percpu_metadata_bytes / MiB,
      stats.percpu_metadata_bytes_res, stats.percpu_metadata_bytes_res / MiB,
      uint64_t(kPageSize),
      uint64_t(kHugePageSize),
      CountAllowedCpus());
  // clang-format on

  PrintExperiments(out);
  out->printf(
      "MALLOC SAMPLED PROFILES: %zu bytes (current), %zu bytes (peak)\n",
      static_cast<size_t>(Static::sampled_objects_size_.value()),
      Static::peak_heap_tracker().CurrentPeakSize());

  MemoryStats memstats;
  if (GetMemoryStats(&memstats)) {
    uint64_t rss = memstats.rss;
    uint64_t vss = memstats.vss;
    // clang-format off
    out->printf(
        "\n"
        "Total process stats (inclusive of non-malloc sources):\n"
        "TOTAL: %12" PRIu64 " (%7.1f MiB) Bytes resident (physical memory used)\n"
        "TOTAL: %12" PRIu64 " (%7.1f MiB) Bytes mapped (virtual memory used)\n",
        rss, rss / MiB, vss, vss / MiB);
    // clang-format on
  }

  out->printf(
      "------------------------------------------------\n"
      "Call ReleaseMemoryToSystem() to release freelist memory to the OS"
      " (via madvise()).\n"
      "Bytes released to the OS take up virtual address space"
      " but no physical memory.\n");
  if (level >= 2) {
    out->printf("------------------------------------------------\n");
    out->printf("Total size of freelists for per-thread and per-CPU caches,\n");
    out->printf("transfer cache, and central cache, as well as number of\n");
    out->printf("live pages, returned/requested spans by size class\n");
    out->printf("------------------------------------------------\n");

    uint64_t cumulative = 0;
    for (int cl = 1; cl < kNumClasses; ++cl) {
      uint64_t class_bytes =
          class_count[cl] * Static::sizemap().class_to_size(cl);

      cumulative += class_bytes;
      // clang-format off
      out->printf(
          "class %3d [ %8zu bytes ] : %8" PRIu64 " objs; %5.1f MiB; %5.1f cum MiB; "
          "%8" PRIu64 " live pages; spans: %6zu ret / %6zu req = %5.4f;\n",
          cl, Static::sizemap().class_to_size(cl), class_count[cl],
          class_bytes / MiB, cumulative / MiB,
          span_stats[cl].num_live_spans()*Static::sizemap().class_to_pages(cl),
          span_stats[cl].num_spans_returned, span_stats[cl].num_spans_requested,
          span_stats[cl].prob_returned());
      // clang-format on
    }

    out->printf("------------------------------------------------\n");
    out->printf("Transfer cache implementation: %s\n",
                TransferCacheImplementationToLabel(
                    Static::transfer_cache().implementation()));

    out->printf("------------------------------------------------\n");
    out->printf("Transfer cache insert/remove hits/misses by size class\n");
    for (int cl = 1; cl < kNumClasses; ++cl) {
      out->printf(
          "class %3d [ %8zu bytes ] : %8" PRIu64 " insert hits; %8" PRIu64
          " insert misses (%8lu partial); %8" PRIu64 " remove hits; %8" PRIu64
          " remove misses (%8lu partial);\n",
          cl, Static::sizemap().class_to_size(cl), tc_stats[cl].insert_hits,
          tc_stats[cl].insert_misses, tc_stats[cl].insert_non_batch_misses,
          tc_stats[cl].remove_hits, tc_stats[cl].remove_misses,
          tc_stats[cl].remove_non_batch_misses);
    }

    if (UsePerCpuCache()) {
      Static::cpu_cache().Print(out);
    }

    Static::page_allocator().Print(out, MemoryTag::kNormal);
    if (Static::numa_topology().active_partitions() > 1) {
      Static::page_allocator().Print(out, MemoryTag::kNormalP1);
    }
    Static::page_allocator().Print(out, MemoryTag::kSampled);
    tracking::Print(out);
    Static::guardedpage_allocator().Print(out);

    uint64_t limit_bytes;
    bool is_hard;
    std::tie(limit_bytes, is_hard) = Static::page_allocator().limit();
    out->printf("PARAMETER desired_usage_limit_bytes %" PRIu64 " %s\n",
                limit_bytes, is_hard ? "(hard)" : "");
    out->printf("Number of times limit was hit: %lld\n",
                Static::page_allocator().limit_hits());

    out->printf("PARAMETER tcmalloc_per_cpu_caches %d\n",
                Parameters::per_cpu_caches() ? 1 : 0);
    out->printf("PARAMETER tcmalloc_max_per_cpu_cache_size %d\n",
                Parameters::max_per_cpu_cache_size());
    out->printf("PARAMETER tcmalloc_max_total_thread_cache_bytes %lld\n",
                Parameters::max_total_thread_cache_bytes());
    out->printf("PARAMETER malloc_release_bytes_per_sec %llu\n",
                Parameters::background_release_rate());
    out->printf(
        "PARAMETER tcmalloc_skip_subrelease_interval %s\n",
        absl::FormatDuration(Parameters::filler_skip_subrelease_interval()));
    out->printf("PARAMETER flat vcpus %d\n",
                subtle::percpu::UsingFlatVirtualCpus() ? 1 : 0);
  }
}

namespace {

/*static*/ void DumpStatsInPbtxt(Printer* out, int level) {
  TCMallocStats stats;
  uint64_t class_count[kNumClasses];
  SpanStats span_stats[kNumClasses];
  TransferCacheStats tc_stats[kNumClasses];
  if (level >= 2) {
    ExtractStats(&stats, class_count, span_stats, nullptr, nullptr, tc_stats,
                 true);
  } else {
    ExtractTCMallocStats(&stats, true);
  }

  const uint64_t bytes_in_use_by_app = InUseByApp(stats);
  const uint64_t virtual_memory_used = VirtualMemoryUsed(stats);
  const uint64_t physical_memory_used = PhysicalMemoryUsed(stats);

  PbtxtRegion region(out, kTop, /*indent=*/0);
  region.PrintI64("in_use_by_app", bytes_in_use_by_app);
  region.PrintI64("page_heap_freelist", stats.pageheap.free_bytes);
  region.PrintI64("central_cache_freelist", stats.central_bytes);
  region.PrintI64("per_cpu_cache_freelist", stats.per_cpu_bytes);
  region.PrintI64("sharded_transfer_cache_freelist",
                  stats.sharded_transfer_bytes);
  region.PrintI64("transfer_cache_freelist", stats.transfer_bytes);
  region.PrintI64("thread_cache_freelists", stats.thread_bytes);
  region.PrintI64("malloc_metadata", stats.metadata_bytes);
  region.PrintI64("actual_mem_used", physical_memory_used);
  region.PrintI64("unmapped", stats.pageheap.unmapped_bytes);
  region.PrintI64("virtual_address_space_used", virtual_memory_used);
  region.PrintI64("num_spans", uint64_t(stats.span_stats.in_use));
  region.PrintI64("num_spans_created", uint64_t(stats.span_stats.total));
  region.PrintI64("num_thread_heaps", uint64_t(stats.tc_stats.in_use));
  region.PrintI64("num_thread_heaps_created", uint64_t(stats.tc_stats.total));
  region.PrintI64("num_stack_traces", uint64_t(stats.stack_stats.in_use));
  region.PrintI64("num_stack_traces_created",
                  uint64_t(stats.stack_stats.total));
  region.PrintI64("num_table_buckets", uint64_t(stats.bucket_stats.in_use));
  region.PrintI64("num_table_buckets_created",
                  uint64_t(stats.bucket_stats.total));
  region.PrintI64("pagemap_size", uint64_t(stats.pagemap_bytes));
  region.PrintI64("pagemap_root_residence", stats.pagemap_root_bytes_res);
  region.PrintI64("percpu_slab_size", stats.percpu_metadata_bytes);
  region.PrintI64("percpu_slab_residence", stats.percpu_metadata_bytes_res);
  region.PrintI64("tcmalloc_page_size", uint64_t(kPageSize));
  region.PrintI64("tcmalloc_huge_page_size", uint64_t(kHugePageSize));
  region.PrintI64("cpus_allowed", CountAllowedCpus());

  {
    auto sampled_profiles = region.CreateSubRegion("sampled_profiles");
    sampled_profiles.PrintI64("current_bytes",
                              Static::sampled_objects_size_.value());
    sampled_profiles.PrintI64("peak_bytes",
                              Static::peak_heap_tracker().CurrentPeakSize());
  }

  // Print total process stats (inclusive of non-malloc sources).
  MemoryStats memstats;
  if (GetMemoryStats(&memstats)) {
    region.PrintI64("total_resident", uint64_t(memstats.rss));
    region.PrintI64("total_mapped", uint64_t(memstats.vss));
  }

  if (level >= 2) {
    {
      for (int cl = 1; cl < kNumClasses; ++cl) {
        uint64_t class_bytes =
            class_count[cl] * Static::sizemap().class_to_size(cl);
        PbtxtRegion entry = region.CreateSubRegion("freelist");
        entry.PrintI64("sizeclass", Static::sizemap().class_to_size(cl));
        entry.PrintI64("bytes", class_bytes);
        entry.PrintI64("num_spans_requested",
                       span_stats[cl].num_spans_requested);
        entry.PrintI64("num_spans_returned", span_stats[cl].num_spans_returned);
        entry.PrintI64("obj_capacity", span_stats[cl].obj_capacity);
      }
    }

    {
      for (int cl = 1; cl < kNumClasses; ++cl) {
        PbtxtRegion entry = region.CreateSubRegion("transfer_cache");
        entry.PrintI64("sizeclass", Static::sizemap().class_to_size(cl));
        entry.PrintI64("insert_hits", tc_stats[cl].insert_hits);
        entry.PrintI64("insert_misses", tc_stats[cl].insert_misses);
        entry.PrintI64("insert_non_batch_misses",
                       tc_stats[cl].insert_non_batch_misses);
        entry.PrintI64("remove_hits", tc_stats[cl].remove_hits);
        entry.PrintI64("remove_misses", tc_stats[cl].remove_misses);
        entry.PrintI64("remove_non_batch_misses",
                       tc_stats[cl].remove_non_batch_misses);
      }
    }

    region.PrintRaw("transfer_cache_implementation",
                    TransferCacheImplementationToLabel(
                        Static::transfer_cache().implementation()));

    if (UsePerCpuCache()) {
      Static::cpu_cache().PrintInPbtxt(&region);
    }
  }
  Static::page_allocator().PrintInPbtxt(&region, MemoryTag::kNormal);
  if (Static::numa_topology().active_partitions() > 1) {
    Static::page_allocator().PrintInPbtxt(&region, MemoryTag::kNormalP1);
  }
  Static::page_allocator().PrintInPbtxt(&region, MemoryTag::kSampled);
  // We do not collect tracking information in pbtxt.

  size_t limit_bytes;
  bool is_hard;
  std::tie(limit_bytes, is_hard) = Static::page_allocator().limit();
  region.PrintI64("desired_usage_limit_bytes", limit_bytes);
  region.PrintBool("hard_limit", is_hard);
  region.PrintI64("limit_hits", Static::page_allocator().limit_hits());

  {
    auto gwp_asan = region.CreateSubRegion("gwp_asan");
    Static::guardedpage_allocator().PrintInPbtxt(&gwp_asan);
  }

  region.PrintI64("memory_release_failures", SystemReleaseErrors());

  region.PrintBool("tcmalloc_per_cpu_caches", Parameters::per_cpu_caches());
  region.PrintI64("tcmalloc_max_per_cpu_cache_size",
                  Parameters::max_per_cpu_cache_size());
  region.PrintI64("tcmalloc_max_total_thread_cache_bytes",
                  Parameters::max_total_thread_cache_bytes());
  region.PrintI64("malloc_release_bytes_per_sec",
                  static_cast<int64_t>(Parameters::background_release_rate()));
  region.PrintI64(
      "tcmalloc_skip_subrelease_interval_ns",
      absl::ToInt64Nanoseconds(Parameters::filler_skip_subrelease_interval()));
  region.PrintRaw("percpu_vcpu_type",
                  subtle::percpu::UsingFlatVirtualCpus() ? "FLAT" : "NONE");
}

}  // namespace

// Gets a human readable description of the current state of the malloc data
// structures. A part of the state is stored in pbtxt format in `buffer`, the
// rest of the state is stored in the old format (the same as in
// MallocExtension::GetStats) in `other_buffer`. Both buffers are
// null-terminated strings in a prefix of "buffer[0,buffer_length-1]" or
// "other_buffer[0,other_buffer_length-1]". Returns the actual written sizes for
// buffer and other_buffer.
//
// REQUIRES: buffer_length > 0 and other_buffer_length > 0.
//
// TODO(b/130249686): This is NOT YET ready to use.
extern "C" ABSL_ATTRIBUTE_UNUSED int MallocExtension_Internal_GetStatsInPbtxt(
    char* buffer, int buffer_length) {
  ASSERT(buffer_length > 0);
  Printer printer(buffer, buffer_length);

  // Print level one stats unless lots of space is available
  if (buffer_length < 10000) {
    DumpStatsInPbtxt(&printer, 1);
  } else {
    DumpStatsInPbtxt(&printer, 2);
  }

  size_t required = printer.SpaceRequired();

  if (buffer_length > required) {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    required += GetRegionFactory()->GetStatsInPbtxt(
        absl::Span<char>(buffer + required, buffer_length - required));
  }

  return required;
}

static void PrintStats(int level) {
  const int kBufferSize = (TCMALLOC_HAVE_TRACKING ? 2 << 20 : 64 << 10);
  char* buffer = new char[kBufferSize];
  Printer printer(buffer, kBufferSize);
  DumpStats(&printer, level);
  (void)write(STDERR_FILENO, buffer, strlen(buffer));
  delete[] buffer;
}

// This function computes a profile that maps a live stack trace to
// the number of bytes of central-cache memory pinned by an allocation
// at that stack trace.
static std::unique_ptr<const ProfileBase> DumpFragmentationProfile() {
  auto profile = absl::make_unique<StackTraceTable>(ProfileType::kFragmentation,
                                                    1, true, true);

  {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    for (Span* s : Static::sampled_objects_) {
      // Compute fragmentation to charge to this sample:
      StackTrace* const t = s->sampled_stack();
      if (t->proxy == nullptr) {
        // There is just one object per-span, and neighboring spans
        // can be released back to the system, so we charge no
        // fragmentation to this sampled object.
        continue;
      }

      // Fetch the span on which the proxy lives so we can examine its
      // co-residents.
      const PageId p = PageIdContaining(t->proxy);
      Span* span = Static::pagemap().GetDescriptor(p);
      if (span == nullptr) {
        // Avoid crashes in production mode code, but report in tests.
        ASSERT(span != nullptr);
        continue;
      }

      const double frag = span->Fragmentation();
      if (frag > 0) {
        profile->AddTrace(frag, *t);
      }
    }
  }
  return profile;
}

// If <unsample> is true, the caller expects a profile where sampling has been
// compensated for (that is, it reports 8000 16-byte objects iff we believe the
// program has that many live objects.)  Otherwise, do not adjust for sampling
// (the caller will do so somehow.)
static std::unique_ptr<const ProfileBase> DumpHeapProfile(bool unsample) {
  auto profile = absl::make_unique<StackTraceTable>(
      ProfileType::kHeap, Sampler::GetSamplePeriod(), true, unsample);
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  for (Span* s : Static::sampled_objects_) {
    profile->AddTrace(1.0, *s->sampled_stack());
  }
  return profile;
}

class AllocationSampleList;

class AllocationSample final : public AllocationProfilingTokenBase {
 public:
  AllocationSample();
  ~AllocationSample() override;

  Profile Stop() && override;

 private:
  std::unique_ptr<StackTraceTable> mallocs_;
  AllocationSample* next ABSL_GUARDED_BY(pageheap_lock);
  friend class AllocationSampleList;
};

class AllocationSampleList {
 public:
  void Add(AllocationSample* as) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    as->next = first_;
    first_ = as;
  }

  // This list is very short and we're nowhere near a hot path, just walk
  void Remove(AllocationSample* as)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    AllocationSample** link = &first_;
    AllocationSample* cur = first_;
    while (cur != as) {
      CHECK_CONDITION(cur != nullptr);
      link = &cur->next;
      cur = cur->next;
    }
    *link = as->next;
  }

  void ReportMalloc(const struct StackTrace& sample)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    AllocationSample* cur = first_;
    while (cur != nullptr) {
      cur->mallocs_->AddTrace(1.0, sample);
      cur = cur->next;
    }
  }

 private:
  AllocationSample* first_;
} allocation_samples_ ABSL_GUARDED_BY(pageheap_lock);

AllocationSample::AllocationSample() {
  mallocs_ = absl::make_unique<StackTraceTable>(
      ProfileType::kAllocations, Sampler::GetSamplePeriod(), true, true);
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  allocation_samples_.Add(this);
}

AllocationSample::~AllocationSample() {
  if (mallocs_ == nullptr) {
    return;
  }

  // deleted before ending profile, do it for them
  {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    allocation_samples_.Remove(this);
  }
}

Profile AllocationSample::Stop() && ABSL_LOCKS_EXCLUDED(pageheap_lock) {
  // We need to remove ourselves from the allocation_samples_ list before we
  // mutate mallocs_;
  if (mallocs_) {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    allocation_samples_.Remove(this);
  }
  return ProfileAccessor::MakeProfile(std::move(mallocs_));
}

extern "C" void MallocExtension_Internal_GetStats(std::string* ret) {
  for (size_t shift = 17; shift < 22; shift++) {
    const size_t size = 1 << shift;
    // Double ret's size until we succeed in writing the buffer without
    // truncation.
    //
    // TODO(b/142931922):  printer only writes data and does not read it.
    // Leverage https://wg21.link/P1072 when it is standardized.
    ret->resize(size - 1);

    size_t written_size = TCMalloc_Internal_GetStats(&*ret->begin(), size - 1);
    if (written_size < size - 1) {
      // We did not truncate.
      ret->resize(written_size);
      break;
    }
  }
}

extern "C" size_t TCMalloc_Internal_GetStats(char* buffer,
                                             size_t buffer_length) {
  Printer printer(buffer, buffer_length);
  if (buffer_length < 10000) {
    DumpStats(&printer, 1);
  } else {
    DumpStats(&printer, 2);
  }

  printer.printf("\nLow-level allocator stats:\n");
  printer.printf("Memory Release Failures: %d\n", SystemReleaseErrors());

  size_t n = printer.SpaceRequired();

  size_t bytes_remaining = buffer_length > n ? buffer_length - n : 0;
  if (bytes_remaining > 0) {
    n += GetRegionFactory()->GetStats(
        absl::Span<char>(buffer + n, bytes_remaining));
  }

  return n;
}

extern "C" const ProfileBase* MallocExtension_Internal_SnapshotCurrent(
    ProfileType type) {
  switch (type) {
    case ProfileType::kHeap:
      return DumpHeapProfile(true).release();
    case ProfileType::kFragmentation:
      return DumpFragmentationProfile().release();
    case ProfileType::kPeakHeap:
      return Static::peak_heap_tracker().DumpSample().release();
    default:
      return nullptr;
  }
}

extern "C" AllocationProfilingTokenBase*
MallocExtension_Internal_StartAllocationProfiling() {
  return new AllocationSample();
}

bool GetNumericProperty(const char* name_data, size_t name_size,
                        size_t* value) {
  ASSERT(name_data != nullptr);
  ASSERT(value != nullptr);
  const absl::string_view name(name_data, name_size);

  // This is near the top since ReleasePerCpuMemoryToOS() calls it frequently.
  if (name == "tcmalloc.per_cpu_caches_active") {
    *value = Static::CPUCacheActive();
    return true;
  }

  if (name == "generic.virtual_memory_used") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = VirtualMemoryUsed(stats);
    return true;
  }

  if (name == "generic.physical_memory_used") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = PhysicalMemoryUsed(stats);
    return true;
  }

  if (name == "generic.current_allocated_bytes" ||
      name == "generic.bytes_in_use_by_app") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = InUseByApp(stats);
    return true;
  }

  if (name == "generic.heap_size") {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    BackingStats stats = Static::page_allocator().stats();
    *value = stats.system_bytes - stats.unmapped_bytes;
    return true;
  }

  if (name == "tcmalloc.central_cache_free") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = stats.central_bytes;
    return true;
  }

  if (name == "tcmalloc.cpu_free") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = stats.per_cpu_bytes;
    return true;
  }

  if (name == "tcmalloc.sharded_transfer_cache_free") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = stats.sharded_transfer_bytes;
    return true;
  }

  if (name == "tcmalloc.slack_bytes") {
    // Kept for backwards compatibility.  Now defined externally as:
    //    pageheap_free_bytes + pageheap_unmapped_bytes.
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    BackingStats stats = Static::page_allocator().stats();
    *value = stats.free_bytes + stats.unmapped_bytes;
    return true;
  }

  if (name == "tcmalloc.pageheap_free_bytes" ||
      name == "tcmalloc.page_heap_free") {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    *value = Static::page_allocator().stats().free_bytes;
    return true;
  }

  if (name == "tcmalloc.pageheap_unmapped_bytes" ||
      name == "tcmalloc.page_heap_unmapped") {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    *value = Static::page_allocator().stats().unmapped_bytes;
    return true;
  }

  if (name == "tcmalloc.page_algorithm") {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    *value = Static::page_allocator().algorithm();
    return true;
  }

  if (name == "tcmalloc.max_total_thread_cache_bytes") {
    absl::base_internal::SpinLockHolder l(&pageheap_lock);
    *value = ThreadCache::overall_thread_cache_size();
    return true;
  }

  if (name == "tcmalloc.current_total_thread_cache_bytes" ||
      name == "tcmalloc.thread_cache_free") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = stats.thread_bytes;
    return true;
  }

  if (name == "tcmalloc.thread_cache_count") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = stats.tc_stats.in_use;
    return true;
  }

  if (name == "tcmalloc.local_bytes") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value =
        stats.thread_bytes + stats.per_cpu_bytes + stats.sharded_transfer_bytes;
    ;
    return true;
  }

  if (name == "tcmalloc.external_fragmentation_bytes") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = (stats.pageheap.free_bytes + stats.central_bytes +
              stats.per_cpu_bytes + stats.sharded_transfer_bytes +
              stats.transfer_bytes + stats.thread_bytes + stats.metadata_bytes);
    return true;
  }

  if (name == "tcmalloc.metadata_bytes") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, true);
    *value = stats.metadata_bytes;
    return true;
  }

  if (name == "tcmalloc.transfer_cache_free") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = stats.transfer_bytes;
    return true;
  }

  bool want_hard_limit = (name == "tcmalloc.hard_usage_limit_bytes");
  if (want_hard_limit || name == "tcmalloc.desired_usage_limit_bytes") {
    size_t amount;
    bool is_hard;
    std::tie(amount, is_hard) = Static::page_allocator().limit();
    if (want_hard_limit != is_hard) {
      amount = std::numeric_limits<size_t>::max();
    }
    *value = amount;
    return true;
  }

  if (name == "tcmalloc.required_bytes") {
    TCMallocStats stats;
    ExtractTCMallocStats(&stats, false);
    *value = RequiredBytes(stats);
    return true;
  }

  const absl::string_view kExperimentPrefix = "tcmalloc.experiment.";
  if (absl::StartsWith(name, kExperimentPrefix)) {
    absl::optional<Experiment> exp =
        FindExperimentByName(absl::StripPrefix(name, kExperimentPrefix));
    if (exp.has_value()) {
      *value = IsExperimentActive(*exp) ? 1 : 0;
      return true;
    }
  }

  return false;
}

MallocExtension::Ownership GetOwnership(const void* ptr) {
  const PageId p = PageIdContaining(ptr);
  return Static::pagemap().GetDescriptor(p)
             ? MallocExtension::Ownership::kOwned
             : MallocExtension::Ownership::kNotOwned;
}

extern "C" bool MallocExtension_Internal_GetNumericProperty(
    const char* name_data, size_t name_size, size_t* value) {
  return GetNumericProperty(name_data, name_size, value);
}

extern "C" void MallocExtension_Internal_GetMemoryLimit(
    MallocExtension::MemoryLimit* limit) {
  ASSERT(limit != nullptr);

  std::tie(limit->limit, limit->hard) = Static::page_allocator().limit();
}

extern "C" void MallocExtension_Internal_SetMemoryLimit(
    const MallocExtension::MemoryLimit* limit) {
  ASSERT(limit != nullptr);

  if (!limit->hard) {
    Parameters::set_heap_size_hard_limit(0);
    Static::page_allocator().set_limit(limit->limit, false /* !hard */);
  } else {
    Parameters::set_heap_size_hard_limit(limit->limit);
  }
}

extern "C" void MallocExtension_Internal_MarkThreadIdle() {
  ThreadCache::BecomeIdle();
}

extern "C" AddressRegionFactory* MallocExtension_Internal_GetRegionFactory() {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  return GetRegionFactory();
}

extern "C" void MallocExtension_Internal_SetRegionFactory(
    AddressRegionFactory* factory) {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  SetRegionFactory(factory);
}

// ReleaseMemoryToSystem drops the page heap lock while actually calling to
// kernel to release pages. To avoid confusing ourselves with
// extra_bytes_released handling, lets do separate lock just for release.
ABSL_CONST_INIT static absl::base_internal::SpinLock release_lock(
    absl::kConstInit, absl::base_internal::SCHEDULE_KERNEL_ONLY);

extern "C" void MallocExtension_Internal_ReleaseMemoryToSystem(
    size_t num_bytes) {
  // ReleaseMemoryToSystem() might release more than the requested bytes because
  // the page heap releases at the span granularity, and spans are of wildly
  // different sizes.  This keeps track of the extra bytes bytes released so
  // that the app can periodically call ReleaseMemoryToSystem() to release
  // memory at a constant rate.
  ABSL_CONST_INIT static size_t extra_bytes_released;

  absl::base_internal::SpinLockHolder rh(&release_lock);

  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  if (num_bytes <= extra_bytes_released) {
    // We released too much on a prior call, so don't release any
    // more this time.
    extra_bytes_released = extra_bytes_released - num_bytes;
    num_bytes = 0;
  } else {
    num_bytes = num_bytes - extra_bytes_released;
  }

  Length num_pages;
  if (num_bytes > 0) {
    // A sub-page size request may round down to zero.  Assume the caller wants
    // some memory released.
    num_pages = BytesToLengthCeil(num_bytes);
    ASSERT(num_pages > Length(0));
  } else {
    num_pages = Length(0);
  }
  size_t bytes_released =
      Static::page_allocator().ReleaseAtLeastNPages(num_pages).in_bytes();
  if (bytes_released > num_bytes) {
    extra_bytes_released = bytes_released - num_bytes;
  } else {
    // The PageHeap wasn't able to release num_bytes.  Don't try to compensate
    // with a big release next time.
    extra_bytes_released = 0;
  }
}

extern "C" void MallocExtension_EnableForkSupport() {
  Static::EnableForkSupport();
}

void TCMallocPreFork() {
  if (!Static::ForkSupportEnabled()) {
    return;
  }

  if (Static::CPUCacheActive()) {
    Static::cpu_cache().AcquireInternalLocks();
  }
  Static::transfer_cache().AcquireInternalLocks();
  guarded_page_lock.Lock();
  release_lock.Lock();
  pageheap_lock.Lock();
  AcquireSystemAllocLock();
}

void TCMallocPostFork() {
  if (!Static::ForkSupportEnabled()) {
    return;
  }

  ReleaseSystemAllocLock();
  pageheap_lock.Unlock();  
  guarded_page_lock.Unlock();
  release_lock.Unlock();
  Static::transfer_cache().ReleaseInternalLocks();
  if (Static::CPUCacheActive()) {
    Static::cpu_cache().ReleaseInternalLocks();
  }
}

extern "C" void MallocExtension_SetSampleUserDataCallbacks(
    MallocExtension::CreateSampleUserDataCallback create,
    MallocExtension::CopySampleUserDataCallback copy,
    MallocExtension::DestroySampleUserDataCallback destroy) {
  Static::SetSampleUserDataCallbacks(create, copy, destroy);
}

// nallocx slow path.
// Moved to a separate function because size_class_with_alignment is not inlined
// which would cause nallocx to become non-leaf function with stack frame and
// stack spills. ABSL_ATTRIBUTE_ALWAYS_INLINE does not work on
// size_class_with_alignment, compiler barks that it can't inline the function
// somewhere.
static ABSL_ATTRIBUTE_NOINLINE size_t nallocx_slow(size_t size, int flags) {
  Static::InitIfNecessary();
  size_t align = static_cast<size_t>(1ull << (flags & 0x3f));
  uint32_t cl;
  if (ABSL_PREDICT_TRUE(Static::sizemap().GetSizeClass(
          CppPolicy().AlignAs(align), size, &cl))) {
    ASSERT(cl != 0);
    return Static::sizemap().class_to_size(cl);
  } else {
    return BytesToLengthCeil(size).in_bytes();
  }
}

// The nallocx function allocates no memory, but it performs the same size
// computation as the malloc function, and returns the real size of the
// allocation that would result from the equivalent malloc function call.
// nallocx is a malloc extension originally implemented by jemalloc:
// http://www.unix.com/man-page/freebsd/3/nallocx/
extern "C" size_t nallocx(size_t size, int flags) noexcept {
  if (ABSL_PREDICT_FALSE(!Static::IsInited() || flags != 0)) {
    return nallocx_slow(size, flags);
  }
  uint32_t cl;
  if (ABSL_PREDICT_TRUE(
          Static::sizemap().GetSizeClass(CppPolicy(), size, &cl))) {
    ASSERT(cl != 0);
    return Static::sizemap().class_to_size(cl);
  } else {
    return BytesToLengthCeil(size).in_bytes();
  }
}

extern "C" MallocExtension::Ownership MallocExtension_Internal_GetOwnership(
    const void* ptr) {
  return GetOwnership(ptr);
}

extern "C" void MallocExtension_Internal_GetProperties(
    std::map<std::string, MallocExtension::Property>* result) {
  TCMallocStats stats;
  ExtractTCMallocStats(&stats, true);

  const uint64_t virtual_memory_used = VirtualMemoryUsed(stats);
  const uint64_t physical_memory_used = PhysicalMemoryUsed(stats);
  const uint64_t bytes_in_use_by_app = InUseByApp(stats);

  result->clear();
  // Virtual Memory Used
  (*result)["generic.virtual_memory_used"].value = virtual_memory_used;
  // Physical Memory used
  (*result)["generic.physical_memory_used"].value = physical_memory_used;
  // Bytes in use By App
  (*result)["generic.bytes_in_use_by_app"].value = bytes_in_use_by_app;
  // Page Heap Free
  (*result)["tcmalloc.page_heap_free"].value = stats.pageheap.free_bytes;
  // Metadata Bytes
  (*result)["tcmalloc.metadata_bytes"].value = stats.metadata_bytes;
  // Heaps in Use
  (*result)["tcmalloc.thread_cache_count"].value = stats.tc_stats.in_use;
  // Central Cache Free List
  (*result)["tcmalloc.central_cache_free"].value = stats.central_bytes;
  // Transfer Cache Free List
  (*result)["tcmalloc.transfer_cache_free"].value = stats.transfer_bytes;
  // Per CPU Cache Free List
  (*result)["tcmalloc.cpu_free"].value = stats.per_cpu_bytes;
  (*result)["tcmalloc.sharded_transfer_cache_free"].value =
      stats.sharded_transfer_bytes;
  (*result)["tcmalloc.per_cpu_caches_active"].value = Static::CPUCacheActive();
  // Thread Cache Free List
  (*result)["tcmalloc.thread_cache_free"].value = stats.thread_bytes;
  // Page Unmapped
  (*result)["tcmalloc.pageheap_unmapped_bytes"].value =
      stats.pageheap.unmapped_bytes;
  (*result)["tcmalloc.page_heap_unmapped"].value =
      stats.pageheap.unmapped_bytes;

  (*result)["tcmalloc.page_algorithm"].value =
      Static::page_allocator().algorithm();

  FillExperimentProperties(result);
  tracking::GetProperties(result);
}

extern "C" size_t MallocExtension_Internal_ReleaseCpuMemory(int cpu) {
  size_t bytes = 0;
  if (Static::CPUCacheActive()) {
    bytes = Static::cpu_cache().Reclaim(cpu);
  }
  return bytes;
}

//-------------------------------------------------------------------
// Helpers for the exported routines below
//-------------------------------------------------------------------

#ifdef ABSL_HAVE_TLS
// See the comment on ThreadCache::thread_local_data_ regarding
// ABSL_ATTRIBUTE_INITIAL_EXEC.
__thread Sampler thread_sampler_ ABSL_ATTRIBUTE_INITIAL_EXEC;

inline Sampler* GetThreadSampler() { return &thread_sampler_; }

#else

inline Sampler* GetThreadSampler() {
  ThreadCache* heap = ThreadCache::GetCache();
  return heap->GetSampler();
}

#endif

enum class Hooks { RUN, NO };

static void FreeSmallSlow(void* ptr, size_t cl);

namespace {

// Sets `*psize` to `size`,
inline void SetCapacity(size_t size, std::nullptr_t) {}
inline void SetCapacity(size_t size, size_t* psize) { *psize = size; }

// Sets `*psize` to the size for the size class in `cl`,
inline void SetClassCapacity(size_t size, std::nullptr_t) {}
inline void SetClassCapacity(uint32_t cl, size_t* psize) {
  *psize = Static::sizemap().class_to_size(cl);
}

// Sets `*psize` to the size for the size class in `cl` if `ptr` is not null,
// else `*psize` is set to 0. This method is overloaded for `nullptr_t` below,
// allowing the compiler to optimize code between regular and size returning
// allocation operations.
inline void SetClassCapacity(const void*, uint32_t, std::nullptr_t) {}
inline void SetClassCapacity(const void* ptr, uint32_t cl, size_t* psize) {
  if (ABSL_PREDICT_TRUE(ptr != nullptr)) {
    *psize = Static::sizemap().class_to_size(cl);
  } else {
    *psize = 0;
  }
}

// Sets `*psize` to the size in pages corresponding to the requested size in
// `size` if `ptr` is not null, else `*psize` is set to 0. This method is
// overloaded for `nullptr_t` below, allowing the compiler to optimize code
// between regular and size returning allocation operations.
inline void SetPagesCapacity(const void*, size_t, std::nullptr_t) {}
inline void SetPagesCapacity(const void* ptr, size_t size, size_t* psize) {
  if (ABSL_PREDICT_TRUE(ptr != nullptr)) {
    *psize = BytesToLengthCeil(size).in_bytes();
  } else {
    *psize = 0;
  }
}

}  // namespace

// In free fast-path we handle delete hooks by delegating work to slower
// function that both performs delete hooks calls and does free. This is done so
// that free fast-path only does tail calls, which allow compiler to avoid
// generating costly prologue/epilogue for fast-path.
template <void F(void*, size_t), Hooks hooks_state>
static ABSL_ATTRIBUTE_SECTION(google_malloc) void invoke_delete_hooks_and_free(
    void* ptr, size_t t) {
  // Refresh the fast path state.
  GetThreadSampler()->UpdateFastPathState();
  return F(ptr, t);
}

template <void F(void*, PageId), Hooks hooks_state>
static ABSL_ATTRIBUTE_SECTION(google_malloc) void invoke_delete_hooks_and_free(
    void* ptr, PageId p) {
  // Refresh the fast path state.
  GetThreadSampler()->UpdateFastPathState();
  return F(ptr, p);
}

// Helper for do_free_with_cl
template <Hooks hooks_state>
static inline ABSL_ATTRIBUTE_ALWAYS_INLINE void FreeSmall(void* ptr,
                                                          size_t cl) {
  if (ABSL_PREDICT_FALSE(!GetThreadSampler()->IsOnFastPath())) {
    // Take the slow path.
    invoke_delete_hooks_and_free<FreeSmallSlow, hooks_state>(ptr, cl);
    return;
  }

#ifndef TCMALLOC_DEPRECATED_PERTHREAD
  // The CPU Cache is enabled, so we're able to take the fastpath.
  ASSERT(Static::CPUCacheActive());
  ASSERT(subtle::percpu::IsFastNoInit());

  Static::cpu_cache().Deallocate(ptr, cl);
#else  // TCMALLOC_DEPRECATED_PERTHREAD
  ThreadCache* cache = ThreadCache::GetCacheIfPresent();

  // IsOnFastPath does not track whether or not we have an active ThreadCache on
  // this thread, so we need to check cache for nullptr.
  if (ABSL_PREDICT_FALSE(cache == nullptr)) {
    FreeSmallSlow(ptr, cl);
    return;
  }

  cache->Deallocate(ptr, cl);
#endif  // TCMALLOC_DEPRECATED_PERTHREAD
}

// this helper function is used when FreeSmall (defined above) hits
// the case of thread state not being in per-cpu mode or hitting case
// of no thread cache. This happens when thread state is not yet
// properly initialized with real thread cache or with per-cpu mode,
// or when thread state is already destroyed as part of thread
// termination.
//
// We explicitly prevent inlining it to keep it out of fast-path, so
// that fast-path only has tail-call, so that fast-path doesn't need
// function prologue/epilogue.
ABSL_ATTRIBUTE_NOINLINE
static void FreeSmallSlow(void* ptr, size_t cl) {
  if (ABSL_PREDICT_TRUE(UsePerCpuCache())) {
    Static::cpu_cache().Deallocate(ptr, cl);
  } else if (ThreadCache* cache = ThreadCache::GetCacheIfPresent()) {
    // TODO(b/134691947):  If we reach this path from the ThreadCache fastpath,
    // we've already checked that UsePerCpuCache is false and cache == nullptr.
    // Consider optimizing this.
    cache->Deallocate(ptr, cl);
  } else {
    // This thread doesn't have thread-cache yet or already. Delete directly
    // into central cache.
    Static::transfer_cache().InsertRange(cl, absl::Span<void*>(&ptr, 1));
  }
}

namespace {

// If this allocation can be guarded, and if it's time to do a guarded sample,
// returns a guarded allocation Span.  Otherwise returns nullptr.
static void* TrySampleGuardedAllocation(size_t size, size_t alignment,
                                        Length num_pages) {
  if (num_pages == Length(1) &&
      GetThreadSampler()->ShouldSampleGuardedAllocation()) {
    // The num_pages == 1 constraint ensures that size <= kPageSize.  And since
    // alignments above kPageSize cause cl == 0, we're also guaranteed
    // alignment <= kPageSize
    //
    // In all cases kPageSize <= GPA::page_size_, so Allocate's preconditions
    // are met.
    return Static::guardedpage_allocator().Allocate(size, alignment);
  }
  return nullptr;
}

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
// Note that cl might not match requested_size in case of
// memalign. I.e. when larger than requested allocation is done to
// satisfy alignment constraint.
//
// In case of out-of-memory condition when allocating span or
// stacktrace struct, this function simply cheats and returns original
// object. As if no sampling was requested.
static void* SampleifyAllocation(size_t requested_size, size_t weight,
                                 size_t requested_alignment, size_t cl,
                                 void* obj, Span* span, size_t* capacity) {
  CHECK_CONDITION((cl != 0 && obj != nullptr && span == nullptr) ||
                  (cl == 0 && obj == nullptr && span != nullptr));

  void* proxy = nullptr;
  void* guarded_alloc = nullptr;
  size_t allocated_size;

  // requested_alignment = 1 means 'small size table alignment was used'
  // Historically this is reported as requested_alignment = 0
  if (requested_alignment == 1) {
    requested_alignment = 0;
  }

  if (cl != 0) {
    ASSERT(cl == Static::pagemap().sizeclass(PageIdContaining(obj)));

    allocated_size = Static::sizemap().class_to_size(cl);

    // If the caller didn't provide a span, allocate one:
    Length num_pages = BytesToLengthCeil(allocated_size);
    if ((guarded_alloc = TrySampleGuardedAllocation(
             requested_size, requested_alignment, num_pages))) {
      ASSERT(IsSampledMemory(guarded_alloc));
      const PageId p = PageIdContaining(guarded_alloc);
      absl::base_internal::SpinLockHolder h(&pageheap_lock);
      span = Span::New(p, num_pages);
      Static::pagemap().Set(p, span);
      // If we report capacity back from a size returning allocation, we can not
      // report the allocated_size, as we guard the size to 'requested_size',
      // and we maintain the invariant that GetAllocatedSize() must match the
      // returned size from size returning allocations. So in that case, we
      // report the requested size for both capacity and GetAllocatedSize().
      if (capacity) allocated_size = requested_size;
    } else if ((span = Static::page_allocator().New(
                    num_pages, MemoryTag::kSampled)) == nullptr) {
      if (capacity) *capacity = allocated_size;
      return obj;
    }

    size_t span_size = Length(Static::sizemap().class_to_pages(cl)).in_bytes();
    size_t objects_per_span = span_size / allocated_size;

    if (objects_per_span != 1) {
      ASSERT(objects_per_span > 1);
      proxy = obj;
      obj = nullptr;
    }
  } else {
    // Set allocated_size to the exact size for a page allocation.
    // NOTE: if we introduce gwp-asan sampling / guarded allocations
    // for page allocations, then we need to revisit do_malloc_pages as
    // the current assumption is that only class sized allocs are sampled
    // for gwp-asan.
    allocated_size = span->bytes_in_span();
  }
  if (capacity) *capacity = allocated_size;

  ASSERT(span != nullptr);

  // Grab the stack trace outside the heap lock
  StackTrace tmp;
  tmp.proxy = proxy;
  tmp.depth = absl::GetStackTrace(tmp.stack, kMaxStackDepth, 1);
  tmp.requested_size = requested_size;
  tmp.requested_alignment = requested_alignment;
  tmp.allocated_size = allocated_size;
  tmp.weight = weight;
  tmp.user_data = Static::CreateSampleUserData();

  {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    // Allocate stack trace
    StackTrace* stack = Static::stacktrace_allocator().New();
    allocation_samples_.ReportMalloc(tmp);
    *stack = tmp;
    span->Sample(stack);
  }

  Static::peak_heap_tracker().MaybeSaveSample();

  if (obj != nullptr) {
#if TCMALLOC_HAVE_TRACKING
    // We delete directly into central cache to avoid tracking this as
    // purely internal deletion. We've already (correctly) tracked
    // this allocation as either malloc hit or malloc miss, and we
    // must not count anything else for this allocation.
    //
    // TODO(b/158678747):  As of cl/315283185, we may occasionally see a hit in
    // the TransferCache here.  Prior to that CL, we always forced a miss.  Both
    // of these may artificially skew our tracking data.
    Static::transfer_cache().InsertRange(cl, absl::Span<void*>(&obj, 1));
#else
    // We are not maintaining precise statistics on malloc hit/miss rates at our
    // cache tiers.  We can deallocate into our ordinary cache.
    ASSERT(cl != 0);
    FreeSmallSlow(obj, cl);
#endif
  }
  return guarded_alloc ? guarded_alloc : span->start_address();
}

// ShouldSampleAllocation() is called when an allocation of the given requested
// size is in progress. It returns the sampling weight of the allocation if it
// should be "sampled," and 0 otherwise. See SampleifyAllocation().
//
// Sampling is done based on requested sizes and later unskewed during profile
// generation.
inline size_t ShouldSampleAllocation(size_t size) {
  return GetThreadSampler()->RecordAllocation(size);
}

template <typename Policy>
inline void* do_malloc_pages(Policy policy, size_t size) {
  // Page allocator does not deal well with num_pages = 0.
  Length num_pages = std::max<Length>(BytesToLengthCeil(size), Length(1));

  MemoryTag tag = MemoryTag::kNormal;
    if (Static::numa_topology().numa_aware()) {
    tag = NumaNormalTag(policy.numa_partition());
  }
  const size_t alignment = policy.align();
  Span* span = Static::page_allocator().NewAligned(
      num_pages, BytesToLengthCeil(alignment), tag);

  if (span == nullptr) {
    return nullptr;
  }

  void* result = span->start_address();
  ASSERT(
      tag == GetMemoryTag(span->start_address()));

  if (size_t weight = ShouldSampleAllocation(size)) {
    CHECK_CONDITION(result == SampleifyAllocation(size, weight, alignment, 0,
                                                  nullptr, span, nullptr));
  }

  return result;
}

template <typename Policy, typename CapacityPtr>
inline void* ABSL_ATTRIBUTE_ALWAYS_INLINE AllocSmall(Policy policy, size_t cl,
                                                     size_t size,
                                                     CapacityPtr capacity) {
  ASSERT(cl != 0);
  void* result;

  if (UsePerCpuCache()) {
    result = Static::cpu_cache().Allocate<Policy::handle_oom>(cl);
  } else {
    result = ThreadCache::GetCache()->Allocate<Policy::handle_oom>(cl);
  }

  if (!Policy::can_return_nullptr()) {
    ASSUME(result != nullptr);
  }

  if (ABSL_PREDICT_FALSE(result == nullptr)) {
    SetCapacity(0, capacity);
    return nullptr;
  }
  size_t weight;
  if (ABSL_PREDICT_FALSE(weight = ShouldSampleAllocation(size))) {
    return SampleifyAllocation(size, weight, policy.align(), cl, result,
                               nullptr, capacity);
  }
  SetClassCapacity(cl, capacity);
  return result;
}

// Handles freeing object that doesn't have size class, i.e. which
// is either large or sampled. We explicitly prevent inlining it to
// keep it out of fast-path. This helps avoid expensive
// prologue/epiloge for fast-path freeing functions.
ABSL_ATTRIBUTE_NOINLINE
static void do_free_pages(void* ptr, const PageId p) {
  void* proxy = nullptr;
  size_t size;
  bool notify_sampled_alloc = false;

  Span* span = Static::pagemap().GetExistingDescriptor(p);
  ASSERT(span != nullptr);
  // Prefetch now to avoid a stall accessing *span while under the lock.
  span->Prefetch();
  {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    ASSERT(span->first_page() == p);
    if (StackTrace* st = span->Unsample()) {
      proxy = st->proxy;
      size = st->allocated_size;
      if (proxy == nullptr && size <= kMaxSize) {
        tracking::Report(kFreeMiss,
                         Static::sizemap().SizeClass(
                             CppPolicy().InSameNumaPartitionAs(ptr), size),
                         1);
      }
      notify_sampled_alloc = true;
      Static::DestroySampleUserData(st->user_data);
      Static::stacktrace_allocator().Delete(st);
    }
    if (IsSampledMemory(ptr)) {
      if (Static::guardedpage_allocator().PointerIsMine(ptr)) {
        // Release lock while calling Deallocate() since it does a system call.
        pageheap_lock.Unlock();
        Static::guardedpage_allocator().Deallocate(ptr);
        pageheap_lock.Lock();
        Span::Delete(span);
      } else {
        ASSERT(reinterpret_cast<uintptr_t>(ptr) % kPageSize == 0);
        Static::page_allocator().Delete(span, MemoryTag::kSampled);
      }
    } else if (kNumaPartitions != 1) {
      ASSERT(reinterpret_cast<uintptr_t>(ptr) % kPageSize == 0);
      Static::page_allocator().Delete(span, GetMemoryTag(ptr));
    } else {
      ASSERT(reinterpret_cast<uintptr_t>(ptr) % kPageSize == 0);
      Static::page_allocator().Delete(span, MemoryTag::kNormal);
    }
  }

  if (notify_sampled_alloc) {
  }

  if (proxy) {
    const auto policy = CppPolicy().InSameNumaPartitionAs(proxy);
    const size_t cl = Static::sizemap().SizeClass(policy, size);
    FreeSmall<Hooks::NO>(proxy, cl);
  }
}

#ifndef NDEBUG
static size_t GetSizeClass(void* ptr) {
  const PageId p = PageIdContaining(ptr);
  return Static::pagemap().sizeclass(p);
}
#endif

// Helper for the object deletion (free, delete, etc.).  Inputs:
//   ptr is object to be freed
//   cl is the size class of that object, or 0 if it's unknown
//   have_cl is true iff cl is known and is non-0.
//
// Note that since have_cl is compile-time constant, genius compiler
// would not need it. Since it would be able to somehow infer that
// GetSizeClass never produces 0 cl, and so it
// would know that places that call this function with explicit 0 is
// "have_cl-case" and others are "!have_cl-case". But we certainly
// don't have such compiler. See also do_free_with_size below.
template <bool have_cl, Hooks hooks_state>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void do_free_with_cl(void* ptr, size_t cl) {
  // !have_cl -> cl == 0
  ASSERT(have_cl || cl == 0);

  const PageId p = PageIdContaining(ptr);

  // if we have_cl, then we've excluded ptr == nullptr case. See
  // comment in do_free_with_size. Thus we only bother testing nullptr
  // in non-sized case.
  //
  // Thus: ptr == nullptr -> !have_cl
  ASSERT(ptr != nullptr || !have_cl);
  if (!have_cl && ABSL_PREDICT_FALSE(ptr == nullptr)) {
    return;
  }

  // ptr must be a result of a previous malloc/memalign/... call, and
  // therefore static initialization must have already occurred.
  ASSERT(Static::IsInited());

  if (!have_cl) {
    cl = Static::pagemap().sizeclass(p);
  }
  if (have_cl || ABSL_PREDICT_TRUE(cl != 0)) {
    ASSERT(cl == GetSizeClass(ptr));
    ASSERT(ptr != nullptr);
    ASSERT(!Static::pagemap().GetExistingDescriptor(p)->sampled());
    FreeSmall<hooks_state>(ptr, cl);
  } else {
    invoke_delete_hooks_and_free<do_free_pages, hooks_state>(ptr, p);
  }
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE void do_free(void* ptr) {
  return do_free_with_cl<false, Hooks::RUN>(ptr, 0);
}

void do_free_no_hooks(void* ptr) {
  return do_free_with_cl<false, Hooks::NO>(ptr, 0);
}

template <typename AlignPolicy>
bool CorrectSize(void* ptr, size_t size, AlignPolicy align);

bool CorrectAlignment(void* ptr, std::align_val_t alignment);

inline ABSL_ATTRIBUTE_ALWAYS_INLINE void FreePages(void* ptr) {
  const PageId p = PageIdContaining(ptr);
  invoke_delete_hooks_and_free<do_free_pages, Hooks::RUN>(ptr, p);
}

template <typename AlignPolicy>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void do_free_with_size(void* ptr,
                                                           size_t size,
                                                           AlignPolicy align) {
  ASSERT(CorrectSize(ptr, size, align));
  ASSERT(CorrectAlignment(ptr, static_cast<std::align_val_t>(align.align())));

  // This is an optimized path that may be taken if the binary is compiled
  // with -fsized-delete. We attempt to discover the size class cheaply
  // without any cache misses by doing a plain computation that
  // maps from size to size-class.
  //
  // The optimized path doesn't work with sampled objects, whose deletions
  // trigger more operations and require to visit metadata.
  if (ABSL_PREDICT_FALSE(IsSampledMemory(ptr))) {
      // we don't know true class size of the ptr
      if (ptr == nullptr) return;
      return FreePages(ptr);
  }

  // At this point, since ptr's tag bit is 1, it means that it
  // cannot be nullptr either. Thus all code below may rely on ptr !=
  // nullptr. And particularly, since we're only caller of
  // do_free_with_cl with have_cl == true, it means have_cl implies
  // ptr != nullptr.
  ASSERT(ptr != nullptr);

  uint32_t cl;
  if (ABSL_PREDICT_FALSE(!Static::sizemap().GetSizeClass(
          CppPolicy().AlignAs(align.align()).InSameNumaPartitionAs(ptr), size,
          &cl))) {
    // We couldn't calculate the size class, which means size > kMaxSize.
    ASSERT(size > kMaxSize || align.align() > alignof(std::max_align_t));
    static_assert(kMaxSize >= kPageSize, "kMaxSize must be at least kPageSize");
    return FreePages(ptr);
  }

  return do_free_with_cl<true, Hooks::RUN>(ptr, cl);
}

inline size_t GetSize(const void* ptr) {
  if (ptr == nullptr) return 0;
  const PageId p = PageIdContaining(ptr);
  size_t cl = Static::pagemap().sizeclass(p);
  if (cl != 0) {
    return Static::sizemap().class_to_size(cl);
  } else {
    const Span* span = Static::pagemap().GetExistingDescriptor(p);
    if (span->sampled()) {
      if (Static::guardedpage_allocator().PointerIsMine(ptr)) {
        return Static::guardedpage_allocator().GetRequestedSize(ptr);
      }
      return span->sampled_stack()->allocated_size;
    } else {
      return span->bytes_in_span();
    }
  }
}

// Checks that an asserted object size for <ptr> is valid.
template <typename AlignPolicy>
bool CorrectSize(void* ptr, size_t size, AlignPolicy align) {
  // size == 0 means we got no hint from sized delete, so we certainly don't
  // have an incorrect one.
  if (size == 0) return true;
  if (ptr == nullptr) return true;
  uint32_t cl = 0;
  // Round-up passed in size to how much tcmalloc allocates for that size.
  if (Static::guardedpage_allocator().PointerIsMine(ptr)) {
    size = Static::guardedpage_allocator().GetRequestedSize(ptr);
  } else if (Static::sizemap().GetSizeClass(CppPolicy().AlignAs(align.align()),
                                            size, &cl)) {
    size = Static::sizemap().class_to_size(cl);
  } else {
    size = BytesToLengthCeil(size).in_bytes();
  }
  size_t actual = GetSize(ptr);
  if (ABSL_PREDICT_TRUE(actual == size)) return true;
  Log(kLog, __FILE__, __LINE__, "size check failed", actual, size, cl);
  return false;
}

// Checks that an asserted object <ptr> has <align> alignment.
bool CorrectAlignment(void* ptr, std::align_val_t alignment) {
  size_t align = static_cast<size_t>(alignment);
  ASSERT(absl::has_single_bit(align));
  return ((reinterpret_cast<uintptr_t>(ptr) & (align - 1)) == 0);
}

// Helpers for use by exported routines below or inside debugallocation.cc:

inline void do_malloc_stats() { PrintStats(1); }

inline int do_mallopt(int cmd, int value) {
  return 1;  // Indicates error
}

#ifdef TCMALLOC_HAVE_STRUCT_MALLINFO
inline struct mallinfo do_mallinfo() {
  TCMallocStats stats;
  ExtractTCMallocStats(&stats, false);

  // Just some of the fields are filled in.
  struct mallinfo info;
  memset(&info, 0, sizeof(info));

  // Unfortunately, the struct contains "int" field, so some of the
  // size values will be truncated.
  info.arena = static_cast<int>(stats.pageheap.system_bytes);
  info.fsmblks = static_cast<int>(stats.thread_bytes + stats.central_bytes +
                                  stats.transfer_bytes);
  info.fordblks = static_cast<int>(stats.pageheap.free_bytes +
                                   stats.pageheap.unmapped_bytes);
  info.uordblks = static_cast<int>(InUseByApp(stats));

  return info;
}
#endif  // TCMALLOC_HAVE_STRUCT_MALLINFO

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

using tcmalloc::tcmalloc_internal::AllocSmall;
using tcmalloc::tcmalloc_internal::CppPolicy;
using tcmalloc::tcmalloc_internal::do_free_no_hooks;
#ifdef TCMALLOC_HAVE_STRUCT_MALLINFO
using tcmalloc::tcmalloc_internal::do_mallinfo;
#endif
using tcmalloc::tcmalloc_internal::do_malloc_pages;
using tcmalloc::tcmalloc_internal::do_malloc_stats;
using tcmalloc::tcmalloc_internal::do_mallopt;
using tcmalloc::tcmalloc_internal::GetThreadSampler;
using tcmalloc::tcmalloc_internal::MallocPolicy;
using tcmalloc::tcmalloc_internal::SetClassCapacity;
using tcmalloc::tcmalloc_internal::SetPagesCapacity;
using tcmalloc::tcmalloc_internal::Static;
using tcmalloc::tcmalloc_internal::UsePerCpuCache;

#ifdef TCMALLOC_DEPRECATED_PERTHREAD
using tcmalloc::tcmalloc_internal::ThreadCache;
#endif  // TCMALLOC_DEPRECATED_PERTHREAD

// Slow path implementation.
// This function is used by `fast_alloc` if the allocation requires page sized
// allocations or some complex logic is required such as initialization,
// invoking new/delete hooks, sampling, etc.
//
// TODO(b/130771275):  This function is marked as static, rather than appearing
// in the anonymous namespace, to workaround incomplete heapz filtering.
template <typename Policy, typename CapacityPtr = std::nullptr_t>
static void* ABSL_ATTRIBUTE_SECTION(google_malloc)
    slow_alloc(Policy policy, size_t size, CapacityPtr capacity = nullptr) {
  Static::InitIfNecessary();
  GetThreadSampler()->UpdateFastPathState();
  void* p;
  uint32_t cl;
  bool is_small = Static::sizemap().GetSizeClass(policy, size, &cl);
  if (ABSL_PREDICT_TRUE(is_small)) {
    p = AllocSmall(policy, cl, size, capacity);
  } else {
    p = do_malloc_pages(policy, size);
    // Set capacity to the exact size for a page allocation.
    // This needs to be revisited if we introduce gwp-asan
    // sampling / guarded allocations to do_malloc_pages().
    SetPagesCapacity(p, size, capacity);
    if (ABSL_PREDICT_FALSE(p == nullptr)) {
      return Policy::handle_oom(size);
    }
  }
  if (Policy::invoke_hooks()) {
  }
  return p;
}

template <typename Policy, typename CapacityPtr = std::nullptr_t>
static inline void* ABSL_ATTRIBUTE_ALWAYS_INLINE
fast_alloc(Policy policy, size_t size, CapacityPtr capacity = nullptr) {
  // If size is larger than kMaxSize, it's not fast-path anymore. In
  // such case, GetSizeClass will return false, and we'll delegate to the slow
  // path. If malloc is not yet initialized, we may end up with cl == 0
  // (regardless of size), but in this case should also delegate to the slow
  // path by the fast path check further down.
  uint32_t cl;
  bool is_small = Static::sizemap().GetSizeClass(policy, size, &cl);
  if (ABSL_PREDICT_FALSE(!is_small)) {
    return slow_alloc(policy, size, capacity);
  }

  // When using per-thread caches, we have to check for the presence of the
  // cache for this thread before we try to sample, as slow_alloc will
  // also try to sample the allocation.
#ifdef TCMALLOC_DEPRECATED_PERTHREAD
  ThreadCache* const cache = ThreadCache::GetCacheIfPresent();
  if (ABSL_PREDICT_FALSE(cache == nullptr)) {
    return slow_alloc(policy, size, capacity);
  }
#endif
  // TryRecordAllocationFast() returns true if no extra logic is required, e.g.:
  // - this allocation does not need to be sampled
  // - no new/delete hooks need to be invoked
  // - no need to initialize thread globals, data or caches.
  // The method updates 'bytes until next sample' thread sampler counters.
  if (ABSL_PREDICT_FALSE(!GetThreadSampler()->TryRecordAllocationFast(size))) {
    return slow_alloc(policy, size, capacity);
  }

  // Fast path implementation for allocating small size memory.
  // This code should only be reached if all of the below conditions are met:
  // - the size does not exceed the maximum size (size class > 0)
  // - cpu / thread cache data has been initialized.
  // - the allocation is not subject to sampling / gwp-asan.
  // - no new/delete hook is installed and required to be called.
  ASSERT(cl != 0);
  void* ret;
#ifndef TCMALLOC_DEPRECATED_PERTHREAD
  // The CPU cache should be ready.
  ret = Static::cpu_cache().Allocate<Policy::handle_oom>(cl);
#else  // !defined(TCMALLOC_DEPRECATED_PERTHREAD)
  // The ThreadCache should be ready.
  ASSERT(cache != nullptr);
  ret = cache->Allocate<Policy::handle_oom>(cl);
#endif  // TCMALLOC_DEPRECATED_PERTHREAD
  if (!Policy::can_return_nullptr()) {
    ASSUME(ret != nullptr);
  }
  SetClassCapacity(ret, cl, capacity);
  return ret;
}

using tcmalloc::tcmalloc_internal::GetOwnership;
using tcmalloc::tcmalloc_internal::GetSize;

extern "C" size_t MallocExtension_Internal_GetAllocatedSize(const void* ptr) {
  ASSERT(!ptr ||
         GetOwnership(ptr) != tcmalloc::MallocExtension::Ownership::kNotOwned);
  return GetSize(ptr);
}

extern "C" void MallocExtension_Internal_MarkThreadBusy() {
  // Allocate to force the creation of a thread cache, but avoid
  // invoking any hooks.
  Static::InitIfNecessary();

  if (UsePerCpuCache()) {
    return;
  }

  do_free_no_hooks(slow_alloc(CppPolicy().Nothrow().WithoutHooks(), 0));
}

//-------------------------------------------------------------------
// Exported routines
//-------------------------------------------------------------------

using tcmalloc::tcmalloc_internal::AlignAsPolicy;
using tcmalloc::tcmalloc_internal::CorrectAlignment;
using tcmalloc::tcmalloc_internal::CorrectSize;
using tcmalloc::tcmalloc_internal::DefaultAlignPolicy;
using tcmalloc::tcmalloc_internal::do_free;
using tcmalloc::tcmalloc_internal::do_free_with_size;

// depends on TCMALLOC_HAVE_STRUCT_MALLINFO, so needs to come after that.
#include "tcmalloc/libc_override.h"

extern "C" ABSL_CACHELINE_ALIGNED void* TCMallocInternalMalloc(
    size_t size) noexcept {
  // Use TCMallocInternalMemalign to avoid requiring size %
  // alignof(std::max_align_t) == 0. TCMallocInternalAlignedAlloc enforces this
  // property.
  return TCMallocInternalMemalign(alignof(std::max_align_t), size);
}

extern "C" ABSL_CACHELINE_ALIGNED void* TCMallocInternalNew(size_t size) {
  return fast_alloc(CppPolicy(), size);
}

extern "C" ABSL_ATTRIBUTE_SECTION(google_malloc) tcmalloc::sized_ptr_t
    tcmalloc_size_returning_operator_new(size_t size) {
  size_t capacity;
  void* p = fast_alloc(CppPolicy(), size, &capacity);
  return {p, capacity};
}

extern "C" ABSL_CACHELINE_ALIGNED void* TCMallocInternalMalloc_aligned(
    size_t size, std::align_val_t alignment) noexcept {
  return fast_alloc(MallocPolicy().AlignAs(alignment), size);
}

extern "C" ABSL_CACHELINE_ALIGNED void* TCMallocInternalNewAligned(
    size_t size, std::align_val_t alignment) {
  return fast_alloc(CppPolicy().AlignAs(alignment), size);
}

#ifdef TCMALLOC_ALIAS
extern "C" void* TCMallocInternalNewAligned_nothrow(
    size_t size, std::align_val_t alignment, const std::nothrow_t& nt) noexcept
    // Note: we use malloc rather than new, as we are allowed to return nullptr.
    // The latter crashes in that case.
    TCMALLOC_ALIAS(TCMallocInternalMalloc_aligned);
#else
extern "C" ABSL_ATTRIBUTE_SECTION(
    google_malloc) void* TCMallocInternalNewAligned_nothrow(size_t size,
                                                            std::align_val_t
                                                                alignment,
                                                            const std::nothrow_t&
                                                                nt) noexcept {
  return fast_alloc(CppPolicy().Nothrow().AlignAs(alignment), size);
}
#endif  // TCMALLOC_ALIAS

extern "C" ABSL_CACHELINE_ALIGNED void TCMallocInternalFree(
    void* ptr) noexcept {
  do_free(ptr);
}

extern "C" void TCMallocInternalSdallocx(void* ptr, size_t size,
                                         int flags) noexcept {
  size_t alignment = alignof(std::max_align_t);

  if (ABSL_PREDICT_FALSE(flags != 0)) {
    ASSERT((flags & ~0x3f) == 0);
    alignment = static_cast<size_t>(1ull << (flags & 0x3f));
  }

  return do_free_with_size(ptr, size, AlignAsPolicy(alignment));
}

extern "C" void* TCMallocInternalCalloc(size_t n, size_t elem_size) noexcept {
  // Overflow check
  const size_t size = n * elem_size;
  if (elem_size != 0 && size / elem_size != n) {
    return MallocPolicy::handle_oom(std::numeric_limits<size_t>::max());
  }
  void* result = fast_alloc(MallocPolicy(), size);
  if (result != nullptr) {
    memset(result, 0, size);
  }
  return result;
}

// Here and below we use TCMALLOC_ALIAS (if supported) to make
// identical functions aliases.  This saves space in L1 instruction
// cache.  As of now it saves ~9K.
extern "C" void TCMallocInternalCfree(void* ptr) noexcept
#ifdef TCMALLOC_ALIAS
    TCMALLOC_ALIAS(TCMallocInternalFree);
#else
{
  do_free(ptr);
}
#endif  // TCMALLOC_ALIAS

static inline ABSL_ATTRIBUTE_ALWAYS_INLINE void* do_realloc(void* old_ptr,
                                                            size_t new_size) {
  Static::InitIfNecessary();
  // Get the size of the old entry
  const size_t old_size = GetSize(old_ptr);

  // Reallocate if the new size is larger than the old size,
  // or if the new size is significantly smaller than the old size.
  // We do hysteresis to avoid resizing ping-pongs:
  //    . If we need to grow, grow to max(new_size, old_size * 1.X)
  //    . Don't shrink unless new_size < old_size * 0.Y
  // X and Y trade-off time for wasted space.  For now we do 1.25 and 0.5.
  const size_t min_growth = std::min(
      old_size / 4,
      std::numeric_limits<size_t>::max() - old_size);  // Avoid overflow.
  const size_t lower_bound_to_grow = old_size + min_growth;
  const size_t upper_bound_to_shrink = old_size / 2;
  if ((new_size > old_size) || (new_size < upper_bound_to_shrink)) {
    // Need to reallocate.
    void* new_ptr = nullptr;

    if (new_size > old_size && new_size < lower_bound_to_grow) {
      // Avoid fast_alloc() reporting a hook with the lower bound size
      // as we the expectation for pointer returning allocation functions
      // is that malloc hooks are invoked with the requested_size.
      new_ptr = fast_alloc(MallocPolicy().Nothrow().WithoutHooks(),
                           lower_bound_to_grow);
      if (new_ptr != nullptr) {
      }
    }
    if (new_ptr == nullptr) {
      // Either new_size is not a tiny increment, or last do_malloc failed.
      new_ptr = fast_alloc(MallocPolicy(), new_size);
    }
    if (new_ptr == nullptr) {
      return nullptr;
    }
    memcpy(new_ptr, old_ptr, ((old_size < new_size) ? old_size : new_size));
    // We could use a variant of do_free() that leverages the fact
    // that we already know the sizeclass of old_ptr.  The benefit
    // would be small, so don't bother.
    do_free(old_ptr);
    return new_ptr;
  } else {
    return old_ptr;
  }
}

extern "C" void* TCMallocInternalRealloc(void* old_ptr,
                                         size_t new_size) noexcept {
  if (old_ptr == NULL) {
    return fast_alloc(MallocPolicy(), new_size);
  }
  if (new_size == 0) {
    do_free(old_ptr);
    return NULL;
  }
  return do_realloc(old_ptr, new_size);
}

extern "C" void* TCMallocInternalNewNothrow(size_t size,
                                            const std::nothrow_t&) noexcept {
  return fast_alloc(CppPolicy().Nothrow(), size);
}

extern "C" tcmalloc::sized_ptr_t tcmalloc_size_returning_operator_new_nothrow(
    size_t size) noexcept {
  size_t capacity;
  void* p = fast_alloc(CppPolicy().Nothrow(), size, &capacity);
  return {p, capacity};
}

extern "C" ABSL_CACHELINE_ALIGNED void TCMallocInternalDelete(void* p) noexcept
#ifdef TCMALLOC_ALIAS
    TCMALLOC_ALIAS(TCMallocInternalFree);
#else
{
  do_free(p);
}
#endif  // TCMALLOC_ALIAS

extern "C" void TCMallocInternalDeleteAligned(
    void* p, std::align_val_t alignment) noexcept
#if defined(TCMALLOC_ALIAS) && defined(NDEBUG)
    TCMALLOC_ALIAS(TCMallocInternalDelete);
#else
{
  // Note: The aligned delete/delete[] implementations differ slightly from
  // their respective aliased implementations to take advantage of checking the
  // passed-in alignment.
  ASSERT(CorrectAlignment(p, alignment));
  return TCMallocInternalDelete(p);
}
#endif

extern "C" ABSL_CACHELINE_ALIGNED void TCMallocInternalDeleteSized(
    void* p, size_t size) noexcept {
  ASSERT(CorrectSize(p, size, DefaultAlignPolicy()));
  do_free_with_size(p, size, DefaultAlignPolicy());
}

extern "C" void TCMallocInternalDeleteSizedAligned(
    void* p, size_t t, std::align_val_t alignment) noexcept {
  return do_free_with_size(p, t, AlignAsPolicy(alignment));
}

extern "C" void TCMallocInternalDeleteArraySized(void* p, size_t size) noexcept
#ifdef TCMALLOC_ALIAS
    TCMALLOC_ALIAS(TCMallocInternalDeleteSized);
#else
{
  do_free_with_size(p, size, DefaultAlignPolicy());
}
#endif

extern "C" void TCMallocInternalDeleteArraySizedAligned(
    void* p, size_t t, std::align_val_t alignment) noexcept
#ifdef TCMALLOC_ALIAS
    TCMALLOC_ALIAS(TCMallocInternalDeleteSizedAligned);
#else
{
  return TCMallocInternalDeleteSizedAligned(p, t, alignment);
}
#endif

// Standard C++ library implementations define and use this
// (via ::operator delete(ptr, nothrow)).
// But it's really the same as normal delete, so we just do the same thing.
extern "C" void TCMallocInternalDeleteNothrow(void* p,
                                              const std::nothrow_t&) noexcept
#ifdef TCMALLOC_ALIAS
    TCMALLOC_ALIAS(TCMallocInternalFree);
#else
{
  do_free(p);
}
#endif  // TCMALLOC_ALIAS

#if defined(TCMALLOC_ALIAS) && defined(NDEBUG)
extern "C" void TCMallocInternalDeleteAligned_nothrow(
    void* p, std::align_val_t alignment, const std::nothrow_t& nt) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDelete);
#else
extern "C" ABSL_ATTRIBUTE_SECTION(
    google_malloc) void TCMallocInternalDeleteAligned_nothrow(void* p,
                                                              std::align_val_t
                                                                  alignment,
                                                              const std::nothrow_t&
                                                                  nt) noexcept {
  ASSERT(CorrectAlignment(p, alignment));
  return TCMallocInternalDelete(p);
}
#endif

extern "C" void* TCMallocInternalNewArray(size_t size)
#ifdef TCMALLOC_ALIAS
    TCMALLOC_ALIAS(TCMallocInternalNew);
#else
{
  return fast_alloc(CppPolicy().WithoutHooks(), size);
}
#endif  // TCMALLOC_ALIAS

extern "C" void* TCMallocInternalNewArrayAligned(size_t size,
                                                 std::align_val_t alignment)
#if defined(TCMALLOC_ALIAS) && defined(NDEBUG)
    TCMALLOC_ALIAS(TCMallocInternalNewAligned);
#else
{
  return TCMallocInternalNewAligned(size, alignment);
}
#endif

extern "C" void* TCMallocInternalNewArrayNothrow(size_t size,
                                                 const std::nothrow_t&) noexcept
#ifdef TCMALLOC_ALIAS
    TCMALLOC_ALIAS(TCMallocInternalNewNothrow);
#else
{
  return fast_alloc(CppPolicy().Nothrow(), size);
}
#endif  // TCMALLOC_ALIAS

// Note: we use malloc rather than new, as we are allowed to return nullptr.
// The latter crashes in that case.
#if defined(TCMALLOC_ALIAS) && defined(NDEBUG)
extern "C" void* TCMallocInternalNewArrayAligned_nothrow(
    size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
    TCMALLOC_ALIAS(TCMallocInternalMalloc_aligned);
#else
extern "C" ABSL_ATTRIBUTE_SECTION(
    google_malloc) void* TCMallocInternalNewArrayAligned_nothrow(size_t size,
                                                                 std::align_val_t
                                                                     alignment,
                                                                 const std::
                                                                     nothrow_t&) noexcept {
  return TCMallocInternalMalloc_aligned(size, alignment);
}
#endif

extern "C" void TCMallocInternalDeleteArray(void* p) noexcept
#ifdef TCMALLOC_ALIAS
    TCMALLOC_ALIAS(TCMallocInternalFree);
#else
{
  do_free(p);
}
#endif  // TCMALLOC_ALIAS

extern "C" void TCMallocInternalDeleteArrayAligned(
    void* p, std::align_val_t alignment) noexcept
#if defined(TCMALLOC_ALIAS) && defined(NDEBUG)
    TCMALLOC_ALIAS(TCMallocInternalDelete);
#else
{
  ASSERT(CorrectAlignment(p, alignment));
  return TCMallocInternalDelete(p);
}
#endif

extern "C" void TCMallocInternalDeleteArrayNothrow(
    void* p, const std::nothrow_t&) noexcept
#ifdef TCMALLOC_ALIAS
    TCMALLOC_ALIAS(TCMallocInternalFree);
#else
{
  do_free(p);
}
#endif  // TCMALLOC_ALIAS

#if defined(TCMALLOC_ALIAS) && defined(NDEBUG)
extern "C" void TCMallocInternalDeleteArrayAligned_nothrow(
    void* p, std::align_val_t alignment, const std::nothrow_t&) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDelete);
#else
extern "C" ABSL_ATTRIBUTE_SECTION(
    google_malloc) void TCMallocInternalDeleteArrayAligned_nothrow(void* p,
                                                                   std::align_val_t
                                                                       alignment,
                                                                   const std::
                                                                       nothrow_t&) noexcept {
  ASSERT(CorrectAlignment(p, alignment));
  return TCMallocInternalDelete(p);
}
#endif

extern "C" void* TCMallocInternalMemalign(size_t align, size_t size) noexcept {
  ASSERT(absl::has_single_bit(align));
  return fast_alloc(MallocPolicy().AlignAs(align), size);
}

extern "C" void* TCMallocInternalAlignedAlloc(size_t align,
                                              size_t size) noexcept
#if defined(TCMALLOC_ALIAS) && defined(NDEBUG)
    TCMALLOC_ALIAS(TCMallocInternalMemalign);
#else
{
  // aligned_alloc is memalign, but with the requirement that:
  //   align be a power of two (like memalign)
  //   size be a multiple of align (for the time being).
  ASSERT(align != 0);
  ASSERT(size % align == 0);

  return TCMallocInternalMemalign(align, size);
}
#endif

extern "C" int TCMallocInternalPosixMemalign(void** result_ptr, size_t align,
                                             size_t size) noexcept {
  if (((align % sizeof(void*)) != 0) || !absl::has_single_bit(align)) {
    return EINVAL;
  }
  void* result = fast_alloc(MallocPolicy().Nothrow().AlignAs(align), size);
  if (result == NULL) {
    return ENOMEM;
  } else {
    *result_ptr = result;
    return 0;
  }
}

static size_t pagesize = 0;

extern "C" void* TCMallocInternalValloc(size_t size) noexcept {
  // Allocate page-aligned object of length >= size bytes
  if (pagesize == 0) pagesize = getpagesize();
  return fast_alloc(MallocPolicy().Nothrow().AlignAs(pagesize), size);
}

extern "C" void* TCMallocInternalPvalloc(size_t size) noexcept {
  // Round up size to a multiple of pagesize
  if (pagesize == 0) pagesize = getpagesize();
  if (size == 0) {    // pvalloc(0) should allocate one page, according to
    size = pagesize;  // http://man.free4web.biz/man3/libmpatrol.3.html
  }
  size = (size + pagesize - 1) & ~(pagesize - 1);
  return fast_alloc(MallocPolicy().Nothrow().AlignAs(pagesize), size);
}

extern "C" void TCMallocInternalMallocStats(void) noexcept {
  do_malloc_stats();
}

extern "C" int TCMallocInternalMallOpt(int cmd, int value) noexcept {
  return do_mallopt(cmd, value);
}

#ifdef TCMALLOC_HAVE_STRUCT_MALLINFO
extern "C" struct mallinfo TCMallocInternalMallocInfo(void) noexcept {
  return do_mallinfo();
}
#endif

extern "C" size_t TCMallocInternalMallocSize(void* ptr) noexcept {
  ASSERT(GetOwnership(ptr) != tcmalloc::MallocExtension::Ownership::kNotOwned);
  return GetSize(ptr);
}

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {
namespace {

// The constructor allocates an object to ensure that initialization
// runs before main(), and therefore we do not have a chance to become
// multi-threaded before initialization.  We also create the TSD key
// here.  Presumably by the time this constructor runs, glibc is in
// good enough shape to handle pthread_key_create().
//
// The destructor prints stats when the program exits.
class TCMallocGuard {
 public:
  TCMallocGuard() {
    TCMallocInternalFree(TCMallocInternalMalloc(1));
    ThreadCache::InitTSD();
    TCMallocInternalFree(TCMallocInternalMalloc(1));
  }
};

static TCMallocGuard module_enter_exit_hook;

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
