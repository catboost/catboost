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

#include "tcmalloc/cpu_cache.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/internal/util.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/static_vars.h"

namespace tcmalloc {
namespace {

void* OOMHandler(size_t) { return nullptr; }

TEST(CpuCacheTest, Metadata) {
  if (!subtle::percpu::IsFast()) {
    return;
  }

  const int num_cpus = absl::base_internal::NumCPUs();

  CPUCache& cache = Static::cpu_cache();
  // Since this test allocates memory, avoid activating the real fast path to
  // minimize allocations against the per-CPU cache.
  cache.Activate(CPUCache::ActivationMode::FastPathOffTestOnly);

  PerCPUMetadataState r = cache.MetadataMemoryUsage();
  EXPECT_EQ(r.virtual_size, num_cpus << CPUCache::kPerCpuShift);
  if (Parameters::lazy_per_cpu_caches()) {
    EXPECT_EQ(r.resident_size, 0);
  } else {
    EXPECT_EQ(r.resident_size, r.virtual_size);
  }

  auto count_cores = [&]() {
    int populated_cores = 0;
    for (int i = 0; i < num_cpus; i++) {
      if (cache.HasPopulated(i)) {
        populated_cores++;
      }
    }
    return populated_cores;
  };

  EXPECT_EQ(0, count_cores());

  int allowed_cpu_id;
  const size_t kSizeClass = 3;
  const size_t num_to_move = Static::sizemap().num_objects_to_move(kSizeClass);
  void* ptr;
  {
    // Restrict this thread to a single core while allocating and processing the
    // slow path.
    //
    // TODO(b/151313823):  Without this restriction, we may access--for reading
    // only--other slabs if we end up being migrated.  These may cause huge
    // pages to be faulted for those cores, leading to test flakiness.
    tcmalloc_internal::ScopedAffinityMask mask(
        tcmalloc_internal::AllowedCpus()[0]);
    allowed_cpu_id = subtle::percpu::GetCurrentVirtualCpuUnsafe();

    ptr = cache.Allocate<OOMHandler>(kSizeClass);

    if (mask.Tampered() ||
        allowed_cpu_id != subtle::percpu::GetCurrentVirtualCpuUnsafe()) {
      return;
    }
  }
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(1, count_cores());

  r = cache.MetadataMemoryUsage();
  EXPECT_EQ(r.virtual_size, num_cpus << CPUCache::kPerCpuShift);
  if (Parameters::lazy_per_cpu_caches()) {
    // We expect to fault in a single core, but we may end up faulting an
    // entire hugepage worth of memory
    const size_t core_slab_size = r.virtual_size / num_cpus;
    const size_t upper_bound =
        ((core_slab_size + kHugePageSize - 1) & ~(kHugePageSize - 1));

    // A single core may be less than the full slab (core_slab_size), since we
    // do not touch every page within the slab.
    EXPECT_GT(r.resident_size, 0);
    EXPECT_LE(r.resident_size, upper_bound) << count_cores();

    // This test is much more sensitive to implementation details of the per-CPU
    // cache.  It may need to be updated from time to time.  These numbers were
    // calculated by MADV_NOHUGEPAGE'ing the memory used for the slab and
    // measuring the resident size.
    //
    // TODO(ckennelly):  Allow CPUCache::Activate to accept a specific arena
    // allocator, so we can MADV_NOHUGEPAGE the backing store in testing for
    // more precise measurements.
    switch (CPUCache::kPerCpuShift) {
      case 12:
        EXPECT_GE(r.resident_size, 4096);
        break;
      case 18:
        EXPECT_GE(r.resident_size, 110592);
        break;
      default:
        ASSUME(false);
        break;
    };

    // Read stats from the CPU caches.  This should not impact resident_size.
    const size_t max_cpu_cache_size = Parameters::max_per_cpu_cache_size();
    size_t total_used_bytes = 0;
    for (int cpu = 0; cpu < num_cpus; ++cpu) {
      size_t used_bytes = cache.UsedBytes(cpu);
      total_used_bytes += used_bytes;

      if (cpu == allowed_cpu_id) {
        EXPECT_GT(used_bytes, 0);
        EXPECT_TRUE(cache.HasPopulated(cpu));
      } else {
        EXPECT_EQ(used_bytes, 0);
        EXPECT_FALSE(cache.HasPopulated(cpu));
      }

      EXPECT_LE(cache.Unallocated(cpu), max_cpu_cache_size);
    }

    for (int cl = 0; cl < kNumClasses; ++cl) {
      // This is sensitive to the current growth policies of CPUCache.  It may
      // require updating from time-to-time.
      EXPECT_EQ(cache.TotalObjectsOfClass(cl),
                (cl == kSizeClass ? num_to_move - 1 : 0))
          << cl;
    }
    EXPECT_EQ(cache.TotalUsedBytes(), total_used_bytes);

    PerCPUMetadataState post_stats = cache.MetadataMemoryUsage();
    // Confirm stats are within expected bounds.
    EXPECT_GT(post_stats.resident_size, 0);
    EXPECT_LE(post_stats.resident_size, upper_bound) << count_cores();
    // Confirm stats are unchanged.
    EXPECT_EQ(r.resident_size, post_stats.resident_size);
  } else {
    EXPECT_EQ(r.resident_size, r.virtual_size);
  }

  // Tear down.
  //
  // TODO(ckennelly):  We're interacting with the real TransferCache.
  cache.Deallocate(ptr, kSizeClass);

  for (int i = 0; i < num_cpus; i++) {
    cache.Reclaim(i);
  }
}

}  // namespace
}  // namespace tcmalloc
