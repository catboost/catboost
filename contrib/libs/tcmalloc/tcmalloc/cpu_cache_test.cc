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

#include <thread>  // NOLINT(build/c++11)

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "absl/random/seed_sequences.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/internal/util.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/testing/testutil.h"

namespace tcmalloc {
namespace tcmalloc_internal {
namespace {

constexpr size_t kStressSlabs = 4;
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
  const size_t virtual_cpu_id_offset = subtle::percpu::UsingFlatVirtualCpus()
                                           ? offsetof(kernel_rseq, vcpu_id)
                                           : offsetof(kernel_rseq, cpu_id);
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
    allowed_cpu_id =
        subtle::percpu::GetCurrentVirtualCpuUnsafe(virtual_cpu_id_offset);

    ptr = cache.Allocate<OOMHandler>(kSizeClass);

    if (mask.Tampered() ||
        allowed_cpu_id !=
            subtle::percpu::GetCurrentVirtualCpuUnsafe(virtual_cpu_id_offset)) {
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
      EXPECT_EQ(cache.Capacity(cpu), max_cpu_cache_size);
      EXPECT_EQ(cache.Allocated(cpu) + cache.Unallocated(cpu),
                cache.Capacity(cpu));
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

TEST(CpuCacheTest, CacheMissStats) {
  if (!subtle::percpu::IsFast()) {
    return;
  }

  const int num_cpus = absl::base_internal::NumCPUs();

  CPUCache& cache = Static::cpu_cache();
  // Since this test allocates memory, avoid activating the real fast path to
  // minimize allocations against the per-CPU cache.
  cache.Activate(CPUCache::ActivationMode::FastPathOffTestOnly);

  //  The number of underflows and overflows must be zero for all the caches.
  for (int cpu = 0; cpu < num_cpus; ++cpu) {
    CPUCache::CpuCacheMissStats total_misses =
        cache.GetTotalCacheMissStats(cpu);
    CPUCache::CpuCacheMissStats interval_misses =
        cache.GetIntervalCacheMissStats(cpu);
    EXPECT_EQ(total_misses.underflows, 0);
    EXPECT_EQ(total_misses.overflows, 0);
    EXPECT_EQ(interval_misses.underflows, 0);
    EXPECT_EQ(interval_misses.overflows, 0);
  }

  int allowed_cpu_id;
  const size_t kSizeClass = 3;
  const size_t virtual_cpu_id_offset = subtle::percpu::UsingFlatVirtualCpus()
                                           ? offsetof(kernel_rseq, vcpu_id)
                                           : offsetof(kernel_rseq, cpu_id);
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
    allowed_cpu_id =
        subtle::percpu::GetCurrentVirtualCpuUnsafe(virtual_cpu_id_offset);

    ptr = cache.Allocate<OOMHandler>(kSizeClass);

    if (mask.Tampered() ||
        allowed_cpu_id !=
            subtle::percpu::GetCurrentVirtualCpuUnsafe(virtual_cpu_id_offset)) {
      return;
    }
  }

  for (int cpu = 0; cpu < num_cpus; ++cpu) {
    CPUCache::CpuCacheMissStats total_misses =
        cache.GetTotalCacheMissStats(cpu);
    CPUCache::CpuCacheMissStats interval_misses =
        cache.GetIntervalCacheMissStats(cpu);
    if (cpu == allowed_cpu_id) {
      EXPECT_EQ(total_misses.underflows, 1);
      EXPECT_EQ(interval_misses.underflows, 1);
    } else {
      EXPECT_EQ(total_misses.underflows, 0);
      EXPECT_EQ(interval_misses.underflows, 0);
    }
    EXPECT_EQ(total_misses.overflows, 0);
    EXPECT_EQ(interval_misses.overflows, 0);
  }

  // Tear down.
  //
  // TODO(ckennelly):  We're interacting with the real TransferCache.
  cache.Deallocate(ptr, kSizeClass);

  for (int i = 0; i < num_cpus; i++) {
    cache.Reclaim(i);
  }
}

static void ShuffleThread(const std::atomic<bool>& stop) {
  if (!subtle::percpu::IsFast()) {
    return;
  }

  CPUCache& cache = Static::cpu_cache();
  // Wake up every 10ms to shuffle the caches so that we can allow misses to
  // accumulate during that interval
  while (!stop) {
    cache.ShuffleCpuCaches();
    absl::SleepFor(absl::Milliseconds(10));
  }
}

static void StressThread(size_t thread_id, const std::atomic<bool>& stop) {
  if (!subtle::percpu::IsFast()) {
    return;
  }

  CPUCache& cache = Static::cpu_cache();
  std::vector<std::pair<size_t, void*>> blocks;
  absl::BitGen rnd;
  while (!stop) {
    const int what = absl::Uniform<int32_t>(rnd, 0, 2);
    if (what) {
      // Allocate an object for a class
      size_t cl = absl::Uniform<int32_t>(rnd, 1, kStressSlabs + 1);
      void* ptr = cache.Allocate<OOMHandler>(cl);
      blocks.emplace_back(std::make_pair(cl, ptr));
    } else {
      // Deallocate an object for a class
      if (!blocks.empty()) {
        cache.Deallocate(blocks.back().second, blocks.back().first);
        blocks.pop_back();
      }
    }
  }

  // Cleaup. Deallocate rest of the allocated memory.
  for (int i = 0; i < blocks.size(); i++) {
    cache.Deallocate(blocks[i].second, blocks[i].first);
  }
}

TEST(CpuCacheTest, StealCpuCache) {
  if (!subtle::percpu::IsFast()) {
    return;
  }

  CPUCache& cache = Static::cpu_cache();
  // Since this test allocates memory, avoid activating the real fast path to
  // minimize allocations against the per-CPU cache.
  cache.Activate(CPUCache::ActivationMode::FastPathOffTestOnly);

  std::vector<std::thread> threads;
  std::thread shuffle_thread;
  const int n_threads = absl::base_internal::NumCPUs();
  std::atomic<bool> stop(false);

  for (size_t t = 0; t < n_threads; ++t) {
    threads.push_back(std::thread(StressThread, t, std::ref(stop)));
  }
  shuffle_thread = std::thread(ShuffleThread, std::ref(stop));

  absl::SleepFor(absl::Seconds(5));
  stop = true;
  for (auto& t : threads) {
    t.join();
  }
  shuffle_thread.join();

  // Check that the total capacity is preserved after the shuffle.
  size_t capacity = 0;
  const int num_cpus = absl::base_internal::NumCPUs();
  const size_t kTotalCapacity = num_cpus * Parameters::max_per_cpu_cache_size();
  for (int cpu = 0; cpu < num_cpus; ++cpu) {
    EXPECT_EQ(cache.Allocated(cpu) + cache.Unallocated(cpu),
              cache.Capacity(cpu));
    capacity += cache.Capacity(cpu);
  }
  EXPECT_EQ(capacity, kTotalCapacity);

  for (int cpu = 0; cpu < num_cpus; ++cpu) {
    cache.Reclaim(cpu);
  }
}

// Runs a single allocate and deallocate operation to warm up the cache. Once a
// few objects are allocated in the cold cache, we can shuffle cpu caches to
// steal that capacity from the cold cache to the hot cache.
static void ColdCacheOperations(int cpu_id) {
  // Temporarily fake being on the given CPU.
  ScopedFakeCpuId fake_cpu_id(cpu_id);

  CPUCache& cache = Static::cpu_cache();
  if (subtle::percpu::UsingFlatVirtualCpus()) {
    subtle::percpu::__rseq_abi.vcpu_id = cpu_id;
  }

  // We allocate and deallocate a single highest cl object.
  // This makes sure that we have a single large object in the cache that faster
  // cache can steal. Allocating a large object ensures that we steal the
  // maximum steal-able capacity for this cache in a short amount of time.
  const size_t cl = kNumClasses - 1;
  void* ptr = cache.Allocate<OOMHandler>(cl);
  cache.Deallocate(ptr, cl);
}

// Runs multiple allocate and deallocate operation on the cpu cache to collect
// misses. Once we collect enough misses on this cache, we can shuffle cpu
// caches to steal capacity from colder caches to the hot cache.
static void HotCacheOperations(int cpu_id) {
  // Temporarily fake being on the given CPU.
  ScopedFakeCpuId fake_cpu_id(cpu_id);

  CPUCache& cache = Static::cpu_cache();
  if (subtle::percpu::UsingFlatVirtualCpus()) {
    subtle::percpu::__rseq_abi.vcpu_id = cpu_id;
  }

  // Allocate and deallocate objects to make sure we have enough misses on the
  // cache. This will make sure we have sufficient disparity in misses between
  // the hotter and colder cache, and that we may be able to steal bytes from
  // the colder cache.
  for (size_t cl = 1; cl <= kStressSlabs; ++cl) {
    void* ptr = cache.Allocate<OOMHandler>(cl);
    cache.Deallocate(ptr, cl);
  }

  // We reclaim the cache to reset it so that we record underflows/overflows the
  // next time we allocate and deallocate objects. Without reclaim, the cache
  // would stay warmed up and it would take more time to drain the colder cache.
  cache.Reclaim(cpu_id);
}

TEST(CpuCacheTest, ColdHotCacheShuffleTest) {
  if (!subtle::percpu::IsFast()) {
    return;
  }

  CPUCache& cache = Static::cpu_cache();
  // Since this test allocates memory, avoid activating the real fast path to
  // minimize allocations against the per-CPU cache.
  cache.Activate(CPUCache::ActivationMode::FastPathOffTestOnly);

  constexpr int hot_cpu_id = 0;
  constexpr int cold_cpu_id = 1;

  const size_t max_cpu_cache_size = Parameters::max_per_cpu_cache_size();

  // Empirical tests suggest that we should be able to steal all the steal-able
  // capacity from colder cache in < 100 tries. Keeping enough buffer here to
  // make sure we steal from colder cache, while at the same time avoid timeouts
  // if something goes bad.
  constexpr int kMaxStealTries = 1000;

  for (int num_tries = 0;
       num_tries < kMaxStealTries &&
       cache.Capacity(cold_cpu_id) >
           CPUCache::kCacheCapacityThreshold * max_cpu_cache_size;
       ++num_tries) {
    ColdCacheOperations(cold_cpu_id);
    HotCacheOperations(hot_cpu_id);
    cache.ShuffleCpuCaches();

    // Check that the capacity is preserved.
    EXPECT_EQ(cache.Allocated(cold_cpu_id) + cache.Unallocated(cold_cpu_id),
              cache.Capacity(cold_cpu_id));
    EXPECT_EQ(cache.Allocated(hot_cpu_id) + cache.Unallocated(hot_cpu_id),
              cache.Capacity(hot_cpu_id));
  }

  size_t cold_cache_capacity = cache.Capacity(cold_cpu_id);
  size_t hot_cache_capacity = cache.Capacity(hot_cpu_id);

  // Check that we drained cold cache to the lower capacity limit.
  // We also keep some tolerance, up to the largest class size, below the lower
  // capacity threshold that we can drain cold cache to.
  EXPECT_GT(cold_cache_capacity,
            CPUCache::kCacheCapacityThreshold * max_cpu_cache_size -
                Static::sizemap().class_to_size(kNumClasses - 1));

  // Check that we have at least stolen some capacity.
  EXPECT_GT(hot_cache_capacity, max_cpu_cache_size);

  // Perform a few more shuffles to make sure that lower cache capacity limit
  // has been reached for the cold cache. A few more shuffles should not
  // change the capacity of either of the caches.
  for (int i = 0; i < 100; ++i) {
    ColdCacheOperations(cold_cpu_id);
    HotCacheOperations(hot_cpu_id);
    cache.ShuffleCpuCaches();

    // Check that the capacity is preserved.
    EXPECT_EQ(cache.Allocated(cold_cpu_id) + cache.Unallocated(cold_cpu_id),
              cache.Capacity(cold_cpu_id));
    EXPECT_EQ(cache.Allocated(hot_cpu_id) + cache.Unallocated(hot_cpu_id),
              cache.Capacity(hot_cpu_id));
  }

  // Check that the capacity of cold and hot caches is same as before.
  EXPECT_EQ(cache.Capacity(cold_cpu_id), cold_cache_capacity);
  EXPECT_EQ(cache.Capacity(hot_cpu_id), hot_cache_capacity);

  // Make sure that the total capacity is preserved.
  EXPECT_EQ(cache.Capacity(cold_cpu_id) + cache.Capacity(hot_cpu_id),
            2 * max_cpu_cache_size);

  // Reclaim caches.
  const int num_cpus = absl::base_internal::NumCPUs();
  for (int cpu = 0; cpu < num_cpus; ++cpu) {
    cache.Reclaim(cpu);
  }
}

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
