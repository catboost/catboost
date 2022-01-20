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

#include <errno.h>

#include "absl/base/internal/sysinfo.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tcmalloc/cpu_cache.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/percpu.h"
#include "tcmalloc/internal_malloc_extension.h"
#include "tcmalloc/malloc_extension.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/static_vars.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {
namespace {

// Called by MallocExtension_Internal_ProcessBackgroundActions.
//
// We use a simple heuristic here:
// We keep track of the set of CPUs that we are allowed to run on.  Whenever a
// CPU is removed from this list, the next call to this routine will detect the
// disappearance and call ReleaseCpuMemory on it.
//
// Note that this heuristic _explicitly_ does not reclaim from isolated cores
// that this process may have set up specific affinities for -- as this thread
// will never have been allowed to run there.
cpu_set_t prev_allowed_cpus;
void ReleasePerCpuMemoryToOS() {
  cpu_set_t allowed_cpus;

  // Only attempt reclaim when per-CPU caches are in use.  While
  // ReleaseCpuMemory() itself is usually a no-op otherwise, we are experiencing
  // failures in non-permissive sandboxes due to calls made to
  // sched_getaffinity() below.  It is expected that a runtime environment
  // supporting per-CPU allocations supports sched_getaffinity().
  // See b/27247854.
  if (!MallocExtension::PerCpuCachesActive()) {
    return;
  }

  if (subtle::percpu::UsingFlatVirtualCpus()) {
    // Our (real) CPU mask does not provide useful information about the state
    // of our virtual CPU set.
    return;
  }

  // This can only fail due to a sandbox or similar intercepting the syscall.
  if (sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus)) {
    // We log periodically as start-up errors are frequently ignored and this is
    // something we do want clients to fix if they are experiencing it.
    Log(kLog, __FILE__, __LINE__,
        "Unexpected sched_getaffinity() failure; errno ", errno);
    return;
  }

  // Note:  This is technically not correct in the presence of hotplug (it is
  // not guaranteed that NumCPUs() is an upper bound on CPU-number).  It is
  // currently safe for Google systems.
  const int num_cpus = absl::base_internal::NumCPUs();
  for (int cpu = 0; cpu < num_cpus; cpu++) {
    if (CPU_ISSET(cpu, &prev_allowed_cpus) && !CPU_ISSET(cpu, &allowed_cpus)) {
      // This is a CPU present in the old mask, but not the new.  Reclaim.
      MallocExtension::ReleaseCpuMemory(cpu);
    }
  }

  // Update cached runnable CPUs for next iteration.
  memcpy(&prev_allowed_cpus, &allowed_cpus, sizeof(cpu_set_t));
}

void ShuffleCpuCaches() {
  if (!MallocExtension::PerCpuCachesActive()) {
    return;
  }

  // Shuffle per-cpu caches
  Static::cpu_cache().ShuffleCpuCaches();
}

// Reclaims per-cpu caches. The CPU mask used in ReleasePerCpuMemoryToOS does
// not provide useful information about virtual CPU state and hence, does not
// reclaim memory when virtual CPUs are enabled.
//
// Here, we use heuristics that are based on cache usage and misses, to
// determine if the caches have been recently inactive and if they may be
// reclaimed.
void ReclaimIdleCpuCaches() {
  // Attempts reclaim only when per-CPU caches are in use.
  if (!MallocExtension::PerCpuCachesActive()) {
    return;
  }

  Static::cpu_cache().TryReclaimingCaches();
}

}  // namespace
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

// Release memory to the system at a constant rate.
void MallocExtension_Internal_ProcessBackgroundActions() {
  tcmalloc::MallocExtension::MarkThreadIdle();

  // Initialize storage for ReleasePerCpuMemoryToOS().
  CPU_ZERO(&tcmalloc::tcmalloc_internal::prev_allowed_cpus);

  absl::Time prev_time = absl::Now();
  constexpr absl::Duration kSleepTime = absl::Seconds(1);

  // Reclaim inactive per-cpu caches once per kCpuCacheReclaimPeriod.
  //
  // We use a longer 30 sec reclaim period to make sure that caches are indeed
  // idle. Reclaim drains entire cache, as opposed to cache shuffle for instance
  // that only shrinks a cache by a few objects at a time. So, we might have
  // larger performance degradation if we use a shorter reclaim interval and
  // drain caches that weren't supposed to.
  constexpr absl::Duration kCpuCacheReclaimPeriod = absl::Seconds(30);
  absl::Time last_reclaim = absl::Now();

  // Shuffle per-cpu caches once per kCpuCacheShufflePeriod secs.
  constexpr absl::Duration kCpuCacheShufflePeriod = absl::Seconds(5);
  absl::Time last_shuffle = absl::Now();

  while (true) {
    absl::Time now = absl::Now();
    const ssize_t bytes_to_release =
        static_cast<size_t>(tcmalloc::tcmalloc_internal::Parameters::
                                background_release_rate()) *
        absl::ToDoubleSeconds(now - prev_time);
    if (bytes_to_release > 0) {  // may be negative if time goes backwards
      tcmalloc::MallocExtension::ReleaseMemoryToSystem(bytes_to_release);
    }

    const bool reclaim_idle_per_cpu_caches =
        tcmalloc::tcmalloc_internal::Parameters::reclaim_idle_per_cpu_caches();

    // If enabled, we use heuristics to determine if the per-cpu caches are
    // inactive. If disabled, we use a more conservative approach, that uses
    // allowed cpu masks, to reclaim cpu caches.
    if (reclaim_idle_per_cpu_caches) {
      // Try to reclaim per-cpu caches once every kCpuCacheReclaimPeriod
      // when enabled.
      if (now - last_reclaim >= kCpuCacheReclaimPeriod) {
        tcmalloc::tcmalloc_internal::ReclaimIdleCpuCaches();
        last_reclaim = now;
      }
    } else {
      tcmalloc::tcmalloc_internal::ReleasePerCpuMemoryToOS();
    }

    const bool shuffle_per_cpu_caches =
        tcmalloc::tcmalloc_internal::Parameters::shuffle_per_cpu_caches();

    if (shuffle_per_cpu_caches) {
      if (now - last_shuffle >= kCpuCacheShufflePeriod) {
        tcmalloc::tcmalloc_internal::ShuffleCpuCaches();
        last_shuffle = now;
      }
    }

    tcmalloc::tcmalloc_internal::Static().sharded_transfer_cache().Plunder();
    prev_time = now;
    absl::SleepFor(kSleepTime);
  }
}
