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
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/percpu.h"
#include "tcmalloc/internal_malloc_extension.h"
#include "tcmalloc/malloc_extension.h"
#include "tcmalloc/parameters.h"

namespace tcmalloc {
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

}  // namespace
}  // namespace tcmalloc

// Release memory to the system at a constant rate.
void MallocExtension_Internal_ProcessBackgroundActions() {
  tcmalloc::MallocExtension::MarkThreadIdle();

  // Initialize storage for ReleasePerCpuMemoryToOS().
  CPU_ZERO(&tcmalloc::prev_allowed_cpus);

  absl::Time prev_time = absl::Now();
  constexpr absl::Duration kSleepTime = absl::Seconds(1);
  while (true) {
    absl::Time now = absl::Now();
    const ssize_t bytes_to_release =
        static_cast<size_t>(tcmalloc::Parameters::background_release_rate()) *
        absl::ToDoubleSeconds(now - prev_time);
    if (bytes_to_release > 0) {  // may be negative if time goes backwards
      tcmalloc::MallocExtension::ReleaseMemoryToSystem(bytes_to_release);
    }

    tcmalloc::ReleasePerCpuMemoryToOS();

    prev_time = now;
    absl::SleepFor(kSleepTime);
  }
}
