// Copyright 2021 The TCMalloc Authors
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

#include "tcmalloc/internal/cache_topology.h"

#include <fcntl.h>
#include <string.h>

#include <cerrno>
#include <cstdio>
#include <optional>

#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/cpu_utils.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/sysinfo.h"
#include "tcmalloc/internal/util.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

namespace {
int OpenSysfsCacheList(size_t cpu) {
  char path[PATH_MAX];
  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpu%zu/cache/index3/shared_cpu_list", cpu);
  return signal_safe_open(path, O_RDONLY | O_CLOEXEC);
}
}  // namespace

void CacheTopology::Init() {
  const auto maybe_numcpus = NumCPUsMaybe();
  if (!maybe_numcpus.has_value()) {
    l3_count_ = 1;
    return;
  }

  cpu_count_ = *maybe_numcpus;
  CpuSet cpus_to_check;
  cpus_to_check.Zero();
  for (int cpu = 0; cpu < cpu_count_; ++cpu) {
    cpus_to_check.Set(cpu);
  }

  while (true) {
    const int cpu = cpus_to_check.FindFirstSet();
    if (cpu == -1) {
      break;
    }
    const int fd = OpenSysfsCacheList(cpu);
    if (fd == -1) {
      // At some point we reach the number of CPU on the system, and
      // we should exit. We verify that there was no other problem.
      TC_CHECK_EQ(errno, ENOENT);
      // For aarch64 if
      // /sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list is missing
      // then L3 is assumed to be shared by all CPUs.
      // TODO(b/210049384): find a better replacement for shared_cpu_list in
      // this case, e.g. based on numa nodes.
#ifdef __aarch64__
      if (l3_count_ == 0) {
        l3_count_ = 1;
      }
#endif
      return;
    }
    // The file contains something like:
    //   0-11,22-33
    // Extract all CPUs from that.

    std::optional<CpuSet> maybe_shared_cpu_list =
        ParseCpulist([&](char* const buf, const size_t count) {
          return signal_safe_read(fd, buf, count, /*bytes_read=*/nullptr);
        });
    signal_safe_close(fd);

    TC_CHECK(maybe_shared_cpu_list.has_value());
    CpuSet& shared_cpu_list = *maybe_shared_cpu_list;
    shared_cpu_list.CLR(cpu);
    cpus_to_check.CLR(cpu);

    const int first_cpu = cpu;
    l3_cache_index_[first_cpu] = l3_count_++;
    // Set the remaining in the parsed cpu set to the l3_cache_index of
    // the first one.
    while (true) {
      int next_cpu = shared_cpu_list.FindFirstSet();
      if (next_cpu == -1) {
        break;
      }
      shared_cpu_list.CLR(next_cpu);
      cpus_to_check.CLR(next_cpu);
      l3_cache_index_[next_cpu] = l3_cache_index_[first_cpu];
    }
  }
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
