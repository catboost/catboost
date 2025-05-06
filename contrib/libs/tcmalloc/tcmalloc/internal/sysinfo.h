#pragma clang system_header
// Copyright 2023 The TCMalloc Authors
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

#ifndef TCMALLOC_INTERNAL_SYSINFO_H_
#define TCMALLOC_INTERNAL_SYSINFO_H_

#include <sched.h>
#include <sys/types.h>

#include <cstddef>
#include <optional>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/functional/function_ref.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/cpu_utils.h"
#include "tcmalloc/internal/logging.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

#if __linux__
// Parse a CPU list in the format used by
// /sys/devices/system/node/nodeX/cpulist files - that is, individual CPU
// numbers or ranges in the format <start>-<end> inclusive all joined by comma
// characters.
//
// Returns std::nullopt on error.
//
// The read function is expected to operate much like the read syscall. It
// should read up to `count` bytes into `buf` and return the number of bytes
// actually read. If an error occurs during reading it should return -1 with
// errno set to an appropriate error code.  read should handle EINTR and retry.
std::optional<CpuSet> ParseCpulist(
    absl::FunctionRef<ssize_t(char* buf, size_t count)> read);

namespace sysinfo_internal {

// Returns the number of possible CPUs on the machine, including currently
// offline CPUs.  If this cannot be retrieved, std::nullopt is returned.
//
// The result of this function is not cached internally.
std::optional<int> NumPossibleCPUsNoCache();

}  // namespace sysinfo_internal
#endif  // __linux__

inline std::optional<int> NumCPUsMaybe() {
  ABSL_CONST_INIT static absl::once_flag flag;
  ABSL_CONST_INIT static std::optional<int> result;
  absl::base_internal::LowLevelCallOnce(
      &flag, [&]() { result = sysinfo_internal::NumPossibleCPUsNoCache(); });
  return result;
}

inline int NumCPUs() {
  std::optional<int> maybe_cpus = NumCPUsMaybe();
  TC_CHECK(maybe_cpus.has_value());
  return *maybe_cpus;
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_INTERNAL_SYSINFO_H_
