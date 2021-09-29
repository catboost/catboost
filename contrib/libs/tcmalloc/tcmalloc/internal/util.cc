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
#include "tcmalloc/internal/util.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tcmalloc/internal/logging.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

int signal_safe_open(const char* path, int flags, ...) {
  int fd;
  va_list ap;

  va_start(ap, flags);
  mode_t mode = va_arg(ap, mode_t);
  va_end(ap);

  do {
    fd = ((flags & O_CREAT) ? open(path, flags, mode) : open(path, flags));
  } while (fd == -1 && errno == EINTR);

  return fd;
}

int signal_safe_close(int fd) {
  int rc;

  do {
    rc = close(fd);
  } while (rc == -1 && errno == EINTR);

  return rc;
}

ssize_t signal_safe_write(int fd, const char* buf, size_t count,
                          size_t* bytes_written) {
  ssize_t rc;
  size_t total_bytes = 0;

  do {
    rc = write(fd, buf + total_bytes, count - total_bytes);
    if (rc > 0) total_bytes += rc;
  } while ((rc > 0 && count > total_bytes) || (rc == -1 && errno == EINTR));

  if (bytes_written != nullptr) *bytes_written = total_bytes;

  return rc;
}

int signal_safe_poll(struct pollfd* fds, int nfds, absl::Duration timeout) {
  int rc = 0;
  absl::Duration elapsed = absl::ZeroDuration();

  // We can't use gettimeofday since it's not async signal safe.  We could use
  // clock_gettime but that would require linking //base against librt.
  // Fortunately, timeout is of sufficiently coarse granularity that we can just
  // approximate it.
  while ((elapsed <= timeout || timeout < absl::ZeroDuration()) && (rc == 0)) {
    if (elapsed > absl::ZeroDuration())
      ::absl::SleepFor(::absl::Milliseconds(1));
    elapsed += absl::Milliseconds(1);
    while ((rc = poll(fds, nfds, 0)) == -1 && errno == EINTR) {
    }
  }

  return rc;
}

ssize_t signal_safe_read(int fd, char* buf, size_t count, size_t* bytes_read) {
  ssize_t rc;
  size_t total_bytes = 0;
  struct pollfd pfd;

  // poll is required for testing whether there is any data left on fd in the
  // case of a signal interrupting a partial read.  This is needed since this
  // case is only defined to return the number of bytes read up to that point,
  // with no indication whether more could have been read (up to count).
  pfd.fd = fd;
  pfd.events = POLL_IN;
  pfd.revents = 0;

  do {
    rc = read(fd, buf + total_bytes, count - total_bytes);
    if (rc > 0) total_bytes += rc;

    if (rc == 0) break;  // EOF
    // try again if there's space to fill, no (non-interrupt) error,
    // and data is available.
  } while (total_bytes < count && (rc > 0 || errno == EINTR) &&
           (signal_safe_poll(&pfd, 1, absl::ZeroDuration()) == 1 ||
            total_bytes == 0));

  if (bytes_read) *bytes_read = total_bytes;

  if (rc != -1 || errno == EINTR)
    rc = total_bytes;  // return the cumulative bytes read
  return rc;
}

std::vector<int> AllowedCpus() {
  // We have no need for dynamically sized sets (currently >1024 CPUs for glibc)
  // at the present time.  We could change this in the future.
  cpu_set_t allowed_cpus;
  CHECK_CONDITION(sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus) ==
                  0);
  int n = CPU_COUNT(&allowed_cpus), c = 0;

  std::vector<int> result(n);
  for (int i = 0; i < CPU_SETSIZE && n; i++) {
    if (CPU_ISSET(i, &allowed_cpus)) {
      result[c++] = i;
      n--;
    }
  }
  CHECK_CONDITION(0 == n);

  return result;
}

static cpu_set_t SpanToCpuSetT(absl::Span<int> mask) {
  cpu_set_t result;
  CPU_ZERO(&result);
  for (int cpu : mask) {
    CPU_SET(cpu, &result);
  }
  return result;
}

ScopedAffinityMask::ScopedAffinityMask(absl::Span<int> allowed_cpus) {
  specified_cpus_ = SpanToCpuSetT(allowed_cpus);
  // getaffinity should never fail.
  CHECK_CONDITION(
      sched_getaffinity(0, sizeof(original_cpus_), &original_cpus_) == 0);
  // See destructor comments on setaffinity interactions.  Tampered() will
  // necessarily be true in this case.
  sched_setaffinity(0, sizeof(specified_cpus_), &specified_cpus_);
}

ScopedAffinityMask::ScopedAffinityMask(int allowed_cpu) {
  CPU_ZERO(&specified_cpus_);
  CPU_SET(allowed_cpu, &specified_cpus_);

  // getaffinity should never fail.
  CHECK_CONDITION(
      sched_getaffinity(0, sizeof(original_cpus_), &original_cpus_) == 0);
  // See destructor comments on setaffinity interactions.  Tampered() will
  // necessarily be true in this case.
  sched_setaffinity(0, sizeof(specified_cpus_), &specified_cpus_);
}

ScopedAffinityMask::~ScopedAffinityMask() {
  // If something else has already reset our affinity, do not attempt to
  // restrict towards our original mask.  This is best-effort as the tampering
  // may obviously occur during the destruction of *this.
  if (!Tampered()) {
    // Note:  We do not assert success here, conflicts may restrict us from all
    // 'original_cpus_'.
    sched_setaffinity(0, sizeof(original_cpus_), &original_cpus_);
  }
}

bool ScopedAffinityMask::Tampered() {
  cpu_set_t current_cpus;
  CHECK_CONDITION(sched_getaffinity(0, sizeof(current_cpus), &current_cpus) ==
                  0);
  return !CPU_EQUAL(&current_cpus, &specified_cpus_);  // Mismatch => modified.
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
