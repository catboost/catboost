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

#ifndef TCMALLOC_INTERNAL_UTIL_H_
#define TCMALLOC_INTERNAL_UTIL_H_

#include <poll.h>  // IWYU pragma: keep
#include <sched.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#include <vector>

#include "absl/base/internal/sysinfo.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tcmalloc/internal/config.h"

#define TCMALLOC_RETRY_ON_TEMP_FAILURE(expression)               \
  (__extension__({                                               \
    long int _temp_failure_retry_result;                         \
    do _temp_failure_retry_result = (long int)(expression);      \
    while (_temp_failure_retry_result == -1L && errno == EINTR); \
    _temp_failure_retry_result;                                  \
  }))

// Useful internal utility functions.  These calls are async-signal safe
// provided the signal handler saves errno at entry and restores it before
// return.
GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// signal_safe_open() - a wrapper for open(2) which ignores signals
// Semantics equivalent to open(2):
//   returns a file-descriptor (>=0) on success, -1 on failure, error in errno
int signal_safe_open(const char *path, int flags, ...);

// signal_safe_close() - a wrapper for close(2) which ignores signals
// Semantics equivalent to close(2):
//   returns 0 on success, -1 on failure, error in errno
int signal_safe_close(int fd);

// signal_safe_write() - a wrapper for write(2) which ignores signals
// Semantics equivalent to write(2):
//   returns number of bytes written, -1 on failure, error in errno
//   additionally, (if not NULL) total bytes written in *bytes_written
//
// In the interrupted (EINTR) case, signal_safe_write will continue attempting
// to write out buf.  This means that in the:
//   write->interrupted by signal->write->error case
// That it is possible for signal_safe_write to return -1 when there were bytes
// flushed from the buffer in the first write.  To handle this case the optional
// bytes_written parameter is provided, when not-NULL, it will always return the
// total bytes written before any error.
ssize_t signal_safe_write(int fd, const char *buf, size_t count,
                          size_t *bytes_written);

// signal_safe_read() - a wrapper for read(2) which ignores signals
// Semantics equivalent to read(2):
//   returns number of bytes written, -1 on failure, error in errno
//   additionally, (if not NULL) total bytes written in *bytes_written
//
// In the interrupted (EINTR) case, signal_safe_read will continue attempting
// to read into buf.  This means that in the:
//   read->interrupted by signal->read->error case
// That it is possible for signal_safe_read to return -1 when there were bytes
// read by a previous read.  To handle this case the optional bytes_written
// parameter is provided, when not-NULL, it will always return the total bytes
// read before any error.
ssize_t signal_safe_read(int fd, char *buf, size_t count, size_t *bytes_read);

// signal_safe_poll() - a wrapper for poll(2) which ignores signals
// Semantics equivalent to poll(2):
//   Returns number of structures with non-zero revent fields.
//
// In the interrupted (EINTR) case, signal_safe_poll will continue attempting to
// poll for data.  Unlike ppoll/pselect, signal_safe_poll is *ignoring* signals
// not attempting to re-enable them.  Protecting us from the traditional races
// involved with the latter.
int signal_safe_poll(struct ::pollfd *fds, int nfds, absl::Duration timeout);

// Affinity helpers.

// Returns a vector of the which cpus the currently allowed thread is allowed to
// run on.  There are no guarantees that this will not change before, after, or
// even during, the call to AllowedCpus().
std::vector<int> AllowedCpus();

// Enacts a scoped affinity mask on the constructing thread.  Attempts to
// restore the original affinity mask on destruction.
//
// REQUIRES: For test-use only.  Do not use this in production code.
class ScopedAffinityMask {
 public:
  // When racing with an external restriction that has a zero-intersection with
  // "allowed_cpus" we will construct, but immediately register as "Tampered()",
  // without actual changes to affinity.
  explicit ScopedAffinityMask(absl::Span<int> allowed_cpus);
  explicit ScopedAffinityMask(int allowed_cpu);

  // Restores original affinity iff our scoped affinity has not been externally
  // modified (i.e. Tampered()).  Otherwise, the updated affinity is preserved.
  ~ScopedAffinityMask();

  // Returns true if the affinity mask no longer matches what was set at point
  // of construction.
  //
  // Note:  This is instantaneous and not fool-proof.  It's possible for an
  // external affinity modification to subsequently align with our originally
  // specified "allowed_cpus".  In this case Tampered() will return false when
  // time may have been spent executing previously on non-specified cpus.
  bool Tampered();

 private:
  cpu_set_t original_cpus_, specified_cpus_;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_INTERNAL_UTIL_H_
