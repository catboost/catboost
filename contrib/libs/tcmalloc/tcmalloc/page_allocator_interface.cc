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

#include "tcmalloc/page_allocator_interface.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tcmalloc/internal/environment.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/internal/util.h"
#include "tcmalloc/static_vars.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

static int OpenLog(MemoryTag tag) {
  const char *fname = [&]() {
    switch (tag) {
      case MemoryTag::kNormal:
        return thread_safe_getenv("TCMALLOC_PAGE_LOG_FILE");
      case MemoryTag::kNormalP1:
        return thread_safe_getenv("TCMALLOC_PAGE_LOG_FILE_P1");
      case MemoryTag::kSampled:
        return thread_safe_getenv("TCMALLOC_SAMPLED_PAGE_LOG_FILE");
      default:
        ASSUME(false);
        __builtin_unreachable();
    }
  }();

  if (ABSL_PREDICT_TRUE(!fname)) return -1;

  if (getuid() != geteuid() || getgid() != getegid()) {
    Log(kLog, __FILE__, __LINE__, "Cannot take a pagetrace from setuid binary");
    return -1;
  }
  char buf[PATH_MAX];
  // Tag file with PID - handles forking children much better.
  int pid = getpid();
  // Blaze tests can output here for recovery of the output file
  const char *test_dir = thread_safe_getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (test_dir) {
    snprintf(buf, sizeof(buf), "%s/%s.%d", test_dir, fname, pid);
  } else {
    snprintf(buf, sizeof(buf), "%s.%d", fname, pid);
  }
  int fd =
      signal_safe_open(buf, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);

  if (fd < 0) {
    Crash(kCrash, __FILE__, __LINE__, fd, errno, fname);
  }

  return fd;
}

PageAllocatorInterface::PageAllocatorInterface(const char *label, MemoryTag tag)
    : PageAllocatorInterface(label, &Static::pagemap(), tag) {}

PageAllocatorInterface::PageAllocatorInterface(const char *label, PageMap *map,
                                               MemoryTag tag)
    : info_(label, OpenLog(tag)), pagemap_(map), tag_(tag) {}

PageAllocatorInterface::~PageAllocatorInterface() {
  // This is part of tcmalloc statics - they must be immortal.
  Crash(kCrash, __FILE__, __LINE__, "should never destroy this");
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
