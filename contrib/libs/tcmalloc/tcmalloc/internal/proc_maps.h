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

#ifndef TCMALLOC_INTERNAL_PROC_MAPS_H_
#define TCMALLOC_INTERNAL_PROC_MAPS_H_

#include <limits.h>
#include <stdint.h>
#include <sys/types.h>

namespace tcmalloc {
namespace tcmalloc_internal {

// A ProcMapsIterator abstracts access to /proc/maps for a given process.
class ProcMapsIterator {
 public:
  struct Buffer {
    static constexpr size_t kBufSize = PATH_MAX + 1024;
    char buf[kBufSize];
  };

  // Create a new iterator for the specified pid.  pid can be 0 for "self".
  explicit ProcMapsIterator(pid_t pid);

  // Create an iterator with specified storage (for use in signal handler).
  // "buffer" should point to a ProcMapsIterator::Buffer buffer can be null in
  // which case a buffer will be allocated.
  ProcMapsIterator(pid_t pid, Buffer* buffer);

  // Returns true if the iterator successfully initialized;
  bool Valid() const;

  bool NextExt(uint64_t* start, uint64_t* end, char** flags, uint64_t* offset,
               int64_t* inode, char** filename, dev_t* dev);

  ~ProcMapsIterator();

 private:
  void Init(pid_t pid, Buffer* buffer);

  char* ibuf_;      // input buffer
  char* stext_;     // start of text
  char* etext_;     // end of text
  char* nextline_;  // start of next line
  char* ebuf_;      // end of buffer (1 char for a nul)
  int fd_;          // filehandle on /proc/*/maps
  pid_t pid_;
  char flags_[10];
  Buffer* dynamic_buffer_;  // dynamically-allocated Buffer
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc

#endif  // TCMALLOC_INTERNAL_PROC_MAPS_H_
