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
//
// Utility class for coalescing sampled stack traces.  Not thread-safe.

#ifndef TCMALLOC_STACK_TRACE_TABLE_H_
#define TCMALLOC_STACK_TRACE_TABLE_H_

#include <stdint.h>

#include <string>

#include "absl/base/thread_annotations.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal_malloc_extension.h"
#include "tcmalloc/malloc_extension.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

class StackTraceTable final : public ProfileBase {
 public:
  // If merge is true, traces with identical size and stack are merged
  // together.  Else they are kept distinct.
  // If unsample is true, Iterate() will scale counts to report estimates
  // of the true total assuming traces were added by the sampler.
  // REQUIRES: L < pageheap_lock
  StackTraceTable(ProfileType type, int64_t period, bool merge, bool unsample);

  // REQUIRES: L < pageheap_lock
  ~StackTraceTable() override;

  // base::Profile methods.
  void Iterate(
      absl::FunctionRef<void(const Profile::Sample&)> func) const override;

  int64_t Period() const override { return period_; }

  ProfileType Type() const override { return type_; }

  // Adds stack trace "t" to table with the specified count.
  // The count is a floating point value to reduce rounding
  // errors when accounting for sampling probabilities.
  void AddTrace(double count, const StackTrace& t)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Exposed for PageHeapAllocator
  struct Bucket {
    // Key
    uintptr_t hash;
    StackTrace trace;

    // Payload
    double count;
    size_t total_weight;
    Bucket* next;

    bool KeyEqual(uintptr_t h, const StackTrace& t) const;
  };

  // For testing
  int depth_total() const { return depth_total_; }
  int bucket_total() const { return bucket_total_; }

 private:
  static constexpr int kHashTableSize = 1 << 14;  // => table_ is 128k

  ProfileType type_;
  int64_t period_;
  int bucket_mask_;
  int depth_total_;
  Bucket** table_;
  int bucket_total_;
  bool merge_;
  bool error_;
  bool unsample_;

  int num_buckets() const { return bucket_mask_ + 1; }
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_STACK_TRACE_TABLE_H_
