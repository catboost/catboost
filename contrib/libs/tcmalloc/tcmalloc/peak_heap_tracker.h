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

#ifndef TCMALLOC_PEAK_HEAP_TRACKER_H_
#define TCMALLOC_PEAK_HEAP_TRACKER_H_

#include "absl/base/thread_annotations.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/atomic_stats_counter.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/malloc_extension.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

class PeakHeapTracker {
 public:
  constexpr PeakHeapTracker() : peak_sampled_span_stacks_(nullptr) {}

  // Possibly save high-water-mark allocation stack traces for peak-heap
  // profile. Should be called immediately after sampling an allocation. If
  // the heap has grown by a sufficient amount since the last high-water-mark,
  // it will save a copy of the sample profile.
  void MaybeSaveSample() ABSL_LOCKS_EXCLUDED(pageheap_lock);

  // Return the saved high-water-mark heap profile, if any.
  std::unique_ptr<ProfileBase> DumpSample() const
      ABSL_LOCKS_EXCLUDED(pageheap_lock);

  size_t CurrentPeakSize() const { return peak_sampled_heap_size_.value(); }

 private:
  // Linked list of stack traces from sampled allocations saved (from
  // sampled_objects_ above) when we allocate memory from the system. The
  // linked list pointer is stored in StackTrace::stack[kMaxStackDepth-1].
  StackTrace* peak_sampled_span_stacks_;

  // Sampled heap size last time peak_sampled_span_stacks_ was saved. Only
  // written under pageheap_lock; may be read without it.
  StatsCounter peak_sampled_heap_size_;

  bool IsNewPeak();
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_PEAK_HEAP_TRACKER_H_
